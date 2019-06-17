from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import six
import sys

import tensorflow as tf
import numpy as np
import json
import dataset
import math

class ProposalEvaluationHook(tf.train.SessionRunHook):
    def __init__(self, eval_fn, eval_data_dir, session_config, validation_variable, params):
        self._session_config = session_config
        self._params = params
        self._eval_fn = eval_fn
        self._eval_data_dir = eval_data_dir
        self._global_step = None
        self._base_dir = params.output
        self._validation_auc = 0
        self._assign_op = tf.assign(
            validation_variable, 
            tf.py_func(
                lambda :np.float32(self._validation_auc),
                [],
                [tf.float32]
            )[0]
        )
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=None, every_steps=params.eval_steps or None
        )
        self._max_auc = 0.
        self._max_auc5 = 0.
        self._max_auc7 = 0.
        self._max_auc57 = 0.

        with open(self._eval_data_dir[1], 'r',encoding='utf-8') as f:
            caption = json.load(f)
        self._vocabulary = caption['vocabulary']

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.eval_input = dataset.get_train_eval_input(self._eval_data_dir[0], self._eval_data_dir[1], False, self._params)
            with tf.device('/gpu:%d' % self._params.gpu):
                self.auc, self.prob_top, self.prop_top, self.idx = self._eval_fn(self.eval_input, self._params)
            sess_creator = tf.train.ChiefSessionCreator(
                config=self._session_config
            )
            self.saver = tf.train.Saver()
            self.sess = tf.train.MonitoredSession(session_creator=sess_creator)
            self.val_num = dataset.val_label_num

    def run_eval(self, save_path):   
        auc_reduce = 0.
        auc_reduce5 = 0.
        auc_reduce7 = 0.
        auc_reduce57 = 0.
        length_reduce = 0.
        step = 0
        results = {}
        check_check = {}
        caculate = [0 for i in range(self._params.start_layer, self._params.end_layer)]
        comfusion = [[0 for i in range(self._params.start_layer, self._params.end_layer)] for i in range(self._params.start_layer, self._params.end_layer)]
        stride = []
        for i in range(self._params.start_layer, self._params.end_layer):
            if i == self._params.start_layer:
                stride.append(2 ** (self._params.anchor_layers - 1 - i))
            else:
                stride.append(stride[-1] + 2 ** (self._params.anchor_layers - 1 - i))
        self.saver.restore(self.sess, save_path)
        validation_iteration = self.val_num // self._params.batch_size
        last_batch = self.val_num - validation_iteration * self._params.batch_size
        if last_batch == 0:
            last_batch = self._params.batch_size
        else:
            validation_iteration += 1
        for iteration in range(validation_iteration):
            re, tl, ts, sen, sl, pt, bt, name, tid, du = self.sess.run(
                [self.auc, self.eval_input["timestamps_length"], self.eval_input["timestamps"], self.eval_input["label"], 
                self.eval_input["label_length"], self.prob_top, self.prop_top, self.eval_input["filename"], self.idx, self.eval_input["duration"]])
            output_list = [re, tl, ts, sen, sl, pt, bt, name, tid, du]
            if iteration == validation_iteration - 1:
                for op in output_list:
                    op = op[:last_batch]

            re5 = re[:,1]
            re57 = re[:,3]
            re7 = re[:,2]
            re = re[:,0]
            for i in range(name.shape[0]):
                name_i = name[i].decode(encoding='UTF-8',errors='strict')
                pt_i = pt[i]
                bt_i = bt[i]
                sen_i = sen[i]
                sl_i = sl[i]
                ts_i = ts[i, 0, :].tolist()
                sentence = []

                for idx_word in range(sl_i):
                    sentence.append(self._vocabulary[sen_i[idx_word]])
                sentence = " ".join(sentence)
                if name_i[:2] == 'v_':
                    name_i = name_i[2:]

                results[str((name_i, sentence, ts_i[0], ts_i[1]))] = []
                assert bt_i.shape[0] == pt_i.shape[0]
                for j in range(min(5, bt_i.shape[0])):
                    results[str((name_i, sentence, ts_i[0], ts_i[1]))].append({'score': float(pt_i[j]), 'segment': bt_i[j].tolist()})
                recall_i = re[i] / tl[i]
                check_check[str((name_i, sentence, ts_i[0], ts_i[1]))] = recall_i

                tid_i = tid[i]
                du_i = du[i]
                layer = tid_i // len(self._params.anchor)
                comfusion_id = self._params.end_layer - 1 - min(
                    max(int(math.log2(du_i / (ts_i[1] - ts_i[0]))), self._params.start_layer), 
                    self._params.end_layer)
                for j in range(len(caculate)):
                    if layer < stride[j]:
                        caculate[j] += 1
                        comfusion[j][comfusion_id] += 1
                        break
            step += 1
            auc_reduce += re.sum()
            auc_reduce5 += re5.sum()
            auc_reduce7 += re7.sum()
            auc_reduce57 += re57.sum()
            length_reduce += float(tl.sum())

        validation_auc = auc_reduce / length_reduce
        validation_auc5 = auc_reduce5 / length_reduce
        validation_auc7 = auc_reduce7 / length_reduce
        validation_auc57 = auc_reduce57 / length_reduce

        if self._max_auc < validation_auc:
            self._max_auc = validation_auc
            self._max_auc5 = validation_auc5
        if self._max_auc7 < validation_auc7:
            self._max_auc7 = validation_auc7
            self._max_auc57 = validation_auc57
        if self._max_auc < validation_auc:
            results_json_true = {}
            results_json_true['results'] = {}
            results_json_true['recall'] = {}
            results_json_false = {}
            results_json_false['results'] = {}
            results_json_false['recall'] = {}
            for k in check_check.keys():
                if check_check[k] > 0.5:
                    results_json_true['results'][k] = results[k]
                    results_json_true['recall'][k] = check_check[k]
                else:
                    results_json_false['results'][k] = results[k]
                    results_json_false['recall'][k] = check_check[k]
            with open(os.path.join(self._params.output, self._params.log_id + '_true.json'), 'w') as f:
                json.dump(results_json_true, f)
            with open(os.path.join(self._params.output, self._params.log_id + '_false.json'), 'w') as f:
                json.dump(results_json_false, f)

        tf.logging.info("Total step: %d, Total Proposal: %d, Validating AUC: %f - %f - %f - %f, Maximum AUC: %f - %f - %f - %f" % (
            step, int(length_reduce), validation_auc, validation_auc7, validation_auc5, validation_auc57, 
            self._max_auc, self._max_auc7, self._max_auc5, self._max_auc57))
        tf.logging.info("Layer Caculate" + str(caculate))
        for i in range(len(comfusion)):
            tf.logging.info("Comfusion Layer Caculate %d:%s" %(i, str(comfusion[i])))
        return validation_auc


    def begin(self):
        if self._timer.last_triggered_step() is None:
            self._timer.update_last_triggered_step(0)
        global_step = tf.train.get_global_step()
        if global_step is None:
            raise RuntimeError("Global step should be created first")
        self._global_step = global_step

    def before_run(self, run_context):
        args = tf.train.SessionRunArgs(self._global_step)
        return args

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results

        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step)

            # Get the real value
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                save_path = os.path.join(self._base_dir, "model.ckpt")
                saver = tf.get_collection(tf.GraphKeys.SAVERS)[0]
                save_path = saver.save(run_context.session, save_path, global_step=global_step)
                tf.logging.info("Saving checkpoints for %d into %s." % (global_step, save_path))
                # Do validation here
                tf.logging.info("Validating model at step %d" % global_step)
                self._validation_auc = self.run_eval(save_path)
                run_context.session.run(self._assign_op)