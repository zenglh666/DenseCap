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
    def __init__(self, model, dataset, session_config, training_saver, params):
        super().__init__()
        self._params = params
        self._base_dir = params.output
        self._training_saver = training_saver
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=None, every_steps=params.eval_steps or None
        )
        self._max_scores = {}

        self.graph = tf.Graph()
        with self.graph.as_default():
            eval_fn = model.get_evaluation_func()
            eval_input = dataset.get_train_eval_input("validation")
            self.val_num = dataset.val_label_num

            with tf.device('/gpu:%d' % self._params.gpu):
                self.scores = eval_fn(eval_input, self._params)

            for k in self.scores.keys():
                self._max_scores[k] = 0.

            sess_creator = tf.train.ChiefSessionCreator(
                config=self._session_config
            )
            self.saver = tf.train.Saver()
            self.sess = tf.train.MonitoredSession(session_creator=sess_creator)

    def run_eval(self, save_path):   
        scores = {}
        for k in self._max_scores.keys():
            scores[k] = 0.
        self.saver.restore(self.sess, save_path)

        validation_iteration = self.val_num // self._params.batch_size
        last_batch = self.val_num - validation_iteration * self._params.batch_size
        validation_iteration += int(last_batch > 0)

        for iteration in range(validation_iteration):
            scores_iter = self.sess.run([self.scores])
            if iteration == validation_iteration - 1 and last_batch:
                for k in scores_iter.keys():
                    scores_iter[k] = scores_iter[k][:last_batch]

            for k in scores_iter.keys():
                scores[k] += scores_iter[k].sum()

        for k in scores_iter.keys():
            scores[k] /= self.val_num
            self._max_scores[k] = max(scores[k], self._max_scores[k])

        log_str = "Total Proposal: %d" % self.val_num
        log_str += " \nValidating ACC: "
        for (k, v) in scores.items():
            log_str += str(k) + " - " + str(v)
        log_str += " \nMaximum ACC: "
        for (k, v) in self._max_scores.items():
            log_str += str(k) + " - " + str(v)


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
                save_path = self._training_saver.save(
                    run_context.session, save_path, global_step=global_step)
                tf.logging.info("Saving checkpoints for %d into %s." % (global_step, save_path))
                # Do validation here
                tf.logging.info("Validating model at step %d" % global_step)
                self._validation_auc = self.run_eval(save_path)
                run_context.session.run(self._assign_op)