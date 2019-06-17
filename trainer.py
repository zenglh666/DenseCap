from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import six
import sys
import json
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
import dataset
import models
from validation import ProposalEvaluationHook


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )
    parser.add_argument("--job_id", type=str, default="",
                        help="id of job")

    # input files
    parser.add_argument("--output", type=str, default="",
                        help="Path to saved models")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset to use")

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        task="caption",
        dataset="",
        input1="D:\\v-lizen\\activitynet\\activitynet_v1-3.c3d.hdf5",
        input2="D:\\v-lizen\\activitynet\\p3d_rgb_feature_per_8f_pca.h5", 
        input3="D:\\v-lizen\\activitynet\\p3d_flow_feature_per_8f_v2_pca.h5",
        input4="D:\\v-lizen\\activitynet\\inception_v4_feature_per_8f_avg_v2_pca.h5",
        input5="D:\\v-lizen\\activitynet\\inception_v4_feature_per_8f_avg_v2_pca_p3d_flow_feature_per_8f_v2_pca.h5",
        input6="D:\\v-lizen\\charades\\charades_p3d_rgb_feature_per_8f_v2_pca.h5",
        input7="D:\\v-lizen\\didemo\\didemo_p3d_rgb_feature_per_8f_v3_pca.h5",
        input8="D:\\v-lizen\\activitynet\\p3d_rgb_feature_per_8f_pca_p3d_flow_feature_per_8f_v2_pca.h5",
        input_reference1="D:\\v-lizen\\charades\\train.json",
        input_reference2="D:\\v-lizen\\activitynet\\train.json",
        input_reference3="D:\\v-lizen\\didemo\\train.json",
        input_reference="",
        bert_file = "",
        bert_file1 = "D:\\v-lizen\\charades\\charades_bert.h5",
        bert_file2 = "D:\\v-lizen\\activitynet\\activitynet_bert.h5",
        bert_file3 = "D:\\v-lizen\\didemo\\didemo_bert.h5",
        bert_file4 = "D:\\v-lizen\\activitynet\\activitynet_skip.h5",
        output="D:\\v-lizen\\DenseCap\\results",
        model="",
        job_id = "",
        log_id= "",
        # Default training hyper parameters
        pre_fetch=32,
        warmup_steps=4000,
        train_steps=200000,
        buffer_size=1024,
        gpu=0,
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        scale_l1=0.0,
        scale_l2=0.0,
        optimizer="Sgd",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        clip_grad_norm=0.,
        learning_rate=1e-2,
        learning_rate_decay="exponential_decay",
        learning_rate_boundaries=[100000, 200000],
        learning_rate_values=[1e-5, 1e-6, 1e-7],
        decay_steps=100000,
        keep_checkpoint_max=1,
        keep_top_checkpoint_max=1,
        # Validation
        eval_steps=10000,
        eval_secs=0,
        validation_reference1="D:\\v-lizen\\charades\\val.json",
        validation_reference2="D:\\v-lizen\\activitynet\\val.json",
        validation_reference3="D:\\v-lizen\\didemo\\val.json",
        validation_reference="",
        save_checkpoint_secs=0,
        save_checkpoint_steps=0,
        # Setting this to True can save disk spaces, but cannot restore
        # training using the saved checkpoint
        only_save_trainable=False,
        restore_params=False,
        infer_in_validation=True,
        random_tile=1,
        random_input=False,
        random_train=False,
        random_val=False,
        k2scale=1024,
        word_length=128,
        timestamps_weights=False,
        proposal_sum=1,
        box_regress=False,
        tf_resize=True,
        hard_rule=False,
        localize=False,
        use_bert=True,
        preload=False,
        use_skip=False,
        earlier_fusion=False,
        one_scale=False,
    )
    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in params.values().keys():
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().items():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().items():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_parameters(params, args):
    params.model = args.model
    params.dataset = args.dataset
    params.output = args.output or params.output
    params.job_id = args.job_id or params.job_id
    params.parse(args.parameters)
    timestr =  datetime.now().isoformat().replace(':','-').replace('.','MS')
    if params.log_id == "":
        params.log_id = timestr
    if params.job_id == "":
        params.job_id = timestr
    params.output = os.path.join(params.output, params.job_id)
    if params.task == 'charades':
        params.input_reference = params.input_reference1
        params.validation_reference = params.validation_reference1
        params.bert_file = params.bert_file1
    elif params.task == 'caption':
        params.input_reference = params.input_reference2
        params.validation_reference = params.validation_reference2
        params.bert_file = params.bert_file2
        if params.use_skip:
            params.bert_file = params.bert_file4
    elif params.task == 'didemo':
        params.input_reference = params.input_reference3
        params.validation_reference = params.validation_reference3
        params.bert_file = params.bert_file3
    else:
        raise(ValueError)

    if params.dataset == "c3d":
        params.input = params.input1
    elif params.dataset == "p3d":
        params.input = params.input2
    elif params.dataset == "p3d_flow":
        params.input = params.input3
    elif params.dataset == "inc":
        params.input = params.input4
    elif params.dataset == "inc_flow":
        params.embedding_size = 1000
        params.input = params.input5
    elif params.dataset == "p3d_cha":
        params.input = params.input6
    elif params.dataset == "p3d_did":
        params.input = params.input7
    elif params.dataset == "p3d_rgb_flow":
        params.embedding_size = 1000
        params.input = params.input8
    else:
        raise ValueError

    if params.one_scale:
        anchor_base = 2 ** (params.anchor_layers - params.end_layer)
        params.anchor=[np.power(anchor_base, 1./3), np.power(anchor_base, 2./3), anchor_base]
    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "exponential_decay":
        return tf.train.exponential_decay(params.learning_rate, global_step, params.decay_steps, decay_rate=0.1, staircase=True)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    gpu_options=tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options,
                            gpu_options=gpu_options)
    device_str = str(params.gpu)
    config.gpu_options.visible_device_list = device_str

    return config


def restore_variables(checkpoint):
    if not checkpoint:
        return tf.no_op("restore_op")

    # Load checkpoints
    tf.logging.info("Loading %s" % checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    values = {}

    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]

        if name in values:
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))

    return tf.group(*ops, name="restore_op")


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    if params.restore_params:
    	params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    export_params(params.output, "params.json", params)
    export_params(
        params.output,
        "%s.json" % args.model,
        collect_params(params, model_cls.get_parameters())
    )

    log = logging.getLogger('tensorflow')
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(message)s')
    fh = logging.FileHandler(os.path.join(params.output, params.log_id + '.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Build Graph
    with tf.Graph().as_default():
        features = dataset.get_train_eval_input(params.input, params.input_reference, True, params)


        # Build model
        initializer = get_initializer(params)
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params.scale_l1, scale_l2=params.scale_l2)
        params.regularizer = regularizer
        model = model_cls(params)
        # Create global step
        global_step = tf.train.get_or_create_global_step()

        # Create optimizer
        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        tf.summary.scalar("learning_rate", learning_rate)

        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate,
                                         beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "LazyAdam":
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2,
                                                   epsilon=params.adam_epsilon)
        elif params.optimizer == "Mom":
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif params.optimizer == "Sgd":
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise RuntimeError("Optimizer %s not supported" % params.optimizer)

        if params.clip_grad_norm > 0.:
            opt = tf.contrib.estimator.clip_gradients_by_norm(opt, params.clip_grad_norm)

        # Get train loss
        with tf.device('/gpu:%d' % params.gpu):
            loss_func = model.get_training_func(initializer, regularizer)
            loss, auc, tl, losses_dict = loss_func(features, params)
            losses_dict['regloss'] = tf.losses.get_regularization_loss()
            loss = loss + losses_dict['regloss']
            ops = opt.minimize(loss, global_step)

        losses_collect = tf.losses.get_losses()

        ema = tf.train.ExponentialMovingAverage(0.997, global_step, name='average')
        ema_op = ema.apply([loss] + [v for v in losses_dict.values()] + [auc, tl])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

        loss_avg = ema.average(loss)
        losses_dict_avg = {}
        for k, v in losses_dict.items():
            losses_dict_avg[k] = ema.average(v)
        auc_avg = ema.average(auc)
        tl_avg = ema.average(tl)
    
        updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies([ops]):
            ops = tf.group(*updates_collection)

        # Print parameters
        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0

        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
            v_size = np.prod(np.array(v.shape.as_list())).tolist()
            total_size += v_size
        tf.logging.info("Total trainable variables size: %d", total_size)


        logging_dict = {
            "step": global_step,
            "loss": loss_avg,
            "train_auc": auc_avg / tl_avg
        }
        logging_dict.update(losses_dict_avg)

        for k,v in logging_dict.items():
            if len(v.get_shape().as_list()) == 0:
                print("Summary %s as %s" % (v.op.name, k))
                tf.summary.scalar(k, v)

        validation_auc = tf.get_variable(shape=[], initializer=tf.zeros_initializer(),name="val_auc")
        tf.summary.scalar("val_auc", validation_auc)
        print("Summary %s as %s" % (validation_auc.op.name, "val_auc"))



        # Add hooks
        save_vars = tf.trainable_variables() + [global_step]
        saver = tf.train.Saver(
            var_list=save_vars if params.only_save_trainable else None,
            max_to_keep=params.keep_checkpoint_max,
            sharded=False
        )
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

        restore_op = restore_variables(args.checkpoint)

        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                logging_dict,
                every_n_iter=100
            ),
            #tf.train.CheckpointSaverHook(
            #    checkpoint_dir=params.output,
            #    save_secs=params.save_checkpoint_secs or None,
            #    save_steps=params.save_checkpoint_steps or None,
            #    saver=saver
            #)
        ]

        # Validation
        config = session_config(params)
        eval_inputs = [params.input, params.validation_reference]

        train_hooks.append(
            ProposalEvaluationHook(
                model.get_evaluation_func(),
                eval_inputs,
                config,
                validation_auc,
                params,
            )
        )

        def restore_fn(step_context):
            step_context.session.run(restore_op)

        def step_fn(step_context):
            # Bypass hook calls
            return step_context.run_with_hooks(ops)

        # Create session, do not use default CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            # Restore pre-trained variables
            sess.run_step_fn(restore_fn)

            while not sess.should_stop():
                sess.run_step_fn(step_fn)


if __name__ == "__main__":
    main(parse_args())
