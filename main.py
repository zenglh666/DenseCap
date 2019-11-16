import argparse
import os
import json
import logging
import numpy as np
import tensorflow as tf
import models
from validation import ProposalEvaluationHook
from datetime import datetime
from dataset import Dataset


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

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        label_file="",
        visual_file="",
        language_file = "",
        output="",
        model="",
        job_id = "",
        log_id= "",
        # Default training hyper parameters
        pre_fetch=4,
        buffer_size=1024,
        batch_size=32,
        gpu=0,
        optimizer="Sgd",
        clip_grad_norm=0.,
        learning_rate_decay="exponential_decay",
        learning_rate=0.1,
        train_steps=120000,
        decay_steps=40000,
        eval_steps=2000,
        log_steps=100,
        eval_secs=0,
        
        # Initializer and Regularizer 
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        scale_l1=0.0,
        scale_l2=0.0,
        keep_checkpoint_max=1,
        keep_top_checkpoint_max=1,
        
        # Store and Restore
        save_checkpoint_secs=0,
        save_checkpoint_steps=0,
        restore_params=False,
        infer_in_validation=True,

        # Dataset Attribute
        word_length=64,
        k2scale=0,
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
    params.output = args.output or params.output
    params.job_id = args.job_id or params.job_id
    params.parse(args.parameters)
    timestr =  datetime.now().isoformat().replace(':','-').replace('.','MS')
    params.log_id = params.log_id or timestr
    params.job_id = params.job_id or timestr

    params.output = os.path.join(params.output, params.job_id)
    assert params.label_file
    assert params.visual_file
    assert params.language_file
    params.k2scale = params.k2scale or 2 ** params.anchor_layers
    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(
            params.initializer_gain, mode="fan_avg", distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(
            params.initializer_gain, mode="fan_avg", distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay == "exponential_decay":
        return tf.train.exponential_decay(
            params.learning_rate, global_step, params.decay_steps, 
            decay_rate=0.1, staircase=True)
    elif params.learning_rate_decay == "none":
        return tf.convert_to_tensor(learning_rate, dtype=tf.float32)
    else:
        raise ValueError("Unknown learning_rate_decay")

def get_optimizer(learning_rate, params):
    if params.optimizer == "Adam":
        opt = tf.train.AdamOptimizer(learning_rate)
    elif params.optimizer == "LazyAdam":
        opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate)
    elif params.optimizer == "Mom":
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    elif params.optimizer == "Sgd":
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise RuntimeError("Optimizer %s not supported" % params.optimizer)
    return opt

    if params.clip_grad_norm > 0.:
        opt = tf.contrib.estimator.clip_gradients_by_norm(opt, params.clip_grad_norm)

def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=str(params.gpu))
    config = tf.ConfigProto(
        allow_soft_placement=True, graph_options=graph_options, gpu_options=gpu_options)

    return config

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
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fh = logging.FileHandler(os.path.join(params.output, params.log_id + '.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    dataset = Dataset(params)
    config = session_config(params)
    model = model_cls(params)

    with tf.Graph().as_default():
        # Build Graph
        features = dataset.get_train_eval_input("train")

        # Build model
        initializer = get_initializer(params)
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params.scale_l1, scale_l2=params.scale_l2)
        
        # Create global step
        global_step = tf.train.get_or_create_global_step()

        # Create optimizer
        learning_rate = get_learning_rate_decay(params.learning_rate, global_step, params)
        tf.summary.scalar("learning_rate", learning_rate)
        opt = get_optimizer(learning_rate, params)

        # Get train loss
        with tf.device('/gpu:%d' % params.gpu):
            loss_func = model.get_training_func(initializer, regularizer)
            loss, metric, losses_dict = loss_func(features, params)
            log_values = [loss] + [v for v in losses_dict.values()] + [metric]
            ops = opt.minimize(loss, global_step)

        losses_collect = tf.losses.get_losses()

        ema = tf.train.ExponentialMovingAverage(0.997, global_step, name='average')
        ema_op = ema.apply(log_values)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

        loss_avg = ema.average(loss)
        losses_dict_avg = {}
        for k, v in losses_dict.items():
            losses_dict_avg[k] = ema.average(v)
        metric_avg = ema.average(metric)

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
            "train_metric": metric_avg
        }
        logging_dict.update(losses_dict_avg)

        for k,v in logging_dict.items():
            if len(v.get_shape().as_list()) == 0:
                print("Summary %s as %s" % (v.op.name, k))
                tf.summary.scalar(k, v)

        # Add hooks
        saver = tf.train.Saver(max_to_keep=1)
        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                logging_dict,
                every_n_iter=params.log_steps
            ),
            ProposalEvaluationHook(
                model,
                dataset,
                config,
                saver,
                params,
            )
        ]

        # Create session, do not use default CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=train_hooks, log_step_count_steps=None,
                save_checkpoint_secs=None, save_checkpoint_steps=None, 
                save_summaries_secs=None, save_summaries_steps=None,
                config=config) as sess:

            while not sess.should_stop():
                sess.run(ops)

if __name__ == "__main__":
    main(parse_args())
