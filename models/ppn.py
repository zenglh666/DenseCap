import copy
import tensorflow as tf
import interface
from utils import *
from .mab import *

def conv_block(x, params, kernel_size, strides):
    x = tf.layers.conv1d(x, params.hidden_size, 
        kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)
    x = layer_process(x, params.layer_convprocess)
    x = tf.nn.relu(x)
    return x


def model_graph(features, mode, params):
    feature_visual = features["feature_visual"]
    feature_language = features["feature_language"]
    timestamps = features["timestamps"]
    duration = features['duration']
    language_length = features["language_length"]

    if params.feature_dropout:
        keep_prob = 1.0 - params.feature_dropout
        distribution = tf.distributions.Bernoulli(probs=keep_prob)
        feature_mask = tf.expand_dims(distribution.sample(tf.shape(feature_visual)[:-1]), axis=-1)
        feature_visual = feature_visual * tf.cast(feature_mask, tf.float32)
    if params.label_dropout:
        keep_prob = 1.0 - params.label_dropout
        distribution = tf.distributions.Bernoulli(probs=keep_prob)
        feature_mask = tf.expand_dims(distribution.sample(tf.shape(feature_language)[:-1]), axis=-1)
        feature_language = feature_language * tf.cast(feature_mask, tf.float32)

    with tf.variable_scope("word_embedding"):
        src_mask = tf.sequence_mask(
            language_length, maxlen=tf.shape(feature_language)[1], dtype=tf.float32)
        feature_language = conv_block(feature_language, params, kernel_size=1, strides=1)
        feature_language = tf.layers.dropout(feature_language, params.relu_dropout)
        feature_language = add_timing_signal(feature_language)
        enc_attn_bias = attention_bias(src_mask, "masking")
        feature_language = feature_language * tf.expand_dims(src_mask, -1)

    with tf.variable_scope("visual_embedding"):
        feature_visual = conv_block(feature_visual, params, kernel_size=1, strides=1)
        feature_visual = tf.layers.dropout(feature_visual, params.relu_dropout)
        feature_visual = add_timing_signal(feature_visual)
        outputs_d_list = []
        outputs_d_size_list = []
        forward = x = feature_visual
        for layer_id in range(params.anchor_layers):
            with tf.variable_scope("layer_%d" % layer_id):
                with tf.variable_scope("input_feed_forward"):
                    x = forward
                    x = conv_block(x, params, kernel_size=3, strides=2)
                    x = tf.layers.dropout(x, params.relu_dropout)
                    forward = x
                    x = conv_block(x, params, kernel_size=3, strides=1)
                    x = tf.layers.dropout(x, params.relu_dropout)
                    outputs_d_list.append(x)
                    outputs_d_size_list.append(tf.shape(x)[1])

    with tf.variable_scope("proposal"):
        output_d = tf.concat(outputs_d_list, axis=1)

        if params.num_mab < 1:
            x = output_d
            memories = tf.reduce_mean(feature_language, axis=1, keepdims=True)
            memories_tile = tf.tile(memories, [1, tf.shape(x)[1], 1])
            x = tf.concat([x, memories_tile], axis=-1)
            x = conv_block(x, params, kernel_size=1, strides=1)
            x = tf.layers.dropout(x, params.relu_dropout)
            output = x
        else:
            output = MAB(output_d, feature_language, enc_attn_bias, params)

        back_event, proposal = get_proposal(params, output, outputs_d_size_list)
        probability = tf.nn.softmax(back_event)[:, :, 1]

    timestamps = timestamps / tf.reshape(duration, [-1, 1])

    if mode == "infer":
        # Prediction
        return probability, proposal

    if mode == "eval":
        acc = get_acc_top1_top5(params, proposal, probability, timestamps)
        return acc

    with tf.variable_scope("loss"):
        acc = get_acc(params, proposal, probability, timestamps)
        loss_dict = {}
        loss_dict['eucloss'] = regress_loss(
            params, proposal, timestamps) * params.eucloss_ratio
        loss = loss_dict['eucloss']
        loss_dict['crossloss_plus'], loss_dict['crossloss_minus'] = class_loss(
            params, proposal, timestamps, back_event)
        loss += loss_dict['crossloss_plus'] + loss_dict['crossloss_minus']

        return loss, tf.reduce_mean(acc), loss_dict


class Model(interface.NMTModel):

    def __init__(self, params, scope="Model"):
        super(Model, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                loss, acc, loss_dict = model_graph(features, "train", params)
                loss_dict['regloss'] = tf.losses.get_regularization_loss()
                loss = loss + loss_dict['regloss']
                return loss, acc, loss_dict

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, reuse=reuse):
                scores = model_graph(features, "eval", params)

            return scores

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                logits = model_graph(features, "infer", params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "PPN"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # model
            language_size=1024,
            visual_size=500,
            anchor_layers=10,
            anchor=[1., 1.25, 1.5],
            hidden_size=256,
            filter_size=256,
            num_heads=8,
            num_mab=1,
            # regularization
            ratio=0.5,
            feature_dropout=0.1,
            label_dropout=0.1,
            eucloss_ratio=10.,
            attention_dropout=0.1,
            residual_dropout=0.1,
            relu_dropout=0.1,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            layer_convprocess="layer_norm",
            #optimizer="Adam",
            #learning_rate=0.001,
        )

        return params
