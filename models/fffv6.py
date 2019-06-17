from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import interface
from utils import *
import attention

def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return attention.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = attention.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = attention.linear(hidden, output_size, True, True)

        return output

def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size
    src_seq = features["label"]
    src_len = features["label_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["label"])[1],
                                dtype=tf.float32)

    src_vocab_size = 10000
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = attention.add_timing_signal(encoder_input)
    enc_attn_bias = attention.attention_bias(src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)

    return encoder_output

def feature_net(params, mode, feature, idx=0):
    outputs = feature
    outputs_list = []
    with tf.variable_scope("feature_extrator_%d" % idx):
        for layer_id in range(params.base_layer):
                if params.dropout > 0.:
                    outputs = tf.layers.dropout(outputs, rate=params.dropout, training=(mode == "train"))
                outputs_pool =  tf.layers.conv1d(
                    outputs, 512, kernel_size=3, strides=1, padding='same', use_bias=False)
                outputs_pool = group_norm(params.batch_size, outputs_pool)

                outputs = tf.layers.conv1d(
                    outputs, 512, kernel_size=3, strides=1, padding='same', use_bias=False)
                outputs = group_norm(params.batch_size, outputs)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.conv1d(
                    outputs, 512, kernel_size=3, strides=1, padding='same', use_bias=False)
                outputs = group_norm(params.batch_size, outputs)
                outputs = outputs + outputs_pool
                outputs = tf.nn.relu(outputs)

        for layer_id in range(params.anchor_layers):
                outputs_pool =  tf.layers.conv1d(
                    outputs, 256, kernel_size=3, strides=2, padding='same', use_bias=False)
                outputs_pool = group_norm(params.batch_size, outputs_pool)

                outputs = tf.layers.conv1d(
                    outputs, 256, kernel_size=3, strides=2, padding='same', use_bias=False)
                outputs = group_norm(params.batch_size, outputs)
                outputs = tf.nn.relu(outputs)
                outputs = tf.layers.conv1d(
                    outputs, 256, kernel_size=3, strides=1, padding='same', use_bias=False)
                outputs = group_norm(params.batch_size, outputs)
                outputs = outputs + outputs_pool
                outputs = tf.nn.relu(outputs)

                outputs_f = tf.layers.conv1d(
                    outputs, 256, kernel_size=3, strides=1, padding='same', use_bias=False)
                outputs_f = group_norm(params.batch_size, outputs_f)
                outputs_f = tf.nn.relu(outputs_f)

                outputs_list.append(outputs_f)
    return outputs_list

def model_graph(features, mode, params):

    feature = features["feature"]
    feature = tf.reshape(feature, [tf.shape(feature)[0], tf.shape(feature)[1], params.embedding_size])
    timestamps = features["timestamps"]
    duration = features['duration']
    timestamps_length = features["timestamps_length"]

    caption_feature = encoding_graph(features, mode, params)
    caption_feature = tf.layers.conv1d(caption_feature, 512, kernel_size=1, strides=1, padding='same', use_bias=False)
    caption_feature = tf.reduce_mean(caption_feature, axis=1, keepdims=True)
    caption_feature = tf.nn.sigmoid(caption_feature)

    feature_list = tf.split(feature, num_or_size_splits=params.multiplier,axis=-1)
    
    with tf.variable_scope("encoder"):
        outputs_list_list = []
        for i in range(len(feature_list)):
            outputs_list = feature_net(params, mode, feature_list[i], i)
            outputs_list_list.append(outputs_list)

        for i in range(len(outputs_list_list)):
            if i > 0:
                assert len(outputs_list_list[i]) == len(outputs_list_list[i - 1])

        outputs_list = []
        outputs_deconv_list = []
        outputs_foconv_list = []
        for j in range(len(outputs_list_list[0])):
            outputs_layer_list = []
            for i in range(len(outputs_list_list)):
                outputs_layer_list.append(outputs_list_list[i][j])
            outputs_list.append(tf.concat(outputs_layer_list, axis=-1))

        for i in range(len(outputs_list)):
            outputs = caption_feature * outputs_list[i]
            outputs = tf.layers.conv1d(
                outputs, 256, kernel_size=3, strides=1, padding='same', use_bias=False)
            outputs = group_norm(params.batch_size, outputs)
            outputs = tf.nn.relu(outputs)
            
            outputs_list[i] = outputs

        if params.deconv:
            outputs_num = len(outputs_list)
            for layer_id in range(outputs_num - 1, 0, -1):
                outputs = tf.expand_dims(outputs_list[layer_id], 1)
                if len(outputs_deconv_list) > 0:
                    outputs = tf.concat([outputs, tf.expand_dims(outputs_deconv_list[0], 1)], axis=-1)
                outputs_pool = tf.layers.conv2d_transpose(outputs, 256, [1,3], [1,2], 'same', use_bias=False)
                outputs_pool = group_norm(params.batch_size, outputs_pool)
                outputs_pool = tf.squeeze(outputs_pool, axis=1)
                outputs_deconv_list.insert(0, outputs_pool)
            outputs_deconv_list.append(0)
        
    back_event, proposal = get_proposal(params, outputs_list, outputs_deconv_list, outputs_foconv_list)
    duration_exp = tf.reshape(duration, [-1, 1, 1])
    timestamps = timestamps / duration_exp

    probability = tf.nn.softmax(back_event)[:, :, 1]
    #proposal = proposal * duration_exp

    prob_top_k, proposal_top_k = choose_top(params, probability, proposal, 100)


    if mode == "infer":
        # Prediction
        return prob_top_k, proposal_top_k

    auc = get_acc(params, proposal_top_k, timestamps)
    if mode == "eval":
        proposal_top_k = proposal_top_k * duration_exp
        return auc, prob_top_k, proposal_top_k

    with tf.variable_scope("loss"):
        loss_dict = {}
        if params.reeval_prop:
            tiou = tIoU(proposal_origin, timestamps)
        else:
            tiou = tIoU(proposal, timestamps)
        
        loss = 0
        if params.timestamps_eucloss or params.debug_all:
            loss_dict['eucloss'] = poposal_thresh_shoot_euclidean_loss(params, tiou, proposal, timestamps) * params.eucloss_ratio
            loss += loss_dict['eucloss']
        if params.timestamps_crosslossv2 or params.debug_all:
            loss_dict['crossloss_plus'], loss_dict['crossloss_minus'] = porbability_thresh_shoot_softmax_crossentropy_loss(params, tiou, back_event)
            loss += loss_dict['crossloss_plus'] + loss_dict['crossloss_minus']

        return loss, tf.reduce_sum(auc), tf.cast(tf.reduce_sum(timestamps_length), tf.float32), loss_dict


class FFF(interface.NMTModel):

    def __init__(self, params, scope="fff"):
        super(FFF, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                loss, auc, tl, loss_dict = model_graph(features, "train", params)
                self.loss_dict = loss_dict
                return loss, auc, tl, loss_dict

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

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
        return "fff"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # model
            ratio=0.3,
            time_signal=False,
            embedding_size=1000,
            base_layer=2,
            anchor_layers=10,
            batch_size=1,
            multiplier=2,
            start_layer=0,
            end_layer=10,
            # regularization
            dropout=0.0,
            K=16,
            top=10,
            tiou_thresh=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            timestamps_eucloss=False,
            timestamps_crossloss=False,
            timestamps_crosslossv2=False,
            eucloss_ratio=10.,
            debug_all=False,
            deconv=False,
            foconv=False,
            proposalv0=False,
            tiou_loss=False,
            reeval_prob=False,
            reeval_prop=False,
            reeval_prop_length=8,
            reeval_prop_ratio=0.1,
            localize=True,
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
        )

        return params