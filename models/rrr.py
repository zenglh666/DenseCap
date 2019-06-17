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


def _residual_fn(x, y, keep_prob=None, reverse=False):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    if reverse:
        return tf.concat([x - y, y],axis=-1)
    else:
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

        return x

def transformer_decoder(inputs, memory, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_decoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
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
                    y, z = _ffn_layer_visual(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        return x

def _ffn_layer_visual(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None, stride=1):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = tf.layers.conv1d(
                    inputs, hidden_size, kernel_size=3, strides=stride, padding='same', use_bias=True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = tf.layers.conv1d(
                    hidden, output_size, kernel_size=3, strides=1, padding='same', use_bias=True)

        return output, hidden


def encoding_graph(src_seq, src_len, mode, params):
    hidden_size = params.hidden_size
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(src_seq)[1],
                                dtype=tf.float32)

    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [params.src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [params.src_vocab_size, hidden_size],
                                        initializer=initializer)

    #bias = tf.get_variable("bias", [hidden_size])

    encoder_input = tf.gather(src_embedding, src_seq)
    encoder_input = tf.nn.relu(encoder_input)

    if params.multiply_embedding_mode == "sqrt_depth":
        encoder_input = encoder_input * (hidden_size ** 0.5)

    encoder_input = encoder_input * tf.expand_dims(src_mask, -1)

    if params.relu_dropout:
        keep_prob = 1.0 - params.relu_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    #encoder_input = tf.nn.bias_add(encoder_input, bias)
    encoder_input = attention.add_timing_signal(encoder_input)
    enc_attn_bias = attention.attention_bias(src_mask, "masking")

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)
    encoder_output = encoder_output * tf.expand_dims(src_mask, -1)

    return encoder_output, enc_attn_bias

def encoding_graph_bert(src_seq, src_len, mode, params):
    hidden_size = params.hidden_size
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(src_seq)[1],
                                dtype=tf.float32)

    inputs = src_seq

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)
    encoder_input = tf.layers.conv1d(
        inputs, hidden_size, kernel_size=1, strides=1, padding='same', use_bias=True)
    encoder_input = tf.nn.relu(encoder_input)

    if params.relu_dropout:
        keep_prob = 1.0 - params.relu_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)
    
    #bias = tf.get_variable("bias", [hidden_size])
    #encoder_input = tf.nn.bias_add(encoder_input, bias)
    encoder_input = attention.add_timing_signal(encoder_input)
    enc_attn_bias = attention.attention_bias(src_mask, "masking")

    

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)
    encoder_output = encoder_output * tf.expand_dims(src_mask, -1)

    enc_attn_bias = attention.attention_bias(src_mask, "masking")

    return encoder_output, enc_attn_bias


def model_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0
        params.feature_dropout = 0.0
        params.label_dropout = 0.0

    feature = features["feature"]
    feature = tf.reshape(feature, [tf.shape(feature)[0], tf.shape(feature)[1], params.embedding_size])
    timestamps = features["timestamps"]
    duration = features['duration']
    timestamps_length = features["timestamps_length"]
    label = features["label"]
    label_length = features["label_length"]
    if params.use_bert:
        label = features["bert"]
        if params.use_skip:
            label = tf.reshape(label, [tf.shape(label)[0], tf.shape(label)[1], params.skip_size])
        else:
            label = tf.reshape(label, [tf.shape(label)[0], tf.shape(label)[1], params.bert_size])

    outputs_list = []
    forward_list = []
    outputs_deconv_list = []
    outputs_foconv_list = []
    outputs_d_list = []

    with tf.variable_scope("proposal"):
        feature = attention.add_timing_signal(feature)
        if params.feature_dropout:
            keep_prob = 1.0 - params.feature_dropout
            distribution = tf.distributions.Bernoulli(probs=keep_prob)
            if params.time_or_scalar_dropout:
                feature_mask = distribution.sample(tf.shape(feature))
            else:
                feature_mask = tf.expand_dims(distribution.sample(tf.shape(feature)[:2]), axis=-1)

            feature = feature * tf.cast(feature_mask, tf.float32)
        if params.label_dropout:
            keep_prob = 1.0 - params.label_dropout
            distribution = tf.distributions.Bernoulli(probs=keep_prob)
            if params.time_or_scalar_dropout:
                labels_mask = distribution.sample(tf.shape(label))
                if params.use_bert:
                    labels_mask = tf.cast(labels_mask, tf.float32)
            else:
                labels_mask = distribution.sample(tf.shape(label)[:2])
                if params.use_bert:
                    labels_mask = tf.expand_dims(tf.cast(labels_mask, tf.float32), axis=-1)
            label = label * labels_mask

        with tf.variable_scope("word_embedding"):
            if params.use_bert:
                caption_feature_base, enc_attn_bias = encoding_graph_bert(label, label_length, mode, params)
            else:
                caption_feature_base, enc_attn_bias = encoding_graph(label, label_length, mode, params)

        feature = tf.layers.conv1d(
            feature, params.hidden_size, kernel_size=1, strides=1, padding='same', use_bias=True)
        feature = tf.nn.relu(feature)
        if params.relu_dropout:
            keep_prob = 1.0 - params.relu_dropout
            feature = tf.nn.dropout(feature, keep_prob)
        foward = feature
        forward_list.append(feature)
        for layer_id in range(params.anchor_layers):
            with tf.variable_scope("layer_%d" % layer_id):
                with tf.variable_scope("input_feed_forward"):
                    x = foward
                    x = tf.layers.average_pooling1d(x, 3, 2, 'same')
                    foward = x
                    forward_list.append(x)
                    if not params.scale_enhance:
                        outputs_d_list.append(x)
                    else:
                        y, z = _ffn_layer_visual(
                            _layer_process(x, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                        )
                        x = _layer_process(y, params.layer_postprocess)
                        x = tf.nn.relu(x)
                        outputs_d_list.append(x)

                

        for layer_id in range(params.anchor_layers):
            with tf.variable_scope("query", reuse=False if layer_id==0 else True):
                x = outputs_d_list[layer_id]
                if params.skip_word:
                    memories = tf.reduce_mean(caption_feature_base, axis=1, keepdims=True)
                    memories_tile = tf.tile(memories, [1, tf.shape(x)[1], 1])
                    x = tf.concat([x, memories_tile], axis=-1)
                    x = tf.layers.conv1d(
                        x, params.hidden_size, kernel_size=1, strides=1, padding='same', use_bias=True)
                    x = tf.nn.relu(x)
                else:
                    x = transformer_decoder(x, caption_feature_base, enc_attn_bias, params)

                outputs = x
            outputs_list.append(outputs)
    
    if params.proposalv2:
        back_event, proposal = get_proposalv3(params, outputs_list, outputs_deconv_list, outputs_foconv_list)
    else:
        back_event, proposal = get_proposal(params, outputs_list, outputs_deconv_list, outputs_foconv_list)
    duration_exp = tf.reshape(duration, [-1, 1, 1])
    timestamps = timestamps / duration_exp

    probability = tf.nn.softmax(back_event)[:, :, 1]
    #proposal = proposal * duration_exp

    prob_top_k, proposal_top_k, idx = choose_top_with_idx(params, probability, proposal, 300)
    distribution = tf.distributions.Uniform()
    prob_random = distribution.sample(tf.shape(prob_top_k))
    proposal_random = distribution.sample(tf.shape(proposal_top_k))
    proposal_random = tf.concat([
        tf.reduce_min(proposal_random, axis=-1, keepdims=True),
        tf.reduce_max(proposal_random, axis=-1, keepdims=True)], axis=-1)
    prob_top_k = prob_random
    proposal_top_k = proposal_random


    if mode == "infer":
        # Prediction
        return prob_top_k, proposal_top_k

    if mode == "eval":
        auc_15 = get_acc_top1_top5(params, proposal_top_k, prob_top_k, timestamps)
        proposal_top_k = proposal_top_k * duration_exp
        return auc_15, prob_top_k, proposal_top_k, idx[:, 0]

    auc = get_acc(params, proposal_top_k, timestamps)

    with tf.variable_scope("loss"):
        loss_dict = {}
        tiou = tIoU(proposal, timestamps)
        
        loss = 0
        if params.timestamps_eucloss or params.debug_all:
            loss_dict['eucloss'] = poposal_thresh_shoot_euclidean_loss(params, tiou, proposal, timestamps) * params.eucloss_ratio
            loss += loss_dict['eucloss']
        if params.timestamps_crosslossv2 or params.debug_all:
            loss_dict['crossloss_plus'], loss_dict['crossloss_minus'] = porbability_thresh_shoot_softmax_crossentropy_loss(params, tiou, back_event)
            loss += loss_dict['crossloss_plus'] + loss_dict['crossloss_minus']

        if params.feature_reeval:
            x = tf.expand_dims(forward_list[-1], 1)
            feature_reeval_loss = 0.
            for i in range(params.anchor_layers, 0, -1):
                x = tf.expand_dims(forward_list[i], 1)
                x = tf.layers.conv2d_transpose(x, params.hidden_size, kernel_size=(1, 3),strides=(1, 2),padding='same')
                x = tf.nn.relu(x)
                y = tf.expand_dims(forward_list[i - 1], 1)
                feature_reeval_loss += tf.losses.mean_squared_error(predictions=x, labels=y)
            
            loss_dict['feature_reeval'] = feature_reeval_loss * params.eucloss_ratio
            loss += loss_dict['feature_reeval']

        return loss, tf.reduce_sum(auc), tf.cast(tf.reduce_sum(timestamps_length), tf.float32), loss_dict


class FFF(interface.NMTModel):

    def __init__(self, params, scope="fff"):
        super(FFF, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
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
            ratio=0.5,
            time_signal=False,
            bert_size=1024,
            skip_size=2400,
            embedding_size=500,
            base_layer=2,
            anchor_layers=10,
            batch_size=8,
            multiplier=1,
            start_layer=0,
            end_layer=10,
            src_vocab_size=10000,
            # regularization
            time_or_scalar_dropout=False,
            feature_dropout=0.0,
            label_dropout=0.0,
            K=16,
            top=10,
            tiou_thresh=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            anchor=[1., 1.25, 1.5],
            timestamps_eucloss=True,
            timestamps_crossloss=False,
            timestamps_crosslossv2=True,
            eucloss_ratio=10.,
            debug_all=False,
            deconv=False,
            foconv=False,
            proposalv2=True,
            tiou_loss=False,
            label_reeval=False,
            feature_reeval=False,
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
            num_encoder_layers=5,
            num_decoder_layers=5,
            attention_dropout=0.1,
            residual_dropout=0.1,
            relu_dropout=0.1,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            skip_word=False,
            skip_global=False,
            skip_attention=False,
            alpha=0.,
            beta=0.,
            scale_enhance=False,
        )

        return params
