from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import numpy as np


def add_timing_signal_1d_given_position(x,
                                        position,
                                        min_timescale=1.0,
                                        max_timescale=1.0e4):
    """Adds sinusoids of diff frequencies to a Tensor, with timing position given.
    Args:
        x: a Tensor with shape [batch, length, channels]
        position: a Tensor with shape [batch, length]
        min_timescale: a float
        max_timescale: a float
    Returns:
        a Tensor the same shape as x.
    """
    with tf.variable_scope("time_signal"):
        channels = tf.shape(x)[2]
        num_timescales = channels // 2
        log_timescale_increment = (
            tf.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = (
            tf.expand_dims(tf.to_float(position), 2) * tf.expand_dims(
                tf.expand_dims(inv_timescales, 0), 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
        signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
        return x + signal

def group_norm(batch_size, x):
    
    x_shape = tf.shape(x)
    if len(x.get_shape().as_list()) == 3:
        kernel = x.get_shape().as_list()[2]
        x = tf.reshape(x, [x_shape[0], x_shape[1], 32, kernel // 32])
        x = tf.layers.batch_normalization(x)
        x = tf.reshape(x, [x_shape[0], x_shape[1], kernel])
    elif len(x.get_shape().as_list()) == 4:
        kernel = x.get_shape().as_list()[3]
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], 32, kernel // 32])
        x = tf.layers.batch_normalization(x)
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], kernel])
    elif len(x.get_shape().as_list()) == 5:
        kernel = x.get_shape().as_list()[4]
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], x_shape[3], 32, kernel // 32])
        x = tf.layers.batch_normalization(x)
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], x_shape[3], kernel])
    
    return x

def tIoU(proposal, timestamps):
    with tf.variable_scope("tiou"):
        proposal_start = tf.expand_dims(proposal, -1)[:, :, 0, :]
        proposal_end = tf.expand_dims(proposal, -1)[:, :, 1, :]
        timestamps_start = tf.expand_dims(timestamps, 1)[:, :, :, 0]
        timestamps_end = tf.expand_dims(timestamps, 1)[:, :, :, 1]
        intersection = tf.maximum(
            0., tf.minimum(proposal_end, timestamps_end) - tf.maximum(proposal_start, timestamps_start))
        union = proposal_end - proposal_start + timestamps_end - timestamps_start - intersection
        tiou = intersection / (union + 1e-8)
        
        return tiou

def get_proposal(params, outputs_list, outputs_deconv_list, outputs_foconv_list):
    with tf.variable_scope("proposal"):
        rd = tf.constant(params.anchor)
        size_rd = rd.get_shape().as_list()[0]
        proposal_list = []
        back_event_list = []
        if params.deconv:
            assert len(outputs_list) == len(outputs_deconv_list)
        if params.foconv:
            assert len(outputs_list) == len(outputs_foconv_list)

        for layer_id in range(params.start_layer, params.end_layer):
            outputs = outputs_list[layer_id]
            if params.foconv and layer_id != 0:
                outputs = tf.concat([outputs_foconv_list[layer_id], outputs], axis=-1)
            if params.deconv and layer_id != (len(outputs_list) - 1):
                outputs = tf.concat([outputs_deconv_list[layer_id], outputs], axis=-1)

            localization = tf.layers.conv1d(outputs, size_rd * 2, kernel_size=1, padding='same', use_bias=False)
            stream_length = tf.shape(localization)[1]
            batch_size = tf.shape(localization)[0]
            localization = tf.reshape(localization, [batch_size, stream_length, size_rd, 2])

            miu_w = rd / tf.cast(stream_length, tf.float32)
            miu_c = (tf.cast(tf.range(stream_length), tf.float32) + 0.5)/ tf.cast(stream_length, tf.float32)
            miu_w = tf.reshape(miu_w, [1, 1, -1])
            miu_c = tf.reshape(miu_c, [1, -1, 1])
            fan_c = miu_c +  miu_w * tf.tanh(0.1 * localization[:, :, :, 0])
            fan_w = miu_w * (tf.sigmoid(0.1 * localization[:, :, :, 1]))
            t_start = tf.expand_dims(fan_c - 0.5 * fan_w, -1)
            t_end = tf.expand_dims(fan_c + 0.5 * fan_w, -1)
            proposal = tf.concat([t_start, t_end], axis=-1)
            proposal = tf.reshape(proposal, [batch_size, -1, 2])
            proposal_list.append(proposal)

            back_event = tf.layers.conv1d(outputs, size_rd * 2, kernel_size=1, padding='same', use_bias=False)
            back_event = tf.reshape(back_event, [batch_size, -1, 2])
            back_event_list.append(back_event)
        back_event = tf.concat(back_event_list, axis=1)
        proposal = tf.concat(proposal_list, axis=1)
        return back_event, proposal

def get_proposalv2(params, outputs_list, outputs_deconv_list, outputs_foconv_list):
    with tf.variable_scope("proposal"):
        rd = tf.constant(params.anchor)
        size_rd = rd.get_shape().as_list()[0]
        proposal_list = []
        back_event_list = []
        if params.deconv:
            assert len(outputs_list) == len(outputs_deconv_list)
        if params.foconv:
            assert len(outputs_list) == len(outputs_foconv_list)

        var_l = tf.get_variable(shape=[1, outputs_list[0].get_shape().as_list()[-1], size_rd * 2], name="localization", regularizer=params.regularizer)
        var_b = tf.get_variable(shape=[1, outputs_list[0].get_shape().as_list()[-1], size_rd * 2], name="back_event", regularizer=params.regularizer)

        for layer_id in range(params.start_layer, params.end_layer):
            outputs = outputs_list[layer_id]
            if params.foconv and layer_id != 0:
                outputs = tf.concat([outputs_foconv_list[layer_id], outputs], axis=-1)
            if params.deconv and layer_id != (len(outputs_list) - 1):
                outputs = tf.concat([outputs_deconv_list[layer_id], outputs], axis=-1)

            localization = tf.nn.conv1d(outputs, var_l, stride=1, padding='SAME')
            stream_length = tf.shape(localization)[1]
            batch_size = tf.shape(localization)[0]
            localization = tf.reshape(localization, [batch_size, stream_length, size_rd, 2])

            miu_w = rd / tf.cast(stream_length, tf.float32)
            miu_c = (tf.cast(tf.range(stream_length), tf.float32) + 0.5)/ tf.cast(stream_length, tf.float32)
            miu_w = tf.reshape(miu_w, [1, 1, -1])
            miu_c = tf.reshape(miu_c, [1, -1, 1])
            fan_c = miu_c +  miu_w * tf.tanh(0.1 * localization[:, :, :, 0])
            fan_w = miu_w * (tf.sigmoid(0.1 * localization[:, :, :, 1]))
            t_start = tf.expand_dims(fan_c - 0.5 * fan_w, -1)
            t_end = tf.expand_dims(fan_c + 0.5 * fan_w, -1)
            proposal = tf.concat([t_start, t_end], axis=-1)
            proposal = tf.reshape(proposal, [batch_size, -1, 2])
            proposal_list.append(proposal)

            back_event = tf.nn.conv1d(outputs, var_b, stride=1, padding='SAME')
            back_event = tf.reshape(back_event, [batch_size, -1, 2])
            back_event_list.append(back_event)
        back_event = tf.concat(back_event_list, axis=1)
        proposal = tf.concat(proposal_list, axis=1)
        return back_event, proposal

def get_proposalv3(params, outputs_list, outputs_deconv_list, outputs_foconv_list):
    with tf.variable_scope("proposal"):
        rd = tf.constant(params.anchor)
        size_rd = rd.get_shape().as_list()[0]
        proposal_list = []
        back_event_list = []
        if params.deconv:
            assert len(outputs_list) == len(outputs_deconv_list)
        if params.foconv:
            assert len(outputs_list) == len(outputs_foconv_list)

        var_l = tf.get_variable(shape=[1, outputs_list[0].get_shape().as_list()[-1], size_rd * 2], name="localization", regularizer=params.regularizer)
        var_b = tf.get_variable(shape=[1, outputs_list[0].get_shape().as_list()[-1], size_rd * 2], name="back_event", regularizer=params.regularizer)

        for layer_id in range(params.start_layer, params.end_layer):
            outputs = outputs_list[layer_id]
            if params.foconv and layer_id != 0:
                outputs = tf.concat([outputs_foconv_list[layer_id], outputs], axis=-1)
            if params.deconv and layer_id != (len(outputs_list) - 1):
                outputs = tf.concat([outputs_deconv_list[layer_id], outputs], axis=-1)

            localization = tf.nn.conv1d(outputs, var_l, stride=1, padding='SAME')
            stream_length = tf.shape(localization)[1]
            batch_size = tf.shape(localization)[0]
            localization = tf.reshape(localization, [batch_size, stream_length, size_rd, 2])

            miu_w = rd / tf.cast(stream_length, tf.float32)
            miu_c = (tf.cast(tf.range(stream_length), tf.float32) + 0.5)/ tf.cast(stream_length, tf.float32)
            miu_w = tf.reshape(miu_w, [1, 1, -1])
            miu_c = tf.reshape(miu_c, [1, -1, 1])
            fan_c = miu_c +  miu_w * (0.1 * localization[:, :, :, 0])
            fan_w = miu_w * (tf.exp(0.1 * localization[:, :, :, 1]))
            t_start = tf.expand_dims(fan_c - 0.5 * fan_w, -1)
            t_end = tf.expand_dims(fan_c + 0.5 * fan_w, -1)
            proposal = tf.concat([t_start, t_end], axis=-1)
            proposal = tf.reshape(proposal, [batch_size, -1, 2])
            proposal_list.append(proposal)

            back_event = tf.nn.conv1d(outputs, var_b, stride=1, padding='SAME')
            back_event = tf.reshape(back_event, [batch_size, -1, 2])
            back_event_list.append(back_event)
        back_event = tf.concat(back_event_list, axis=1)
        proposal = tf.concat(proposal_list, axis=1)
        return back_event, proposal

def choose_top(params, probability, proposal, top_k):
    with tf.device('/cpu:0'), tf.variable_scope("choose_top"):
        proposal_number = tf.shape(probability)[1]
        top_num = tf.minimum(proposal_number, top_k)
        prob_top, top_id = tf.nn.top_k(probability, top_num)
        batch_id = tf.tile(
            tf.reshape(tf.range(tf.shape(probability)[0]), [-1, 1, 1]), 
            [1, top_num, 1])
        top_id = tf.concat([batch_id, tf.expand_dims(top_id, -1)], axis=-1)
        proposal_top = tf.gather_nd(proposal, top_id)
        return prob_top, proposal_top

def choose_top_with_idx(params, probability, proposal, top_k):
    with tf.device('/cpu:0'), tf.variable_scope("choose_top"):
        proposal_number = tf.shape(probability)[1]
        top_num = tf.minimum(proposal_number, top_k)
        prob_top, top_id = tf.nn.top_k(probability, top_num)
        batch_id = tf.tile(
            tf.reshape(tf.range(tf.shape(probability)[0]), [-1, 1, 1]), 
            [1, top_num, 1])
        top_id_batch = tf.concat([batch_id, tf.expand_dims(top_id, -1)], axis=-1)
        proposal_top = tf.gather_nd(proposal, top_id_batch)
        return prob_top, proposal_top, top_id

def get_recall(params, proposal_top_k, timestamps):
    with tf.variable_scope("recall"), tf.device('/cpu:0'):
        top_list = [i for i in range(1, 101)]
        recall_list = []
        tiou = tIoU(proposal_top_k, timestamps)
        k = tf.cast(tf.minimum(tf.shape(tiou)[1], top_list[-1]), tf.float32) / float(len(top_list))
        top_list_tensor = tf.cast(tf.cast(tf.constant(top_list), tf.float32) * k, tf.int32)

        for thresh in params.tiou_thresh:
            shoot = tf.cast(tf.greater(tiou, thresh), tf.int32)
            for i in range(len(top_list)):
                shoot_i = shoot[:, :top_list_tensor[i], :]
                tiou_k = tf.reduce_max(shoot_i, axis=1)
                recall = tf.reduce_sum(tiou_k, axis=1)
                if i == 0 or i == (len(top_list) -1):
                    recall = tf.cast(recall, tf.float32) / 2.
                else:
                    recall = tf.cast(recall, tf.float32)
                recall_list.append(recall)
        return tf.add_n(recall_list) * k  / float(len(params.tiou_thresh))

def get_acc(params, proposal_top_k, timestamps):
    with tf.variable_scope("recall"), tf.device('/cpu:0'):
        tiou = tIoU(proposal_top_k, timestamps)
        shoot = tf.cast(tf.greater(tiou, params.ratio), tf.float32)
        shoot_i = shoot[:, :1, :]
        tiou_k = tf.reduce_max(shoot_i, axis=1)
        recall = tf.reduce_sum(tiou_k, axis=1)

        return recall * 100

def tiou_pyfunc(proposal, timestamps):
    proposal_start = np.expand_dims(proposal, -1)[:, :, 0, :]
    proposal_end = np.expand_dims(proposal, -1)[:, :, 1, :]
    timestamps_start = np.expand_dims(timestamps, 1)[:, :, :, 0]
    timestamps_end = np.expand_dims(timestamps, 1)[:, :, :, 1]
    intersection = np.maximum(
        0., np.minimum(proposal_end, timestamps_end) - np.maximum(proposal_start, timestamps_start))
    union = proposal_end - proposal_start + timestamps_end - timestamps_start - intersection
    tiou = intersection / (union + 1e-8)
        
    return tiou

def generate_top5(proposal_top_k, prob_top_k):
    top_proposal_list = []
    for i in range(5):
        top_idx = np.argmax(prob_top_k, axis=1)
        top_proposal = []
        for j in range(top_idx.size):
            top_proposal.append(np.expand_dims(proposal_top_k[j, top_idx[j], :], 0))
            prob_top_k[j,top_idx[j]] = -1
        top_proposal = np.expand_dims(np.concatenate(top_proposal, axis=0), 1)
        top_proposal_list.append(top_proposal)
        tiou = tiou_pyfunc(top_proposal, proposal_top_k)[:,0,:]
        prob_top_k = prob_top_k * (tiou < 0.5).astype(np.float32)
    output = np.concatenate(top_proposal_list, axis=1)
    return output

def get_acc_top1_top5(params, proposal_top_k, prob_top_k, timestamps):
    with tf.variable_scope("recall"), tf.device('/cpu:0'):

        tiou = tIoU(proposal_top_k, timestamps)
        shoot = tf.cast(tf.greater(tiou, 0.5), tf.float32)
        shoot = shoot[:, :1, :]
        recall15 = tf.reduce_max(shoot, axis=1)

        shoot = tf.cast(tf.greater(tiou, 0.7), tf.float32)
        shoot = shoot[:, :1, :]
        recall17 = tf.reduce_max(shoot, axis=1)

        top_5 = tf.py_func(
            generate_top5,
            [proposal_top_k, prob_top_k],
            [tf.float32]
        )[0]
        tiou_top5 = tIoU(top_5, timestamps)
        shoot = tf.cast(tf.greater(tiou_top5, 0.5), tf.float32)
        recall55 = tf.reduce_max(shoot, axis=1)

        shoot = tf.cast(tf.greater(tiou_top5, 0.7), tf.float32)
        recall57 = tf.reduce_max(shoot, axis=1)

        recall = tf.concat([recall15, recall55, recall17, recall57], axis=1)

        return recall * 100

def resblock_1d(params, inputs, kernel):
    outputs_pool =  tf.layers.conv1d(
        inputs, kernel, kernel_size=3, strides=1, padding='same', use_bias=False)
    outputs_pool = group_norm(params.batch_size, outputs_pool)

    outputs = tf.layers.conv1d(
        inputs, kernel, kernel_size=3, strides=1, padding='same', use_bias=False)
    outputs = group_norm(params.batch_size, outputs)
    outputs = tf.nn.relu(outputs)
    outputs = tf.layers.conv1d(
        outputs, kernel, kernel_size=3, strides=1, padding='same', use_bias=False)
    outputs = group_norm(params.batch_size, outputs)
    outputs = outputs + outputs_pool
    outputs = tf.nn.relu(outputs)

    return outputs

def reeval_v1(params, proposal, timestamps, timestamps_length, base_layer):
    with tf.variable_scope("reevalv1"):
        outputs = resblock_1d(params, base_layer, 512)
        outputs = resblock_1d(params, outputs, 256)
        outputs = resblock_1d(params, outputs, 128)
        feature = outputs

        stream_length = tf.cast(tf.shape(outputs)[1], tf.float32)

        outputs = tf.layers.conv1d(
            feature, 2, kernel_size=1, strides=1, padding='same', use_bias=False)

        timestamps_id = tf.reshape(timestamps[:,:,0], [tf.shape(timestamps)[0], tf.shape(timestamps)[1], 1])
        timestamps_id = tf.cast(tf.minimum(tf.maximum(timestamps_id, 0.), 1.) * stream_length, tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(tf.shape(timestamps_id)[0]), [-1, 1, 1]), [1, tf.shape(timestamps_id)[1], 1])
        timestamps_id = tf.concat([batch_id, timestamps_id], axis=-1)
        timestamps_shoot = tf.scatter_nd(
            timestamps_id,
            tf.ones(tf.shape(timestamps_id)[:-1], dtype=tf.int32), 
            [tf.shape(outputs)[0], tf.shape(outputs)[1]],
        )
        timestamps_shoot = tf.cast(tf.cast(timestamps_shoot, tf.bool), tf.int32)
        weights = tf.cast(timestamps_shoot, tf.float32)
        weights = weights * tf.cast(tf.shape(timestamps_shoot)[1], tf.float32) 
        weights += 1
        weights = tf.stop_gradient(weights)
        with tf.variable_scope("start_end_loss"):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                logits=outputs, labels=timestamps_shoot, weights=weights)

        outputs = tf.layers.conv1d(
            feature, 2, kernel_size=1, strides=1, padding='same', use_bias=False)

        timestamps_id = tf.reshape(timestamps[:,:,1], [tf.shape(timestamps)[0], tf.shape(timestamps)[1], 1])
        timestamps_id = tf.cast(tf.minimum(tf.maximum(timestamps_id, 0.), 1.) * stream_length, tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(tf.shape(timestamps_id)[0]), [-1, 1, 1]), [1, tf.shape(timestamps_id)[1], 1])
        timestamps_id = tf.concat([batch_id, timestamps_id], axis=-1)
        timestamps_shoot = tf.scatter_nd(
            timestamps_id,
            tf.ones(tf.shape(timestamps_id)[:-1], dtype=tf.int32), 
            [tf.shape(outputs)[0], tf.shape(outputs)[1]],
        )
        timestamps_shoot = tf.cast(tf.cast(timestamps_shoot, tf.bool), tf.int32)
        weights = tf.cast(timestamps_shoot, tf.float32)
        weights = weights * tf.cast(tf.shape(timestamps_shoot)[1], tf.float32) 
        weights += 1
        weights = tf.stop_gradient(weights)
        with tf.variable_scope("start_end_loss"):
            cross_entropy2 = tf.losses.sparse_softmax_cross_entropy(
                logits=outputs, labels=timestamps_shoot, weights=weights) 


        outputs = tf.layers.conv1d(
            tf.stop_gradient(feature), 64, kernel_size=1, strides=1, padding='same', use_bias=False)
        position_id = tf.cast(tf.expand_dims(tf.minimum(tf.maximum(proposal, 0.), 1.), -1) * stream_length, tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(tf.shape(proposal)[0]), [-1, 1, 1, 1]), [1, tf.shape(proposal)[1], tf.shape(proposal)[2], 1])
        proposal_id = tf.concat([batch_id, position_id], axis=-1)
        back_event = tf.gather_nd(outputs, tf.stop_gradient(proposal_id))

        return back_event, cross_entropy + cross_entropy2

def resblock_1d_s1(params, inputs, kernel):
    outputs_pool =  tf.layers.conv1d(
        inputs, kernel, kernel_size=1, strides=1, padding='same', use_bias=False)
    outputs_pool = group_norm(params.batch_size, outputs_pool)

    outputs = tf.layers.conv1d(
        inputs, kernel, kernel_size=1, strides=1, padding='same', use_bias=False)
    outputs = group_norm(params.batch_size, outputs)
    outputs = tf.nn.relu(outputs)
    outputs = tf.layers.conv1d(
        outputs, kernel, kernel_size=1, strides=1, padding='same', use_bias=False)
    outputs = group_norm(params.batch_size, outputs)
    outputs = outputs + outputs_pool
    outputs = tf.nn.relu(outputs)
    return outputs 

def conv_block_2d(params, inputs, kernel):
    outputs = tf.layers.conv2d(
        inputs, kernel, kernel_size=[1, 1], strides=[1, 1], padding='same', use_bias=False)
    outputs = group_norm(params.batch_size, outputs)
    outputs = tf.nn.relu(outputs)
    return outputs 


def generate_reeval_feature(feature, proposal, offset=0.1, num=8):
    batch_size = tf.shape(proposal)[0]
    proposal_num = tf.shape(proposal)[1]
    idx_num = 2 * num
    idx_feature_len = idx_num * feature.get_shape().as_list()[-1]

    proposal_length = tf.reshape(proposal[:,:,1] - proposal[:,:,0], [batch_size, proposal_num, 1, 1])

    start_end_range = tf.reshape(tf.cast(tf.range(num) - num // 2, tf.float32) / tf.cast(num // 2, tf.float32) * offset, [1,1,1,-1])
    start_end_pos = tf.expand_dims(proposal, -1) + proposal_length * start_end_range
    start_end_pos = tf.reshape(start_end_pos, [batch_size, proposal_num, 2, num])
    
    #middle_range = tf.reshape(tf.cast(tf.range(num - 1), tf.float32) / tf.cast(num - 1, tf.float32) * (1 - offset) + (offset / 2), [1,1,1,-1])
    #mideel_pos = tf.reshape(proposal[:,:,0], [batch_size, proposal_num, 1, 1]) + proposal_length * middle_range
    #mideel_pos = tf.reshape(mideel_pos, [batch_size, proposal_num, -1])

    #start_end_middle_pos = tf.concat([start_end_pos, mideel_pos], axis=2)

    start_end_pos = tf.minimum(tf.maximum(start_end_pos, 0.), 1.)
    start_end_pos = tf.cast(start_end_pos * tf.cast(tf.shape(feature)[1], tf.float32), tf.int32)
    batch_id = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1, 1, 1]), [1, proposal_num, 2, num, 1])
    idx = tf.concat([batch_id, tf.expand_dims(start_end_pos, -1)], axis=-1)

    reeval_feature = tf.gather_nd(feature, tf.stop_gradient(idx))
    reeval_feature = tf.reshape(reeval_feature, [batch_size, proposal_num, 2, num * feature.get_shape().as_list()[-1]])
    return reeval_feature


def reeval_v2(params, proposal, base_layer):
    with tf.variable_scope("reevalv2"):
        outputs = resblock_1d(params, base_layer, 512)
        outputs = resblock_1d(params, outputs, 256)
        outputs = resblock_1d(params, outputs, 128)


        feature = generate_reeval_feature(outputs, proposal, offset=params.reeval_prop_ratio, num=params.reeval_prop_length)
        proposal_length = tf.expand_dims(proposal[:,:,1] - proposal[:,:,0], -1)

        outputs = feature[:,:,0,:]
        outputs = resblock_1d_s1(params, outputs, 512)
        outputs = resblock_1d_s1(params, outputs, 256)
        outputs = resblock_1d_s1(params, outputs, 128)
        left = tf.layers.conv1d(
            outputs, 1, kernel_size=1, padding='valid', use_bias=False)

        outputs = feature[:,:,1,:]
        outputs = resblock_1d_s1(params, outputs, 512)
        outputs = resblock_1d_s1(params, outputs, 256)
        outputs = resblock_1d_s1(params, outputs, 128)
        right = tf.layers.conv1d(
            outputs, 1, kernel_size=1, padding='valid', use_bias=False)
        
        outputs = tf.concat([left, right], axis=-1)
        outputs = tf.stop_gradient(proposal_length) * tf.nn.tanh(outputs) * params.reeval_prop_ratio

        return outputs


def timestamps_shoot_euclidean_loss(params, tiou, proposal, timestamps, timestamps_length):
    with tf.variable_scope("timestamps_eucloss"):
        timestamps_shoot_id =  tf.argmax(tiou, axis=1, output_type=tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(tf.shape(tiou)[0]), [-1, 1, 1]), [1, tf.shape(timestamps_shoot_id)[1], 1])
        timestamps_shoot_id = tf.concat([batch_id, tf.expand_dims(timestamps_shoot_id, -1)], axis=-1)
        proposal_shoot = tf.gather_nd(proposal, timestamps_shoot_id)

        mask = tf.expand_dims(tf.cast(tf.sequence_mask(timestamps_length), tf.float32), -1) / params.batch_size
        if params.timestamps_weights:
            proposal_length = tf.expand_dims(timestamps[:,:,1] - timestamps[:,:,0], -1) + 1e-3
            mask = mask / proposal_length
        euclidean = tf.losses.absolute_difference(timestamps, proposal_shoot, weights=mask, reduction=tf.losses.Reduction.SUM) 
        return euclidean

def timestamps_shoot_euclidean_loss_v2(params, tiou, proposal, timestamps, timestamps_length, duration):
    with tf.variable_scope("timestamps_euclossv2"):
        timestamps_shoot_id =  tf.argmax(tiou, axis=1, output_type=tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(tf.shape(tiou)[0]), [-1, 1, 1]), [1, tf.shape(timestamps_shoot_id)[1], 1])
        timestamps_shoot_id = tf.concat([batch_id, tf.expand_dims(timestamps_shoot_id, -1)], axis=-1)
        proposal_shoot = tf.gather_nd(proposal, timestamps_shoot_id)

        mask = tf.expand_dims(tf.cast(tf.sequence_mask(timestamps_length), tf.float32), -1)

        proposal_length = tf.expand_dims(proposal_shoot[:,:,1] - proposal_shoot[:,:,0], -1)
        distance_mask = tf.greater_equal(tf.abs(proposal_shoot - timestamps), proposal_length * params.reeval_prop_ratio)
        mask = tf.stop_gradient(mask * tf.cast(distance_mask, tf.float32) * tf.reduce_sum(mask) / params.batch_size)
        euclidean = tf.losses.absolute_difference(timestamps, proposal_shoot, weights=mask) 
        return euclidean, distance_mask

def timestamps_shoot_euclidean_loss_v3(params, tiou, mask, proposal, timestamps, timestamps_length, duration):
    with tf.variable_scope("timestamps_euclossv3"):
        timestamps_shoot_id =  tf.argmax(tiou, axis=1, output_type=tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(tf.shape(tiou)[0]), [-1, 1, 1]), [1, tf.shape(timestamps_shoot_id)[1], 1])
        timestamps_shoot_id = tf.concat([batch_id, tf.expand_dims(timestamps_shoot_id, -1)], axis=-1)
        proposal_shoot = tf.gather_nd(proposal, timestamps_shoot_id)

        mask = tf.cast(mask, tf.float32) * tf.expand_dims(tf.cast(tf.sequence_mask(timestamps_length), tf.float32), -1)
        mask = tf.stop_gradient(mask * tf.reduce_sum(mask) / params.batch_size)

        euclidean = tf.losses.absolute_difference(timestamps,proposal_shoot, weights=mask) 
        return euclidean

def timestamps_shoot_cross_entropy_loss(params, tiou, back_event):
    with tf.variable_scope("timestamps_cross"):
        timestamps_shoot_id =  tf.argmax(tiou, axis=1, output_type=tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(tf.shape(tiou)[0]), [-1, 1, 1]), [1, tf.shape(timestamps_shoot_id)[1], 1])
        timestamps_shoot_id = tf.concat([batch_id, tf.expand_dims(timestamps_shoot_id, -1)], axis=-1)
        back_event_shoot = tf.gather_nd(back_event, timestamps_shoot_id)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event_shoot, labels=tf.ones(tf.shape(back_event_shoot)[:-1], dtype=tf.int32),)

        return cross_entropy

def repair_shoot_mask(timestamps_not_shoot, timestamps_shoot_id, timestamps_length):
    for i in range(timestamps_shoot_id.shape[0]):
        for j in range(timestamps_length[i], timestamps_shoot_id.shape[1]):
            index = timestamps_shoot_id[i, j]
            timestamps_not_shoot[i, index] = True

    return timestamps_not_shoot

def timestamps_shoot_cross_entropy_lossv2(params, tiou, back_event, timestamps_length):
    with tf.variable_scope("timestamps_crossv2"):
        timestamps_shoot_id =  tf.argmax(tiou, axis=1, output_type=tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(tf.shape(tiou)[0]), [-1, 1, 1]), [1, tf.shape(timestamps_shoot_id)[1], 1])
        timestamps_shoot_id = tf.concat([batch_id, tf.expand_dims(timestamps_shoot_id, -1)], axis=-1)
        back_event_shoot = tf.gather_nd(back_event, timestamps_shoot_id)

        mask = tf.expand_dims(tf.cast(tf.sequence_mask(timestamps_length), tf.float32), -1) / params.batch_size
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event_shoot, labels=tf.ones(tf.shape(back_event_shoot)[:-1], dtype=tf.int32), 
            weights=mask, reduction=tf.losses.Reduction.SUM)

        timestamps_not_shoot = tf.scatter_nd(
            timestamps_shoot_id,
            tf.ones(tf.shape(timestamps_shoot_id)[:-1], dtype=tf.int32), 
            [tf.shape(back_event)[0], tf.shape(back_event)[1]],
        )
        timestamps_not_shoot = tf.logical_not(tf.cast(timestamps_not_shoot, tf.bool))
        timestamps_not_shoot = tf.py_func(
            repair_shoot_mask, 
            [timestamps_not_shoot, timestamps_shoot_id, timestamps_length],
            [tf.bool])[0]
        timestamps_not_shoot.set_shape([tf.Dimension(None), tf.Dimension(None)])
        timestamps_not_shoot = tf.cast(timestamps_not_shoot, tf.float32) * tf.cast(tf.expand_dims(timestamps_length, -1), tf.float32)
            
        cross_entropy2 = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event, labels=tf.zeros(tf.shape(back_event)[:-1], dtype=tf.int32), weights=timestamps_not_shoot)

        return cross_entropy, cross_entropy2

def porbability_thresh_shoot_softmax_crossentropy_loss(params, tiou, back_event):
    with tf.variable_scope("top_k_croloss"):
        max_tiou =  tf.reduce_max(tiou, axis=2)
        shoot_mask = tf.greater(max_tiou, params.ratio)
        back_event_shoot = tf.expand_dims(tf.boolean_mask(back_event, shoot_mask), 0)
        cross_entropy1 = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event_shoot, labels=tf.ones(tf.shape(back_event_shoot)[:-1], dtype=tf.int32))

        shoot_mask = tf.less(max_tiou, (1 - params.ratio))
        back_event_shoot = tf.expand_dims(tf.boolean_mask(back_event, shoot_mask), 0)
        cross_entropy2 = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event_shoot, labels=tf.zeros(tf.shape(back_event_shoot)[:-1], dtype=tf.int32))

        return cross_entropy1, cross_entropy2

def poposal_thresh_shoot_euclidean_loss(params, tiou, proposal, timestamps):
    with tf.variable_scope("top_k_eucloss"):
        max_tiou =  tf.reduce_max(tiou, axis=2)
        shoot_mask = tf.greater(max_tiou, params.ratio)
        proposal_shoot = tf.expand_dims(tf.boolean_mask(proposal, shoot_mask), 0)

        timestamps_shoot_id = tf.argmax(tiou, axis=2, output_type=tf.int32)
        batch_id = tf.tile(tf.reshape(tf.range(params.batch_size), [-1, 1, 1]), [1, tf.shape(timestamps_shoot_id)[1], 1])
        timestamps_shoot_id = tf.concat([batch_id, tf.expand_dims(timestamps_shoot_id, -1)], axis=-1)
        proposal_shoot_gd = tf.gather_nd(timestamps, timestamps_shoot_id)
        proposal_shoot_gd = tf.expand_dims(tf.boolean_mask(proposal_shoot_gd, shoot_mask), 0)

        euclidean = tf.losses.absolute_difference(proposal_shoot, proposal_shoot_gd)
        return euclidean