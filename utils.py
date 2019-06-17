from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import numpy as np

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

def get_proposal(params, outputs_list):
    with tf.variable_scope("proposal"):
        rd = tf.constant(params.anchor)
        size_rd = rd.get_shape().as_list()[0]
        proposal_list = []
        back_event_list = []

        var_l = tf.get_variable(shape=[1, params.hidden_size, size_rd * 2], 
            name="localization", regularizer=params.regularizer)
        var_b = tf.get_variable(shape=[1, params.hidden_size, size_rd * 2], 
            name="back_event", regularizer=params.regularizer)

        for layer_id in range(params.start_layer, params.end_layer):
            outputs = outputs_list[layer_id]

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


def get_acc(params, proposal_top_k, timestamps):
    with tf.variable_scope("recall"), tf.device('/cpu:0'):
        tiou = tIoU(proposal_top_k, timestamps)
        shoot = tf.cast(tf.greater(tiou, params.ratio), tf.float32)
        shoot_i = shoot[:, :1, :]
        tiou_k = tf.reduce_max(shoot_i, axis=1)
        acc = tf.reduce_sum(tiou_k, axis=1)
        return acc * 100

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
        acc15 = tf.reduce_max(shoot, axis=1)

        shoot = tf.cast(tf.greater(tiou, 0.7), tf.float32)
        shoot = shoot[:, :1, :]
        acc17 = tf.reduce_max(shoot, axis=1)

        top_5 = tf.py_func(
            generate_top5,
            [proposal_top_k, prob_top_k],
            [tf.float32]
        )[0]
        tiou_top5 = tIoU(top_5, timestamps)
        shoot = tf.cast(tf.greater(tiou_top5, 0.5), tf.float32)
        acc55 = tf.reduce_max(shoot, axis=1)

        shoot = tf.cast(tf.greater(tiou_top5, 0.7), tf.float32)
        acc57 = tf.reduce_max(shoot, axis=1)

        acc_dict = {"acc_R1_t0.5":acc15, "acc_R1_t0.7":acc17, 
            "acc_R5_t0.5":acc55, "acc_R5_t0.7":acc57}

        return recall * 100

def crossentropy_loss(params, tiou, back_event):
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

def euclidean_loss(params, tiou, proposal, timestamps):
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