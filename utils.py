import copy
import tensorflow as tf
import numpy as np

def tIoU(proposal, timestamps):
    with tf.variable_scope("tiou"):
        proposal_start = proposal[:, :, 0]
        proposal_end = proposal[:, :, 1]
        timestamps_start = tf.expand_dims(timestamps[:, 0], -1)
        timestamps_end = tf.expand_dims(timestamps[:, 1], -1)
        intersection = tf.maximum(
            0., tf.minimum(proposal_end, timestamps_end) - tf.maximum(proposal_start, timestamps_start))
        union = proposal_end - proposal_start + timestamps_end - timestamps_start - intersection
        tiou = intersection / (union + 1e-8)
        
        return tf.stop_gradient(tiou)

def get_proposal(params, inputs, inputs_size_list):
    batch_size = tf.shape(inputs)[0]
    stream_length = tf.shape(inputs)[1]

    with tf.variable_scope("proposal"):
        rd = tf.constant(params.anchor)
        size_rd = rd.get_shape().as_list()[0]
        localization = tf.layers.conv1d(inputs, size_rd * 2, kernel_size=1, padding='same', use_bias=False)
        localization = tf.reshape(localization, [batch_size, stream_length, size_rd, 2])

        with tf.device('/cpu:0'):
            miu_w_list = []
            miu_c_list = []
            for size in inputs_size_list:
                miu_w = rd / tf.cast(size, tf.float32)
                miu_c = (tf.cast(tf.range(size), tf.float32) + 0.5)/ tf.cast(size, tf.float32)
                miu_w = tf.reshape(miu_w, [1, 1, size_rd])
                miu_w = tf.tile(miu_w, [1, size, 1])
                miu_c = tf.reshape(miu_c, [1, size, 1])
                miu_w_list.append(miu_w)
                miu_c_list.append(miu_c)
            miu_w = tf.concat(miu_w_list, axis=1)
            miu_c = tf.concat(miu_c_list, axis=1)

        fan_c = miu_c +  miu_w * tf.tanh(0.1 * localization[:, :, :, 0])
        fan_w = miu_w * (tf.sigmoid(0.1 * localization[:, :, :, 1]) * 0.5 + 0.5)
        t_start = tf.expand_dims(fan_c - 0.5 * fan_w, -1)
        t_end = tf.expand_dims(fan_c + 0.5 * fan_w, -1)
        proposal = tf.concat([t_start, t_end], axis=-1)
        proposal = tf.reshape(proposal, [batch_size, stream_length*size_rd, 2])

    with tf.variable_scope("back_event"):
        back_event = tf.layers.conv1d(inputs, size_rd * 2, kernel_size=1, padding='same', use_bias=False)
        back_event = tf.reshape(back_event, [batch_size, stream_length*size_rd, 2])

    return back_event, proposal

def get_proposalv2(params, inputs, inputs_size_list):
    batch_size = tf.shape(inputs)[0]
    stream_length = tf.shape(inputs)[1]

    with tf.variable_scope("proposal"):
        rd = tf.constant(params.anchor)
        size_rd = rd.get_shape().as_list()[0]
        localization = tf.layers.conv1d(inputs, size_rd * 2, kernel_size=1, padding='same', use_bias=False)
        localization = tf.reshape(localization, [batch_size, stream_length, size_rd, 2])

        with tf.device('/cpu:0'):
            miu_w_list = []
            miu_c_list = []
            for size in inputs_size_list:
                miu_w = rd / tf.cast(size, tf.float32)
                miu_c = (tf.cast(tf.range(size), tf.float32) + 0.5)/ tf.cast(size, tf.float32)
                miu_w = tf.reshape(miu_w, [1, 1, size_rd])
                miu_w = tf.tile(miu_w, [1, size, 1])
                miu_c = tf.reshape(miu_c, [1, size, 1])
                miu_w_list.append(miu_w)
                miu_c_list.append(miu_c)
            miu_w = tf.concat(miu_w_list, axis=1)
            miu_c = tf.concat(miu_c_list, axis=1)

        fan_c = miu_c +  miu_w * (0.1 * localization[:, :, :, 0])
        fan_w = miu_w * tf.exp(0.1 * localization[:, :, :, 1])
        t_start = tf.expand_dims(fan_c - 0.5 * fan_w, -1)
        t_end = tf.expand_dims(fan_c + 0.5 * fan_w, -1)
        proposal = tf.concat([t_start, t_end], axis=-1)
        proposal = tf.reshape(proposal, [batch_size, stream_length*size_rd, 2])

    with tf.variable_scope("back_event"):
        back_event = tf.layers.conv1d(inputs, size_rd * 2, kernel_size=1, padding='same', use_bias=False)
        back_event = tf.reshape(back_event, [batch_size, stream_length*size_rd, 2])

    return back_event, proposal

def choose_top(proposal, probability):
    top_idx = tf.cast(tf.math.argmax(probability, axis=1), tf.int32)
    top_idx = tf.expand_dims(top_idx, 1)
    batch_id = tf.reshape(tf.range(tf.shape(probability)[0]), [-1, 1])
    top_idx = tf.concat([batch_id, top_idx], axis=-1)
    top_proposal = tf.gather_nd(proposal, top_idx)
    return top_proposal

def nms(probability, proposal, top_proposal_gather, thresh):
    top_proposal = choose_top(proposal, probability)
    tiou_prop_top = tIoU(proposal, top_proposal)
    top_proposal = tf.expand_dims(top_proposal, axis=1)
    probability = probability * tf.cast(tf.less(tiou_prop_top, thresh), tf.float32)
    top_proposal_gather = tf.concat([top_proposal_gather, top_proposal], axis=1)
    return probability, proposal, top_proposal_gather, thresh

def get_acc(params, proposal, probability, timestamps):
    with tf.variable_scope("accuracy"), tf.device('/cpu:0'):
        top_proposal = choose_top(proposal, probability)
        top_proposal = tf.expand_dims(top_proposal, axis=1)
        top1 = tIoU(top_proposal, timestamps)
        shoot = tf.cast(tf.greater(top1, params.ratio), tf.float32)
        return shoot

def get_acc_top1_top5(params, proposal, probability, timestamps):
    with tf.variable_scope("accuracy"), tf.device('/cpu:0'):
        top_proposal = choose_top(proposal, probability)
        top_proposal = tf.expand_dims(top_proposal, axis=1)
        top1 = tIoU(top_proposal, timestamps)
        acc11 = tf.cast(tf.greater(top1, 0.1), tf.float32)
        acc13 = tf.cast(tf.greater(top1, 0.3), tf.float32)
        acc15 = tf.cast(tf.greater(top1, 0.5), tf.float32)
        acc17 = tf.cast(tf.greater(top1, 0.7), tf.float32)

        mIoU = tf.cast(tf.cast((top_proposal * 6.), tf.int32), tf.float32) / 6.
        mIoU = tIoU(mIoU, timestamps)

        stop = lambda probability, proposal, top_proposal, thresh: tf.less(tf.shape(top_proposal)[1], 6)
        _, _, top_proposal, _ = tf.while_loop(stop, nms, 
            [probability, proposal, tf.zeros([tf.shape(proposal)[0], 1, 2]), 0.5], 
            parallel_iterations=1, back_prop=False, 
            shape_invariants=[
                probability.get_shape(), proposal.get_shape(), 
                tf.TensorShape([None, None, 2]), tf.TensorShape([])
            ]
        )
        top_proposal = top_proposal[:, 1:, :]
        tiou_top5 = tIoU(top_proposal, timestamps)
        shoot = tf.cast(tf.greater(tiou_top5, 0.5), tf.float32)
        acc55 = tf.reduce_max(shoot, axis=1)

        _, _, top_proposal, _ = tf.while_loop(stop, nms, 
            [probability, proposal, tf.zeros([tf.shape(proposal)[0], 1, 2]), 0.7], 
            parallel_iterations=1, back_prop=False, 
            shape_invariants=[
                probability.get_shape(), proposal.get_shape(), 
                tf.TensorShape([None, None, 2]), tf.TensorShape([])
            ]
        )
        top_proposal = top_proposal[:, 1:, :]
        tiou_top5 = tIoU(top_proposal, timestamps)
        shoot = tf.cast(tf.greater(tiou_top5, 0.7), tf.float32)
        acc57 = tf.reduce_max(shoot, axis=1)

        _, _, top_proposal, _ = tf.while_loop(stop, nms, 
            [probability, proposal, tf.zeros([tf.shape(proposal)[0], 1, 2]), 0.3], 
            parallel_iterations=1, back_prop=False, 
            shape_invariants=[
                probability.get_shape(), proposal.get_shape(), 
                tf.TensorShape([None, None, 2]), tf.TensorShape([])
            ]
        )
        top_proposal = top_proposal[:, 1:, :]
        tiou_top5 = tIoU(top_proposal, timestamps)
        shoot = tf.cast(tf.greater(tiou_top5, 0.3), tf.float32)
        acc53 = tf.reduce_max(shoot, axis=1)

        _, _, top_proposal, _ = tf.while_loop(stop, nms, 
            [probability, proposal, tf.zeros([tf.shape(proposal)[0], 1, 2]), 0.1], 
            parallel_iterations=1, back_prop=False, 
            shape_invariants=[
                probability.get_shape(), proposal.get_shape(), 
                tf.TensorShape([None, None, 2]), tf.TensorShape([])
            ]
        )
        top_proposal = top_proposal[:, 1:, :]
        tiou_top5 = tIoU(top_proposal, timestamps)
        shoot = tf.cast(tf.greater(tiou_top5, 0.1), tf.float32)
        acc51 = tf.reduce_max(shoot, axis=1)

        acc_dict = {"acc_R1_t0.1":acc11, "acc_R1_t0.3":acc13, "acc_R1_t0.5":acc15, "acc_R1_t0.7":acc17, 
            "acc_R5_t0.1":acc51, "acc_R5_t0.3":acc53, "acc_R5_t0.5":acc55, "acc_R5_t0.7":acc57, "mIoU":top1}

        return acc_dict

def class_loss(params, proposal, timestamps, back_event):
    with tf.variable_scope("croloss"):
        tiou = tIoU(proposal, timestamps)
        shoot_mask = tf.greater(tiou, params.ratio)
        back_event_shoot = tf.boolean_mask(back_event, shoot_mask)
        cross_entropy1 = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event_shoot, labels=tf.ones(tf.shape(back_event_shoot)[:-1], dtype=tf.int32))

        shoot_mask = tf.logical_not(shoot_mask)
        back_event_shoot = tf.boolean_mask(back_event, shoot_mask)
        cross_entropy2 = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event_shoot, labels=tf.zeros(tf.shape(back_event_shoot)[:-1], dtype=tf.int32))

        return cross_entropy1, cross_entropy2

def class_lossv2(params, proposal, timestamps, back_event):
    with tf.variable_scope("croloss"):
        tiou = tIoU(proposal, timestamps)
        shoot_mask = tf.greater(tiou, params.ratio)
        back_event_shoot = tf.boolean_mask(back_event, shoot_mask)
        tiou_shoot = tf.boolean_mask(tiou, shoot_mask)
        cross_entropy1 = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event_shoot, labels=tf.ones(tf.shape(back_event_shoot)[:-1], dtype=tf.int32), weights=tiou_shoot)

        shoot_mask = tf.logical_not(shoot_mask)
        tiou_shoot = 1 - tf.boolean_mask(tiou, shoot_mask)
        back_event_shoot = tf.boolean_mask(back_event, shoot_mask)
        cross_entropy2 = tf.losses.sparse_softmax_cross_entropy(
            logits=back_event_shoot, labels=tf.zeros(tf.shape(back_event_shoot)[:-1], dtype=tf.int32), weights=tiou_shoot)

        return cross_entropy1, cross_entropy2

def regress_loss(params, proposal, timestamps):
    with tf.variable_scope("eucloss"):
        tiou = tIoU(proposal, timestamps)
        shoot_mask = tf.expand_dims(tf.greater(tiou, params.ratio), -1)
        timestamps = tf.expand_dims(timestamps, axis=1)

        euclidean = tf.losses.absolute_difference(proposal, timestamps, weights=shoot_mask)
        return euclidean

def regress_lossv2(params, proposal, timestamps):
    with tf.variable_scope("eucloss"):
        proposal = proposal * params.k2scale
        timestamps = timestamps * params.k2scale
        tiou = tIoU(proposal, timestamps)
        shoot_mask = tf.expand_dims(tf.greater(tiou, params.ratio), -1)
        timestamps = tf.expand_dims(timestamps, axis=1)

        diff = tf.abs(proposal - timestamps)
        weights = tf.greater(diff, 1)
        weights_float = tf.cast(tf.logical_and(shoot_mask, weights), tf.float32)
        absolute = tf.reduce_sum((diff - 0.5) * weights_float) / (tf.reduce_sum(weights_float) + 0.001)
        weights_float = tf.cast(tf.logical_and(shoot_mask, tf.logical_not(weights)), tf.float32)
        euclidean = tf.reduce_sum((diff * diff) * weights_float) / (tf.reduce_sum(weights_float) + 0.001) * 0.5
        return absolute + euclidean