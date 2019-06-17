# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator

import numpy as np
import tensorflow as tf
import h5py
import json
import sys
from random import shuffle
from block_list import *

timestamps_pad_size = 30
label_pad_size = 64

features_hdf5 = None
bert_hdf5 = None

def get_train_eval_input(feature_filename, caption_filename, train, params):
    global timestamps_pad_size
    if params.localize:
        timestamps_pad_size = 1
        if params.use_bert:
            return get_train_eval_input_v3(feature_filename, caption_filename, train, params)
        else:
            return get_train_eval_input_v2(feature_filename, caption_filename, train, params)
    else:
        return get_train_eval_input_v1(feature_filename, caption_filename, train, params)

def get_train_eval_input_v1(feature_filename, caption_filename, train, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        features_hdf5 = h5py.File(feature_filename,'r')
        tf.add_to_collection('feature_file', features_hdf5)
        with open(caption_filename, 'r',encoding='utf-8') as f:
            caption = json.load(f)
        keys = list(caption['caption'].keys())

        count = 0
        keys_refresh = []
        for k in keys:
            if k not in features_hdf5:
                count += 1
            elif (not train) and (params.dataset == 'proposal') and (k in block_list):
                count += 1
            else:
                keys_refresh.append(k)
        tf.logging.info('total unable files: %d' % count)
        keys = keys_refresh

        dataset = tf.data.Dataset.range(len(keys))
        if params.task == "caption" or params.task == "charades":
            label = "sentences"
        elif params.task == "proposal":
            label = "labels"

        shuffle(keys)

        dataset = dataset.map(
            lambda ind: tf.py_func(
                lambda ind: (
                    features_hdf5[keys[ind]]['c3d_features'][:].astype(np.float32),
                    np.array(caption['caption'][keys[ind]]['timestamps']).astype(np.float32),
                    len(caption['caption'][keys[ind]]['timestamps']),
                    np.float32(caption['caption'][keys[ind]]['duration']),
                    keys[ind],
                    np.array(
                        [np.pad(i, ((0, label_pad_size - i.size)), 'constant') for i in 
                            [np.reshape(np.array(j), [-1]) for j in caption['caption'][keys[ind]][label]]], np.int32),
                    np.array(
                        [np.array(j).size for j in caption['caption'][keys[ind]][label]], np.int32)
                ),
                [ind],
                [tf.float32, tf.float32, tf.int32, tf.float32, tf.string, tf.int32, tf.int32]
            )
        )
            
        if params.k2scale > 1:
            dataset = dataset.map(
                lambda feature, timestamps, timestamps_length, duration, fname, label, label_length: (
                    tf.squeeze(
                        tf.image.resize_images(
                            tf.reshape(feature, [-1, params.embedding_size, 1]), 
                            [params.k2scale, params.embedding_size]
                        ), 
                    -1),
                    timestamps, 
                    timestamps_length,
                    duration, 
                    fname,
                    label,
                    label_length,
                ),
            )

        if train:
            dataset = dataset.cache()
            dataset = dataset.shuffle(params.buffer_size)
            dataset = dataset.repeat()

        dataset = dataset.map(
            lambda feature, timestamps, timestamps_length, duration, fname, label, label_length: (
                feature,
                tf.pad(timestamps, [[0, timestamps_pad_size - tf.shape(timestamps)[0]], [0, 0]]), 
                timestamps_length,
                duration, 
                fname,
                tf.pad(label, [[0, timestamps_pad_size - tf.shape(label)[0]], [0, 0]]),
                tf.pad(label_length, [[0, timestamps_pad_size - tf.shape(label_length)[0]]]),
            ),
        )

        if params.random_input or (train and params.random_train) or ((not train) and params.random_val):
            dataset = dataset.map(
                lambda feature, timestamps, timestamps_length, duration, fname, label, label_length: (
                    #tf.random_uniform(shape=tf.shape(feature), minval=-1., maxval=1.),
                    tf.ones_like(feature),
                    timestamps,
                    timestamps_length,
                    duration,
                    fname, 
                    label, 
                    label_length
                ),
            )

        # Create iterator
        dataset = dataset.batch(params.batch_size)
        dataset = dataset.prefetch(params.pre_fetch)
        dataset = dataset.map(
            lambda feature, timestamps, timestamps_length, duration, fname, label, label_length: {
                "feature": feature,
                "timestamps": timestamps,
                "timestamps_length": timestamps_length,
                "duration": duration,
                "filename": fname,
                "label": label,
                "label_length": label_length
            },
        )
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Convert to dictionary
        features["feature"].set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(None)])
        features["duration"].set_shape([tf.Dimension(None)])
        features["timestamps"].set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(None)])
        features["timestamps_length"].set_shape([tf.Dimension(None)])
        features["filename"].set_shape([tf.Dimension(None)])
        features["label"].set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(None)])
        features["label_length"].set_shape([tf.Dimension(None), tf.Dimension(None)])

        features["timestamps"] = features["timestamps"][:, :tf.reduce_max(features["timestamps_length"]), :]
        features["label"] = features["label"][:, :tf.reduce_max(features["timestamps_length"]), :tf.reduce_max(features["label_length"])]

        return features

def get_train_eval_input_v2(feature_filename, caption_filename, train, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        global features_hdf5
        if features_hdf5 is None:
            features_hdf5 = h5py.File(feature_filename,'r')
            tf.add_to_collection('feature_file', features_hdf5)
        with open(caption_filename, 'r',encoding='utf-8') as f:
            caption = json.load(f)
        keys = list(caption['caption'].keys())

        count = 0
        keys_refresh = []
        for k in keys:
            if k not in features_hdf5:
                count += 1
            elif (not train) and (params.dataset == 'proposal') and (k in block_list):
                count += 1
            else:
                keys_refresh.append(k)
        tf.logging.info('total unable files: %d' % count)
        keys = keys_refresh

        if params.task == "proposal":
            label = "labels"
        else:
            label = "sentences"
        labels = []
        for k in keys:
            for idx in range(len(caption['caption'][k][label])):
                labels.append((k, caption['caption'][k]["timestamps"][idx], caption['caption'][k][label][idx]))

        dataset = tf.data.Dataset.range(len(labels))
        shuffle(labels)

        dataset = dataset.map(
            lambda ind: tf.py_func(
                lambda ind: (
                    features_hdf5[labels[ind][0]]['c3d_features'][:].astype(np.float32),
                    np.array([labels[ind][1]]).astype(np.float32),
                    np.int32(1),
                    np.float32(caption['caption'][labels[ind][0]]['duration']),
                    labels[ind][0],
                    np.array(labels[ind][2], np.int32),
                    np.array(len(labels[ind][2]), np.int32)
                ),
                [ind],
                [tf.float32, tf.float32, tf.int32, tf.float32, tf.string, tf.int32, tf.int32]
            )
        )
            
        if params.k2scale > 1:
            dataset = dataset.map(
                lambda feature, timestamps, timestamps_length, duration, fname, label, label_length: (
                    tf.squeeze(
                        tf.image.resize_images(
                            tf.reshape(feature, [-1, params.embedding_size, 1]), 
                            [params.k2scale, params.embedding_size]
                        ), 
                    -1),
                    timestamps, 
                    timestamps_length,
                    duration, 
                    fname,
                    label,
                    label_length,
                ),
            )

        dataset = dataset.cache()
        dataset = dataset.repeat()

        if train:
            dataset = dataset.shuffle(params.buffer_size)
            global train_label_num
            train_label_num = len(labels)
        else:
            global val_label_num
            val_label_num = len(labels)

        dataset = dataset.map(
            lambda feature, timestamps, timestamps_length, duration, fname, label, label_length: (
                feature,
                tf.pad(timestamps, [[0, timestamps_pad_size - tf.shape(timestamps)[0]], [0, 0]]), 
                timestamps_length,
                duration, 
                fname,
                tf.pad(label, [[0, tf.maximum(params.word_length - tf.shape(label)[0], 0)]])[:params.word_length],
                tf.minimum(label_length, params.word_length)
            ),
        )

        if params.random_input or (train and params.random_train) or ((not train) and params.random_val):
            dataset = dataset.map(
                lambda feature, timestamps, timestamps_length, duration, fname, label, label_length: (
                    #tf.random_uniform(shape=tf.shape(feature), minval=-1., maxval=1.),
                    tf.ones_like(feature),
                    timestamps,
                    timestamps_length,
                    duration,
                    fname, 
                    label, 
                    label_length
                ),
            )

        # Create iterator
        dataset = dataset.batch(params.batch_size)
        dataset = dataset.prefetch(params.pre_fetch)
        dataset = dataset.map(
            lambda feature, timestamps, timestamps_length, duration, fname, label, label_length: {
                "feature": feature,
                "timestamps": timestamps,
                "timestamps_length": timestamps_length,
                "duration": duration,
                "filename": fname,
                "label": label,
                "label_length": label_length
            },
        )
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Convert to dictionary
        features["feature"].set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(None)])
        features["duration"].set_shape([tf.Dimension(None)])
        features["timestamps"].set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(None)])
        features["timestamps_length"].set_shape([tf.Dimension(None)])
        features["filename"].set_shape([tf.Dimension(None)])
        features["label"].set_shape([tf.Dimension(None), tf.Dimension(None)])
        features["label_length"].set_shape([tf.Dimension(None)])

        features["timestamps"] = features["timestamps"][:, :tf.reduce_max(features["timestamps_length"]), :]
        max_len = tf.reduce_max(features["label_length"])
        features["label"] = features["label"][:, :max_len]

        return features


def get_train_eval_input_v3(feature_filename, caption_filename, train, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        global features_hdf5, bert_hdf5
        if features_hdf5 is None:
            features_hdf5 = h5py.File(feature_filename,'r')
        if bert_hdf5 is None:
            bert_hdf5 = h5py.File(params.bert_file,'r')

        with open(caption_filename, 'r',encoding='utf-8') as f:
            caption = json.load(f)
        keys = list(caption['caption'].keys())

        count = 0
        keys_refresh = []
        for k in keys:
            if k not in features_hdf5 or k not in bert_hdf5:
                count += 1
            elif (not train) and (params.dataset == 'proposal') and (k in block_list):
                count += 1
            else:
                keys_refresh.append(k)
        tf.logging.info('total unable files: %d' % count)
        keys = keys_refresh

        if params.task == "proposal":
            label = "labels"
        else:
            label = "sentences"
            
        labels = []
        for k in keys:
            for idx in range(len(caption['caption'][k][label])):
                labels.append((k, idx, caption['caption'][k]["timestamps"][idx], caption['caption'][k][label][idx]))

        for l in labels:
            if l[0] not in bert_hdf5:
                print(l[0])
            elif str(l[1]) not in bert_hdf5[l[0]]:
                print(l[0], l[1])

        dataset = tf.data.Dataset.range(len(labels))
        shuffle(labels)

        dataset = dataset.map(
            lambda ind: tf.py_func(
                lambda ind: (
                    features_hdf5[labels[ind][0]]['c3d_features'][:].astype(np.float32),
                    np.array([labels[ind][2]]).astype(np.float32),
                    np.int32(1),
                    np.float32(caption['caption'][labels[ind][0]]['duration']),
                    labels[ind][0],
                    np.array(labels[ind][3], np.int32),
                    np.array(len(labels[ind][3]), np.int32),
                    bert_hdf5[labels[ind][0]][str(labels[ind][1])][:].astype(np.float32)
                ),
                [ind],
                [tf.float32, tf.float32, tf.int32, tf.float32, tf.string, tf.int32, tf.int32, tf.float32]
            )
        )
            
        if params.k2scale > 1:
            dataset = dataset.map(
                lambda feature, timestamps, timestamps_length, duration, fname, label, label_length, bert: (
                    tf.squeeze(
                        tf.image.resize_images(
                            tf.reshape(feature, [-1, params.embedding_size, 1]), 
                            [params.k2scale, params.embedding_size]
                        ), 
                    -1),
                    timestamps, 
                    timestamps_length,
                    duration, 
                    fname,
                    label,
                    tf.shape(bert)[0],
                    bert
                ),
            )

        dataset = dataset.cache()
        dataset = dataset.repeat()

        if train:
            dataset = dataset.shuffle(params.buffer_size)
            global train_label_num
            train_label_num = len(labels)
        else:
            global val_label_num
            val_label_num = len(labels)
            

        dataset = dataset.map(
            lambda feature, timestamps, timestamps_length, duration, fname, label, label_length, bert: (
                feature,
                tf.pad(timestamps, [[0, timestamps_pad_size - tf.shape(timestamps)[0]], [0, 0]]), 
                timestamps_length,
                duration, 
                fname,
                tf.pad(label, [[0, tf.maximum(params.word_length - tf.shape(label)[0], 0)]])[:params.word_length],
                tf.minimum(label_length, params.word_length),
                tf.pad(bert, [[0, tf.maximum(params.word_length - tf.shape(bert)[0], 0)], [0, 0]])[:params.word_length],
            ),
        )

        # Create iterator
        dataset = dataset.batch(params.batch_size)
        dataset = dataset.prefetch(params.pre_fetch)
        dataset = dataset.map(
            lambda feature, timestamps, timestamps_length, duration, fname, label, label_length, bert: {
                "feature": feature,
                "timestamps": timestamps,
                "timestamps_length": timestamps_length,
                "duration": duration,
                "filename": fname,
                "label": label,
                "label_length": label_length,
                "bert": bert
            },
        )
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Convert to dictionary
        features["feature"].set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(None)])
        features["duration"].set_shape([tf.Dimension(None)])
        features["timestamps"].set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(None)])
        features["timestamps_length"].set_shape([tf.Dimension(None)])
        features["filename"].set_shape([tf.Dimension(None)])
        features["label"].set_shape([tf.Dimension(None), tf.Dimension(None)])
        features["label_length"].set_shape([tf.Dimension(None)])
        features["bert"].set_shape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(None)])

        features["timestamps"] = features["timestamps"][:, :tf.reduce_max(features["timestamps_length"]), :]
        max_len = tf.reduce_max(features["label_length"])
        features["label"] = features["label"][:, :max_len]
        features["bert"] = features["bert"][:, :max_len]
        return features

