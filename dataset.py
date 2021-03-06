import numpy as np
import tensorflow as tf
import h5py
import json
import sys
from random import shuffle
from block_list import *

class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, params, block=False):
        super().__init__()
        self.params = params
        self.label_file = params.label_file
        self.visual_file = params.visual_file
        self.language_file = params.language_file

        with open(self.label_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        self.feature_visual = h5py.File(self.visual_file, 'r')
        self.feature_language = h5py.File(self.language_file, 'r')

        self.keys = list(self.labels.keys())
        miss_count = 0
        keys_refresh = []
        for k in self.keys:
            if k not in self.feature_visual or k not in self.feature_language:
                miss_count += 1
            elif block and k in block_list:
                miss_count += 1
            else:
                keys_refresh.append(k)     
        tf.logging.info('total unable files: %d' % miss_count)
        self.keys = keys_refresh


    def get_train_eval_input(self, stage):
        with tf.device("/cpu:0"):
            used_labels = []
            for k in self.keys:
                if self.labels[k]["subset"] == stage:
                    for idx in range(len(self.labels[k]["timestamps"])):
                        if "didemo" in self.params.label_file:
                            lf_id = idx // 4
                        else:
                            lf_id =  idx
                        used_labels.append([k, lf_id, self.labels[k]["duration"], 
                            self.labels[k]["timestamps"][idx], self.labels[k]["sentences"][idx]])

            if stage == "validation":
                self.val_label_num = len(used_labels)

            dataset = tf.data.Dataset.range(len(used_labels))
            if stage == "train":
                shuffle(used_labels)

            dataset = dataset.map(
                lambda ind: tf.py_func(
                    lambda ind: (
                        used_labels[ind][0],
                        np.float32(used_labels[ind][2]),
                        np.float32(used_labels[ind][3]),
                        used_labels[ind][4],
                        self.feature_visual[used_labels[ind][0]][:].astype(np.float32),
                        self.feature_language[used_labels[ind][0]][str(used_labels[ind][1])][:].astype(np.float32),
                    ),
                    [ind],
                    [tf.string, tf.float32, tf.float32, tf.string, tf.float32, tf.float32]
                )
            )
                
            dataset = dataset.map(
                lambda video, duration, timestamps, sentences, feature_visual, feature_language: (
                    video,
                    duration,
                    timestamps,
                    sentences,
                    tf.squeeze(tf.image.resize_images(
                        tf.reshape(feature_visual, [-1, self.params.visual_size, 1]), 
                        [self.params.k2scale, self.params.visual_size]
                    ), -1),
                    tf.pad(
                        feature_language, 
                        [[0, tf.maximum(self.params.word_length - tf.shape(feature_language)[0], 0)], [0, 0]]
                    )[:self.params.word_length, :],
                    tf.shape(feature_language)[0]
                ),
            )

            #dataset = dataset.cache()
            dataset = dataset.repeat()
            if stage == "train":
                dataset = dataset.shuffle(self.params.buffer_size)

            # Create iterator
            dataset = dataset.batch(self.params.batch_size)
            dataset = dataset.prefetch(self.params.pre_fetch)
            dataset = dataset.map(
                lambda video, duration, timestamps, sentences, feature_visual, feature_language, language_length: {
                    "video": video,
                    "duration": duration,
                    "timestamps": timestamps,
                    "sentences": sentences,
                    "feature_visual": feature_visual,
                    "feature_language": feature_language,
                    "language_length": language_length
                },
            )
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()

            # Convert to dictionary
            features["video"].set_shape(tf.TensorShape([None]))
            features["duration"].set_shape(tf.TensorShape([None]))
            features["timestamps"].set_shape(tf.TensorShape([None, 2]))
            features["feature_visual"].set_shape(tf.TensorShape([None, self.params.k2scale, self.params.visual_size]))
            features["feature_language"].set_shape(tf.TensorShape([None, None, self.params.language_size]))
            features["sentences"].set_shape(tf.TensorShape([None]))
            features["language_length"].set_shape(tf.TensorShape([None]))
            
            max_len = tf.maximum(tf.reduce_max(features["language_length"]), self.params.word_length)
            features["feature_language"] = features["feature_language"][:, :max_len, :]
            return features

