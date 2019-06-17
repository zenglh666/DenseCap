from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import interface
from utils import *
def generate_proposal(multiply_gap, multiply_number):
    proposal = []
    proposal.append([0., 1.])
    count = 1
    number = 1.
    gap = 1.
    while count < 100:
        gap = gap / multiply_gap
        number = np.maximum(number * multiply_number, 2.)
        start_middle = gap / 2.
        end_middle = 1. - start_middle
        duration = end_middle - start_middle
        stride = duration / float(int(number) - 1)
        for i in range(int(number)):
            middel = start_middle + i * stride
            proposal.append([middel - gap / 2., middel + gap / 2.])
            count += 1
            if count >= 100:
                break
    return np.array(proposal, dtype=np.float32)


def model_graph(features, mode, params):
    timestamps = features["timestamps"]
    duration = features['duration']
    timestamps_length = features["timestamps_length"]

    proposal = tf.py_func(
        generate_proposal,
        [params.multiply_gap, params.multiply_number],
        [tf.float32]
    )

    probability = tf.cast(tf.range(100), tf.float32) / 100.

    proposal = tf.tile(tf.reshape(proposal, [1, -1, 2]), [tf.shape(timestamps)[0], 1, 1])
    probability = tf.tile(tf.reshape(probability, [1, -1]), [tf.shape(timestamps)[0], 1])

    duration_exp = tf.reshape(duration, [-1, 1, 1])
    timestamps = timestamps / duration_exp


    if mode == "infer":
        # Prediction
        return prob_top_k, proposal_top_k

    auc = get_recall(params, proposal, timestamps)
    if mode == "eval":
        proposal = proposal * duration_exp
        return auc, probability, proposal

    with tf.variable_scope("loss"):
        loss_dict = {}
        loss = tf.get_variable(shape=[1], name="localization")

        return loss, tf.reduce_sum(auc), tf.cast(tf.reduce_sum(timestamps_length), tf.float32), loss_dict


class GGG(interface.NMTModel):

    def __init__(self, params, scope="ggg"):
        super(GGG, self).__init__(params=params, scope=scope)

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
        return "ggg"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # model
            embedding_size=500,
            base_layer=2,
            anchor_layers=10,
            batch_size=1,
            multiplier=1,
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
            multiply_gap = 2.,
            multiply_number = 3.,
        )

        return params
