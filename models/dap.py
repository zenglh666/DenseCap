from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import interface
from utils import *

def model_graph(features, mode, params):

    feature = features["feature"]
    feature_len = features["feature_length"]
    timestamps = features['timestamps'] / features['duration']
    batch_size = tf.shape(feature)[0]
    batch_len = tf.shape(feature)[1]

    if params.reverse_source:
        src_seq = tf.reverse_sequence(src_seq, seq_dim=1,
                                      seq_lengths=feature_len)


    if params.dropout and not params.use_variational_dropout:
        feature = tf.nn.dropout(feature, 1.0 - params.dropout)

    cell_enc = []
    for _ in range(params.num_hidden_layers):
        if params.rnn_cell == "LSTMCell":
            cell_e = tf.nn.rnn_cell.BasicLSTMCell(params.hidden_size)
        elif params.rnn_cell == "GRUCell":
            cell_e = tf.nn.rnn_cell.GRUCell(params.hidden_size)
        else:
            raise ValueError("%s not supported" % params.rnn_cell)

        cell_e = tf.nn.rnn_cell.DropoutWrapper(
            cell_e,
            output_keep_prob=1.0 - params.dropout,
            variational_recurrent=params.use_variational_dropout,
            input_size=params.embedding_size,
            dtype=tf.float32
        )

        if params.use_residual:
            cell_e = tf.nn.rnn_cell.ResidualWrapper(cell_e)

        cell_enc.append(cell_e)

    cell_enc = tf.nn.rnn_cell.MultiRNNCell(cell_enc)

    with tf.variable_scope("encoder"):
        _, final_state = tf.nn.dynamic_rnn(cell_enc, feature, feature_len,
                                           dtype=tf.float32)

    outputs = tf.concat([final_state[0].c, final_state[0].h], axis=1)
    outputs = tf.expand_dims(outputs, 0)
    
    back_event, probability, proposal = get_proposal(outputs, params.K)

    prob_top_k_list, proposal_top_k_list = choose_top(probability, proposal, params.k)

    if mode == "infer":
        # Prediction
        return prob_top_k_list, proposal_top_k_list

    
    recall_list = get_recall(proposal_top_k_list, timestamps, params.tiou)
    if mode == "eval":
        return recall_list

    with tf.variable_scope("loss"):
        tiou = tIoU(proposal, timestamps)

        euclidean = timestamps_shoot_euclidean_loss(tiou, proposal, timestamps)
        cross_entropy = timestamps_shoot_cross_entropy_loss(tiou, back_event)

        loss =  cross_entropy + euclidean
        return loss


class Dap(interface.NMTModel):

    def __init__(self, params, scope="dap"):
        super(Dap, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

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
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, "infer", params)

            return logits

        return inference_fn

    @staticmethod
    def get_name():
        return "dap"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # model
            rnn_cell="LSTMCell",
            embedding_size=500,
            hidden_size=512,
            num_hidden_layers=1,
            use_residual=False,
            # regularization
            dropout=0.0,
            use_variational_dropout=False,
            label_smoothing=0.1,
            max_length=1000,
            reverse_source=False,
            K=512,
            k=[5, 10, 15, 20, 30 ,50, 100],
            top=1000,
            tiou=[0.2, 0.5, 0.7, 0.8, 0.9]
        )

        return params
