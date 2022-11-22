# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/9 下午5:04
# @Project: unified_chinese_multi_tasking_framework
# @File: layers.py
# @Software: PyCharm

'''
python
meta_cell = rnn_cell.BasicLSTMCell(meta_cell_unit_nums, state_is_tuple=False)
lstm_cell = MetaLSTMCell(unit_nums, meta_cell)
outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, ph_seqLen,
                        dtype=tf.float32 ,swap_memory=True, scope = 'meta-lstm-')

init a meta_cell with a shared scope, this work can be solved.  Pay attention to the  `reuse_variables()`
'''
from __future__ import division

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
from tensorflow.python.ops import math_ops

from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid, tanh
LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


class MyLSTM(tf.nn.rnn_cell.BasicLSTMCell):

    def __call__(self, inputs, state, scope=None):
        """LSTM as mentioned in paper."""
        with vs.variable_scope(scope or "basic_lstm_cell"):
            # Parameters of gates are concatenated into one multiply for
            # efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(
                    value=state, num_or_size_splits=2, split_dim=1)
            g = tf.concat(1, [inputs, h])
            concat = linear([g], 4 * self._num_units, True, scope=scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(
                value=concat, num_split=4, split_dim=1)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat_v2([new_c, new_h], 1)
            return new_h, new_state


class MetaLSTM(rnn_cell.RNNCell):
    def __init__(self, num_units, meta_lstm_cell=None, activation=tf.tanh):
        self._num_units = num_units
        self._activation = activation
        self._meta_lstm_cell = meta_lstm_cell
        self._meta_num_units = self._meta_lstm_cell.output_size
        self._total_num_units = self._num_units + self._meta_num_units

    @property
    def state_size(self):
        return 2 * self._total_num_units

    @property
    def output_size(self):
        return self._num_units

    def get_meta_results(self, meta_output, input, dimensions, scope='meta'):
        with tf.variable_scope(scope):
            W_matrix_list = []
            input_shape = int(input.get_shape()[-1])

        for i in np.arange(4):
            P = tf.get_variable('P{}'.format(i), shape=[self._meta_num_units, dimensions],
                                initializer=tf.uniform_unit_scaling_initializer(), dtype=tf.float32)
            Q = tf.get_variable('Q{}'.format(i), shape=[self._meta_num_units, input_shape],
                                initializer=tf.uniform_unit_scaling_initializer(), dtype=tf.float32)

            _W_matrix = tf.matmul(tf.reshape(tf.matrix_diag(meta_output), [-1, self._meta_num_units]), P)
            _W_matrix = tf.reshape(_W_matrix, [-1, self._meta_num_units, dimensions])
            _W_matrix = tf.matmul(tf.reshape(tf.transpose(_W_matrix, [0, 2, 1]), [-1, self._meta_num_units]), Q)
            _W_matrix = tf.reshape(_W_matrix, [-1, dimensions, input_shape])

            W_matrix_list.append(_W_matrix)

        W = tf.concat(values=W_matrix_list, axis=1)
        B = rnn_cell._linear(meta_output, 4 * dimensions, False)

        results = tf.matmul(W, tf.expand_dims(input, -1))
        results = tf.add(tf.reshape(results, [-1, 4 * dimensions]), B)

        return results

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            total_h, total_c = tf.split(axis=1, num_or_size_splits=2, value=state)
            h = total_h[:, 0:self._num_units]
            c = total_c[:, 0:self._num_units]
            meta_state = tf.concat(values=[total_h[:, self._num_units:], total_c[:, self._num_units]], axis=1)
            meta_input = tf.concat(values=[inputs, h], axis=1)

            meta_output, meta_new_state = self._meta_lstm_cell(meta_input, meta_state)

            input_concat = tf.concat(values=[inputs, h], axis=1)
            lstm_gates = self.get_meta_results(meta_output, input_concat, self._num_units, scope='meta_result')
            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=lstm_gates)
            new_c = (c * math_ops.sigmoid(f) + math_ops.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * math_ops.sigmoid(o)

            meta_h, meta_c = tf.split(axis=1, num_or_size_splits=2, value=meta_new_state)
            new_total_h = tf.concat(values=[new_h, meta_h], axis=1)
            new_total_c = tf.concat(values=[new_c, meta_c], axis=1)
            new_total_state = tf.concat(values=[new_total_h, new_total_c], axis=1)

            return new_h, new_total_state

