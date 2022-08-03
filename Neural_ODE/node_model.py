# coding utf-8

import os
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops


class NODE(tf.nn.rnn_cell.RNNCell):
    # https://github.com/raminmh/liquid_time_constant_networks/blob/master/experiments_with_ltcs/ctrnn_model.py
    def __init__(self,
                 num_units,
                 recurrent_kernel_initializer,
                 input_kernel_initializer,
                 recurrent_bias_initializer=None,
                 cell_clip=-1,
                 activation=None,
                 reuse=None,
                 name=None):
        super(NODE, self).__init__(_reuse=reuse, name=name)

        self._num_units = num_units
        self._recurrent_initializer = recurrent_kernel_initializer
        self._input_initializer = input_kernel_initializer
        self._bias_initializer = recurrent_bias_initializer
        self._activation = activation or math_ops.tanh

        # Number of ODE solver steps
        self._unfolds = 6
        # Time of each ODE solver step, for variable time RNN change this
        # to a placeholder/non-trainable variable
        self._delta_t = 0.01

        self.cell_clip = cell_clip

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def export_weights(self, dirname, sess, output_weights=None):
        os.makedirs(dirname, exist_ok=True)
        w_in, w_rec, b_rec = sess.run([self._input_kernel, self._recurrent_kernel, self._bias])

        if (not output_weights is None):
            output_w, output_b = sess.run(output_weights)
            np.savetxt(os.path.join(dirname, "output_w.txt"), output_w)
            np.savetxt(os.path.join(dirname, "output_b.txt"), output_b)

        np.savetxt(os.path.join(dirname, "w_in.txt"), w_in)
        np.savetxt(os.path.join(dirname, "w_rec.txt"), w_rec)
        np.savetxt(os.path.join(dirname, "b_rec.txt"), b_rec)

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self, input_shape):
        pass

    def _dense(self, inputs, state):
        input_size = int(inputs.shape[-1])
        self._input_kernel = self.add_variable(
            "input_kernel",
            shape=[input_size, self._num_units],
            initializer=self._input_initializer
        )

        self._recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            shape=[self._num_units, self._num_units],
            initializer=self._recurrent_initializer
        )

        if self._bias_initializer is None:
            self._bias_initializer = init_ops.zeros_initializer()
        self._bias = self.add_variable(
            "recurrent_bias",
            shape=[self._num_units],
            initializer=self._bias_initializer
        )

        gate_inputs = math_ops.matmul(inputs, self._input_kernel)
        recurrent_update = math_ops.matmul(state, self._recurrent_kernel)
        gate_inputs = math_ops.add(gate_inputs, recurrent_update)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        y = self._activation(gate_inputs)

        return y

    def __call__(self, inputs, state, scope=None):

        self._input_size = int(inputs.shape[-1])
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):  # Reset gate and update gate.

                for i in range(self._unfolds):
                    k1 = self._delta_t * self._dense(inputs, state)
                    k2 = self._delta_t * self._dense(inputs, state + k1 * 0.5)
                    k3 = self._delta_t * self._dense(inputs, state + k2 * 0.5)
                    k4 = self._delta_t * self._dense(inputs, state + k3)

                    state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

                    # Optional clipping of the RNN cell to enforce stability (not needed)
                    if (self.cell_clip > 0):
                        state = tf.clip_by_value(state, -self.cell_clip, self.cell_clip)

        return state, state
