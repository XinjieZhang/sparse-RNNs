# coding utf-8

import os
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops


class CTRNN(tf.nn.rnn_cell.RNNCell):
    # https://github.com/raminmh/liquid_time_constant_networks/blob/master/experiments_with_ltcs/ctrnn_model.py
    def __init__(self,
                 num_units,
                 recurrent_kernel_initializer,
                 input_kernel_initializer,
                 recurrent_bias_initializer=None,
                 cell_clip=-1,
                 fix_tau=True,
                 activation=None,
                 reuse=None,
                 name=None):
        super(CTRNN, self).__init__(_reuse=reuse, name=name)

        self._num_units = num_units
        self._recurrent_initializer = recurrent_kernel_initializer
        self._input_initializer = input_kernel_initializer
        self._bias_initializer = recurrent_bias_initializer
        self._activation = activation or math_ops.tanh

        # Number of ODE solver steps
        self._unfolds = 6
        # Time of each ODE solver step, for variable time RNN change this
        # to a placeholder/non-trainable variable
        self._delta_t = 0.1
        
        # Time-constant of the cell
        self.fix_tau = fix_tau
        self.tau = 0.5
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

        tau = np.ones(1)
        if (not self.fix_tau):
            sp_op = tf.nn.softplus(self._tau_var)
            tau = sess.run(sp_op)

        if (not output_weights is None):
            output_w, output_b = sess.run(output_weights)
            np.savetxt(os.path.join(dirname, "output_w.txt"), output_w)
            np.savetxt(os.path.join(dirname, "output_b.txt"), output_b)

        np.savetxt(os.path.join(dirname, "w_in.txt"), w_in)
        np.savetxt(os.path.join(dirname, "w_rec.txt"), w_rec)
        np.savetxt(os.path.join(dirname, "b_rec.txt"), b_rec)
        np.savetxt(os.path.join(dirname, "tau.txt"), tau)

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
        # CTRNN ODE is: df/dt = NN(x) - f
        # where x is the input, and NN is a MLP.
        # Input could be: 1: just the input of the RNN cell
        # or 2: input of the RNN cell merged with the current state

        self._input_size = int(inputs.shape[-1])
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):  # Reset gate and update gate.
                if (not self.fix_tau):
                    tau = tf.get_variable('tau', [], initializer=tf.constant_initializer(self.tau))
                    self._tau_var = tau
                    tau = tf.nn.softplus(tau)  # Make sure tau is positive
                else:
                    tau = self.tau

                for i in range(self._unfolds):
                    input_f_prime = self._dense(inputs=inputs, state=state)

                    # df/dt
                    f_prime = -state / tau + input_f_prime

                    # If we solve this ODE with explicit euler we get
                    # f(t+deltaT) = f(t) + deltaT * df/dt
                    state = state + self._delta_t * f_prime

                    # Optional clipping of the RNN cell to enforce stability (not needed)
                    if (self.cell_clip > 0):
                        state = tf.clip_by_value(state, -self.cell_clip, self.cell_clip)

        return state, state
