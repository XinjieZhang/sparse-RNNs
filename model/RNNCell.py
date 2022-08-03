# coding utf-8
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops

from tensorflow.python.ops.rnn_cell_impl import RNNCell


class BasicRNNCell(RNNCell):
    # https://github.com/gyyang/multitask/blob/master/network.py
    def __init__(self,
                 num_units,
                 recurrent_kernel_initializer,
                 input_kernel_initializer,
                 recurrent_bias_initializer=None,
                 activation=None,
                 reuse=None,
                 name=None):
        super(BasicRNNCell, self).__init__(_reuse=reuse, name=name)

        self._num_units = num_units
        self._recurrent_initializer = recurrent_kernel_initializer
        self._input_initializer = input_kernel_initializer
        self._bias_initializer = recurrent_bias_initializer
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._input_kernel = self.add_variable(
            "input_kernel",
            shape=[input_depth, self._num_units],
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

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        gate_inputs = math_ops.matmul(inputs, self._input_kernel)
        recurrent_update = math_ops.matmul(state, self._recurrent_kernel)
        gate_inputs = math_ops.add(gate_inputs, recurrent_update)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        output = self._activation(gate_inputs)
        return output, output


class FixedRNNCell(RNNCell):
    def __init__(self,
                 num_units,
                 recurrent_kernel_initializer,
                 input_kernel_initializer,
                 recurrent_bias_initializer=None,
                 activation=None,
                 reuse=None,
                 name=None):
        super(FixedRNNCell, self).__init__(_reuse=reuse, name=name)

        self._num_units = num_units
        self._recurrent_initializer = recurrent_kernel_initializer
        self._input_initializer = input_kernel_initializer
        self._bias_initializer = recurrent_bias_initializer
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._input_kernel = self.add_variable(
            "input_kernel",
            shape=[input_depth, self._num_units],
            trainable=True,
            initializer=self._input_initializer
        )

        self._recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            shape=[self._num_units, self._num_units],
            trainable=False,
            initializer=self._recurrent_initializer,
        )

        if self._bias_initializer is None:
            self._bias_initializer = init_ops.zeros_initializer()
        self._bias = self.add_variable(
            "recurrent_bias",
            shape=[self._num_units],
            trainable=True,
            initializer=self._bias_initializer
        )

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        gate_inputs = math_ops.matmul(inputs, self._input_kernel)
        recurrent_update = math_ops.matmul(state, self._recurrent_kernel)
        gate_inputs = math_ops.add(gate_inputs, recurrent_update)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        output = self._activation(gate_inputs)
        return output, output

