import os
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from enum import Enum

# https://github.com/raminmh/liquid_time_constant_networks/blob/master/experiments_with_ltcs/ltc_model.py

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2


class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2


class LTCCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 params_init
                 ):

        self._input_size = -1
        self._num_units = num_units
        self._is_built = False

        # Number of ODE solver steps in one RNN step
        self._ode_solver_unfolds = 6
        self._solver = ODESolver.SemiImplicit

        self._input_mapping = MappingType.Affine

        self._w_min_value = 0
        self._w_max_value = 1000
        self._gleak_min_value = 0.00001
        self._gleak_max_value = 1000
        self._cm_t_min_value = 0.000001
        self._cm_t_max_value = 1000

        self.sensory_mu_init = params_init['sensory_mu']
        self.sensory_sigma_init = params_init['sensory_sigma']
        self.sensory_W_init = params_init['sensory_W']
        self.sensory_erev_init = params_init["sensory_erev"]

        self.mu_init = params_init["mu"]
        self.sigma_init = params_init["sigma"]
        self.W_init = params_init["W"]
        self.erev_init = params_init["erev"]

        self.vleak_init = params_init["vleak"]
        self.gleak_init = params_init["gleak"]
        self.cm_t_init = params_init["cm_t"]

        self.input_w_init = params_init['input_w']
        self.input_b_init = params_init['input_b']

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _map_inputs(self, inputs, resuse_scope=False):
        varscope = "sensory_mapping"
        reuse = tf.AUTO_REUSE
        if (resuse_scope):
            varscope = self._sensory_varscope
            reuse = True

        with tf.variable_scope(varscope, reuse=reuse) as scope:
            self._sensory_varscope = scope
            if (self._input_mapping == MappingType.Affine or self._input_mapping == MappingType.Linear):
                w = tf.get_variable(
                    name='input_w',
                    shape=[self._input_size],
                    trainable=True,
                    initializer=self.input_w_init
                )
                inputs = inputs * w
            if (self._input_mapping == MappingType.Affine):
                b = tf.get_variable(
                    name='input_b',
                    shape=[self._input_size],
                    trainable=True,
                    initializer=self.input_b_init
                )
                inputs = inputs + b
        return inputs

    # TODO: Implement RNNLayer properly,i.e, allocate variables here
    def build(self, input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope("ltc"):
            if (not self._is_built):
                # TODO: Move this part into the build method inherited form tf.Layers
                self._is_built = True
                self._input_size = int(inputs.shape[-1])

                self._get_variables()

            elif (self._input_size != int(inputs.shape[-1])):
                raise ValueError(
                    "You first feed an input with {} features and now one with {} features, that is not possible".format(
                        self._input_size,
                        int(inputs[-1])
                    ))

            inputs = self._map_inputs(inputs)

            if (self._solver == ODESolver.Explicit):
                next_state = self._ode_step_explicit(inputs, state, _ode_solver_unfolds=self._ode_solver_unfolds)
            elif (self._solver == ODESolver.SemiImplicit):
                next_state = self._ode_step(inputs, state)
            elif (self._solver == ODESolver.RungeKutta):
                next_state = self._ode_step_runge_kutta(inputs, state)
            else:
                raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))

            outputs = next_state

        return outputs, next_state

    # Create tf variables
    def _get_variables(self):
        self.sensory_mu = tf.get_variable(
            name='sensory_mu',
            shape=[self._input_size, self._num_units],
            trainable=True,
            initializer=self.sensory_mu_init
        )
        self.sensory_sigma = tf.get_variable(
            name='sensory_sigma',
            shape=[self._input_size, self._num_units],
            trainable=True,
            initializer=self.sensory_sigma_init
        )
        self.sensory_W = tf.get_variable(
            name='sensory_W',
            shape=[self._input_size, self._num_units],
            trainable=True,
            initializer=self.sensory_W_init
        )
        self.sensory_erev = tf.get_variable(
            name='sensory_erev',
            shape=[self._input_size, self._num_units],
            trainable=True,
            initializer=self.sensory_erev_init
        )

        self.mu = tf.get_variable(
            name='mu',
            shape=[self._num_units, self._num_units],
            trainable=True,
            initializer=self.mu_init
        )
        self.sigma = tf.get_variable(
            name='sigma',
            shape=[self._num_units, self._num_units],
            trainable=True,
            initializer=self.sigma_init
        )
        self.W = tf.get_variable(
            name='W',
            shape=[self._num_units, self._num_units],
            trainable=True,
            initializer=self.W_init
        )
        self.erev = tf.get_variable(
            name='erev',
            shape=[self._num_units, self._num_units],
            trainable=True,
            initializer=self.erev_init
        )
        self.vleak = tf.get_variable(
            name='v_leak',
            shape=[self._num_units],
            trainable=True,
            initializer=self.vleak_init
        )
        self.gleak = tf.get_variable(
            name='gleak',
            shape=[self._num_units],
            trainable=True,
            initializer=self.gleak_init
        )
        self.cm_t = tf.get_variable(
            name='cm_t',
            shape=[self._num_units],
            trainable=True,
            initializer=self.cm_t_init
        )

    # Hybrid euler method
    def _ode_step(self, inputs, state):
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = tf.reduce_sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = tf.reduce_sum(sensory_w_activation, axis=1)

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            rev_activation = w_activation * self.erev

            # Reduce over dimension 1 (=source neurons)
            w_numerator = tf.reduce_sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = tf.reduce_sum(w_activation, axis=1) + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        return v_pre

    def _f_prime(self, inputs, state):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = tf.reduce_sum(sensory_w_activation, axis=1)

        # Unfold the mutliply ODE multiple times into one RNN step
        w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

        w_reduced_synapse = tf.reduce_sum(w_activation, axis=1)

        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation

        sum_in = tf.reduce_sum(sensory_in, axis=1) - v_pre * w_reduced_synapse + tf.reduce_sum(synapse_in,
                                                                                               axis=1) - v_pre * w_reduced_sensory

        f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

        return f_prime

    def _ode_step_runge_kutta(self, inputs, state):

        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + k1 * 0.5)
            k3 = h * self._f_prime(inputs, state + k2 * 0.5)
            k4 = h * self._f_prime(inputs, state + k3)

            state = state + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return state

    def _ode_step_explicit(self, inputs, state, _ode_solver_unfolds):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        w_reduced_sensory = tf.reduce_sum(sensory_w_activation, axis=1)

        # Unfold the mutliply ODE multiple times into one RNN step
        for t in range(_ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            w_reduced_synapse = tf.reduce_sum(w_activation, axis=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = tf.reduce_sum(sensory_in, axis=1) - v_pre * w_reduced_synapse + tf.reduce_sum(synapse_in,
                                                                                                   axis=1) - v_pre * w_reduced_sensory

            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = tf.reshape(v_pre, [-1, v_pre.shape[-1], 1])
        mues = v_pre - mu
        x = sigma * mues
        return tf.nn.sigmoid(x)

    def get_param_constrain_op(self):

        cm_clipping_op = tf.assign(self.cm_t, tf.clip_by_value(self.cm_t, self._cm_t_min_value, self._cm_t_max_value))
        gleak_clipping_op = tf.assign(self.gleak, tf.clip_by_value(self.gleak, self._gleak_min_value, self._gleak_max_value))
        w_clipping_op = tf.assign(self.W, tf.clip_by_value(self.W, self._w_min_value, self._w_max_value))
        sensory_w_clipping_op = tf.assign(self.sensory_W, tf.clip_by_value(self.sensory_W, self._w_min_value, self._w_max_value))

        return [cm_clipping_op, gleak_clipping_op, w_clipping_op, sensory_w_clipping_op]

    def export_weights(self, dirname, sess, output_weights=None):
        os.makedirs(dirname, exist_ok=True)
        w, erev, mu, sigma = sess.run([self.W, self.erev, self.mu, self.sigma])
        sensory_w, sensory_erev, sensory_mu, sensory_sigma = sess.run([self.sensory_W, self.sensory_erev, self.sensory_mu, self.sensory_sigma])
        vleak, gleak, cm = sess.run([self.vleak, self.gleak, self.cm_t])

        if (not output_weights is None):
            output_w, output_b = sess.run(output_weights)
            np.savetxt(os.path.join(dirname, "output_w.txt"), output_w)
            np.savetxt(os.path.join(dirname, "output_b.txt"), output_b)
        np.savetxt(os.path.join(dirname, "w.txt"), w)
        np.savetxt(os.path.join(dirname, "erev.txt"), erev)
        np.savetxt(os.path.join(dirname, "mu.txt"), mu)
        np.savetxt(os.path.join(dirname, "sigma.txt"), sigma)
        np.savetxt(os.path.join(dirname, "sensory_w.txt"), sensory_w)
        np.savetxt(os.path.join(dirname, "sensory_erev.txt"), sensory_erev)
        np.savetxt(os.path.join(dirname, "sensory_mu.txt"), sensory_mu)
        np.savetxt(os.path.join(dirname, "sensory_sigma.txt"), sensory_sigma)
        np.savetxt(os.path.join(dirname, "vleak.txt"), vleak)
        np.savetxt(os.path.join(dirname, "gleak.txt"), gleak)
        np.savetxt(os.path.join(dirname, "cm.txt"), cm)
