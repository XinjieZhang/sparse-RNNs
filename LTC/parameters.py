# coding utf-8

import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


def get_variables(hp):

    params_init = {}

    n_input = hp['n_input']
    n_hidden = hp['n_hidden']

    sensory_mu = np.random.uniform(low=0.3, high=0.8, size=[n_input, n_hidden])
    sensory_mu_initializer = tf.constant_initializer(sensory_mu, dtype=tf.float32)
    params_init["sensory_mu"] = sensory_mu_initializer
    sensory_sigma = np.random.uniform(low=3, high=8, size=[n_input, n_hidden])
    sensory_sigma_initializer = tf.constant_initializer(sensory_sigma, dtype=tf.float32)
    params_init["sensory_sigma"] = sensory_sigma_initializer
    sensory_W = np.random.uniform(low=0.01, high=1.0, size=[n_input, n_hidden])
    sensory_W_initializer = tf.constant_initializer(sensory_W, dtype=tf.float32)
    params_init["sensory_W"] = sensory_W_initializer
    sensory_erev = 2 * np.random.randint(low=0, high=2, size=[n_input, n_hidden]) - 1
    sensory_erev_initializer = tf.constant_initializer(sensory_erev)
    params_init["sensory_erev"] = sensory_erev_initializer

    mu = np.random.uniform(low=0.3, high=0.8, size=[n_hidden, n_hidden])
    mu_initializer = tf.constant_initializer(mu, dtype=tf.float32)
    params_init["mu"] = mu_initializer
    sigma = np.random.uniform(low=3, high=8, size=[n_hidden, n_hidden])
    sigma_initializer = tf.constant_initializer(sigma, dtype=tf.float32)
    params_init["sigma"] = sigma_initializer
    W = np.random.uniform(low=0.01, high=1, size=[n_hidden, n_hidden])
    W_initializer = tf.constant_initializer(W, dtype=tf.float32)
    params_init["W"] = W_initializer
    erev = 2 * np.random.randint(low=0, high=2, size=[n_hidden, n_hidden]) - 1
    erev_initializer = tf.constant_initializer(erev)
    params_init["erev"] = erev_initializer

    # vleak = np.random.uniform(low=-0.2, high=0.2, size=[n_hidden])
    # vleak_initializer = tf.constant_initializer(vleak, dtype=tf.float32)
    vleak_initializer = tf.initializers.random_uniform(minval=-0.2, maxval=0.2)
    params_init["vleak"] = vleak_initializer
    # gleak = np.random.uniform(low=0.001, high=1.0, size=[n_hidden])
    # gleak_initializer = tf.constant_initializer(gleak, dtype=tf.float32)
    gleak_initializer = tf.initializers.constant(1)
    params_init["gleak"] = gleak_initializer
    # cm_t = np.random.uniform(low=0.4, high=0.6, size=[n_hidden])
    # cm_t_initializer = tf.constant_initializer(cm_t, dtype=tf.float32)
    cm_t_initializer = tf.initializers.constant(0.5)
    params_init["cm_t"] = cm_t_initializer

    # input_w = np.ones([n_input])
    # input_w_initializer = tf.constant_initializer(input_w)
    input_w_initializer = tf.initializers.constant(1)
    params_init["input_w"] = input_w_initializer
    # input_b = np.zeros([n_input])
    # input_b_initializer = tf.constant_initializer(input_b)
    input_b_initializer = tf.initializers.constant(0)
    params_init["input_b"] = input_b_initializer

    params_init['output_w'] = None
    params_init['output_b'] = None

    return params_init

