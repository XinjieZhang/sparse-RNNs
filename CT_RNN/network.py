# coding utf-8

import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from ctrnn_model import CTRNN

from utils.tools import load_hp, print_variables


class Model:
    def __init__(self,
                 model_dir,
                 input_kernel_initializer,
                 recurrent_kernel_initializer,
                 recurrent_bias_initializer=None,
                 output_kernel_initializer=None,
                 output_bias_initializer=None,
                 hp=None
                 ):

        # Reset Tensorflow before running anything
        tf.reset_default_graph()

        if hp is None:
            hp = load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])

        self.model_dir = model_dir
        self.hp = hp

        n_input = hp['n_input']
        n_classes = hp['n_classes']
        n_hidden = hp['n_hidden']

        self.x = tf.placeholder(tf.float32, [None, None, n_input])
        self.y = tf.placeholder(tf.int32, [None, None])

        self.rec_kernel_ini = recurrent_kernel_initializer
        self.input_kernel_ini = input_kernel_initializer
        self.rec_bias_ini = recurrent_bias_initializer
        self.output_kernel_ini = output_kernel_initializer
        self.output_bias_ini = output_bias_initializer

        head = self.x
        self.cell = CTRNN(n_hidden,
                          recurrent_kernel_initializer=self.rec_kernel_ini,
                          input_kernel_initializer=self.input_kernel_ini,
                          recurrent_bias_initializer=self.rec_bias_ini)

        head, _ = tf.nn.dynamic_rnn(self.cell, head, dtype=tf.float32, time_major=True)

        if self.output_kernel_ini is None:
            self.output_kernel_ini = tf.random_normal_initializer(dtype=tf.float32)
        if self.output_bias_ini is None:
            self.output_bias_ini = tf.constant_initializer(0.0, dtype=tf.float32)

        self.y_hat = tf.layers.Dense(units=n_classes,
                                     activation=None,
                                     kernel_initializer=self.output_kernel_ini,
                                     bias_initializer=self.output_bias_ini)(head)
        self.cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.y_hat))
        model_prediction = tf.argmax(input=self.y_hat, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.y, tf.int64)), tf.float32))

        self.var_list = tf.trainable_variables()

        for v in self.var_list:
            if 'input_kernel' in v.name:
                self.w_in = v
            elif 'recurrent_kernel' in v.name:
                self.w_rec = v
            elif 'recurrent_bias' in v.name:
                self.b_rec = v
            elif 'dense/kernel' in v.name:
                self.w_out = v
            elif 'dense/bias' in v.name:
                self.b_out = v

        # Create an optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=hp['learning_rate'])

        # set cost
        self.set_optimizer()

        # Variable saver
        self.saver = tf.train.Saver()

    def set_optimizer(self):

        # print("Variables being optimized:")
        # print_variables()

        self.grads_and_vars = self.opt.compute_gradients(self.cost, self.var_list)
        # Apply any applicable weights masks to the gradient and clip
        capped_gvs = []
        for grad, var in self.grads_and_vars:
            if 'input_kernel' in var.op.name:
                if 'w_in_mask' in self.hp:
                    grad *= self.hp['w_in_mask']
            elif 'recurrent_kernel' in var.op.name:
                if 'w_rec_mask' in self.hp:
                    grad *= self.hp['w_rec_mask']
            elif 'dense/kernel' in var.op.name:
                if 'w_out_mask' in self.hp:
                    grad *= self.hp['w_out_mask']
            capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
        self.train_step = self.opt.apply_gradients(capped_gvs)

    # https://github.com/gyyang/multitask/blob/master/network.py
    def restore(self, load_dir=None):
        """restore the model"""
        sess = tf.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    # https://github.com/gyyang/multitask/blob/master/network.py
    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)
