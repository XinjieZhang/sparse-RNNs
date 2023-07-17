# coding utf-8

import os
import tensorflow as tf
from model.RNNCell import BasicRNNCell, FixedRNNCell

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

        if hp is None:
            hp = load_hp(model_dir)
            if hp is None:
                raise ValueError('No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])

        self.model_dir = model_dir
        self.hp = hp

        n_steps = hp['n_steps']
        n_input = hp['n_input']
        n_classes = hp['n_classes']
        n_hidden = hp['n_hidden']

        self.x = tf.placeholder(tf.float32, [None, n_steps, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_classes])

        self.rec_kernel_ini = recurrent_kernel_initializer
        self.input_kernel_ini = input_kernel_initializer
        self.rec_bias_ini = recurrent_bias_initializer
        self.output_kernel_ini = output_kernel_initializer
        self.output_bias_ini = output_bias_initializer

        if 'RNNCell' not in hp or hp['RNNCell'] == 'Basic':
            cell = BasicRNNCell(n_hidden,
                                recurrent_kernel_initializer=self.rec_kernel_ini,
                                input_kernel_initializer=self.input_kernel_ini,
                                recurrent_bias_initializer=self.rec_bias_ini)
        elif hp['RNNCell'] == 'Fixed':
            cell = FixedRNNCell(n_hidden,
                                recurrent_kernel_initializer=self.rec_kernel_ini,
                                input_kernel_initializer=self.input_kernel_ini,
                                recurrent_bias_initializer=self.rec_bias_ini)

        self.h, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)

        # Output
        with tf.variable_scope("output"):
            if self.output_kernel_ini is None:
                self.output_kernel_ini = tf.random_normal_initializer(dtype=tf.float32)
            w_out = tf.get_variable(
                'output_kernel',
                [n_hidden, n_classes],
                dtype=tf.float32,
                trainable=True,
                initializer=self.output_kernel_ini
            )
            if self.output_bias_ini is None:
                self.output_bias_ini = tf.constant_initializer(0.0, dtype=tf.float32)
            b_out = tf.get_variable(
                'output_bias',
                [n_classes],
                dtype=tf.float32,
                trainable=True,
                initializer=self.output_bias_ini
            )

            self.pred = tf.matmul(self.h[:, -1, :], w_out) + b_out
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
            
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.var_list = tf.trainable_variables()

        for v in self.var_list:
            if 'input_kernel' in v.name:
                self.w_in = v
            elif 'recurrent_kernel' in v.name:
                self.w_rec = v
            elif 'recurrent_bias' in v.name:
                self.b_rec = v
            elif 'output_kernel' in v.name:
                self.w_out = v
            elif 'output_bias' in v.name:
                self.b_out = v

        # Create an optimizer
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=hp['learning_rate'])

        # set cost
        self.set_optimizer()

        # Variable saver
        self.saver = tf.train.Saver()

    def set_optimizer(self):
        
        # Print Trainable Variables
        # print_variables()
        
        self.grads_and_vars = self.opt.compute_gradients(self.cost)
        # Apply any applicable weights masks to the gradient and clip
        capped_gvs = []
        for grad, var in self.grads_and_vars:
            if 'input_kernel' in var.op.name:
                if 'w_in_mask' in self.hp:
                    grad *= self.hp['w_in_mask']
            elif 'recurrent_kernel' in var.op.name:
                if 'w_rec_mask' in self.hp:
                    grad *= self.hp['w_rec_mask']
            elif 'output_kernel' in var.op.name:
                if 'w_out_mask' in self.hp:
                    grad *= self.hp['w_out_mask']
            capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
        self.train_step = self.opt.apply_gradients(capped_gvs)
        # self.train_step = self.opt.minimize(self.cost)

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

    def ablation_units(self, sess, mask):
        if mask is None:
            return

        for v in self.var_list:
            v_val = sess.run(v)
            if 'recurrent_kernel' in v.name:
                v_val = v_val * mask
            sess.run(v.assign(v_val))