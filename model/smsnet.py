# coding utf-8

import os
import tensorflow as tf
from model.RNNCell import BasicRNNCell, FixedRNNCell

from utils.tools import load_hp, print_variables


class Model:
    # https://github.com/NetworkRanger/tensorflow-ml-exercise
    def __init__(self,
                 model_dir,
                 input_kernel_initializer,
                 recurrent_kernel_initializer,
                 embedding_mat_initializer,
                 recurrent_bias_initializer=None,
                 output_kernel_initializer=None,
                 output_bias_initializer=None,
                 hp=None
                 ):
        
        # Reset Tensorflow graphs
        tf.reset_default_graph() # must be in the beginning

        if hp is None:
            hp = load_hp(model_dir)
            if hp is None:
                raise ValueError('No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])

        self.model_dir = model_dir
        self.hp = hp

        max_sequence_length = hp['max_sequence_length']
        vocab_size = hp['vocab_size']
        embedding_size = hp['embedding_size']
        n_hidden = hp['hidden_size']

        self.x = tf.placeholder(tf.int32, [None, max_sequence_length])
        self.y = tf.placeholder(tf.int32, [None])

        # Create embedding
        self.embedding_mat_ini = embedding_mat_initializer
        if self.embedding_mat_ini is None:
            self.embedding_mat_ini = tf.random_uniform_initializer(-1.0, 1.0, dtype=tf.float32)
        self.embed_matrix = tf.get_variable(
            'embedding_matrix',
            [vocab_size, embedding_size],
            dtype=tf.float32,
            initializer=self.embedding_mat_ini,
            trainable=True
        )
        self.embedding_output = tf.nn.embedding_lookup(self.embed_matrix, self.x)

        # RNN Cell
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

        self.h, states = tf.nn.dynamic_rnn(cell, self.embedding_output, dtype=tf.float32)
        self.output = tf.transpose(self.h, [1, 0, 2])
        self.y_hat = tf.gather(self.output, int(self.output.get_shape()[0]) - 1)

        # Output
        with tf.variable_scope("output"):
            if self.output_kernel_ini is None:
                self.output_kernel_ini = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
            w_out = tf.get_variable(
                'output_kernel',
                [n_hidden, 2],
                dtype=tf.float32,
                trainable=True,
                initializer=self.output_kernel_ini
            )
            if self.output_bias_ini is None:
                self.output_bias_ini = tf.constant_initializer(0.1, dtype=tf.float32)
            b_out = tf.get_variable(
                'output_bias',
                [2],
                dtype=tf.float32,
                trainable=True,
                initializer=self.output_bias_ini
            )

            self.logist = tf.nn.softmax(tf.matmul(self.y_hat, w_out) + b_out)

            # loss function
            self.cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logist, labels=self.y))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logist, 1),
                                                       tf.cast(self.y, tf.int64)), tf.float32))
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
        elif hp['optimizer'] == 'RMSProp':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=hp['learning_rate'])
            
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