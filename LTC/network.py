# coding utf-8

import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import ltc_model as ltc

from utils.tools import load_hp, print_variables


class Model:
    def __init__(self,
                 model_dir,
                 params_init,
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
        self.params_init = params_init
        self.hp = hp

        n_input = hp['n_input']
        n_hidden = hp['n_hidden']
        n_classes = hp['n_classes']

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, n_input])
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, None])

        self.output_w_init = self.params_init['output_w']
        self.output_b_init = self.params_init['output_b']

        head = self.x
        self.wm = ltc.LTCCell(n_hidden, self.params_init)
        self.wm._solver = ltc.ODESolver.SemiImplicit

        # Dynamic rnn with time major
        head, _ = tf.nn.dynamic_rnn(self.wm, head, dtype=tf.float32, time_major=True)
        self.constrain_op = self.wm.get_param_constrain_op()

        # Output
        self.y_hat = tf.layers.Dense(units=n_classes,
                                     activation=None,
                                     kernel_initializer=self.output_w_init,
                                     bias_initializer=self.output_b_init)(head)
        self.cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=self.y,
            logits=self.y_hat)
        )

        pred = tf.argmax(input=self.y_hat, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(self.y, tf.int64)), tf.float32))

        self.var_list = tf.trainable_variables()

        for v in self.var_list:
            if 'input_w' in v.name:
                self.input_w = v
            elif 'input_b' in v.name:
                self.input_b = v
            elif 'kernel' in v.name:
                self.output_w = v
            elif 'bias' in v.name:
                self.output_b = v

        # Create an optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=hp['learning_rate'])

        # set cost
        self.set_optimizer()

        # Variable saver
        self.saver = tf.train.Saver()

    def set_optimizer(self):

        var_list = self.var_list
        # print("Variables being optimized:")
        # for v in var_list:
        #     print(v)

        self.grads_and_vars = self.opt.compute_gradients(self.cost, var_list)
        # Apply any applicable weights masks to the gradient and clip
        capped_gvs = []
        # LTC can be sparsified by only setting w[i,j] to 0
        # both input and recurrent matrix will be sparsified
        for grad, var in self.grads_and_vars:
            if 'rnn/ltc/W' in var.op.name:
                if 'sparsity_mask' in self.hp:
                    grad *= self.hp['sparsity_mask']
            if 'rnn/ltc/sensory_W' in var.op.name:
                if 'sensory_sparsity_mask' in self.hp:
                    grad *= self.hp['sensory_sparsity_mask']
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
