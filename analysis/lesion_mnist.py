# coding utf-8
# Before running the code you need to run mnist.py to generate the network

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import random

import sys
sys.path.append('../')
from model.smnistnet import Model
from utils.tools import load_hp
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class Disruption_balance(object):

    def __init__(self, model_dir):

        self.data = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)

        self.model_dir = model_dir
        self.hp = load_hp(model_dir)

        self.n_input = self.hp['n_input']
        self.n_steps = self.hp['n_steps']
        self.n_hidden = self.hp['n_hidden']
        self.feedforward_motifs = self.hp['feedforward_motifs']
        rng = np.random.RandomState(self.hp['seed'])

        w_in0 = rng.randn(self.n_input, self.n_hidden) / np.sqrt(self.n_input)
        input_kernel_initializer = tf.constant_initializer(w_in0, dtype=tf.float32)

        w_rec0 = rng.uniform(low=-0.7, high=0.7, size=[self.n_hidden, self.n_hidden])
        recurrent_kernel_initializer = tf.constant_initializer(w_rec0, dtype=tf.float32)

        tf.reset_default_graph()

        self.model = Model(model_dir,
                           input_kernel_initializer=input_kernel_initializer,
                           recurrent_kernel_initializer=recurrent_kernel_initializer,
                           )

    def disruption(self):

        results = []
        for i in range(10):
            with tf.Session() as sess:
                model = self.model
                model.restore()

                w_rec = sess.run(model.w_rec)

                test_acc = self.prediction(sess, model)
                # print("Testing Accuracy: {:.6f}".format(test_acc))

                unbalanced_motifs = []
                for i, j, k in self.feedforward_motifs:
                    if w_rec[i][k] * w_rec[i][j] * w_rec[j][k] < 0:
                        unbalanced_motifs.append((i, j, k))
                random.shuffle(unbalanced_motifs)
                # print(unbalanced_motifs)

                is_connected = np.greater(abs(w_rec), 0).astype(int)

                w_mask = is_connected.copy()
                prediction_acc = [test_acc*100]
                for (i, j, k) in unbalanced_motifs:
                    w_mask[i][k] = 0
                    w_mask[i][j] = 0
                    w_mask[j][k] = 0

                    model.ablation_units(sess, w_mask)

                    new_test_acc = self.prediction(sess, model)
                    prediction_acc.append(new_test_acc*100)
                    # print("Testing Accuracy: {:.6f}".format(new_test_acc))
            results.append(prediction_acc)

        results = np.array(results)

        return results.mean(axis=0), results.std(axis=0)

    # validation
    def prediction(self, sess, model):
        # mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
        test_len = 10000
        test_data = self.data.test.images[:test_len].reshape((-1, 28, 28))
        test_label = self.data.test.labels[:test_len]
        test_acc = sess.run(model.accuracy, feed_dict={model.x: test_data, model.y: test_label})

        return test_acc


if __name__ == '__main__':

    results = []
    dy = []
    for i in range(1, 11):
        DATAPATH = os.path.join(os.getcwd(), 'lesion_experiments', 'MNIST')
        root_dir = 'fixed_sparse_n_150_p_0.05_'+str(i)
        model_dir = os.path.join(DATAPATH, root_dir)

        accuracy = Disruption_balance(model_dir=model_dir).disruption()
        results.append(accuracy[0])
        dy.append(accuracy[1])

    fig = plt.figure(figsize=(4, 3))
    fig.add_axes([0.2, 0.18, 0.7, 0.7])
    for i in range(len(results)):
        plt.errorbar(range(len(results[i])), results[i], dy[i], fmt='o-', linewidth=1,
                     elinewidth=0.3, capsize=1.5, ms=2.5, label='trial ' + str(i+1))
        # plt.plt(range(len(results[i])), results[i], '*-', linewidth=1,
        #          markersize=3, label='trial ' + str(i+1))
    plt.xlabel("removed unbalanced motifs", fontsize=8)
    plt.ylabel("Test accuracy (%)", fontsize=8)
    plt.title("MNIST", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=6)
    plt.show()
