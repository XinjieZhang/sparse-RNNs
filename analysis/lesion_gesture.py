# coding utf-8
# Before running the code you need to run gesture.py to generate the network


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append('../')
from model.gesturenet import Model
from data.Gesture_preprocess import GestureData
from utils.tools import load_hp
import matplotlib.pyplot as plt


class Disruption_balance(object):

    def __init__(self, model_dir):

        self.data = GestureData()

        self.model_dir = model_dir
        self.hp = load_hp(model_dir)

        self.n_input = self.hp['n_input']
        self.n_hidden = self.hp['n_hidden']
        self.feedforward_motifs = self.hp['feedforward_motifs']
        rng = np.random.RandomState(self.hp['seed'])

        w_in0 = rng.randn(self.n_input, self.n_hidden) / np.sqrt(self.n_input)
        input_kernel_initializer = tf.constant_initializer(w_in0, dtype=tf.float32)

        w_rec0 = rng.uniform(low=-0.6, high=0.6, size=[self.n_hidden, self.n_hidden])
        recurrent_kernel_initializer = tf.constant_initializer(w_rec0, dtype=tf.float32)

        self.model = Model(model_dir,
                           input_kernel_initializer=input_kernel_initializer,
                           recurrent_kernel_initializer=recurrent_kernel_initializer
                           )

    def disruption(self):

        with tf.Session() as sess:
            model = self.model
            model.restore()

            w_rec = sess.run(model.w_rec)

            test_acc0 = sess.run(model.accuracy, feed_dict={model.x: self.data.test_x, model.y: self.data.test_y})
            print("Testing Accuracy: {:.6f}".format(test_acc0))

            unbalanced_motifs = []
            for i, j, k in self.feedforward_motifs:
                if w_rec[i][k] * w_rec[i][j] * w_rec[j][k] < 0:
                    unbalanced_motifs.append((i, j, k))
            # print(unbalanced_motifs)

            is_connected = np.greater(abs(w_rec), 0).astype(int)

            w_mask = is_connected.copy()
            prediction_acc = [test_acc0 * 100]
            for (i, j, k) in unbalanced_motifs:
                w_mask[i][k] = 0
                w_mask[i][j] = 0
                w_mask[j][k] = 0

                model.ablation_units(sess, w_mask)

                new_test_acc = sess.run(model.accuracy, feed_dict={model.x: self.data.test_x, model.y: self.data.test_y})
                prediction_acc.append(new_test_acc * 100)
                print("Testing Accuracy: {:.6f}".format(new_test_acc))

        return prediction_acc


if __name__ == '__main__':

    results = []

    for i in range(1, 11):

        DATAPATH = os.path.join(os.getcwd(), 'lesion_experiments', 'Gesture')
        root_dir = 'fixed_sparse_n_128_p_0.05_'+str(i)
        model_dir = os.path.join(DATAPATH, root_dir)

        accuracy = Disruption_balance(model_dir=model_dir).disruption()
        results.append(accuracy)

    fig = plt.figure()
    for i in range(len(results)):
        plt.plot(range(len(results[i])), results[i], '*-', label='trial ' + str(i + 1))
        plt.xlabel('removed unbalanced motifs')
        plt.ylabel('test accuracy (%)')
    plt.title('Gesture dataset')
    plt.legend()
    plt.show()
