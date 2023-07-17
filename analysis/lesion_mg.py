# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import random

import sys
sys.path.append('../')
from utils.tools import load_hp
from model.mgnet import Model
from data.MG_preprocess import seq_iterator
import matplotlib.pyplot as plt


class Disruption_balance(object):

    def __init__(self,
                 model_dir,
                 ):

        self.model_dir = model_dir
        self.hp = load_hp(model_dir)

        # get the data set
        data_dir = '../data/MG/MackeyGlass/'
        data_file = 'MackeyGlass_t17.txt'

        # open the data from txt file
        text_data = np.loadtxt(os.path.join(data_dir, data_file))

        data_len = len(text_data)

        split = [0.8, 0.2]  # train : valid = 8:2
        cumusplit = [np.sum(split[:i]) for i, s in enumerate(split)]
        self.data_train = text_data[int((np.dot(data_len, cumusplit))[0]): int((np.dot(data_len, cumusplit))[1])]
        # data_test = text_data[int((np.dot(data_len, cumusplit))[1]): int((np.dot(data_len, cumusplit))[2])]
        self.data_valid = text_data[int((np.dot(data_len, cumusplit))[1]):]

        self.n_input = self.hp['n_input']
        self.n_hidden = self.hp['hidden_size']
        self.feedforward_motifs = self.hp['feedforward_motifs']
        rng = np.random.RandomState(self.hp['seed'])

        w_in0 = rng.randn(self.n_input, self.n_hidden)
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

                test_mse0 = self.validation(sess, model, data=self.data_valid, hp=self.hp)
                # print("Testing mse: {:.6f}".format(test_mse0))

                unbalanced_motifs = []
                for i, j, k in self.feedforward_motifs:
                    if w_rec[i][k] * w_rec[i][j] * w_rec[j][k] < 0:
                        unbalanced_motifs.append((i, j, k))
                random.shuffle(unbalanced_motifs)
                # print(unbalanced_motifs)

                is_connected = np.greater(abs(w_rec), 0).astype(int)

                w_mask = is_connected.copy()
                prediction_mse = [test_mse0]
                for (i, j, k) in unbalanced_motifs:
                    w_mask[i][k] = 0
                    w_mask[i][j] = 0
                    w_mask[j][k] = 0

                    model.ablation_units(sess, w_mask)

                    new_test_mse = self.validation(sess, model, data=self.data_valid, hp=self.hp)
                    prediction_mse.append(new_test_mse)
                    # print("Testing MSE: {:.6f}".format(new_test_mse))
            results.append(prediction_mse)

        results = np.array(results)

        return results.mean(axis=0), results.std(axis=0)

    def validation(self, session, model, data, hp):
        batch_size = 1
        n_input = hp['n_input']
        n_output = hp['n_output']

        costs = 0.0
        for step, (batch_x, batch_y) in enumerate(seq_iterator(data, hp['input_width'], hp['label_width'], batch_size)):
            batch_x = batch_x.reshape((batch_size, hp['input_width'], n_input))
            batch_y = batch_y.reshape((batch_size, hp['label_width'], n_output))

            cost = session.run(model.cost, feed_dict={model.x: batch_x, model.y: batch_y})
            costs += cost
        return costs / (step + 1)


if __name__ == '__main__':

    results = []
    dy = []
    for i in range(1, 11):
        DATAPATH = os.path.join(os.getcwd(), 'lesion_experiments', 'MG')
        root_dir = 'fixed_sparse_n_128_p_0.05_'+str(i)
        model_dir = os.path.join(DATAPATH, root_dir)

        mse = Disruption_balance(model_dir=model_dir).disruption()
        results.append(mse[0])
        dy.append(mse[1])

    fig = plt.figure(figsize=(4, 3))
    fig.add_axes([0.2, 0.18, 0.7, 0.7])
    for i in range(len(results)):
        plt.errorbar(range(len(results[i])), results[i], dy[i], fmt='o-', linewidth=1,
                     elinewidth=0.3, capsize=1.5, ms=2.5, label='trial ' + str(i + 1))
        # plt.plot(range(len(results[i])), results[i], '*-', linewidth=1,
        #          markersize=3, label='trial ' + str(i + 1))
    plt.xlabel('removed unbalanced motifs', fontsize=8)
    plt.ylabel('Test MSE', fontsize=8)
    plt.title('MG', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=6)
    plt.show()
