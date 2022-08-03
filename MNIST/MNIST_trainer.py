# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from model.smnistnet import Model
from model.structure_evolution import createWeightMask
from tensorflow.examples.tutorials.mnist import input_data
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_input': 28,  # MNIST data input (img shape: 28*28)
        'n_steps': 28,  # time steps
        'n_hidden': 150,  # hidden layer num of features
        'n_classes': 10,  # MNIST total classes (0-9 digits)
        'learning_rate': 0.001,
        'training_iters': 200,
        'batch_size': 128,
        'epsilon_rec': None
    }
    return hp


def get_network_model(txt_path):
    # get weighted edge list
    n = 150 # number of nodes
    M = np.zeros([n, n])
    f = open(txt_path, 'r')
    list = f.readlines()

    for line in list:
        s, t, w = map(int, line.strip().split())
        if w == 1:
            M[s, t] = 1
        else:
            M[s, t] = -1

    return np.sign(M)


def train(model_dir,
          hp=None,
          display_step=50,
          seed=0,
          load_dir=None):

    mkdir_p(model_dir)

    mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)

    # Network parameters
    default_hp = get_default_hp()
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # initialize weights
    n_input = hp['n_input']
    n_hidden = hp['n_hidden']
    n_steps = hp['n_steps']
    rng = np.random.RandomState(hp['seed'])

    w_in0 = rng.randn(n_input, n_hidden) / np.sqrt(n_input)
    input_kernel_initializer = tf.constant_initializer(w_in0, dtype=tf.float32)

    if 'network_edge_list' not in hp or hp['network_edge_list'] is None:
        w_rec0 = rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
        if (hp['epsilon_rec'] is not None and
                hp['epsilon_rec'] <= 1.0):
            p_rec = hp['epsilon_rec']
            _, w_rec_mask = createWeightMask(p_rec, n_hidden, n_hidden, hp)
            hp['w_rec_mask'] = w_rec_mask.tolist()
            w_rec0 *= hp['w_rec_mask']
    else:
        w_rec0 = rng.uniform(low=-0.7, high=0.7, size=[n_hidden, n_hidden])
        txt_file = hp['network_edge_list']
        txt_path = os.path.join("..\\model", "network", "generated", "MNIST", "Trial 1", txt_file)
        w_sign = get_network_model(txt_path=txt_path)
        hp['w_rec_mask'] = abs(w_sign).tolist()
        w_rec0 = w_sign * abs(w_rec0)

    save_hp(hp, model_dir)

    recurrent_kernel_initializer = tf.constant_initializer(w_rec0, dtype=tf.float32)

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    # Reset Tensorflow before running anything
    tf.reset_default_graph()

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=tf.ConfigProto()) as sess:

        gpu_id = hp['gpu_id']
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            # Build the model
            model = Model(model_dir,
                          input_kernel_initializer=input_kernel_initializer,
                          recurrent_kernel_initializer=recurrent_kernel_initializer,
                          hp=hp)

        if load_dir is not None:
            model.restore(load_dir) # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        train_accuracy = []
        test_accuracy = []
        for epoch in range(hp['training_iters']):

            train_acc = 0.0
            costs = 0.0
            for step in range(display_step):
                batch_x, batch_y = mnist.train.next_batch(hp['batch_size'])
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((hp['batch_size'], n_steps, n_input))
                _, acc, cost = sess.run([model.train_step, model.accuracy, model.cost],
                                        feed_dict={model.x: batch_x, model.y: batch_y})

                train_acc += acc
                costs += cost
            train_accuracy.append(train_acc/display_step)
            print('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.
                  format((epoch + 1) * display_step, costs/display_step, train_acc/display_step))

            # validation
            if (epoch+1) % 1 == 0:
                test_len = 500
                test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
                test_label = mnist.test.labels[:test_len]
                test_acc = sess.run(model.accuracy, feed_dict={model.x: test_data, model.y: test_label})
                test_accuracy.append(test_acc)
                print("Testing Accuracy: {:.6f}, Time: {:.8f}".format(test_acc, time.time() - t_start))

        print("optimization finished!")
        # save the model
        model.save()
        # save train loss and train accuracy over all epochs
        np.savetxt(os.path.join(model_dir, 'train_accuracy.txt'), np.asarray(train_accuracy))
        np.savetxt(os.path.join(model_dir, 'test_accuracy.txt'), np.asarray(test_accuracy))
        # print("time:", time.time() - t_start)

        # plot accuracy over time
        epoch_seq = np.arange(1, hp['training_iters']+1)
        plt.plot(epoch_seq, train_accuracy)
        plt.title('train accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()


if __name__ == '__main__':
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='../results/MNIST/test')
    args = parser.parse_args()

    try:
        gpu_id = sys.argv[1]
        print('Selecting GPU', gpu_id)
    except:
        gpu_id = None

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'epsilon_rec': None,
          'RNNCell': 'Fixed',
          'optimizer': 'adam',
          'network_edge_list': 'BN_n_150_p_0.05_0_0.597.txt',
          'gpu_id': gpu_id
          }
    train(args.modeldir,
          seed=0,
          hp=hp)
