# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from collections import defaultdict

import random
import networkx as nx
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from model.smnistnet import Model
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


def createWeightMask(n, p):

    # n = 150  # number of nodes
    # p = 0.05  # connectivity

    num_unbalanced_motifs = 15
    G = nx.erdos_renyi_graph(n=n, p=p, directed=True)  # generated directed random network
    move_nodes = np.random.choice(G.nodes, size=int(num_unbalanced_motifs * 2), replace=False)
    G.remove_nodes_from(move_nodes)

    vertexs = np.random.choice(G.nodes, num_unbalanced_motifs, replace=False)
    feedforward_motifs = []
    for index in range(num_unbalanced_motifs):
        motif_nodes = [vertexs[index], move_nodes[2*index], move_nodes[2*index+1]]
        random.shuffle(motif_nodes)
        i, j, k = motif_nodes
        G.add_edge(i, k)
        G.add_edge(i, j)
        G.add_edge(j, k)
        feedforward_motifs.append((i, j, k))

    w_mask = np.zeros((n, n))
    for s, t in G.edges:
        w_mask[s, t] = 1

    return w_mask, feedforward_motifs


# validation
def prediction(sess, model, data):
    test_len = 10000
    test_data = data.test.images[:test_len].reshape((-1, 28, 28))
    test_label = data.test.labels[:test_len]
    test_acc = sess.run(model.accuracy, feed_dict={model.x: test_data, model.y: test_label})

    return test_acc


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

    w_rec0 = rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
    if (hp['epsilon_rec'] is not None and
            hp['epsilon_rec'] <= 1.0):
        p_rec = hp['epsilon_rec']
        w_rec_mask, feedforward_motifs = createWeightMask(n_hidden, p_rec)
        w_rec0 = w_rec0 * w_rec_mask
        hp['w_rec_mask'] = w_rec_mask.tolist()
        hp['feedforward_motifs'] = np.array(feedforward_motifs).tolist()
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
            model.restore(load_dir)  # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        train_accuracy = []
        for epoch in range(hp['training_iters']):

            train_acc = 0.0
            costs = 0.0
            for step in range(display_step):
                batch_x, batch_y = mnist.train.next_batch(hp['batch_size'])
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((hp['batch_size'], n_steps, n_input))
                sess.run(model.train_step, feed_dict={model.x: batch_x, model.y: batch_y})

                acc = sess.run(model.accuracy, feed_dict={model.x: batch_x, model.y: batch_y})
                cost = sess.run(model.cost, feed_dict={model.x: batch_x, model.y: batch_y})
                train_acc += acc
                costs += cost
            train_accuracy.append(train_acc/display_step)
            print('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.
                  format((epoch + 1) * display_step, costs/display_step, train_acc/display_step))

        print("optimization finished!")
        # save the model
        model.save()
        test_acc = prediction(sess, model, data=mnist)
        print("Testing Accuracy: {:.6f}, Time: {:.8f}".format(test_acc, time.time() - t_start))


if __name__ == '__main__':
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='lesion_experiments/MNIST/fixed_sparse_n_150_p_0.05')
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
        'epsilon_rec': 0.05,
        'gpu_id': gpu_id
    }
    train(args.modeldir,
          seed=0,
          hp=hp)