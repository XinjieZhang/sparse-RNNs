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
from model.mgnet import Model
from data.MG_preprocess import seq_iterator
from model.structure_evolution import createWeightMask
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_epochs': 100,
        'batch_size': 12,
        'hidden_size': 128,
        'input_width': 500,
        'label_width': 50,
        'noise': False,  # default:False
        'learning_rate': 1e-3,  # fully_connected:lr=1e-3, fixed_sparse:lr=2e-3
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


def validation(session, model, data, hp):
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


def train(model_dir,
          hp=None,
          display_step=10,
          seed=0,
          load_dir=None):

    mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp()
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # get the data set
    data_dir = '../data/MG/MackeyGlass/'
    data_file = 'MackeyGlass_t17.txt'

    # open the data from txt file
    text_data = np.loadtxt(os.path.join(data_dir, data_file))

    data_len = len(text_data)
    n_input = 1
    n_output = 1

    hp['n_input'] = n_input
    hp['n_output'] = n_output

    split = [0.8, 0.2] # train : valid = 8:2
    cumusplit = [np.sum(split[:i]) for i, s in enumerate(split)]
    data_train = text_data[int((np.dot(data_len, cumusplit))[0]): int((np.dot(data_len, cumusplit))[1])]
    # data_test = text_data[int((np.dot(data_len, cumusplit))[1]): int((np.dot(data_len, cumusplit))[2])]
    data_valid = text_data[int((np.dot(data_len, cumusplit))[1]):]

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # initialize weights
    n_hidden = hp['hidden_size']
    rng = np.random.RandomState(hp['seed'])

    w_in0 = rng.randn(n_input, n_hidden)
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
    # tf.reset_default_graph()

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

        train_loss = []
        valid_loss = []
        for epoch in range(1, hp['n_epochs']*display_step+1):
            batch_size = hp['batch_size']
            costs = 0.0
            for step, (batch_x, batch_y) in enumerate(seq_iterator(data_train, hp['input_width'], hp['label_width'],
                                                                   hp['batch_size'], hp['noise'])):
                batch_x = batch_x.reshape((batch_size, hp['input_width'], n_input))
                batch_y = batch_y.reshape((batch_size, hp['label_width'], n_output))
                sess.run(model.train_step, feed_dict={model.x: batch_x, model.y: batch_y})

                cost = sess.run(model.cost, feed_dict={model.x: batch_x, model.y: batch_y})
                costs += cost
            num_batchs = step + 1
            train_mse = costs / num_batchs

            if epoch % display_step == 0:
                train_loss.append(train_mse)

                # validation
                valid_mse = validation(sess, model, data=data_valid, hp=hp)
                valid_loss.append(valid_mse)
                print("Steps: %d - train mse: %.6f - test mse: %.6f - time: %.4f " %
                      (epoch*num_batchs, train_mse, valid_mse, time.time() - t_start))

        # save the model
        model.save()
        # save train loss and train accuracy over all epochs
        np.savetxt(os.path.join(model_dir, 'train_loss.txt'), np.asarray(train_loss))
        np.savetxt(os.path.join(model_dir, 'valid_loss.txt'), np.asarray(valid_loss))
        print("optimization finished!")


if __name__ == '__main__':
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='lesion_experiments/MG/fixed_sparse_n_128_p_0.05')
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
        'gpu_id': gpu_id,
        'learning_rate': 2e-3  # fully_connected:lr=1e-3, fixed_sparse:lr=2e-3
    }
    train(args.modeldir,
          seed=0,
          hp=hp)
