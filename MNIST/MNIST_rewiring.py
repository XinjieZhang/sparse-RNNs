# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from model.smnistnet import Model
from model.structure_evolution import *
from tensorflow.examples.tutorials.mnist import input_data
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_input': 28,  # MNIST data input (img shape: 28*28)
        'n_steps': 28,  # timesteps
        'n_hidden': 150,  # hidden layer num of features
        'n_classes': 10,  # MNIST total classes (0-9 digits)
        'learning_rate': 0.001,
        'training_iters': 201,
        'batch_size': 128,
        'epsilon_rec': None
    }
    return hp


def create_model(model_dir, hp, w_in, w_rec, b_rec=None, w_out=None, b_out=None):

    input_kernel_initializer = tf.constant_initializer(w_in, dtype=tf.float32)
    recurrent_kernel_initializer = tf.constant_initializer(w_rec, dtype=tf.float32)

    recurrent_bias_initializer = tf.constant_initializer(b_rec, dtype=tf.float32) if (b_rec is not None) else b_rec
    output_kernel_initializer = tf.constant_initializer(w_out, dtype=tf.float32) if (w_out is not None) else w_out
    output_bias_initializer = tf.constant_initializer(b_out, dtype=tf.float32) if (b_out is not None) else b_out

    model = Model(model_dir,
                  input_kernel_initializer=input_kernel_initializer,
                  recurrent_kernel_initializer=recurrent_kernel_initializer,
                  recurrent_bias_initializer=recurrent_bias_initializer,
                  output_kernel_initializer=output_kernel_initializer,
                  output_bias_initializer=output_bias_initializer,
                  hp=hp)

    return model


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
    w_rec0 = rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
    if hp['rewiring_algorithm'] == 'DeepR':
        if (hp['epsilon_rec'] is not None and
                hp['epsilon_rec'] <= 1.0):
            p_rec = hp['epsilon_rec']
            nb_non_zero = int(n_hidden * n_hidden * p_rec)
            w_rec0, w_rec_mask, w_rec_sign_0 = weight_sampler_strict_number(w_rec0, n_hidden, n_hidden, nb_non_zero)
            hp['w_rec_mask'] = w_rec_mask.tolist()
            save_hp(hp, model_dir)
    elif hp['rewiring_algorithm'] == 'SET':
        if (hp['epsilon_rec'] is not None and
                hp['epsilon_rec'] <= 1.0):
            p_rec = hp['epsilon_rec']
            no_w_rec, w_rec_mask = createWeightMask(p_rec, n_hidden, n_hidden, hp)
            hp['w_rec_mask'] = w_rec_mask.tolist()
            w_rec0 *= hp['w_rec_mask']
            save_hp(hp, model_dir)
    [w_in, w_rec, b_rec, w_out, b_out] = [w_in0, w_rec0, None, None, None]

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    train_accuracy = []
    test_accuracy = []
    for epoch in range(hp['training_iters']):
        # Reset Tensorflow before running anything
        tf.reset_default_graph()

        # Build the model
        model = create_model(model_dir, hp, w_in, w_rec, b_rec, w_out, b_out)

        with tf.Session() as sess:
            if load_dir is not None:
                model.restore(load_dir) # complete restore
            else:
                # Assume everything is restored
                sess.run(tf.global_variables_initializer())

            w_rec0 = sess.run(model.w_rec)

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
            train_accuracy.append(train_acc / display_step)
            print('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.
                  format((epoch + 1) * display_step, costs / display_step, train_acc / display_step))

            # Validation
            if (epoch + 1) % 5 == 0:
                test_len = 500
                test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
                test_label = mnist.test.labels[:test_len]
                test_acc = sess.run(model.accuracy, feed_dict={model.x: test_data, model.y: test_label})
                test_accuracy.append(test_acc)
                print("Testing Accuracy: {:.6f}, Time: {:.8f}".format(test_acc, time.time() - t_start))

            # save the model
            model.save()

            # ------------- weights evolution ---------------
            # Get weight list
            w_in = sess.run(model.w_in)
            w_rec = sess.run(model.w_rec)
            w_out = sess.run(model.w_out)
            b_rec = sess.run(model.b_rec)
            b_out = sess.run(model.b_out)

            if epoch % 5 == 0:
                fname = open(os.path.join(model_dir, 'edge_list_weighted_' + str(epoch) + '.csv'), 'w', newline='')
                csv.writer(fname).writerow(('Id', 'Source', 'Target', 'Weight'))
                k = 0
                for i in range(w_rec.shape[0]):
                    for j in range(w_rec.shape[1]):
                        if w_rec[i, j] != 0:
                            source = i
                            target = j
                            csv.writer(fname).writerow((k, source, target, w_rec[i, j]))
                            k += 1
                fname.close()

            if hp['rewiring_algorithm'] == 'DeepR':
                # deep rewiring
                # Guillaume Bellec et al. (2017) DEEP REWIRING: TRAINING VERY SPARSE DEEP NETWORKS WORKS
                # arXiv:1711.05136v1
                mask_connected = lambda th: (np.greater(th, 0)).astype(int)
                noise_update = lambda th: np.random.normal(scale=1e-5, size=th.shape)

                l1 = 1e-5  # regulation coefficient
                add_gradient_op = w_rec + mask_connected(abs(w_rec)) * noise_update(w_rec)
                apply_l1_reg = - mask_connected(abs(w_rec)) * np.sign(w_rec) * l1
                w_rec1 = add_gradient_op + apply_l1_reg

                w_rec, w_rec_mask, nb_reconnect = rewiring(w_rec0 * w_rec1, w_rec1, nb_non_zero, w_rec_sign_0)
                assert_connection_number(abs(w_rec), nb_non_zero)
                hp['w_rec_mask'] = w_rec_mask.tolist()
                save_hp(hp, model_dir)
            elif hp['rewiring_algorithm'] == 'SET':
                # SET
                # It removes the weights closest to zero in each layer and add new random weights
                w_rec_mask, w_rec_core = rewireMask(w_rec, no_w_rec, zeta=hp['zeta'])
                w_rec *= w_rec_core
                hp['w_rec_mask'] = w_rec_mask.tolist()
                save_hp(hp, model_dir)

                if epoch % 5 == 0:
                    fname = open(os.path.join(model_dir, 'core_edge_list_weighted_' + str(epoch) + '.csv'), 'w',
                                 newline='')
                    csv.writer(fname).writerow(('Id', 'Source', 'Target', 'Weight'))
                    k = 0
                    for i in range(w_rec.shape[0]):
                        for j in range(w_rec.shape[1]):
                            if w_rec[i, j] != 0:
                                source = i
                                target = j
                                csv.writer(fname).writerow((k, source, target, w_rec[i, j]))
                                k += 1
                    fname.close()

        sess.close()

    print("optimization finished!")
    # save train loss and train accuracy over all epochs
    np.savetxt(os.path.join(model_dir, 'train_accuracy.txt'), np.asarray(train_accuracy))
    np.savetxt(os.path.join(model_dir, 'test_accuracy.txt'), np.asarray(test_accuracy))

    # plot accuracy over time
    epoch_seq = np.arange(1, hp['training_iters'] + 1)
    plt.plot(epoch_seq, train_accuracy)
    plt.title('train accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='../results/MNIST/SET_n_150_p_0.05_zeta_0.2')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'epsilon_rec': 0.05,
          'rewiring_algorithm': 'SET', # DeepR or SET
          'zeta': 0.2,  # The proportion of parameters to prune in the SET
          }
    train(args.modeldir,
          seed=1,
          hp=hp)
