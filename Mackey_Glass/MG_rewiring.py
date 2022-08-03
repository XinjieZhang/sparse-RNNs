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
from model.mgnet import Model
from data.MG_preprocess import seq_iterator
from model.structure_evolution import *
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_epochs': 101,
        'batch_size': 12,
        'hidden_size': 128,
        'input_width': 500,
        'label_width': 50,
        'noise': False, # default: False
        'learning_rate': 1e-3,
        'epsilon_rec': None
    }
    return hp


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

    split = [0.8, 0.2]  # train: test = 8:2
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

    train_loss = []
    valid_loss = []
    for epoch in range(hp['n_epochs']):
        # Reset Tensorflow before running anything
        tf.reset_default_graph()

        # Build the model
        model = create_model(model_dir, hp, w_in, w_rec, b_rec, w_out, b_out)

        with tf.Session() as sess:
            if load_dir is not None:
                model.restore(load_dir)  # complete restore
            else:
                # Assume everything is restored
                sess.run(tf.global_variables_initializer())

            w_rec0 = sess.run(model.w_rec)

            for i in range(display_step):
                # Training
                costs = 0.0
                for step, (batch_x, batch_y) in enumerate(seq_iterator(data_train, hp['input_width'], hp['label_width'],
                                                                       hp['batch_size'], hp['noise'])):
                    batch_x = batch_x.reshape((hp['batch_size'], hp['input_width'], n_input))
                    batch_y = batch_y.reshape((hp['batch_size'], hp['label_width'], n_output))
                    sess.run(model.train_step, feed_dict={model.x: batch_x, model.y: batch_y})

                    cost = sess.run(model.cost, feed_dict={model.x: batch_x, model.y: batch_y})
                    costs += cost
                num_batchs = step + 1
                train_mse = costs / num_batchs
            train_loss.append(train_mse)

            # validation
            valid_mse = validation(sess, model, data=data_valid, hp=hp)
            valid_loss.append(valid_mse)
            print("Steps: %d - train mse: %.6f - test mse: %.6f - time: %.4f " %
                  ((epoch+1)*display_step*num_batchs, train_mse, valid_mse, time.time() - t_start))
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

    # save train loss and train accuracy over all epochs
    np.savetxt(os.path.join(model_dir, 'train_loss.txt'), np.asarray(train_loss))
    np.savetxt(os.path.join(model_dir, 'valid_loss.txt'), np.asarray(valid_loss))
    print("optimization finished!")

    # Plot loss over time
    epoch_seq = np.arange(1, hp['n_epochs'] + 1)
    plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
    plt.plot(epoch_seq, valid_loss, 'r-', label='Test Set')
    plt.title('Mean square error')
    plt.xlabel('Epochs')
    plt.ylabel('mean square error')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='../results/MG/SET_n_128_p_0.05_zeta_0.2')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'epsilon_rec': 0.05,
          'learning_rate': 2e-3,
          'rewiring_algorithm': 'SET',  # DeepR or SET
          'zeta': 0.2,  # The proportion of parameters to prune in the SET
          }
    train(args.modeldir,
          seed=1,
          hp=hp)

