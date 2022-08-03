# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import sys
sys.path.append('../')
from CT_RNN.network import Model
from data.Gesture_preprocess import GestureData
from model.structure_evolution import *
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_input': 32,
        'n_hidden': 128,
        'n_classes': 5,
        'learning_rate': 0.005,
        'batch_size': 32,
        'training_iters': 201,
        'log_period': 1,
        'sparsity': None,
    }
    return hp


def sparse_var(v, sparsity_level):
    mask = np.random.choice([0, 1],
                            size=v.shape,
                            p=[1-sparsity_level, sparsity_level]).astype(int)
    v = v * mask
    return [v, mask]


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
          seed=0,
          load_dir=None):

    mkdir_p(model_dir)

    gesture_data = GestureData()

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

    # initializer weights
    n_input = hp['n_input']
    n_hidden = hp['n_hidden']
    rng = np.random.RandomState(hp['seed'])

    w_in0 = rng.randn(n_input, n_hidden)
    w_rec0 = rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
    if hp['rewiring_algorithm'] == 'DeepR':
        if (hp['sparsity'] is not None and
                hp['sparsity'] <= 1.0):
            p_rec = hp['sparsity']
            nb_non_zero = int(n_hidden * n_hidden * p_rec)
            w_rec0, sparsity_mask, w_rec_sign_0 = weight_sampler_strict_number(w_rec0, n_hidden, n_hidden, nb_non_zero)
            hp['w_rec_mask'] = sparsity_mask.tolist()
            save_hp(hp, model_dir)
    elif hp['rewiring_algorithm'] == 'SET':
        if (hp['sparsity'] is not None and
                hp['sparsity'] <= 1.0):
            w_rec0, sparsity_mask = sparse_var(w_rec0, hp['sparsity'])
            no_w_rec = np.sum(sparsity_mask)
            hp['w_rec_mask'] = sparsity_mask.tolist()
            save_hp(hp, model_dir)
    [w_in, w_rec, b_rec, w_out, b_out] = [w_in0, w_rec0, None, None, None]

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    best_valid_accuracy = 0
    best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
    for epoch in range(hp['training_iters']):
        # Build the model
        model = create_model(model_dir, hp, w_in, w_rec, b_rec, w_out, b_out)

        with tf.Session() as sess:
            if load_dir is not None:
                model.restore(load_dir)  # complete restore
            else:
                # Assume everything is restored
                sess.run(tf.global_variables_initializer())

            w_rec0 = sess.run(model.w_rec)

            losses = []
            accs = []
            for batch_x, batch_y in gesture_data.iterate_train(batch_size=hp['batch_size']):
                _, acc, loss = sess.run([model.train_step, model.accuracy, model.cost],
                                        feed_dict={model.x: batch_x, model.y: batch_y})

                losses.append(loss)
                accs.append(acc)
            train_accuracy.append(np.mean(accs))

            # Validation
            if (epoch + 1) % hp['log_period'] == 0:
                test_acc, test_loss = sess.run([model.accuracy, model.cost],
                                               feed_dict={model.x: gesture_data.test_x, model.y: gesture_data.test_y})
                valid_acc, valid_loss = sess.run([model.accuracy, model.cost],
                                                 feed_dict={model.x: gesture_data.valid_x, model.y: gesture_data.valid_y})
                test_accuracy.append(test_acc)
                valid_accuracy.append(valid_acc)
                print(
                    "Epochs {:03d}, train loss: {:0.4f}, train accuracy: {:0.4f}%, valid loss: {:0.4f}, "
                    "valid accuracy: {:0.4f}%, test loss: {:0.4f}, test accuracy: {:0.4f}%, Time: {:0.6f}".format(
                        epoch + 1,
                        np.mean(losses), np.mean(accs) * 100,
                        valid_loss, valid_acc * 100,
                        test_loss, test_acc * 100,
                        time.time() - t_start))
                # Accuracy metric -> higher is better
                if (valid_acc > best_valid_accuracy and epoch > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        epoch + 1,
                        np.mean(losses), np.mean(acc) * 100,
                        valid_loss, valid_acc * 100,
                        test_loss, test_acc * 100
                    )
                # save the model
                model.save()

            # ------------- weights evolution ---------------
            # Get weight list
            w_in = sess.run(model.w_in)
            w_rec = sess.run(model.w_rec)
            w_out = sess.run(model.w_out)
            b_rec = sess.run(model.b_rec)
            b_out = sess.run(model.b_out)

            # save the recurrent network model
            if epoch % 10 == 0:
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
                noise_update = lambda th: np.random.normal(scale=1e-3, size=th.shape)

                l1 = 1e-3  # regulation coefficient
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

                if epoch % 10 == 0:
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

            if epoch+1 == hp['training_iters']:
                # export_weights
                model.cell.export_weights(os.path.join(model_dir, 'model_params'), sess=sess)

                print("optimization finished!")
                best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = best_valid_stats
                print("Best epoch {:03d}, train loss: {:0.6f}, train accuracy: {:0.6f}, valid loss: {:0.6f}, "
                      "valid accuracy: {:0.6f}, test loss: {:0.6f}, test accuracy: {:0.6f}".format(
                      best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc
                      ))

                # save train loss and train accuracy over all epochs
                np.savetxt(os.path.join(model_dir, 'train_accuracy.txt'), np.asarray(train_accuracy))
                np.savetxt(os.path.join(model_dir, 'valid_accuracy.txt'), np.asarray(valid_accuracy))
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

    parser.add_argument('--modeldir', type=str, default='../results/Gesture_ctrnn/deep_rewiring_n_128_p_0.05')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'learning_rate': 0.01,
          'sparsity': 0.05,
          'rewiring_algorithm': 'DeepR',  # DeepR or SET
          'zeta': 0.1
          }
    train(args.modeldir,
          seed=0,
          hp=hp)
