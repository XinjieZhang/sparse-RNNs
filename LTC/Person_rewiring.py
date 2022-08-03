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
from LTC.network import Model
from LTC.parameters import get_variables
from data.Person_preprocess import PersonData
from model.structure_evolution import *
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_input': 3+4,
        'n_hidden': 64,
        'n_classes': 7,
        'learning_rate': 0.005,
        'batch_size': 64,
        'training_iters': 101,
        'log_period': 1,
        'sparsity': None,
        'sensory_sparsity': None
    }
    return hp


def sparse_var(v, sparsity_level):
    mask = np.random.choice([0, 1],
                            size=v.shape,
                            p=[1-sparsity_level, sparsity_level]).astype(int)
    v = v * mask
    return [v, mask]


def train(model_dir,
          hp=None,
          seed=0,
          load_dir=None):

    mkdir_p(model_dir)

    person_data = PersonData()

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
    params_init = get_variables(hp)

    if hp['rewiring_algorithm'] == 'DeepR':
        if (hp['sparsity'] is not None and
                hp['sparsity'] <= 1.0):
            W = np.random.uniform(low=0.01, high=1, size=[hp['n_hidden'], hp['n_hidden']])
            nb_non_zero = int(hp['n_hidden'] * hp['n_hidden'] * hp['sparsity'])
            W, sparsity_mask, W_sign_0 = weight_sampler_strict_number(W, hp['n_hidden'], hp['n_hidden'], nb_non_zero)
            hp['sparsity_mask'] = sparsity_mask.tolist()
            W_initializer = tf.constant_initializer(W, dtype=tf.float32)
            params_init.update({'W': W_initializer})
            save_hp(hp, model_dir)

        if (hp['sensory_sparsity'] is not None and
                hp['sensory_sparsity'] <= 1.0):
            sensory_W = np.random.uniform(low=0.01, high=1, size=[hp['n_input'], hp['n_hidden']])
            nb_non_zero_sensory = int(hp['n_input'] * hp['n_hidden'] * hp['sensory_sparsity'])
            sensory_W, sensory_sparsity_mask, sensory_W_sign_0 = weight_sampler_strict_number(sensory_W, hp['n_input'],
                                                                                              hp['n_hidden'],
                                                                                              nb_non_zero_sensory)
            hp['sensory_sparsity_mask'] = sensory_sparsity_mask.tolist()
            sensory_W_initializer = tf.constant_initializer(sensory_W, dtype=tf.float32)
            params_init.update({'sensory_W': sensory_W_initializer})
            save_hp(hp, model_dir)
    elif hp['rewiring_algorithm'] == 'SET':
        if (hp['sparsity'] is not None and
                hp['sparsity'] <= 1.0):
            W = np.random.uniform(low=0.01, high=1, size=[hp['n_hidden'], hp['n_hidden']])
            W, sparsity_mask = sparse_var(W, hp['sparsity'])
            W_num = np.sum(sparsity_mask)
            hp['sparsity_mask'] = sparsity_mask.tolist()
            W_initializer = tf.constant_initializer(W, dtype=tf.float32)
            params_init.update({'W': W_initializer})
            save_hp(hp, model_dir)

        if (hp['sensory_sparsity'] is not None and
                hp['sensory_sparsity'] <= 1.0):
            sensory_W = np.random.uniform(low=0.01, high=1, size=[hp['n_input'], hp['n_hidden']])
            sensory_W, sensory_sparsity_mask = sparse_var(sensory_W, hp['sensory_sparsity'])
            sensory_W_num = np.sum(sensory_sparsity_mask)
            hp['sensory_sparsity_mask'] = sensory_sparsity_mask.tolist()
            sensory_W_initializer = tf.constant_initializer(sensory_W, dtype=tf.float32)
            params_init.update({'sensory_W': sensory_W_initializer})
            save_hp(hp, model_dir)

    # Build the model
    # model = Model(model_dir, params_init, hp=hp)

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
        model = Model(model_dir, params_init, hp=hp)

        with tf.Session() as sess:
            if load_dir is not None:
                model.restore(load_dir)  # complete restore
            else:
                # Assume everything is restored
                sess.run(tf.global_variables_initializer())

            W_0 = sess.run(model.wm.W)
            sensory_W_0 = sess.run(model.wm.sensory_W)

            losses = []
            accs = []
            for batch_x, batch_y in person_data.iterate_train(batch_size=hp['batch_size']):
                _, acc, loss = sess.run([model.train_step, model.accuracy, model.cost],
                                        feed_dict={model.x: batch_x, model.y: batch_y})
                sess.run(model.constrain_op)

                losses.append(loss)
                accs.append(acc)
            train_accuracy.append(np.mean(accs))

            # Validation
            if (epoch + 1) % hp['log_period'] == 0:
                test_acc, test_loss = sess.run([model.accuracy, model.cost],
                                               feed_dict={model.x: person_data.test_x, model.y: person_data.test_y})
                valid_acc, valid_loss = sess.run([model.accuracy, model.cost],
                                                 feed_dict={model.x: person_data.valid_x, model.y: person_data.valid_y})
                test_accuracy.append(test_acc)
                valid_accuracy.append(valid_acc)
                print(
                    "Epochs {:03d}, train loss: {:0.4f}, train accuracy: {:0.4f}%, valid loss: {:0.4f}, "
                    "valid accuracy: {:0.4f}%, test loss: {:0.4f}, test accuracy: {:0.4f}%, Time: {:0.2f}".format(
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

            # get variables
            W, erev, mu, sigma = sess.run([model.wm.W, model.wm.erev, model.wm.mu, model.wm.sigma])
            sensory_W, sensory_erev, sensory_mu, sensory_sigma = sess.run([
                model.wm.sensory_W, model.wm.sensory_erev, model.wm.sensory_mu, model.wm.sensory_sigma])

            if epoch % 5 == 0:
                fname = open(os.path.join(model_dir, 'signed_edge_list_' + str(epoch) + '.csv'), 'w', newline='')
                csv.writer(fname).writerow(('Id', 'Source', 'Target', 'Weight'))
                k = 0
                sparse_model = erev * hp['sparsity_mask']
                for i in range(sparse_model.shape[0]):
                    for j in range(sparse_model.shape[1]):
                        if sparse_model[i, j] != 0:
                            source = i
                            target = j
                            csv.writer(fname).writerow((k, source, target, sparse_model[i, j]))
                            k += 1
                fname.close()

            if hp['rewiring_algorithm'] == 'DeepR':
                # deep rewiring
                # Guillaume Bellec et al. (2017) DEEP REWIRING: TRAINING VERY SPARSE DEEP NETWORKS WORKS
                # arXiv:1711.05136v1
                mask_connected = lambda th: (np.greater(th, 0)).astype(int)
                noise_update = lambda th: np.random.normal(scale=1e-3, size=th.shape)

                l1 = 1e-3  # regulation coefficient
                add_gradient_op = W + mask_connected(abs(W)) * noise_update(W)
                apply_l1_reg = - mask_connected(abs(W)) * np.sign(W) * l1
                W_1 = add_gradient_op + apply_l1_reg

                W, sparsity_mask, nb_reconnect, sparsity_mask_core = rewiring(W_0 * W_1, W_1, nb_non_zero, W_sign_0)
                assert_connection_number(abs(W), nb_non_zero)
                hp['sparsity_mask'] = sparsity_mask.tolist()

                add_gradient_op_sensory = sensory_W + mask_connected(abs(sensory_W)) * noise_update(sensory_W)
                apply_l1_reg_sensory = - mask_connected(abs(sensory_W)) * np.sign(sensory_W) * l1
                sensory_W_1 = add_gradient_op_sensory + apply_l1_reg_sensory

                sensory_W, sensory_sparsity_mask, nb_reconnect_sensory, _ = rewiring(sensory_W_0 * sensory_W_1, sensory_W_1,
                                                                                     nb_non_zero_sensory, sensory_W_sign_0)
                assert_connection_number(abs(sensory_W), nb_non_zero_sensory)
                hp['sensory_sparsity_mask'] = sensory_sparsity_mask.tolist()
            elif hp['rewiring_algorithm'] == 'SET':
                # SET
                # It removes the weights closest to zero in each layer and add new random weights
                sparsity_mask, sparsity_mask_core = rewireMask(W, W_num, zeta=hp['zeta'])
                W *= sparsity_mask_core
                hp['sparsity_mask'] = sparsity_mask.tolist()

                sensory_sparsity_mask, sensory_sparsity_mask_core = rewireMask(sensory_W, sensory_W_num,
                                                                               zeta=hp['zeta'])
                sensory_W *= sensory_sparsity_mask_core
                hp['sensory_sparsity_mask'] = sensory_sparsity_mask.tolist()

                # save the recurrent network model
                if epoch % 5 == 0:
                    fname = open(os.path.join(model_dir, 'core_signed_edge_list_' + str(epoch) + '.csv'), 'w',
                                 newline='')
                    csv.writer(fname).writerow(('Id', 'Source', 'Target', 'Weight'))
                    k = 0
                    sparse_model = erev * sparsity_mask_core
                    for i in range(sparse_model.shape[0]):
                        for j in range(sparse_model.shape[1]):
                            if sparse_model[i, j] != 0:
                                source = i
                                target = j
                                csv.writer(fname).writerow((k, source, target, erev[i, j]))
                                k += 1
                    fname.close()

            params_init.update({'W': tf.constant_initializer(W)})
            params_init.update({'erev': tf.constant_initializer(erev)})
            params_init.update({'mu': tf.constant_initializer(mu)})
            params_init.update({'sigma': tf.constant_initializer(sigma)})

            params_init.update({'sensory_W': tf.constant_initializer(sensory_W)})
            params_init.update({'sensory_erev': tf.constant_initializer(sensory_erev)})
            params_init.update({'sensory_mu': tf.constant_initializer(sensory_mu)})
            params_init.update({'sensory_sigma': tf.constant_initializer(sensory_sigma)})

            vleak, gleak, cm = sess.run([model.wm.vleak, model.wm.gleak, model.wm.cm_t])
            params_init.update({'vleak': tf.constant_initializer(vleak)})
            params_init.update({'gleak': tf.constant_initializer(gleak)})
            params_init.update({'cm_t': tf.constant_initializer(cm)})

            input_w, input_b = sess.run([model.input_w, model.input_b])
            params_init.update({'input_w': tf.constant_initializer(input_w)})
            params_init.update({'input_b': tf.constant_initializer(input_b)})

            output_w, output_b = sess.run([model.output_w, model.output_b])
            params_init.update({'output_w': tf.constant_initializer(output_w)})
            params_init.update({'output_b': tf.constant_initializer(output_b)})

            if epoch+1 == hp['training_iters']:
                print("optimization finished!")

                # export_weights
                model.wm.export_weights(os.path.join(model_dir, 'model_params'), sess=sess)

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

    parser.add_argument('--modeldir', type=str, default='../results/Person_LTC/SET_n_64_p_0.05_zeta_0.1')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'sparsity': 0.05,
          'sensory_sparsity': 0.05,
          'rewiring_algorithm': 'SET',  # DeepR or SET
          'zeta': 0.1
          }
    train(args.modeldir,
          hp=hp,
          seed=0)

