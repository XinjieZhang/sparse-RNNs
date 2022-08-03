# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_input': 3+4,
        'n_hidden': 64,
        'n_classes': 7,
        'learning_rate': 0.01,
        'batch_size': 64,
        'training_iters': 100,
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

    if (hp['sparsity'] is not None and
            hp['sparsity'] <= 1.0):
        W = np.random.uniform(low=0.01, high=1, size=[hp['n_hidden'], hp['n_hidden']])
        W, sparsity_mask = sparse_var(W, hp['sparsity'])
        hp['sparsity_mask'] = sparsity_mask.tolist()
        W_initializer = tf.constant_initializer(W, dtype=tf.float32)
        params_init.update({'W': W_initializer})
        save_hp(hp, model_dir)

    if (hp['sensory_sparsity'] is not None and
            hp['sensory_sparsity'] <= 1.0):
        sensory_W = np.random.uniform(low=0.01, high=1, size=[hp['n_input'], hp['n_hidden']])
        sensory_W, sensory_sparsity_mask = sparse_var(sensory_W, hp['sensory_sparsity'])
        hp['sensory_sparsity_mask'] = sensory_sparsity_mask.tolist()
        sensory_W_initializer = tf.constant_initializer(sensory_W, dtype=tf.float32)
        params_init.update({'sensory_W': sensory_W_initializer})
        save_hp(hp, model_dir)

    # Build the model
    model = Model(model_dir, params_init, hp=hp)

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    with tf.Session() as sess:
        if load_dir is not None:
            model.restore(load_dir)  # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        train_accuracy = []
        valid_accuracy = []
        test_accuracy = []
        best_valid_accuracy = 0
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        for epoch in range(hp['training_iters']):

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
                    model.save()

        # export_weights
        model.wm.export_weights(os.path.join(model_dir, 'model_params'), sess=sess)

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

    parser.add_argument('--modeldir', type=str, default='../results/Person_LTC/fully_connected_n_64')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'n_hidden': 64,
          'sparsity': None,
          'sensory_sparsity': None
          }
    train(args.modeldir,
          hp=hp,
          seed=0)

