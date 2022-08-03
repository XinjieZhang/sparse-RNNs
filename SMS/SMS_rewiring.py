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
from model.smsnet import Model
from model.structure_evolution import *
from data.SMS_preprocess import SMS
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_epochs': 101,
        'batch_size': 128,
        'max_sequence_length': 25,
        'hidden_size': 100,
        'embedding_size': 50,
        'min_word_frequency': 10,
        'learning_rate': 1e-3,
        'epsilon_rec': None
    }
    return hp


def create_model(model_dir, hp, w_in, w_rec, embedding_mat, b_rec=None, w_out=None, b_out=None):

    input_kernel_initializer = tf.constant_initializer(w_in, dtype=tf.float32)
    recurrent_kernel_initializer = tf.constant_initializer(w_rec, dtype=tf.float32)
    embedding_mat_initializer = tf.constant_initializer(embedding_mat)

    recurrent_bias_initializer = tf.constant_initializer(b_rec, dtype=tf.float32) if (b_rec is not None) else b_rec
    output_kernel_initializer = tf.constant_initializer(w_out, dtype=tf.float32) if (w_out is not None) else w_out
    output_bias_initializer = tf.constant_initializer(b_out, dtype=tf.float32) if (b_out is not None) else b_out

    model = Model(model_dir,
                  input_kernel_initializer=input_kernel_initializer,
                  recurrent_kernel_initializer=recurrent_kernel_initializer,
                  embedding_mat_initializer=embedding_mat_initializer,
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

    # Network parameters
    default_hp = get_default_hp()
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # get the data set
    sms_data = SMS(hp)
    x_train = sms_data.x_train
    y_train = sms_data.y_train
    x_test = sms_data.x_test
    y_test = sms_data.y_test
    vocab_size = sms_data.vocab_size
    print("Vocabulary Size: {:d}".format(vocab_size))
    print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

    hp['vocab_size'] = vocab_size
    save_hp(hp, model_dir)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Create embedding
    embedding_size = hp['embedding_size']
    embedding_mat = np.random.uniform(-1.0, 1.0, [vocab_size, embedding_size])

    # initialize weights
    n_input = embedding_size
    n_hidden = hp['hidden_size']
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

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    batch_size = hp['batch_size']
    for epoch in range(hp['n_epochs']):
        # Build the model
        model = create_model(model_dir, hp, w_in, w_rec, embedding_mat, b_rec, w_out, b_out)

        with tf.Session() as sess:
            if load_dir is not None:
                model.restore(load_dir)  # complete restore
            else:
                # Assume everything is restored
                sess.run(tf.global_variables_initializer())

            w_rec0 = sess.run(model.w_rec)

            # Shuffle training data
            shuffled_ix = np.random.permutation(np.arange(len(x_train)))
            x_train = x_train[shuffled_ix]
            y_train = y_train[shuffled_ix]
            num_batches = int(len(x_train) / batch_size) + 1

            costs = 0.0
            acc = 0.0
            for i in range(num_batches):
                min_ix = i * batch_size
                max_ix = np.min([len(x_train), ((i + 1) * batch_size)])
                batch_x = x_train[min_ix:max_ix]
                batch_y = y_train[min_ix:max_ix]

                sess.run(model.train_step, feed_dict={model.x: batch_x, model.y: batch_y})
                costs += sess.run(model.cost, feed_dict={model.x: batch_x, model.y: batch_y})
                acc += sess.run(model.acc, feed_dict={model.x: batch_x, model.y: batch_y})
            temp_train_loss = costs / num_batches
            temp_train_acc = acc / num_batches
            train_loss.append(temp_train_loss)
            train_accuracy.append(temp_train_acc)

            # validation
            temp_test_loss = sess.run(model.cost, feed_dict={model.x: x_test, model.y: y_test})
            temp_test_acc = sess.run(model.acc, feed_dict={model.x: x_test, model.y: y_test})
            test_loss.append(temp_test_loss)
            test_accuracy.append(temp_test_acc)
            print('Epoch: {}, Test Loss: {:.6f}, Test Acc: {:.6f}'.format(epoch + 1, temp_test_loss, temp_test_acc))

            # save the model
            model.save()

            # ----------------- weights evolution -----------------
            # Get weight list
            embedding_mat = sess.run(model.embed_matrix)
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
                noise_update = lambda th: np.random.normal(scale=1e-6, size=th.shape)

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
                    fname = open(os.path.join(model_dir, 'core_edge_list_weighted_' + str(epoch) + '.csv'), 'w', newline='')
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
    np.savetxt(os.path.join(model_dir, 'train_accuracy.txt'), np.asarray(train_accuracy))
    np.savetxt(os.path.join(model_dir, 'test_loss.txt'), np.asarray(test_loss))
    np.savetxt(os.path.join(model_dir, 'test_accuracy.txt'), np.asarray(test_accuracy))
    print("optimization finished!")

    print("time:", time.time() - t_start)

    # Plot loss over time
    epoch_seq = np.arange(1, hp['n_epochs'] + 1)
    plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
    plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
    plt.title('Softmax Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Softmax Loss')
    plt.legend(loc='upper left')
    plt.show()

    # Plot accuracy over time
    plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
    plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='../results/SMS/SET_n_100_p_0.05_zeta_0.1')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'epsilon_rec': 0.05,
          'rewiring_algorithm': 'SET',  # DeepR or SET
          'learning_rate': 5e-4,
          'zeta': 0.1,
          # 'optimizer': 'RMSProp'
          }
    train(args.modeldir,
          seed=1,
          hp=hp)

