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
from model.smsnet import Model
from model.structure_evolution import createWeightMask
from data.SMS_preprocess import SMS
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_epochs': 200,
        'batch_size': 128,
        'max_sequence_length': 25,
        'hidden_size': 100,
        'embedding_size': 50,
        'min_word_frequency': 10,
        'learning_rate': 5e-4,
        'epsilon_rec': None
    }
    return hp


def get_network_model(txt_path):
    # get weighted edge list
    n = 100 # number of nodes
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
    embedding_mat_ini = tf.constant_initializer(embedding_mat)

    # initialize weights
    n_input = embedding_size
    n_hidden = hp['hidden_size']
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
        w_rec0 = rng.uniform(low=-0.6, high=0.6, size=[n_hidden, n_hidden])
        txt_file = hp['network_edge_list']
        txt_path = os.path.join("..\\model", "network", "generated", "SMS", "Trial1", txt_file)
        w_sign = get_network_model(txt_path=txt_path)
        hp['w_rec_mask'] = abs(w_sign).tolist()
        w_rec0 = w_sign * abs(w_rec0)
    save_hp(hp, model_dir)

    recurrent_kernel_initializer = tf.constant_initializer(w_rec0, dtype=tf.float32)
    
    # Build the model
    model = Model(model_dir,
                  input_kernel_initializer=input_kernel_initializer,
                  recurrent_kernel_initializer=recurrent_kernel_initializer,
                  embedding_mat_initializer=embedding_mat_ini,
                  hp=hp)

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

        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        batch_size = hp['batch_size']
        for epoch in range(hp['n_epochs']):
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

    parser.add_argument('--modeldir', type=str, default='../results/SMS/test')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'epsilon_rec': None,
          'learning_rate': 5e-4,
          'RNNCell': 'Fixed',
          'optimizer': 'adam',
          'network_edge_list': 'BN_n_100_p_0.05_0_0.646.txt'
          }
    train(args.modeldir,
          seed=0,
          hp=hp)

