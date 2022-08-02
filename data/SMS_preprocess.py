# coding utf-8

import io
import os
import re
import requests
from zipfile import ZipFile
import numpy as np
import tensorflow as tf

# https://github.com/NetworkRanger/tensorflow-ml-exercise


# Create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string


class SMS:
    def __init__(self, hp):

        # Download or open data
        data_dir = '../data/SMS_data'
        data_file = 'text_data.txt'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if not os.path.isfile(os.path.join(data_dir, data_file)):
            zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
            r = requests.get(zip_url)
            z = ZipFile(io.BytesIO(r.content))
            file = z.read('SMSSpamCollection')
            # Format Data
            text_data = file.decode()
            text_data = text_data.encode('ascii', errors='ignore')
            text_data = text_data.decode().split('\n')

            # Save data to text file
            with open(os.path.join(data_dir, data_file), 'w') as file_conn:
                for text in text_data:
                    file_conn.write("{}\n".format(text))
        else:
            # Open data from text file
            text_data = []
            with open(os.path.join(data_dir, data_file), 'r') as file_conn:
                for row in file_conn:
                    text_data.append(row)
            text_data = text_data[:-1]

        text_data = [x.split('\t') for x in text_data if len(x) >= 1]
        [text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

        # Clean texts
        text_data_train = [clean_text(x) for x in text_data_train]

        # Change texts into numeric vectors
        max_sequence_length = hp['max_sequence_length']
        min_word_frequency = hp['min_word_frequency']
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                             min_frequency=min_word_frequency)
        text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

        # Shuffle and split data
        text_processed = np.array(text_processed)
        text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
        shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
        x_shuffled = text_processed[shuffled_ix]
        y_shuffled = text_data_target[shuffled_ix]

        # Split train/test set
        ix_cutoff = int(len(y_shuffled) * 0.80)
        self.x_train, self.x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
        self.y_train, self.y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
        self.vocab_size = len(vocab_processor.vocabulary_)
