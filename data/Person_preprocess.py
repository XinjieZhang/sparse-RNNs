# coding utf-8

import numpy as np

# https://github.com/raminmh/liquid_time_constant_networks/blob/master/experiments_with_ltcs/person.py

class_map = {
    'lying down': 0,
    'lying': 0,
    'sitting down': 1,
    'sitting': 1,
    'standing up from lying': 2,
    'standing up from sitting': 2,
    'standing up from sitting on the ground': 2,
    "walking": 3,
    "falling": 4,
    'on all fours': 5,
    'sitting on the ground': 6,
}  # 11 to 7

sensor_ids = {
    "010-000-024-033": 0,
    "010-000-030-096": 1,
    "020-000-033-111": 2,
    "020-000-032-221": 3
}


def one_hot(x, n):
    y = np.zeros(n, dtype=np.float32)
    y[x] = 1
    return y


def load_crappy_formated_csv():

    all_x = []
    all_y = []

    series_x = []
    series_y = []

    all_feats = []
    all_labels = []
    with open("../data/person/ConfLongDemo_JSI.txt", "r") as f:
        current_person = "A01"

        for line in f:
            arr = line.split(",")
            if (len(arr) < 6):
                break
            if (arr[0] != current_person):
                # Enque and reset
                series_x = np.stack(series_x, axis=0)
                series_y = np.array(series_y, dtype=np.int32)
                all_x.append(series_x)
                all_y.append(series_y)
                series_x = []
                series_y = []
            current_person = arr[0]
            sensor_id = sensor_ids[arr[1]]
            label_col = class_map[arr[7].replace("\n", "")]
            feature_col_2 = np.array(arr[4:7], dtype=np.float32)

            feature_col_1 = np.zeros(4, dtype=np.float32)
            feature_col_1[sensor_id] = 1

            feature_col = np.concatenate([feature_col_1, feature_col_2])
            # 100ms sampling time
            # print("feature_col: ",str(feature_col))
            series_x.append(feature_col)
            all_feats.append(feature_col)
            all_labels.append(one_hot(label_col, 7))
            series_y.append(label_col)

    all_labels = np.stack(all_labels, axis=0)
    print("all_labels.shape: ", str(all_labels.shape))
    prior = np.mean(all_labels, axis=0)
    print("Resampled Prior: ", str(prior * 100))
    all_feats = np.stack(all_feats, axis=0)
    print("all_feats.shape: ", str(all_feats.shape))

    all_mean = np.mean(all_feats, axis=0)
    all_std = np.std(all_feats, axis=0)
    all_mean[3:] = 0
    all_std[3:] = 1
    print("all_mean: ", str(all_mean))
    print("all_std: ", str(all_std))

    return all_x, all_y


def cut_in_sequences(all_x, all_y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for i in range(len(all_x)):
        x, y = all_x[i], all_y[i]

        for s in range(0, x.shape[0] - seq_len, inc):
            start = s
            end = start + seq_len
            sequences_x.append(x[start:end])
            sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class PersonData:

    def __init__(self, seq_len=32):

        all_x, all_y = load_crappy_formated_csv()
        all_x, all_y = cut_in_sequences(all_x, all_y, seq_len=seq_len, inc=seq_len // 2)

        total_seqs = all_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(27731).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = all_x[:, permutation[:valid_size]]
        self.valid_y = all_y[:, permutation[:valid_size]]
        self.test_x = all_x[:, permutation[valid_size:valid_size + test_size]]
        self.test_y = all_y[:, permutation[valid_size:valid_size + test_size]]
        self.train_x = all_x[:, permutation[valid_size + test_size:]]
        self.train_y = all_y[:, permutation[valid_size + test_size:]]

        print("Total number of test sequences: {}".format(self.test_x.shape[1]))

    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]
            yield (batch_x, batch_y)