"""Utility functions."""

import os
import errno
import json
import numpy as np
import tensorflow as tf


# https://github.com/gyyang/multitask/blob/master/tools.py
def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    if not os.path.isfile(fname):
        fname = os.path.join(model_dir, 'hparams.json')  # backward compat
        if not os.path.isfile(fname):
            return None

    with open(fname, 'r') as f:
        hp = json.load(f)

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp['rng'] = np.random.RandomState(hp['seed']+1000)
    return hp


def print_variables():
    """
    print trainable variables

    """
    print("[*] Model Trainable Variables:")
    parm_cnt = 0
    variable = [v for v in tf.trainable_variables()]
    for v in variable:
        print("   ", v.name, v.get_shape())
        parm_cnt_v = 1
        for i in v.get_shape().as_list():
            parm_cnt_v *= i
        parm_cnt += parm_cnt_v
    print("[*] Model Param Size: %.4fM" % (parm_cnt / 1024 / 1024))


# https://github.com/gyyang/multitask/blob/master/tools.py
def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# https://github.com/gyyang/multitask/blob/master/tools.py
def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)




