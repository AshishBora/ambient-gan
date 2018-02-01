# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

from __future__ import division

import os
import shutil
import cPickle as pickle
import tensorflow as tf


def get_trainable_vars(scope_name):
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
    return train_vars


def print_hparams(hparams):
    hparam_values = hparams.values()
    keys = sorted(hparam_values.keys())
    print ''
    for key in keys:
        print '{} = {}'.format(key, hparam_values[key])
    print ''


def save_hparams(hparams):
    pkl_filepath = hparams.hparams_dir + 'hparams.pkl'
    with open(pkl_filepath, 'wb') as f:
        pickle.dump(hparams, f)


def read_hparams(pkl_filepath):
    with open(pkl_filepath, 'rb') as f:
        hparams = pickle.load(f)
    return hparams


def get_ckpt_path(ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = os.path.join(ckpt_dir,
                                 ckpt.model_checkpoint_path)
    else:
        ckpt_path = None
    return ckpt_path


def try_restore(hparams, sess, model_saver):
    # Attempt to restore variables from checkpoint
    ckpt_path = get_ckpt_path(hparams.ckpt_dir)
    if ckpt_path:  # if a previous ckpt exists
        model_saver.restore(sess, ckpt_path)
        init_train_iter = int(ckpt_path.split('/')[-1].split('-')[-1])
        print 'Succesfully loaded model from {0} at train_iter = {1}'.format(ckpt_path, init_train_iter)
    else:
        print 'No checkpoint found'
        init_train_iter = -1
    return init_train_iter


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)


def get_optimizer(hparams, lr):
    if hparams.opt_type == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    if hparams.opt_type == 'momentum':
        return tf.train.MomentumOptimizer(lr, hparams.opt_param1)
    elif hparams.opt_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr, decay=hparams.opt_param1)
    elif hparams.opt_type == 'adam':
        return tf.train.AdamOptimizer(lr, beta1=hparams.opt_param1, beta2=hparams.opt_param2)
    elif hparams.opt_type == 'adagrad':
        return tf.train.AdagradOptimizer(lr)
    else:
        raise Exception('Optimizer {} not supported'.format(hparams.opt_type))


def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data


def save_to_pickle(data, pkl_filepath):
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
