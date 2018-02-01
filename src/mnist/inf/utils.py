# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

from __future__ import division
import basic_utils


def setup(hparams):
    expt_dir = get_expt_dir(hparams)
    hparams.ckpt_dir = './ckpt/' + expt_dir
    basic_utils.set_up_dir(hparams.ckpt_dir)


def get_expt_dir(hparams):
    expt_dir = 'bs{}_lr{}/'.format(
        hparams.batch_size,
        hparams.learning_rate
    )
    return expt_dir
