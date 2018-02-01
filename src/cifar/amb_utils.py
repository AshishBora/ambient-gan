import amb_basic_utils
import amb_dir_def


def setup_vals(hparams):
    """Setup some values in hparams"""

    # hparams.c_dim = 3
    hparams.image_dims = [32, 32, 3]
    # hparams.train_size = 50000
    # hparams.y_dim = 10  # [Optional] Number of labels
    hparams.signal_power = 104.953
    hparams.x_min = 0.0
    hparams.x_max = 255.0

    expt_dir = amb_dir_def.get_expt_dir(hparams)
    hparams.hparams_dir = hparams.results_dir + 'hparams/' + expt_dir
    hparams.sample_dir = hparams.results_dir + 'samples/' + expt_dir
    hparams.metrics_dir = hparams.results_dir + 'metrics/' + expt_dir
    hparams.ckpt_dir = hparams.results_dir + 'ckpt/' + expt_dir


def setup_dirs(hparams):
    """Setup the dirs"""
    amb_basic_utils.set_up_dir(hparams.hparams_dir)
    amb_basic_utils.set_up_dir(hparams.sample_dir)
    amb_basic_utils.set_up_dir(hparams.metrics_dir)
    amb_basic_utils.set_up_dir(hparams.ckpt_dir)
