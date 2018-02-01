# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

from __future__ import division

from argparse import ArgumentParser

from commons import basic_utils
from commons import utils
from commons import arch
from commons import measure
from commons import hparams_def

from mnist.gen import utils as mnist_utils
from mnist.gen import gan_def as mnist_gan_def
from mnist.inf import inf_def as mnist_inf_def

from celebA.gen import utils as celebA_utils
from celebA.gen import gan_def as celebA_gan_def

# from cifar10 import utils as cifar10_utils
# from cifar10 import gan_def as cifar10_gan_def


def main(hparams):

    # Set up some stuff according to hparams
    utils.setup_vals(hparams)
    utils.setup_dirs(hparams)

    # print and save hparams.pkl
    basic_utils.print_hparams(hparams)
    basic_utils.save_hparams(hparams)

    # Get the measurement device
    mdevice = measure.get_mdevice(hparams)

    # Get network definitions
    if hparams.dataset == 'mnist':
        gan_def = mnist_gan_def
        inf_def = mnist_inf_def
        real_val_iterator = mnist_utils.RealValIterator()
    elif hparams.dataset == 'celebA':
        gan_def = celebA_gan_def
        inf_def = None
        real_val_iterator = celebA_utils.RealValIterator()
    # elif hparams.dataset == 'cifar10':
    #     gan_def = cifar10_gan_def
    #     real_val_iterator = cifar10_utils.RealValIterator()
    else:
        raise NotImplementedError

    # Get generator, discriminator
    if hparams.model_type == 'dcgan':
        generator = gan_def.generator_dcgan
        discriminator = gan_def.discriminator_dcgan
    elif hparams.model_type == 'wgangp':
        generator = gan_def.generator_wgangp
        discriminator = gan_def.discriminator_wgangp
    # elif hparams.model_type == 'acwgangp':
    #     generator = gan_def.generator_acwgangp
    #     discriminator = gan_def.discriminator_acwgangp
    else:
        raise NotImplementedError

    # Define the connections according to model class and run
    if hparams.model_class == 'unconditional':
        z_ph, x_ph = utils.get_phs_uncond(hparams)
        phs = (z_ph, x_ph)
        if mdevice.output_type == 'vector':
            discriminator = gan_def.discriminator_fc
        x_lossy, x_sample, \
        theta_ph, theta_gen_ph, \
        d_loss, g_loss, \
        d_update_op, g_update_op, iter_ph = arch.model_fn_uncond(hparams, z_ph, x_ph, generator, discriminator, mdevice)
        sess = utils.train(hparams, phs, d_update_op, g_update_op, d_loss, g_loss, x_sample, x_lossy, real_val_iterator,
                           theta_ph, theta_gen_ph, mdevice, iter_ph, inf_def)
        # utils.visualize_uncond(hparams, sess, z_ph, x_sample)
    elif hparams.model_class == 'conditional':
        z_ph, x_ph, y_ph = utils.get_phs_cond(hparams)
        phs = (z_ph, x_ph, y_ph)
        if mdevice.output_type == 'vector':
            discriminator = gan_def.discriminator_fc_cond
        x_lossy, x_sample, \
        theta_ph, theta_gen_ph, \
        d_loss, g_loss, \
        d_update_op, g_update_op, iter_ph = arch.model_fn_cond(hparams, z_ph, x_ph, y_ph, generator, discriminator, mdevice)
        sess = utils.train(hparams, phs, d_update_op, g_update_op, d_loss, g_loss, x_sample, x_lossy, real_val_iterator,
                           theta_ph, theta_gen_ph, mdevice, iter_ph, inf_def)
        # utils.visualize_cond(hparams, sess, z_ph, y_ph, x_sample)
    # elif hparams.model_class == 'auxcond':
    #     z_ph, x_ph, y_ph = utils.get_phs_cond(hparams)
    #     phs = (z_ph, x_ph, y_ph)
    #     if mdevice.output_type == 'vector':
    #         discriminator = gan_def.discriminator_fc_cond
    #     x_lossy, x_sample, \
    #     theta_ph, theta_gen_ph, \
    #     d_loss, g_loss, \
    #     d_update_op, g_update_op, iter_ph = arch.model_fn_auxcond(hparams, z_ph, x_ph, y_ph, generator, discriminator, mdevice)
    #     sess = utils.train(hparams, phs, d_update_op, g_update_op, d_loss, g_loss, x_sample, x_lossy, real_val_iterator,
    #                        theta_ph, theta_gen_ph, mdevice, iter_ph)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--hparams', type=str, help='Comma separated list of "name=value" pairs.')
    ARGS = PARSER.parse_args()
    HPARAMS = hparams_def.get_hparams(ARGS)
    main(HPARAMS)
