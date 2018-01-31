""" Define architectures for generative models.

Generation modes: Unconditional, Conditional, Auxillary Conditional
Loss types: Vanilla, Wasserstein
Loss addons: Gradient Penalty

"""

# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

import tensorflow as tf
import utils


def loss_vanilla(d_logit, d_gen_logit):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit, labels=tf.ones_like(d_logit)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_gen_logit, labels=tf.zeros_like(d_gen_logit)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_gen_logit, labels=tf.ones_like(d_gen_logit)))
    return d_loss, g_loss


def loss_wasserstein(d_logit, d_gen_logit):
    ###### NOTE #####
    # The signs of the losses are flipped inadvertantly as compared to usual presentation.
    # Since the discriminator outputs a single number, this doesn't change the objective --
    # we can just interpret the discriminator output as prob(fake) instead of prob(real)
    #################
    d_loss = tf.reduce_mean(d_logit - d_gen_logit)
    g_loss = tf.reduce_mean(d_gen_logit)
    return d_loss, g_loss


def loss_auxcond(aclogits, aclogits_gen, y):
    d_acloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=aclogits, labels=y))
    d_acloss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=aclogits_gen, labels=y))
    d_acloss = (d_acloss_real + d_acloss_gen) / 2.0
    g_acloss = d_acloss_gen
    return d_acloss, g_acloss


def loss_gradient_penalty(hparams, discriminator, x_lossy, x_gen_lossy, y):

    x_lossy_dims = len(x_lossy.get_shape())

    alpha_shape = [1 for _ in range(x_lossy_dims)]
    alpha_shape[0] = hparams.batch_size
    alpha = tf.random_uniform(shape=alpha_shape, minval=0., maxval=1.)
    differences = x_gen_lossy - x_lossy
    interpolates = x_lossy + (alpha * differences)

    if hparams.model_class in ['conditional', 'auxcond']:
        _, disc_interp, _ = discriminator(hparams, interpolates, y, 'discrim', train=False, reuse=True)
    else:
        _, disc_interp = discriminator(hparams, interpolates, 'discrim', train=False, reuse=True)

    gradients = tf.gradients(disc_interp, [interpolates])[0]
    reduction_indices = range(1, x_lossy_dims)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=reduction_indices))
    d_gp_loss = tf.reduce_mean((slopes - 1.0)**2)

    return d_gp_loss


def get_loss(hparams,
             d_logit, d_gen_logit,
             x_lossy, x_gen_lossy, discriminator,
             ac_logits, ac_logits_gen, y):

    if hparams.model_type == 'dcgan':
        d_loss, g_loss = loss_vanilla(d_logit, d_gen_logit)
    elif hparams.model_type == 'wgan':
        d_loss, g_loss = loss_wasserstein(d_logit, d_gen_logit)
    elif hparams.model_type == 'wgangp':
        d_w_loss, g_loss = loss_wasserstein(d_logit, d_gen_logit)
        d_gp_loss = loss_gradient_penalty(hparams, discriminator, x_lossy, x_gen_lossy, None)
        d_loss = d_w_loss + hparams.gp_lambda * d_gp_loss
    elif hparams.model_type == 'acwgangp':

        # Get wasserstein losses
        d_w_loss = tf.reduce_mean(d_gen_logit - d_logit)
        g_w_loss = tf.reduce_mean(-d_gen_logit)

        # Get gradient penalty loss
        d_gp_loss = loss_gradient_penalty(hparams, discriminator, x_lossy, x_gen_lossy, y)

        # Get auxillary conditional loss
        # in the original implementation, fake y are also created. Here, we just reuse real labels.
        d_ac_loss, g_ac_loss = loss_auxcond(ac_logits, ac_logits_gen, y)

        # Get total loss
        d_loss = d_w_loss + hparams.d_ac_lambda*d_ac_loss + hparams.gp_lambda*d_gp_loss
        g_loss = g_w_loss + hparams.g_ac_lambda*g_ac_loss

    else:
        raise NotImplementedError

    return d_loss, g_loss


def get_lossy(hparams, mdevice, x, theta_ph, x_gen, theta_gen_ph):

    # Measurements / Unmeasurements
    if hparams.train_mode == 'ambient':
        x_lossy = mdevice.measure(hparams, x, theta_ph)  # measure real
        x_gen_lossy = mdevice.measure(hparams, x_gen, theta_gen_ph)  # measure generated
    elif hparams.train_mode == 'unmeasure':
        x_lossy = x  # measure and then unmeasure on real happens outside tensorflow
        x_gen_lossy = x_gen  # do not measure generated
    elif hparams.train_mode == 'baseline':
        # baseline can be used only for cases where measurements are same dimension as original
        x_lossy = mdevice.measure(hparams, x, theta_ph) # measure real
        x_gen_lossy = x_gen  # do not measure generated
    else:
        raise NotImplementedError

    return x_lossy, x_gen_lossy



def model_fn_auxcond(hparams, z, x, y, generator, discriminator, mdevice):

    # Get theta placeholders
    theta_ph = mdevice.get_theta_ph(hparams)
    theta_gen_ph = mdevice.get_theta_ph(hparams)

    # Get generations
    x_gen = generator(hparams, y, z, 'gen', train=True, reuse=False)
    x_sample = generator(hparams, y, z, 'gen', train=False, reuse=True)

    # Get lossy versions
    x_lossy, x_gen_lossy = get_lossy(hparams, mdevice, x, theta_ph, x_gen, theta_gen_ph)

    # Apply discriminator
    _, d_logit, ac_logits = discriminator(hparams, x_lossy, y, 'discrim', train=True, reuse=False)
    _, d_gen_logit, ac_logits_gen = discriminator(hparams, x_gen_lossy, y, 'discrim', train=True, reuse=True)

    # Get loss
    d_loss, g_loss = get_loss(hparams,
                              d_logit, d_gen_logit,
                              x_lossy, x_gen_lossy, discriminator,
                              ac_logits, ac_logits_gen, y)

    # Get train ops
    d_update_op, g_update_op, iter_ph = utils.get_train_ops(hparams, d_loss, g_loss)

    return x_lossy, x_sample, theta_ph, theta_gen_ph, d_loss, g_loss, d_update_op, g_update_op, iter_ph



def model_fn_cond(hparams, z, x, y, generator, discriminator, mdevice):

    # Get theta placeholders
    theta_ph = mdevice.get_theta_ph(hparams)
    theta_gen_ph = mdevice.get_theta_ph(hparams)

    # Get generations
    x_gen = generator(hparams, y, z, 'gen', train=True, reuse=False)
    x_sample = generator(hparams, y, z, 'gen', train=False, reuse=True)

    # Get lossy versions
    x_lossy, x_gen_lossy = get_lossy(hparams, mdevice, x, theta_ph, x_gen, theta_gen_ph)

    # Apply discriminator
    _, d_logit = discriminator(hparams, x_lossy, y, 'discrim', train=True, reuse=False)
    _, d_gen_logit = discriminator(hparams, x_gen_lossy, y, 'discrim', train=True, reuse=True)

    # Get loss
    d_loss, g_loss = get_loss(hparams,
                              d_logit, d_gen_logit,
                              x_lossy, x_gen_lossy, discriminator,
                              None, None, None)

    # Get train ops
    d_update_op, g_update_op, iter_ph = utils.get_train_ops(hparams, d_loss, g_loss)

    return x_lossy, x_sample, theta_ph, theta_gen_ph, d_loss, g_loss, d_update_op, g_update_op, iter_ph


def model_fn_uncond(hparams, z, x, generator, discriminator, mdevice):

    # Get theta placeholders
    theta_ph = mdevice.get_theta_ph(hparams)
    theta_gen_ph = mdevice.get_theta_ph(hparams)

    # Get generations
    x_gen = generator(hparams, z, 'gen', train=True, reuse=False)
    x_sample = generator(hparams, z, 'gen', train=False, reuse=True)

    # Get lossy versions
    x_lossy, x_gen_lossy = get_lossy(hparams, mdevice, x, theta_ph, x_gen, theta_gen_ph)

    # Apply discriminator
    _, d_logit = discriminator(hparams, x_lossy, 'discrim', train=True, reuse=False)
    _, d_gen_logit = discriminator(hparams, x_gen_lossy, 'discrim', train=True, reuse=True)

    # Get loss
    d_loss, g_loss = get_loss(hparams,
                              d_logit, d_gen_logit,
                              x_lossy, x_gen_lossy, discriminator,
                              None, None, None)

    # Get train ops
    d_update_op, g_update_op, iter_ph = utils.get_train_ops(hparams, d_loss, g_loss)

    return x_lossy, x_sample, theta_ph, theta_gen_ph, d_loss, g_loss, d_update_op, g_update_op, iter_ph
