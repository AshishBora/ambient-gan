# pylint: disable = C0103, C0111, C0301, R0914

import tensorflow as tf
from commons import ops
import wgan_utils


def generator_dcgan(hparams, z, scope_name, train, reuse):

    gf_dim = 64  # dimension of generator filters in first conv layer

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        s = 64
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        g_bn0 = ops.batch_norm(name='g_bn0')
        g_bn1 = ops.batch_norm(name='g_bn1')
        g_bn2 = ops.batch_norm(name='g_bn2')
        g_bn3 = ops.batch_norm(name='g_bn3')

        # project `z` and reshape
        h0 = tf.reshape(ops.linear(z, gf_dim*8*s16*s16, 'g_h0_lin'), [-1, s16, s16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0, train=train))

        h1 = ops.deconv2d(h0, [hparams.batch_size, s8, s8, gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(g_bn1(h1, train=train))

        h2 = ops.deconv2d(h1, [hparams.batch_size, s4, s4, gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2, train=train))

        h3 = ops.deconv2d(h2, [hparams.batch_size, s2, s2, gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3, train=train))

        h4 = ops.deconv2d(h3, [hparams.batch_size, s, s, hparams.c_dim], name='g_h4')
        x_gen = tf.nn.tanh(h4)

    return x_gen


def discriminator_dcgan(hparams, x, scope_name, train, reuse):

    df_dim = 64  # dimension of discriminator filters in first conv layer

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        d_bn1 = ops.batch_norm(name='d_bn1')
        d_bn2 = ops.batch_norm(name='d_bn2')
        d_bn3 = ops.batch_norm(name='d_bn3')

        h0 = ops.lrelu(ops.conv2d(x, df_dim, name='d_h0_conv'))

        h1 = ops.conv2d(h0, df_dim*2, name='d_h1_conv')
        h1 = ops.lrelu(d_bn1(h1, train=train))

        h2 = ops.conv2d(h1, df_dim*4, name='d_h2_conv')
        h2 = ops.lrelu(d_bn2(h2, train=train))

        h3 = ops.conv2d(h2, df_dim*8, name='d_h3_conv')
        h3 = ops.lrelu(d_bn3(h3, train=train))

        h4 = ops.linear(tf.reshape(h3, [hparams.batch_size, -1]), 1, 'd_h3_lin')

        d_logit = h4
        d = tf.nn.sigmoid(d_logit)

    return d, d_logit


def discriminator_fc(hparams, x, scope_name, train, reuse):  # pylint: disable = W0613
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        d_bn0 = ops.batch_norm(name='d_bn0')
        h0 = ops.linear(x, 100, 'd_h0_lin')
        h0 = ops.lrelu(d_bn0(h0, train=train))

        d_bn1 = ops.batch_norm(name='d_bn1')
        h1 = ops.linear(h0, 100, 'd_h1_lin')
        h1 = ops.lrelu(d_bn1(h1, train=train))

        d_logit = ops.linear(h1, 1, 'd_h2_lin')
        d = tf.nn.sigmoid(d_logit)

        return d, d_logit


def generator_wgangp(hparams, z, scope_name, train, reuse):  # pylint: disable = W0613

    gen_dim = 16
    dims = [64 * gen_dim, 64 * gen_dim // 2, 64 * gen_dim // 4, 64 * gen_dim // 8, 3]
    resized_image_size = 64
    z_dim = 100
    activation = tf.nn.relu

    N = len(dims)
    image_size = resized_image_size // (2 ** (N - 1))

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        W_z = wgan_utils.weight_variable([z_dim, dims[0] * image_size * image_size], name="W_z")
        h_z = tf.matmul(z, W_z)
        h_z = tf.reshape(h_z, [-1, image_size, image_size, dims[0]])
        h_bnz = wgan_utils.batch_norm(h_z, dims[0], train, scope="gen_bnz")
        h = activation(h_bnz, name='h_z')
        # wgan_utils.add_activation_summary(h)

        for index in range(N - 2):
            image_size *= 2
            W = wgan_utils.weight_variable([4, 4, dims[index + 1], dims[index]], name="W_%d" % index)
            b = tf.zeros([dims[index + 1]])
            deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[index + 1]])
            h_conv_t = wgan_utils.conv2d_transpose_strided(h, W, b, output_shape=deconv_shape)
            h_bn = wgan_utils.batch_norm(h_conv_t, dims[index + 1], train, scope="gen_bn%d" % index)
            h = activation(h_bn, name='h_%d' % index)
            # wgan_utils.add_activation_summary(h)

        image_size *= 2
        W_pred = wgan_utils.weight_variable([4, 4, dims[-1], dims[-2]], name="W_pred")
        b = tf.zeros([dims[-1]])
        deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[-1]])
        h_conv_t = wgan_utils.conv2d_transpose_strided(h, W_pred, b, output_shape=deconv_shape)
        x_gen = tf.nn.tanh(h_conv_t, name='x_gen')
        # wgan_utils.add_activation_summary(x_gen)

    return x_gen


def discriminator_wgangp(hparams, x, scope_name, train, reuse):  # pylint: disable = W0613

    dims = [3, 64, 64 * 2, 64 * 4, 64 * 8, 1]
    activation = tf.nn.relu

    N = len(dims)

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        skip_bn = True  # First layer of discriminator skips batch norm
        for index in range(N - 2):
            W = wgan_utils.weight_variable([4, 4, dims[index], dims[index + 1]], name="W_%d" % index)
            b = tf.zeros([dims[index+1]])
            h_conv = wgan_utils.conv2d_strided(x, W, b)
            if skip_bn:
                h_bn = h_conv
                skip_bn = False
            else:
                h_bn = tf.contrib.layers.batch_norm(inputs=h_conv, decay=0.9, epsilon=1e-5, is_training=train, scope="disc_bn%d" % index)
            h = activation(h_bn, name="h_%d" % index)
            # wgan_utils.add_activation_summary(h)

        W_pred = wgan_utils.weight_variable([4, 4, dims[-2], dims[-1]], name="W_pred")
        b = tf.zeros([dims[-1]])
        h_pred = wgan_utils.conv2d_strided(h, W_pred, b)

    return None, h_pred
