# pylint: disable = C0103, C0111, C0301, R0913, R0914

import tensorflow as tf
from commons import ops
import wganlib


def NCHW_to_NHWC(inputs):
    outputs = tf.transpose(inputs, [0, 2, 3, 1], name='NCHW_to_NHWC')
    return outputs


def NHWC_to_NCHW(inputs):
    outputs = tf.transpose(inputs, [0, 3, 1, 2], name='NHWC_to_NCHW')
    return outputs


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def generator_dcgan(hparams, y, z, scope_name, train, reuse):
    """
    The architecture is from
    https://github.com/carpedm20/DCGAN-tensorflow
    License: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
    """

    s = 28
    gf_dim = 64  # dimension of generator filters in first conv layer
    gfc_dim = 1024  # dimension of generator units for for fully connected layer

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        s2, s4 = int(s/2), int(s/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [hparams.batch_size, 1, 1, hparams.y_dim])
        z = tf.concat([z, y], 1)

        g_bn0 = ops.batch_norm(name='g_bn0')
        h0 = ops.linear(z, gfc_dim, 'g_h0_lin')
        h0 = tf.nn.relu(g_bn0(h0, train=train))
        h0 = tf.concat([h0, y], 1)

        g_bn1 = ops.batch_norm(name='g_bn1')
        h1 = ops.linear(h0, gf_dim*2*s4*s4, 'g_h1_lin')
        h1 = tf.nn.relu(g_bn1(h1, train=train))
        h1 = tf.reshape(h1, [hparams.batch_size, s4, s4, gf_dim * 2])
        h1 = ops.conv_cond_concat(h1, yb)

        g_bn2 = ops.batch_norm(name='g_bn2')
        h2 = ops.deconv2d(h1, [hparams.batch_size, s2, s2, gf_dim * 2], name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2, train=train))
        h2 = ops.conv_cond_concat(h2, yb)

        x_gen = tf.nn.sigmoid(ops.deconv2d(h2, [hparams.batch_size, s, s, hparams.c_dim], name='g_h3'))

    return x_gen


def discriminator_dcgan(hparams, x, y, scope_name, train, reuse):
    """
    The architecture is from
    https://github.com/carpedm20/DCGAN-tensorflow
    License: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
    """

    df_dim = 64  # dimension of discriminator filters in first conv layer
    dfc_dim = 1024  # dimension of discriminator units for fully connected layer

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        yb = tf.reshape(y, [hparams.batch_size, 1, 1, hparams.y_dim])
        x = ops.conv_cond_concat(x, yb)

        h0 = ops.lrelu(ops.conv2d(x, hparams.c_dim + hparams.y_dim, name='d_h0_conv'))
        h0 = ops.conv_cond_concat(h0, yb)

        d_bn1 = ops.batch_norm(name='d_bn1')
        h1 = ops.conv2d(h0, df_dim + hparams.y_dim, name='d_h1_conv')
        h1 = ops.lrelu(d_bn1(h1, train=train))
        h1 = tf.reshape(h1, [hparams.batch_size, -1])
        h1 = tf.concat([h1, y], 1)

        d_bn2 = ops.batch_norm(name='d_bn2')
        h2 = ops.linear(h1, dfc_dim, 'd_h2_lin')
        h2 = ops.lrelu(d_bn2(h2, train=train))
        h2 = tf.concat([h2, y], 1)

        h3 = ops.linear(h2, 1, 'd_h3_lin')

        d_logit = h3
        d = tf.nn.sigmoid(d_logit)

    return d, d_logit


def discriminator_fc_cond(hparams, x, y, scope_name, train, reuse):  # pylint: disable = W0613
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        x = tf.concat([x, y], 1)

        d_bn0 = ops.batch_norm(name='d_bn0')
        h0 = ops.linear(x, 25, 'd_h0_lin')
        h0 = ops.lrelu(d_bn0(h0, train=train))
        h0 = tf.concat([h0, y], 1)

        d_bn1 = ops.batch_norm(name='d_bn1')
        h1 = ops.linear(h0, 25, 'd_h1_lin')
        h1 = ops.lrelu(d_bn1(h1, train=train))
        h1 = tf.concat([h1, y], 1)

        d_logit = ops.linear(h1, 1, 'd_h2_lin')
        d = tf.nn.sigmoid(d_logit)

    return d, d_logit


def discriminator_fc(hparams, x, scope_name, train, reuse):  # pylint: disable = W0613
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        d_bn0 = ops.batch_norm(name='d_bn0')
        h0 = ops.linear(x, 25, 'd_h0_lin')
        h0 = ops.lrelu(d_bn0(h0, train=train))

        d_bn1 = ops.batch_norm(name='d_bn1')
        h1 = ops.linear(h0, 25, 'd_h1_lin')
        h1 = ops.lrelu(d_bn1(h1, train=train))

        d_logit = ops.linear(h1, 1, 'd_h2_lin')
        d = tf.nn.sigmoid(d_logit)

    return d, d_logit



def generator_wgangp(hparams, z, scope_name, train, reuse):  # pylint: disable = W0613
    """
    The architecture is from
    https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py
    License: https://github.com/igul222/improved_wgan_training/blob/master/LICENSE
    """

    DIM = 64 # Model dimensionality
    assert hparams.z_dim == 128

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        output = wganlib.linear.Linear('Generator.Input', 128, 4*4*4*DIM, z)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*DIM, 4, 4])

        output = wganlib.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
        output = tf.nn.relu(output)

        output = output[:, :, :7, :7]

        output = wganlib.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
        output = tf.nn.relu(output)

        output = wganlib.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
        output = tf.nn.sigmoid(output)

        x_gen = NCHW_to_NHWC(output)
        x_gen = tf.reshape(x_gen, [hparams.batch_size, 28, 28, 1])

    return x_gen


def discriminator_wgangp(hparams, x, scope_name, train, reuse):  # pylint: disable = W0613
    """
    The architecture is from
    https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py
    License: https://github.com/igul222/improved_wgan_training/blob/master/LICENSE
    """

    DIM = 64 # Model dimensionality

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        output = NHWC_to_NCHW(x)

        output = wganlib.conv2d.Conv2D('Discriminator.1', 1, DIM, 5, output, stride=2)
        output = LeakyReLU(output)

        output = wganlib.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
        output = LeakyReLU(output)

        output = wganlib.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
        output = LeakyReLU(output)

        output = tf.reshape(output, [-1, 4*4*4*DIM])
        output = wganlib.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return None, tf.reshape(output, [-1])
