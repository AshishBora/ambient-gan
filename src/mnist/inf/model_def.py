# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

"""Define the inference model

The network definition and some functions are from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py
"""

import tensorflow as tf


def get_hparams():
    hparams = tf.contrib.training.HParams(
        input_dim=784,
        output_dim=10,
        batch_size=64,
        max_train_iter=20000,
        learning_rate=1e-4,
        max_checkpoints=1,
    )
    return hparams


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)


def bias_variable(name, shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)
    # return tf.Variable(initial)


def infer(inputs, scope_name='inf'):

    with tf.variable_scope(scope_name):

        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            x_image = tf.reshape(inputs, [-1, 28, 28, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable('w_c1', [5, 5, 1, 32])
            b_conv1 = bias_variable('b_c1', [32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable('w_c2', [5, 5, 32, 64])
            b_conv2 = bias_variable('b_c2', [64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable('w_fc1', [7 * 7 * 64, 1024])
            b_fc1 = bias_variable('b_fc1', [1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable('w_fc2', [1024, 10])
            b_fc2 = bias_variable('b_fc2', [10])
            logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        predictions = tf.nn.softmax(logits)

    return logits, predictions, keep_prob


def inf_restore_vars():
    restore_vars = [
        'inf/w_c1',
        'inf/b_c1',
        'inf/w_c2',
        'inf/b_c2',
        'inf/w_fc1',
        'inf/b_fc1',
        'inf/w_fc2',
        'inf/b_fc2',
    ]
    return restore_vars
