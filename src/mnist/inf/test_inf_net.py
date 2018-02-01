import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import inf_def

def main():
    inf_net = inf_def.InferenceNetwork()
    targets = tf.placeholder(tf.float32, [None, 10])
    correct_prediction = tf.equal(tf.argmax(inf_net.logits, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)
    test_images_reshaped = np.reshape(mnist.test.images, [10000, 28, 28, 1])
    feed_dict = {inf_net.x_ph: test_images_reshaped,
                 targets: mnist.test.labels,
                 inf_net.keep_prob: 1.0}

    print inf_net.sess.run(accuracy, feed_dict=feed_dict)


if __name__ == '__main__':
    main()
