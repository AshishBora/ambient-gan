"""Train a network on MNIST.
For best results, run this file from the same folder that contains it.
"""

from __future__ import division

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

import model_def
import basic_utils
import utils


def main():

    # Get the hparams
    hparams = model_def.get_hparams()

    # Set up some things according to hparams
    utils.setup(hparams)

    # Create the model
    inputs = tf.placeholder(tf.float32, [None, hparams.input_dim])
    targets = tf.placeholder(tf.float32, [None, hparams.output_dim])
    logits, _, keep_prob = model_def.infer(hparams, inputs, 'inf')

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    train_op = tf.train.AdamOptimizer(hparams.learning_rate).minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Get a new TF session
    sess = tf.Session()

    # Model checkpointing setup
    model_saver = tf.train.Saver(max_to_keep=hparams.max_checkpoints)

    # Initialization
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Attempt to restore variables from checkpoint
    init_train_iter = basic_utils.try_restore(hparams, sess, model_saver)

    # Import data
    mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)

    # print hparams
    for var in  basic_utils.get_trainable_vars('inf'):
        print var.op.name

    # Train
    for train_iter in range(init_train_iter, hparams.max_train_iter):
        # Save checkpoint
        if train_iter % 100 == 0:
            save_path = os.path.join(hparams.ckpt_dir, 'snapshot')
            model_saver.save(sess, save_path, global_step=train_iter)
        # Train iteration
        batch_xs, batch_ys = mnist.train.next_batch(hparams.batch_size)
        train_accuracy = sess.run(accuracy, feed_dict={inputs: batch_xs, targets: batch_ys, keep_prob: 1.0})
        print 'step {}, training accuracy {}'.format(train_iter, train_accuracy)
        sess.run(train_op, feed_dict={inputs: batch_xs,
                                      targets: batch_ys,
                                      keep_prob: 0.5})

    # Save final checkpoint
    save_path = os.path.join(hparams.ckpt_dir, 'snapshot')
    model_saver.save(sess, save_path, global_step=hparams.max_train_iter)

    # Test trained model
    accuracy_val = sess.run(accuracy, feed_dict={inputs: mnist.test.images,
                                                 targets: mnist.test.labels,
                                                 keep_prob: 0.5})
    print 'Test accuracy = {}'.format(accuracy_val)


if __name__ == '__main__':
    main()
