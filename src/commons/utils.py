# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101, C0103

from __future__ import division

import os
import scipy.misc
import numpy as np
import tensorflow as tf

import basic_utils
import dir_def
# import inception_score


def setup_vals(hparams):
    """Setup some values in hparams"""
    if hparams.dataset == 'mnist':
        hparams.c_dim = 1
        hparams.image_dims = [28, 28, 1]
        hparams.train_size = 60000
        hparams.y_dim = 10  # [Optional] Number of labels
        hparams.x_min = 0
        hparams.x_max = 1
        hparams.signal_power = 0.11204461  # Assuming each pixel in [0, 1]
    elif hparams.dataset == 'celebA':
        hparams.c_dim = 3
        hparams.image_dims = [64, 64, 3]
        hparams.train_size = 180000
        hparams.x_min = -1
        hparams.x_max = 1
        hparams.signal_power = 0.2885201  # Assuming each pixel in [-1, 1]
    # elif hparams.dataset == 'cifar10':
    #     hparams.c_dim = 3
    #     hparams.image_dims = [32, 32, 3]
    #     hparams.train_size = 50000
    #     hparams.y_dim = 10  # [Optional] Number of labels
    else:
        raise NotImplementedError

    expt_dir = dir_def.get_expt_dir(hparams)
    hparams.hparams_dir = hparams.results_dir + 'hparams/' + expt_dir
    hparams.ckpt_dir    = hparams.results_dir + 'ckpt/'    + expt_dir
    hparams.summary_dir = hparams.results_dir + 'summ/'    + expt_dir
    hparams.sample_dir  = hparams.results_dir + 'samples/' + expt_dir


def setup_dirs(hparams):
    """Setup the dirs"""
    basic_utils.set_up_dir(hparams.hparams_dir)
    basic_utils.set_up_dir(hparams.ckpt_dir)
    basic_utils.set_up_dir(hparams.summary_dir)
    basic_utils.set_up_dir(hparams.sample_dir)


def save_images(images, size, image_path):
    if len(images.shape) != 2:
        images_inv = (images + 1.0) / 2.0
        scipy.misc.imsave(image_path, merge(images_inv, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def sample_z_val(hparams, rng=np.random):
    if hparams.z_dist == 'uniform':
        z_val = rng.uniform(-1, 1, size=(hparams.batch_size, hparams.z_dim))
    elif hparams.z_dist == 'gaussian':
        z_val = rng.randn(hparams.batch_size, hparams.z_dim)
    # elif sample_spec[0] == 'inspect_z_idx':
    #     z_idx = sample_spec[1]
    #     z_val = np.zeros([hparams.batch_size, hparams.z_dim])
    #     z_val[:, z_idx] = np.arange(0, 1, 1./hparams.batch_size)
    else:
        raise NotImplementedError
    return z_val


def sample_y_val(hparams, sample_type, num_samples):
    if sample_type == 'fixed':
        reps = int(np.ceil(1.0 * num_samples / hparams.y_dim))
        y_idx = np.tile(np.arange(hparams.y_dim), reps)
        y_idx = y_idx[:num_samples]
        y_val = np.zeros((num_samples, hparams.y_dim))
        y_val[np.arange(num_samples), y_idx] = 1
    elif sample_type == 'random':
        y_idx = np.random.choice(hparams.y_dim, num_samples)
        y_val = np.zeros((num_samples, hparams.y_dim))
        y_val[np.arange(num_samples), y_idx] = 1
    else:
        raise NotImplementedError
    return y_val


def get_phs_cond(hparams):
    z_ph = tf.placeholder(tf.float32, [None, hparams.z_dim], name='z_ph')
    x_ph = tf.placeholder(tf.float32, [hparams.batch_size] + hparams.image_dims, name='x_ph')
    y_ph = tf.placeholder(tf.float32, [hparams.batch_size, hparams.y_dim], name='y_ph')
    return z_ph, x_ph, y_ph


def get_phs_uncond(hparams):
    z_ph = tf.placeholder(tf.float32, [None, hparams.z_dim], name='z_ph')
    x_ph = tf.placeholder(tf.float32, [hparams.batch_size] + hparams.image_dims, name='x_ph')
    return z_ph, x_ph


def get_train_ops(hparams, d_loss, g_loss):

    d_vars = basic_utils.get_trainable_vars('discrim')
    g_vars = basic_utils.get_trainable_vars('gen')

    print 'Trainable generator vars:'
    for var in g_vars:
        print var.op.name
    print ''

    print 'Trainable discriminator vars:'
    for var in d_vars:
        print var.op.name
    print ''

    iter_ph = tf.placeholder(tf.int32, shape=None)

    if hparams.lr_decay == 'linear':
        lr_decay = tf.maximum(0.0, 1.0 - (tf.cast(iter_ph, tf.float32) / hparams.linear_decay_max_iter))
    elif hparams.lr_decay == 'false':
        lr_decay = 1.0
    else:
        raise NotImplementedError

    d_opt = basic_utils.get_optimizer(hparams, hparams.d_lr * lr_decay)
    g_opt = basic_utils.get_optimizer(hparams, hparams.g_lr * lr_decay)
    d_update_op = d_opt.minimize(d_loss, var_list=d_vars)
    g_update_op = g_opt.minimize(g_loss, var_list=g_vars)

    # assert hparams.d_lr == hparams.g_lr
    # opt = basic_utils.get_optimizer(hparams, hparams.d_lr * lr_decay)
    # d_update_op = opt.minimize(d_loss, var_list=d_vars)
    # g_update_op = opt.minimize(g_loss, var_list=g_vars)

    return d_update_op, g_update_op, iter_ph


def sample_cond(hparams, y_vals, mdevice, phs, theta_ph, real_vals, x_sample, x_lossy, sess, num_samples, rng=np.random):

    z_ph, x_ph, y_ph = phs[0], phs[1], phs[2]
    x_sample_vals = []
    x_lossy_vals = []
    num_batches = int(np.ceil(1.0 * num_samples / hparams.batch_size))
    for i in range(num_batches):
        y_val = np.zeros((hparams.batch_size, hparams.y_dim))
        start = i*hparams.batch_size
        stop = min((i+1)*hparams.batch_size, len(y_vals))
        y_val[:stop-start, :] = y_vals[start:stop, :]
        z_val = sample_z_val(hparams, rng)
        feed_dict = {z_ph: z_val, x_ph: real_vals[0], y_ph: y_val}
        feed_dict[theta_ph] = mdevice.sample_theta(hparams)
        feed_dict.pop(None, None)
        x_sample_val = sess.run(x_sample, feed_dict=feed_dict)
        x_lossy_val = sess.run(x_lossy, feed_dict=feed_dict)
        x_sample_vals.append(x_sample_val[:stop-start])
        x_lossy_vals.append(x_lossy_val[:stop-start])

    x_sample_val = np.concatenate(x_sample_vals, axis=0)
    x_lossy_val = np.concatenate(x_lossy_vals, axis=0)

    return x_sample_val, x_lossy_val


def save_samples(hparams, phs, theta_ph, x_sample, x_lossy,
                 mdevice, sess,
                 epoch, batch_num,
                 real_vals, x_measured_val=None):
    if len(phs) == 3:
        y_vals = sample_y_val(hparams, 'fixed', hparams.sample_num)
        rng = np.random.RandomState(0)
        x_sample_val, x_lossy_val = sample_cond(hparams, y_vals, mdevice,
                                                phs, theta_ph, real_vals,
                                                x_sample, x_lossy,
                                                sess, hparams.sample_num, rng)
        size = int(np.sqrt(hparams.sample_num))
        save_images(x_sample_val, [size, size], './{}/x_sample_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
        if epoch == 0:
            save_images(x_lossy_val, [size, size], './{}/x_lossy_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
            if hparams.train_mode == 'unmeasure':
                save_images(x_measured_val, [8, 16], './{}/x_measured_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                x_measured_val_clipped = np.minimum(np.maximum(x_measured_val, hparams.x_min), hparams.x_max)
                save_images(x_measured_val_clipped, [8, 16], './{}/x_measured_clipped_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
    else:
        z_ph, x_ph = phs[0], phs[1]
        rng = np.random.RandomState(0)
        z_val = sample_z_val(hparams, rng)
        feed_dict = {z_ph: z_val, x_ph: real_vals[0]}
        feed_dict[theta_ph] = mdevice.sample_theta(hparams)
        feed_dict.pop(None, None)
        x_sample_val = sess.run(x_sample, feed_dict=feed_dict)
        x_lossy_val = sess.run(x_lossy, feed_dict=feed_dict)
        save_images(x_sample_val, [8, 8], './{}/x_sample_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
        if epoch == 0:
            save_images(x_lossy_val, [8, 8], './{}/x_lossy_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
            if hparams.train_mode == 'unmeasure':
                save_images(x_measured_val, [8, 8], './{}/x_measured_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))
                x_measured_val_clipped = np.minimum(np.maximum(x_measured_val, hparams.x_min), hparams.x_max)
                save_images(x_measured_val_clipped, [8, 8], './{}/x_measured_clipped_{:02d}_{:04d}.png'.format(hparams.sample_dir, epoch, batch_num))

    print 'Saved samples at epoch {}, batch_num {}'.format(epoch, batch_num)



def train(hparams, phs, d_update_op, g_update_op, d_loss, g_loss, x_sample, x_lossy, real_val_iterator,
          theta_ph, theta_gen_ph, mdevice, iter_ph):

    # z_ph, x_ph = phs[0], phs[1]

    # Get a new TF session
    sess = tf.Session()

    # Add placeholder and summary for inception score
    # if hparams.dataset == 'cifar10':
    #     inception_mean_ph = tf.placeholder(tf.float32, (), name='inception_mean_ph')
    #     inception_std_ph = tf.placeholder(tf.float32, (), name='inception_std_ph')
    #     inception_summaries = [
    #         tf.summary.scalar('inception_mean', inception_mean_ph),
    #         tf.summary.scalar('inception_std', inception_std_ph)
    #     ]
    #     inception_summary = tf.summary.merge(inception_summaries)

    # Summary writing setup
    scalar_summaries = []
    scalar_summaries.append(tf.summary.scalar('d_loss', d_loss))
    scalar_summaries.append(tf.summary.scalar('g_loss', g_loss))
    scalar_summary = tf.summary.merge(scalar_summaries)
    summary_writer = tf.summary.FileWriter(hparams.summary_dir)

    # Model checkpointing setup
    model_saver = tf.train.Saver(max_to_keep=hparams.max_checkpoints)

    # initialization
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Attempt to restore variables from checkpoint
    init_train_iter = basic_utils.try_restore(hparams, sess, model_saver)

    # Define some constants
    num_batches = hparams.train_size // hparams.batch_size

    # Training loop
    for train_iter in range(init_train_iter+1, hparams.max_train_iter):

        # Save checkpoint
        if train_iter % 500 == 0:
            save_path = os.path.join(hparams.ckpt_dir, 'snapshot')
            model_saver.save(sess, save_path, global_step=train_iter)
            print 'Saved model at iteration {}'.format(train_iter)

        # Get random samples for the data, theta, and z
        real_vals = real_val_iterator.next(hparams)
        theta_val = mdevice.sample_theta(hparams)
        theta_gen_val = mdevice.sample_theta(hparams)
        z_val = sample_z_val(hparams)

        x_measured_val = None
        if hparams.train_mode == 'unmeasure':
            x_real_val = real_vals[0]
            x_measured_val = mdevice.measure_np(hparams, x_real_val, theta_val)  # measure real
            x_lossy_val = mdevice.unmeasure_np(hparams, x_measured_val, theta_val)  # unmeasure real
            real_vals[0] = x_lossy_val

        vals = [z_val] + real_vals
        feed_dict = {ph: val for (ph, val) in zip(phs, vals)}
        feed_dict[theta_ph] = theta_val
        feed_dict[theta_gen_ph] = theta_gen_val
        feed_dict[iter_ph] = train_iter
        feed_dict.pop(None, None)

        for _ in range(hparams.d_iters):
            sess.run(d_update_op, feed_dict=feed_dict)
        for _ in range(hparams.g_iters):
            sess.run(g_update_op, feed_dict=feed_dict)

        # if (train_iter % 500 == 0) and (hparams.dataset == 'cifar10'):
        #     print 'Computing and saving inception score...',
        #     # Compute and save inception score
        #     y_vals = sample_y_val(hparams, 'fixed', hparams.inception_num_samples)
        #     x_sample_val, _ = sample_cond(hparams, y_vals, mdevice,
        #                                   phs, theta_ph, real_vals,
        #                                   x_sample, x_lossy,
        #                                   sess, hparams.inception_num_samples)
        #     x_sample_val = ((x_sample_val+1.)*(255.99/2)).astype('int32')
        #     mean, std = inception_score.get_inception_score(list(x_sample_val))
        #     inception_summary_val = sess.run(
        #         inception_summary,
        #         feed_dict={
        #             inception_mean_ph: mean,
        #             inception_std_ph: std,
        #         }
        #     )
        #     summary_writer.add_summary(inception_summary_val, train_iter)
        #     print 'Done. iter = {}, mean = {}, std = {}'.format(train_iter, mean, std)

        # Logging
        epoch = train_iter // num_batches
        scalar_summary_str = sess.run(scalar_summary, feed_dict=feed_dict)
        batch_num = train_iter % num_batches
        print 'Epoch: [{}] [{}/{}], [{}/{}]'.format(
            epoch, batch_num, num_batches,
            train_iter, hparams.max_train_iter)
        summary_writer.add_summary(scalar_summary_str, train_iter)

        # Save samples
        if batch_num % 100 == 1:
            save_samples(hparams, phs, theta_ph, x_sample, x_lossy,
                         mdevice, sess,
                         epoch, batch_num,
                         real_vals, x_measured_val)

    # Save final checkpoint
    save_path = os.path.join(hparams.ckpt_dir, 'snapshot')
    model_saver.save(sess, save_path, global_step=hparams.max_train_iter-1)
    print 'Saved model at iteration {}'.format(hparams.max_train_iter-1)

    # Save final samples
    train_iter = hparams.max_train_iter - 1
    epoch = train_iter // num_batches
    batch_num = train_iter % num_batches

    real_vals = real_val_iterator.next(hparams)
    theta_val = mdevice.sample_theta(hparams)
    x_measured_val = None
    if hparams.train_mode == 'unmeasure':
        x_real_val = real_vals[0]
        x_measured_val = mdevice.measure_np(hparams, x_real_val, theta_val)  # measure real
        x_lossy_val = mdevice.unmeasure_np(hparams, x_measured_val, theta_val)  # unmeasure real
        real_vals[0] = x_lossy_val

    save_samples(hparams, phs, theta_ph, x_sample, x_lossy,
                 mdevice, sess,
                 epoch, batch_num,
                 real_vals, x_measured_val)

    return sess


############## Some old functions ##############

# def visualize_cond(hparams, sess, z_ph, y_ph, x_sample):
#     image_frame_dim = int(math.ceil(hparams.batch_size**.5))
#     for z_idx in range(hparams.z_dim):
#         y_val = sample_y_val(hparams, 'random')
#         z_val = sample_z_val(hparams, ['inspect_z_idx', z_idx])
#         feed_dict = {y_ph: y_val,
#                      z_ph: z_val}
#         x_sample_val = sess.run(x_sample, feed_dict=feed_dict)
#         save_path = os.path.join(hparams.sample_dir, 'test_arange_{0}.png'.format(z_idx))
#         save_images(x_sample_val, [image_frame_dim, image_frame_dim], save_path)

# def visualize_uncond(hparams, sess, z_ph, x_sample):
#     image_frame_dim = int(math.ceil(hparams.batch_size**.5))
#     for z_idx in range(hparams.z_dim):
#         z_val = sample_z_val(hparams, ['inspect_z_idx', z_idx])
#         feed_dict = {z_ph: z_val}
#         x_sample_val = sess.run(x_sample, feed_dict=feed_dict)
#         save_path = os.path.join(hparams.sample_dir, 'test_arange_{0}.png'.format(z_idx))
#         save_images(x_sample_val, [image_frame_dim, image_frame_dim], save_path)

# def imsave(images, size, path):
#     return scipy.misc.imsave(path, merge(images, size))

# def inverse_transform(images):
#     return (images+1.)/2.

