# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

from __future__ import division

import numpy as np
from scipy import mgrid, ndimage
import tensorflow as tf
import cvxpy
import cv2

import im_rotate


def get_gaussian_filter(radius, size):
    x, y = mgrid[-(size-1)/2:size/2, -(size-1)/2:size/2]
    g = np.exp(-(x**2/float(2*radius**2) + y**2/float(2*radius**2)))
    g = g / g.sum()
    return g


def blur(hparams, x):
    size = hparams.blur_filter_size  # set size=1 for no blurring
    gaussian_filter = get_gaussian_filter(hparams.blur_radius, size)
    gaussian_filter = np.reshape(gaussian_filter, [size, size, 1, 1])
    x_blurred_list = []
    for i in range(hparams.image_dims[-1]):
        x_blurred = tf.nn.conv2d(x[:, :, :, i:i+1], gaussian_filter, strides=[1, 1, 1, 1], padding="SAME")
        x_blurred_list.append(x_blurred)
    x_blurred = tf.concat(x_blurred_list, axis=3)
    return x_blurred


def blur_np(hparams, x):
    size = hparams.blur_filter_size  # set size=1 for no blurring
    gaussian_filter = get_gaussian_filter(hparams.blur_radius, size)
    gaussian_filter = np.reshape(gaussian_filter, [1, size, size, 1])
    x_blurred = ndimage.filters.convolve(x, gaussian_filter, mode='constant')
    return x_blurred


def wiener_deconv(hparams, x):
    # https://gist.github.com/danstowell/f2d81a897df9e23cc1da

    noise_power = hparams.additive_noise_std**2
    nsr = noise_power / hparams.signal_power  # nsr = 1/snr

    size = hparams.image_dims[0]
    gaussian_filter = get_gaussian_filter(hparams.blur_radius, size)
    filter_fft = np.fft.fftn(np.fft.fftshift(gaussian_filter))
    filter_fft_conj = np.conj(filter_fft)
    den = filter_fft*filter_fft_conj + nsr + 1e-6

    x_deconved = np.zeros_like(x)
    for i in range(x.shape[0]):
        for c in range(x.shape[-1]):
            x_fft = np.fft.fftn(x[i, :, :, c])
            x_deconved_fft = x_fft * filter_fft_conj / den
            x_deconved[i, :, :, c] = np.real(np.fft.ifftn(x_deconved_fft))

    x_deconved = np.minimum(np.maximum(x_deconved, hparams.x_min), hparams.x_max)

    return x_deconved


def get_inpaint_func_opencv(hparams, inpaint_type):
    x_min = hparams.x_min
    x_max = hparams.x_max
    def inpaint_func(image, mask):
        mask = np.prod(mask, axis=2, keepdims=True)
        unknown = (1-mask).astype(np.uint8)
        image = 255 * (image - x_min) / (x_max - x_min)
        image = image.astype(np.uint8)
        inpainted = cv2.inpaint(image, unknown, 3, inpaint_type)
        inpainted = inpainted.astype(np.float32)
        inpainted = inpainted / 255.0 * (x_max - x_min) + x_min
        inpainted = np.reshape(inpainted, image.shape)
        return inpainted
    return inpaint_func


def get_inpaint_func_tv():
    def inpaint_func(image, mask):
        """Total variation inpainting"""
        inpainted = np.zeros_like(image)
        for c in range(image.shape[2]):
            image_c = image[:, :, c]
            mask_c = mask[:, :, c]
            if np.min(mask_c) > 0:
                # if mask is all ones
                inpainted[:, :, c] = image_c
            else:
                h, w = image_c.shape
                inpainted_c_var = cvxpy.Variable(h, w)
                obj = cvxpy.Minimize(cvxpy.tv(inpainted_c_var))
                constraints = [cvxpy.mul_elemwise(mask_c, inpainted_c_var) == cvxpy.mul_elemwise(mask_c, image_c)]
                prob = cvxpy.Problem(obj, constraints)
                # prob.solve(solver=cvxpy.SCS, max_iters=100, eps=1e-2)  # scs solver
                prob.solve()  # default solver
                inpainted[:, :, c] = inpainted_c_var.value
        return inpainted
    return inpaint_func


def get_padding_ep(hparams):
    """Get padding for extract_patch measurements"""
    k = hparams.patch_size
    if hparams.dataset == 'mnist':
        size = 28
    elif hparams.dataset == 'celebA':
        size = 64
    else:
        raise NotImplementedError
    pad_size = (size - k) // 2
    paddings = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
    return paddings


def get_padding_prp(hparams):
    """Get padding for pad_rotate_project measurements"""
    if hparams.dataset == 'mnist':
        paddings = [[0, 0], [6, 6], [6, 6], [0, 0]]
    elif hparams.dataset == 'celebA':
        paddings = [[0, 0], [14, 14], [14, 14], [0, 0]]
    else:
        raise NotImplementedError
    return paddings


def pad(hparams, inputs):
    paddings = get_padding_prp(hparams)
    outputs = tf.pad(inputs, paddings, "CONSTANT")
    return outputs


def rotate(inputs, angles):
    outputs = im_rotate.tf_image_rotate(inputs, angles)
    outputs = tf.reshape(outputs, inputs.get_shape())
    return outputs


def project(hparams, inputs):
    outputs = tf.reduce_sum(inputs, axis=2)
    outputs = tf.reshape(outputs, [hparams.batch_size, -1])
    return outputs


def concat(projected, angles):
    angles = tf.reshape(angles, [-1, 1])
    concatenated = tf.concat([projected, angles], 1)
    return concatenated
