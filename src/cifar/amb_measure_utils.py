# # pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

from __future__ import division

import numpy as np
from scipy import mgrid, ndimage
import tensorflow as tf
# import cvxpy
# import cv2

# from commons import im_rotate


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

# def get_inpaint_func_opencv(inpaint_type):
#     def inpaint_func(image, mask):
#         unknown = (1-mask).astype(np.uint8)
#         image = image.astype(np.float32)
#         inpainted = cv2.inpaint(image, unknown, 3, inpaint_type)
#         inpainted = inpainted.astype(np.float32)
#         inpainted = np.reshape(inpainted, image.shape)
#         return inpainted
#     return inpaint_func


# def get_inpaint_func_tv():
#     def inpaint_func(image, mask):
#         """Total variation inpainting"""
#         assert image.shape[2] == 1
#         image = image[:, :, 0]
#         h, w = image.shape
#         inpainted_var = cvxpy.Variable(h, w)
#         obj = cvxpy.Minimize(cvxpy.tv(inpainted_var))
#         constraints = [cvxpy.mul_elemwise(mask, inpainted_var) == cvxpy.mul_elemwise(mask, image)]
#         prob = cvxpy.Problem(obj, constraints)
#         # Use SCS to solve the problem.
#         # prob.solve(solver=cvxpy.SCS, max_iters=100, eps=1e-2)
#         prob.solve()  # default solver
#         inpainted = inpainted_var.value
#         inpainted = np.expand_dims(inpainted, 2)
#         return inpainted
#     return inpaint_func


# def get_padding_ep(hparams):
#     """Get padding for extract_patch measurements"""
#     k = hparams.drop_patch_k
#     if hparams.dataset == 'mnist':
#         size = 28
#     elif hparams.dataset == 'celebA':
#         size = 64
#     else:
#         raise NotImplementedError
#     pad_size = (size - k) // 2
#     paddings = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]
#     return paddings


# def get_padding_prp(hparams):
#     """Get padding for pad_rotate_project measurements"""
#     if hparams.dataset == 'mnist':
#         paddings = [[0, 0], [6, 6], [6, 6], [0, 0]]
#     elif hparams.dataset == 'celebA':
#         paddings = [[0, 0], [14, 14], [14, 14], [0, 0]]
#     else:
#         raise NotImplementedError
#     return paddings


# def pad(hparams, inputs):
#     paddings = get_padding_prp(hparams)
#     outputs = tf.pad(inputs, paddings, "CONSTANT")
#     return outputs


# def rotate(inputs, angles):
#     outputs = im_rotate.tf_image_rotate(inputs, angles)
#     outputs = tf.reshape(outputs, inputs.get_shape())
#     return outputs


# def project(hparams, inputs):
#     outputs = tf.reduce_sum(inputs, axis=2)
#     outputs = tf.reshape(outputs, [hparams.batch_size, -1])
#     return outputs


# def concat(projected, angles):
#     angles = tf.reshape(angles, [-1, 1])
#     concatenated = tf.concat([projected, angles], 1)
#     return concatenated
