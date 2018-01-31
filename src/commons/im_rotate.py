# pylint: disable = C0103, C0111, C0301, R0913, R0903

import tensorflow as tf
import numpy as np
# from tensorflow.python.framework import ops
from scipy.ndimage.interpolation import rotate


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def image_rotate(images, angles):
    rotated_images = np.zeros_like(images)
    for i in range(images.shape[0]):
        rotated_images[i] = rotate(images[i], angles[i]*180.0/np.pi, axes=(1, 0),
                                   reshape=False, order=0,
                                   mode='constant', cval=0.0, prefilter=False)
    return rotated_images


def image_rotate_grad(op, grad):
    images = op.inputs[0] # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )
    angles = op.inputs[1] # the second argument
    grad_reshaped = tf.reshape(grad, images.get_shape())
    return tf.contrib.image.rotate(grad_reshaped, -angles), None


def tf_image_rotate(images, angles, name=None):

    with tf.name_scope(name, "image_rotate", [images, angles]) as name:
        z = py_func(image_rotate,
                    [images, angles],
                    [tf.float32],
                    name=name,
                    grad=image_rotate_grad)  # <-- here's the call to the gradient

        return z[0]
