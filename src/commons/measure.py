# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

"""Implementations of measurement and unmeasurement"""

from __future__ import division
import copy
import tensorflow as tf
import numpy as np
from scipy import signal
import cv2

import measure_utils


def get_mdevice(hparams):
    if hparams.measurement_type == 'drop_independent':
        mdevice = DropIndependent(hparams)
    elif hparams.measurement_type == 'drop_row':
        mdevice = DropRow(hparams)
    elif hparams.measurement_type == 'drop_col':
        mdevice = DropCol(hparams)
    elif hparams.measurement_type == 'drop_rowcol':
        mdevice = DropRowCol(hparams)
    elif hparams.measurement_type == 'drop_patch':
        mdevice = DropPatch(hparams)
    elif hparams.measurement_type == 'keep_patch':
        mdevice = KeepPatch(hparams)
    elif hparams.measurement_type == 'extract_patch':
        mdevice = ExtractPatch(hparams)
    elif hparams.measurement_type == 'blur_addnoise':
        mdevice = BlurAddNoise(hparams)
    elif hparams.measurement_type == 'pad_rotate_project':
        mdevice = PadRotateProject(hparams)
    elif hparams.measurement_type == 'pad_rotate_project_with_theta':
        mdevice = PadRotateProjectWithTheta(hparams)
    else:
        raise NotImplementedError
    return mdevice


class MeasurementDevice(object):
    """Base class for measurement devices"""

    def __init__(self, hparams):
        # self.measurement_type = hparams.measurement_type
        self.batch_dims = [hparams.batch_size] + hparams.image_dims
        self.output_type = None  # indicate whether image or vector

    def get_theta_ph(self, hparams):
        """Abstract Method"""
        # Should return theta_ph
        raise NotImplementedError

    def sample_theta(self, hparams):
        """Abstract Method"""
        # Should return theta_val
        raise NotImplementedError

    def measure(self, hparams, x, theta_ph):
        """Abstract Method"""
        # Tensorflow implementation of measurement. Must be differentiable wrt x.
        # Should return x_measured
        raise NotImplementedError

    def measure_np(self, hparams, x_val, theta_val):
        # Calling tf.Seesion() every time is quite slow
        # x_measured = self.measure(hparams, x_val, theta_val)
        # with tf.Session() as sess:
        #     x_measured_val = sess.run(x_measured)
        # return x_measured_val
        raise NotImplementedError

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        """Abstract Method"""
        # Should return x_hat
        raise NotImplementedError


class DropDevice(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def get_theta_ph(self, hparams):
        theta_ph = tf.placeholder(tf.float32, shape=self.batch_dims, name='theta_ph')
        return theta_ph

    def sample_theta(self, hparams):
        """Abstract Method"""
        # Should return theta_val
        raise NotImplementedError

    def measure(self, hparams, x, theta_ph):
        x_measured = tf.multiply(theta_ph, x, name='x_measured')
        return x_measured

    def measure_np(self, hparams, x_val, theta_val):
        x_measured_val = theta_val * x_val
        return x_measured_val

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        if hparams.unmeasure_type == 'medfilt':
            unmeasure_func = lambda image, mask: signal.medfilt(image)
        elif hparams.unmeasure_type == 'inpaint-telea':
            inpaint_type = cv2.INPAINT_TELEA
            unmeasure_func = measure_utils.get_inpaint_func_opencv(hparams, inpaint_type)
        elif hparams.unmeasure_type == 'inpaint-ns':
            inpaint_type = cv2.INPAINT_NS
            unmeasure_func = measure_utils.get_inpaint_func_opencv(hparams, inpaint_type)
        elif hparams.unmeasure_type == 'inpaint-tv':
            unmeasure_func = measure_utils.get_inpaint_func_tv()
        elif hparams.unmeasure_type == 'blur':
            unmeasure_func = measure_utils.get_blur_func()
        else:
            raise NotImplementedError

        x_unmeasured_val = np.zeros_like(x_measured_val)
        for i in range(x_measured_val.shape[0]):
            x_unmeasured_val[i] = unmeasure_func(x_measured_val[i], theta_val[i])

        return x_unmeasured_val


class DropMaskType1(DropDevice):

    def get_noise_shape(self):
        """Abstract Method"""
        # Should return noise_shape
        raise NotImplementedError

    def sample_theta(self, hparams):
        noise_shape = self.get_noise_shape()
        mask = np.random.uniform(size=noise_shape)
        p = hparams.drop_prob
        mask = np.float32(mask >= p) / (1 - p)
        theta_val = np.ones(shape=self.batch_dims)
        theta_val = theta_val * mask
        return theta_val


class DropIndependent(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        return noise_shape


class DropRow(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        noise_shape[2] = 1
        return noise_shape


class DropCol(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        noise_shape[1] = 1
        return noise_shape


class DropRowCol(DropDevice):

    def sample_theta(self, hparams):
        drop_row = DropRow(hparams)
        mask1 = drop_row.sample_theta(hparams)
        drop_col = DropCol(hparams)
        mask2 = drop_col.sample_theta(hparams)
        theta_val = mask1 * mask2
        return theta_val


class DropMaskType2(DropDevice):

    def sample_theta(self, hparams):
        raise NotImplementedError

    def patch_mask(self, hparams):
        k = hparams.patch_size
        h, w = hparams.image_dims[0:2]
        patch_mask = np.ones(self.batch_dims)
        for i in range(hparams.batch_size):
            x, y = np.random.choice(h-k), np.random.choice(w-k)
            patch_mask[i, x:x+k, y:y+k, :] = 0
        return patch_mask


class DropPatch(DropMaskType2):

    def sample_theta(self, hparams):
        return self.patch_mask(hparams)


class KeepPatch(DropMaskType2):

    def sample_theta(self, hparams):
        return 1 - self.patch_mask(hparams)


class ExtractPatch(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def get_theta_ph(self, hparams):
        theta_ph = tf.placeholder(tf.int32, shape=(hparams.batch_size, 2), name='theta_ph')
        return theta_ph

    def sample_theta(self, hparams):
        k = hparams.patch_size
        h, w = hparams.image_dims[0:2]
        theta_val = np.zeros([hparams.batch_size, 2])
        for i in range(hparams.batch_size):
            x, y = np.random.choice(h-k), np.random.choice(w-k)
            theta_val[i, :] = [x, y]
        return theta_val

    def measure(self, hparams, x, theta_ph):
        k = hparams.patch_size
        patch_list = []
        for t in range(hparams.batch_size):
            i, j = theta_ph[t, 0], theta_ph[t, 1]
            patch = x[t, i:i+k, j:j+k, :]
            patch = tf.reshape(patch, [1, k, k, hparams.image_dims[-1]])
            patch_list.append(patch)
        patches = tf.concat(patch_list, axis=0)
        # TODO(abora): Remove padding by using a custom discriminator
        paddings = measure_utils.get_padding_ep(hparams)
        x_measured = tf.pad(patches, paddings, "CONSTANT", name='x_measured')
        return x_measured

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        # How to implement this?
        raise NotImplementedError


class BlurAddNoise(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def get_theta_ph(self, hparams):
        theta_ph = tf.placeholder(tf.float32, shape=self.batch_dims, name='theta_ph')
        return theta_ph

    def sample_theta(self, hparams):
        theta_val = hparams.additive_noise_std * np.random.randn(*(self.batch_dims))
        return theta_val

    def measure(self, hparams, x, theta_ph):
        x_blurred = measure_utils.blur(hparams, x)
        x_measured = tf.add(x_blurred, theta_ph, name='x_measured')
        return x_measured

    def measure_np(self, hparams, x_val, theta_val):
        x_blurred = measure_utils.blur_np(hparams, x_val)
        x_measured = x_blurred + theta_val
        return x_measured

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        if hparams.unmeasure_type == 'wiener':
            x_unmeasured_val = measure_utils.wiener_deconv(hparams, x_measured_val)
        else:
            raise NotImplementedError
        return x_unmeasured_val


class PadRotateProjectDevice(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'vector'

    def get_theta_ph(self, hparams):
        theta_ph = tf.placeholder(tf.float32, shape=[hparams.batch_size, hparams.num_angles], name='theta_ph')
        return theta_ph

    def sample_theta(self, hparams):
        theta_val = (2*np.pi)*np.random.random((hparams.batch_size, hparams.num_angles)) - np.pi
        return theta_val

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        raise NotImplementedError


class PadRotateProject(PadRotateProjectDevice):

    def measure(self, hparams, x, theta_ph):
        x_padded = measure_utils.pad(hparams, x)
        x_measured_list = []
        for i in range(hparams.num_angles):
            angles = theta_ph[:, i]
            x_rotated = measure_utils.rotate(x_padded, angles)
            x_measured = measure_utils.project(hparams, x_rotated)
            x_measured_list.append(x_measured)
        x_measured = tf.concat(x_measured_list, axis=1, name='x_measured')
        return x_measured

    def measure_np(self, hparams, x_val, theta_val):
        raise NotImplementedError

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        raise NotImplementedError


class PadRotateProjectWithTheta(PadRotateProjectDevice):

    def measure(self, hparams, x, theta_ph):
        x_padded = measure_utils.pad(hparams, x)
        x_measured_list = []
        for i in range(hparams.num_angles):
            angles = theta_ph[:, i]
            x_rotated = measure_utils.rotate(x_padded, angles)
            x_projected = measure_utils.project(hparams, x_rotated)
            x_measured = measure_utils.concat(x_projected, angles)
            x_measured_list.append(x_measured)
        x_measured = tf.concat(x_measured_list, axis=1, name='x_measured')
        return x_measured

    def measure_np(self, hparams, x_val, theta_val):
        raise NotImplementedError

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        raise NotImplementedError
