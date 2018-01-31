# pylint: disable = C0103, C0111, C0301, R0913, R0903

from tensorflow.examples.tutorials.mnist import input_data


class RealValIterator(object):

    def __init__(self):
        self.data = input_data.read_data_sets('./data/mnist', one_hot=True)

    def next(self, hparams):
        x_real, y_real = self.data.train.next_batch(hparams.batch_size)
        x_real = x_real.reshape(hparams.batch_size, 28, 28, 1)
        if hparams.model_class == 'conditional':
            return [x_real, y_real]
        elif hparams.model_class == 'unconditional':
            return [x_real]
        else:
            raise NotImplementedError
