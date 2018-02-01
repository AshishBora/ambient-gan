import tensorflow as tf
import model_def


class InferenceNetwork(object):
    def __init__(self):
        self.sess = tf.Session()
        self.x_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x_ph')
        self.logits, self.y_hat, self.keep_prob = model_def.infer(self.x_ph, 'inf')
        restore_vars = model_def.inf_restore_vars()
        restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
        self.default_feed_dict = {self.keep_prob : 1.0}
        inf_restorer = tf.train.Saver(var_list=restore_dict)
        inf_restore_path = tf.train.latest_checkpoint('./src/mnist/inf/ckpt/bs64_lr0.0001/')
        inf_restorer.restore(self.sess, inf_restore_path)

    def get_y_hat_val(self, x_sample_val):
        self.default_feed_dict[self.x_ph] = x_sample_val
        y_hat_val = self.sess.run(self.y_hat, feed_dict=self.default_feed_dict)
        return y_hat_val
