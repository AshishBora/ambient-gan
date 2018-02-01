# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

"""WGAN-GP ResNet for CIFAR-10"""

import cPickle as pickle

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!

# DATA_DIR = '/home/abora/datasets/cifar10/cifar-10-batches-py'
DATA_DIR = './data/cifar10/cifar-10-batches-py/'

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1, 2]:
    raise Exception('Only 1 or 2 GPUs supported!')

BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 1 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 100000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000 # How frequently to calculate Inception score

CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss


if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None


########## ambient stuff ##############

AMBIENT = True

def flat_to_NCHW(inputs, batch_size, local_lib=tf):
    outputs = local_lib.reshape(inputs, (batch_size, 3, 32, 32))
    return outputs


def NCHW_to_NHWC(inputs, local_lib=tf):
    outputs = local_lib.transpose(inputs, [0, 2, 3, 1])
    return outputs


def NHWC_to_NCHW(inputs, local_lib=tf):
    outputs = local_lib.transpose(inputs, [0, 3, 1, 2])
    return outputs


def NCHW_to_flat(inputs, batch_size, local_lib=tf):
    outputs = local_lib.reshape(inputs, (batch_size, 3072))
    return outputs


def amb_get_lossy(local_x, local_theta_ph):
    batch_size = BATCH_SIZE/len(DEVICES)
    local_x_nhwc = NCHW_to_NHWC(flat_to_NCHW(local_x, batch_size))
    local_x_lossy = amb_mdevice.measure(AMB_HPARAMS, local_x_nhwc, local_theta_ph)
    local_x_lossy = NCHW_to_flat(NHWC_to_NCHW(local_x_lossy), batch_size)
    return local_x_lossy


def amb_measure_unmeasure_np(local_x, local_theta_val):
    local_x_nhwc = NCHW_to_NHWC(flat_to_NCHW(local_x, BATCH_SIZE, local_lib=np), local_lib=np)
    local_x_lossy = amb_mdevice.measure_np(AMB_HPARAMS, local_x_nhwc, local_theta_val)
    local_x_lossy = amb_mdevice.unmeasure_np(AMB_HPARAMS, local_x_lossy, local_theta_val)
    local_x_lossy = NCHW_to_flat(NHWC_to_NCHW(local_x_lossy, local_lib=np), BATCH_SIZE, local_lib=np)
    return local_x_lossy


from argparse import ArgumentParser

import amb_utils
import amb_hparams_def
import amb_basic_utils
import amb_measure

def amb_setup():
    parser = ArgumentParser()
    parser.add_argument('--hparams', type=str, help='Comma separated list of "name=value" pairs.')
    args = parser.parse_args()
    hparams = amb_hparams_def.get_hparams(args)

    # Set up some stuff according to hparams
    amb_utils.setup_vals(hparams)
    amb_utils.setup_dirs(hparams)

    # print and save hparams.pkl
    amb_basic_utils.print_hparams(hparams)
    amb_basic_utils.save_hparams(hparams)

    return hparams

# Call our setup
AMB_HPARAMS = amb_setup()
amb_mdevice = amb_measure.get_mdevice(AMB_HPARAMS)
amb_theta1_phs = [amb_mdevice.get_theta_ph(AMB_HPARAMS, 'theta1_ph'+str(i)) for i in range(len(DEVICES))]  # real data, real labels
amb_theta2_phs = [amb_mdevice.get_theta_ph(AMB_HPARAMS, 'theta2_ph'+str(i)) for i in range(len(DEVICES))]  # real data, real labels
amb_theta3_phs = [amb_mdevice.get_theta_ph(AMB_HPARAMS, 'theta3_ph'+str(i)) for i in range(len(DEVICES))]  # real data, real labels

########################################



##### Original code starts

with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    fake_data_splits = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            x_gen = Generator(BATCH_SIZE/len(DEVICES), labels_splits[i])
            if AMBIENT:
                if AMB_HPARAMS.train_mode == 'ambient':
                    x_gen = amb_get_lossy(x_gen, amb_theta2_phs[i])
                elif AMB_HPARAMS.train_mode in ['baseline', 'unmeasure']:
                    pass
                else:
                    raise NotImplementedError
            fake_data_splits.append(x_gen)

    all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)
    if AMBIENT:
        if AMB_HPARAMS.train_mode in ['ambient', 'baseline']:
            all_real_data_splits = [
                amb_get_lossy(dsplit_i, amb_theta1_phs[i]) for (i, dsplit_i) in enumerate(all_real_data_splits)
            ]
        elif AMB_HPARAMS.train_mode == 'unmeasure':
            pass
        else:
            raise NotImplementedError

    DEVICES_B = DEVICES[:len(DEVICES)/2]
    DEVICES_A = DEVICES[len(DEVICES)/2:]

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    for i, device in enumerate(DEVICES_A):
        with tf.device(device):
            real_and_fake_data = tf.concat([
                all_real_data_splits[i],
                all_real_data_splits[len(DEVICES_A)+i],
                fake_data_splits[i],
                fake_data_splits[len(DEVICES_A)+i]
            ], axis=0)
            real_and_fake_labels = tf.concat([
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i],
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i]
            ], axis=0)
            disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)
            disc_real = disc_all[:BATCH_SIZE/len(DEVICES_A)]
            disc_fake = disc_all[BATCH_SIZE/len(DEVICES_A):]
            disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
            if CONDITIONAL and ACGAN:
                disc_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], labels=real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)])
                ))
                disc_acgan_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], dimension=1)),
                            real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)]
                        ),
                        tf.float32
                    )
                ))
                disc_acgan_fake_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[BATCH_SIZE/len(DEVICES_A):], dimension=1)),
                            real_and_fake_labels[BATCH_SIZE/len(DEVICES_A):]
                        ),
                        tf.float32
                    )
                ))


    for i, device in enumerate(DEVICES_B):
        with tf.device(device):
            real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
            fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
            labels = tf.concat([
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i],
            ], axis=0)
            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES_A),1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
            disc_costs.append(gradient_penalty)

    disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
    if CONDITIONAL and ACGAN:
        disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
        disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
        disc_cost = disc_wgan + (ACGAN_SCALE*disc_acgan)
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

    disc_params = lib.params_with_name('Discriminator.')

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.

    gen_costs = []
    gen_acgan_costs = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            n_samples = GEN_BS_MULTIPLE * BATCH_SIZE / len(DEVICES)
            fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
            if CONDITIONAL and ACGAN:
                x_gen = Generator(n_samples,fake_labels)
                if AMBIENT:
                    if AMB_HPARAMS.train_mode == 'ambient':
                        x_gen = amb_get_lossy(x_gen, amb_theta3_phs[i])
                    elif AMB_HPARAMS.train_mode in ['baseline', 'unmeasure']:
                        pass
                    else:
                        raise NotImplementedError
                disc_fake, disc_fake_acgan = Discriminator(x_gen, fake_labels)
                gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                ))
            # else:
            #     gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]))
    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
    if CONDITIONAL and ACGAN:
        gen_cost += (ACGAN_SCALE_G*(tf.add_n(gen_acgan_costs) / len(DEVICES)))


    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), AMB_HPARAMS.sample_dir +'x_sample_{}.png'.format(frame))

    # Function to save x_lossy
    def save_x_lossy():
        feed_dict = {
            all_real_data_int: _data,
        }
        for j in range(len(DEVICES)):
            feed_dict[amb_theta1_phs[j]] = amb_mdevice.sample_theta(AMB_HPARAMS)
        x_lossy_val = session.run(real_data, feed_dict=feed_dict)
        x_lossy_val = ((x_lossy_val+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(x_lossy_val.reshape((64, 3, 32, 32)), AMB_HPARAMS.sample_dir +'x_lossy.png')

    # Function for calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100 = Generator(100, fake_labels_100)
    def get_inception_score(n):
        all_samples = []
        for i in xrange(n/100):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))

    train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR)
    def inf_train_gen():
        while True:
            for images,_labels in train_gen():
                yield images,_labels


    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print "{} Params:".format(name)
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g == None:
                print "\t{} ({}) [no grad!]".format(v.name, shape_str)
            else:
                print "\t{} ({})".format(v.name, shape_str)
        print "Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        )

    session.run(tf.initialize_all_variables())

    # Model checkpointing setup
    model_saver = tf.train.Saver(max_to_keep=AMB_HPARAMS.max_checkpoints)

    # Attempt to restore variables from checkpoint
    init_train_iter = amb_basic_utils.try_restore(AMB_HPARAMS, session, model_saver)
    amb_inception_list_path = AMB_HPARAMS.metrics_dir + 'inception.pkl'
    if os.path.exists(amb_inception_list_path):
        with open(amb_inception_list_path, 'rb') as f:
            amb_inception_list = pickle.load(f)
    else:
        amb_inception_list = []

    gen = inf_train_gen()
    lib.plot._iter = [init_train_iter+1]

    for iteration in xrange(init_train_iter+1, AMB_HPARAMS.max_train_iter):

        start_time = time.time()

        # Save checkpoint
        if iteration % 500 == 0:
            save_path = os.path.join(AMB_HPARAMS.ckpt_dir, 'snapshot')
            model_saver.save(session, save_path, global_step=iteration)
            print 'Saved model at iteration {}'.format(iteration)

        if iteration > 0:
            feed_dict = {_iteration: iteration}
            for i in range(len(DEVICES)):
                feed_dict[amb_theta3_phs[i]] = amb_mdevice.sample_theta(AMB_HPARAMS)
            _ = session.run([gen_train_op], feed_dict=feed_dict)

        for i in xrange(N_CRITIC):
            _data, _labels = gen.next()

            if AMB_HPARAMS.train_mode == 'unmeasure':
                theta_val1 = amb_mdevice.sample_theta(AMB_HPARAMS)
                theta_val2 = amb_mdevice.sample_theta(AMB_HPARAMS)
                theta_val = np.concatenate([theta_val1, theta_val2], axis=0)
                _data = amb_measure_unmeasure_np(_data, theta_val)

            if CONDITIONAL and ACGAN:
                feed_dict = {
                    all_real_data_int: _data,
                    all_real_labels:_labels,
                    _iteration:iteration
                }
                for i in range(len(DEVICES)):
                    feed_dict[amb_theta1_phs[i]] = amb_mdevice.sample_theta(AMB_HPARAMS)
                    feed_dict[amb_theta2_phs[i]] = amb_mdevice.sample_theta(AMB_HPARAMS)
                _disc_cost, \
                _disc_wgan, _disc_acgan, \
                _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run(
                    [disc_cost, disc_wgan, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op],
                    feed_dict=feed_dict
                )
            else:
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

        lib.plot.plot('cost', _disc_cost)
        if CONDITIONAL and ACGAN:
            lib.plot.plot('wgan', _disc_wgan)
            lib.plot.plot('acgan', _disc_acgan)
            lib.plot.plot('acc_real', _disc_acgan_acc)
            lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
            inception_score = get_inception_score(50000)
            lib.plot.plot('inception_50k', inception_score[0])
            lib.plot.plot('inception_50k_std', inception_score[1])
            amb_inception_list.append((iteration, inception_score))

        if iteration == 0:
            save_x_lossy()

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 0:
            dev_disc_costs = []
            for images, _labels in dev_gen():
                feed_dict = {
                    all_real_data_int: images,
                    all_real_labels: _labels
                }
                for i in range(len(DEVICES)):
                    feed_dict[amb_theta1_phs[i]] = amb_mdevice.sample_theta(AMB_HPARAMS)
                    feed_dict[amb_theta2_phs[i]] = amb_mdevice.sample_theta(AMB_HPARAMS)
                _dev_disc_cost = session.run([disc_cost], feed_dict=feed_dict)
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush(AMB_HPARAMS.metrics_dir)
            with open(AMB_HPARAMS.metrics_dir + 'inception.pkl', 'wb') as f:
                pickle.dump(amb_inception_list, f)

        lib.plot.tick()

    # Save final checkpoint
    save_path = os.path.join(AMB_HPARAMS.ckpt_dir, 'snapshot')
    model_saver.save(session, save_path, global_step=AMB_HPARAMS.max_train_iter-1)
    print 'Saved model at iteration {}'.format(AMB_HPARAMS.max_train_iter-1)

    # Save final inception score
    inception_score = get_inception_score(50000)
    lib.plot.plot('inception_50k', inception_score[0])
    lib.plot.plot('inception_50k_std', inception_score[1])
    amb_inception_list.append((AMB_HPARAMS.max_train_iter-1, inception_score))
    lib.plot.flush(AMB_HPARAMS.metrics_dir)
    with open(AMB_HPARAMS.metrics_dir + 'inception.pkl', 'wb') as f:
        pickle.dump(amb_inception_list, f)
