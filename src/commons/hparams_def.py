import tensorflow as tf

def get_hparams(args=None):

    # Create a HParams object specifying the names and values of the model hyperparameters:

    hparams = tf.contrib.training.HParams(

        # task
        dataset='mnist',
        measurement_type='drop_independent',
        drop_prob=0.0,  # drop probability
        patch_size=10,  # size of patch to drop
        blur_radius=1.0,  # Radius for gaussian blurring
        blur_filter_size=1,  # Size of the blurring filter
        additive_noise_std=0.0,  # std deviation of noise to add
        num_angles=1,  # Number of rotate + project measurements

        # mode
        train_mode='ambient',  # ambient, unmeasure, or baseline
        unmeasure_type='medfilt',

        # model
        model_class='unconditional',
        model_type='wgangp',
        z_dim=100,
        z_dist='uniform',
        gp_lambda=10.0,  # gradient penalty lambda
        d_ac_lambda=1.0,  # How to scale the critic's ACGAN loss relative to WGAN loss
        g_ac_lambda=0.1,  # How to scale generator's ACGAN loss relative to WGAN loss

        # optimization
        opt_type='adam',  # optimizer type
        batch_size=64,  # how many examples are processed together
        g_lr=0.0002,  # Learning rate for the generator
        d_lr=0.0002,  # Learning rate for the disciminator
        lr_decay='false',
        linear_decay_max_iter=100000,
        opt_param1=0.9,  # parameter1 to optimizer
        opt_param2=0.999,  # parameter2 to optimizer
        g_iters=1,  # number of generator updates per train_iter
        d_iters=1,  # number of discriminator updates per train_iter

        # monitoring, saving, running
        results_dir='./results/', # Where to store the results
        sample_num=64,  # how many samples are visualized
        max_checkpoints=1,  # maximum number of checkpoints to keep
        inception_num_samples=50000,  # maximum number of checkpoints to keep
        max_train_iter=25000,
    )

    # Override hyperparameters values by parsing the command line
    if args is not None:
        hparams.parse(args.hparams)

    return hparams
