import tensorflow as tf

def get_hparams(args=None):

    # Create a HParams object specifying the names and values of the model hyperparameters:

    hparams = tf.contrib.training.HParams(

        # task
        dataset='cifar10',
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

        # optimization
        batch_size=32,  # how many examples are processed together per GPU

        # monitoring, saving, running
        results_dir='./results/', # Where to store the results
        max_checkpoints=1,  # maximum number of checkpoints to keep
        max_train_iter=50000,
    )

    # Override hyperparameters values by parsing the command line
    if args is not None:
        hparams.parse(args.hparams)

    return hparams
