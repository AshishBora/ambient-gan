HPARAMS="\
measurement_type=drop_independent,\
drop_prob=0.0,\
patch_size=10,\
blur_radius=1.0,\
blur_filter_size=1,\
additive_noise_std=0.0,\
num_angles=1,\
\
train_mode=ambient,\
unmeasure_type=None,\
\
results_dir=./results/,\
max_train_iter=50000,\
"

python src/cifar/gan_cifar_resnet.py \
    --hparams $HPARAMS
