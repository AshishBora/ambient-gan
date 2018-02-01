----
model_type          dcgan
measurement_type    blur_addnoise
blur_radius         1.0
blur_filter_size    5
additive_noise_std  0.0 0.1 0.2
train_mode          ambient

----
model_type          dcgan
measurement_type    blur_addnoise
blur_radius         1.0
blur_filter_size    5
additive_noise_std  0.2
train_mode          unmeasure
unmeasure_type      wiener
