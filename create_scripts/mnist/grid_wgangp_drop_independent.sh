----
model_type          wgangp
measurement_type    drop_independent
drop_prob           0.0 0.1 0.5 0.8 0.9 0.95 0.99
train_mode          ambient baseline

----
model_type          wgangp
measurement_type    drop_independent
drop_prob           0.0 0.1 0.5 0.8 0.9 0.95 0.99
train_mode          unmeasure
unmeasure_type      blur

----
model_type          wgangp
measurement_type    drop_independent
drop_prob           0.1 0.5 0.8 0.9 0.95 0.99
train_mode          unmeasure
unmeasure_type      inpaint-tv
max_train_iter      1000

# Run p = 0 for more iterations
----
model_type          wgangp
measurement_type    drop_independent
drop_prob           0.0
train_mode          unmeasure
unmeasure_type      inpaint-tv
