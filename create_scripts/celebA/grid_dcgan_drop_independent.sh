----
model_type          dcgan
measurement_type    drop_independent
drop_prob           0.5 0.8 0.9 0.95 0.98
train_mode          ambient

----
model_type          dcgan
measurement_type    drop_independent
drop_prob           0.95
train_mode          unmeasure
unmeasure_type      blur
