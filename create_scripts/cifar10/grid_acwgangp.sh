----
measurement_type         drop_independent
drop_prob                0.0
train_mode               ambient
max_train_iter           100000

----
measurement_type         drop_independent
drop_prob                0.1 0.5 0.8 0.9 0.95 0.99
train_mode               ambient baseline
max_train_iter           100000

----
measurement_type         drop_independent
drop_prob                0.1 0.5 0.8 0.9 0.95 0.99
train_mode               unmeasure
unmeasure_type           blur
max_train_iter           100000
