----
model_type          dcgan
measurement_type    drop_patch
patch_size          32
train_mode          ambient

----
model_type          dcgan
measurement_type    drop_patch
patch_size          32
train_mode          unmeasure
unmeasure_type      inpaint-ns


----
model_type          dcgan
measurement_type    keep_patch
patch_size          32
train_mode          ambient
