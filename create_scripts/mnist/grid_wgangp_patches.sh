# -- Drop Patch ----------------------------------
----
model_type          wgangp
measurement_type    drop_patch
patch_size          14
train_mode          ambient

----
model_type          wgangp
measurement_type    drop_patch
patch_size          14
train_mode          unmeasure
unmeasure_type      inpaint-ns


# -- Extract Patch ----------------------------------
----
model_type          wgangp
measurement_type    extract_patch
patch_size          14
train_mode          ambient


# -- Keep Patch -------------------------------------

----
model_type          wgangp
measurement_type    keep_patch
patch_size          14
train_mode          ambient
