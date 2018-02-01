mkdir scripts


# -----------


# # mnist dcgan 1d
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/mnist/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/mnist/grid_dcgan_1d.sh

# # mnist wgangp drop independent
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/mnist/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/mnist/grid_wgangp_drop_independent.sh

# # mnist wgangp blur + noise
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/mnist/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/mnist/grid_wgangp_blur_addnoise.sh

# # mnist wgangp patches
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/mnist/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/mnist/grid_wgangp_patches.sh


# # -----------


# # celebA dcgan 1d
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/celebA/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/celebA/grid_dcgan_1d.sh

# # celebA dcgan drop_independent
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/celebA/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/celebA/grid_dcgan_drop_independent.sh

# # celebA dcgan blur + noise
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/celebA/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/celebA/grid_dcgan_blur_addnoise.sh

# # celebA dcgan patches
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/celebA/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/celebA/grid_dcgan_patches.sh


# -----------


# cifar10 acwgangp
python ./create_scripts/create_scripts.py \
    --base-script-path ./create_scripts/cifar10/base_script.sh \
    --scripts-base-dir ./scripts/ \
    --grid-path ./create_scripts/cifar10/grid_acwgangp.sh


# -----------


# Make sure everything in scripts is executable
chmod +x ./scripts/*.sh
