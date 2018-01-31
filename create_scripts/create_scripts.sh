mkdir scripts

# mnist dcgan 1d
python ./create_scripts/create_scripts.py \
    --base-script-path ./create_scripts/mnist/base_script.sh \
    --scripts-base-dir ./scripts/ \
    --grid-path ./create_scripts/mnist/grid_dcgan_1d.sh

# mnist wgangp drop independent
python ./create_scripts/create_scripts.py \
    --base-script-path ./create_scripts/mnist/base_script.sh \
    --scripts-base-dir ./scripts/ \
    --grid-path ./create_scripts/mnist/grid_wgangp_drop_independent.sh

# mnist wgangp blur + noise
python ./create_scripts/create_scripts.py \
    --base-script-path ./create_scripts/mnist/base_script.sh \
    --scripts-base-dir ./scripts/ \
    --grid-path ./create_scripts/mnist/grid_wgangp_blur_addnoise.sh

# mnist wgangp patches
python ./create_scripts/create_scripts.py \
    --base-script-path ./create_scripts/mnist/base_script.sh \
    --scripts-base-dir ./scripts/ \
    --grid-path ./create_scripts/mnist/grid_wgangp_patches.sh

# # celebA dcgan
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/celebA/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/celebA/grid_dcgan.sh

# Make sure everything in scripts is executable
chmod +x ./scripts/*.sh
