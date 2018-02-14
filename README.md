# AmbientGAN: Generative models from lossy measurements

This repository provides code to reproduce results from the paper [AmbientGAN: Generative models from lossy measurements](https://openreview.net/forum?id=Hy7fDog0b).

The training setup is as in the following diagram:

<img src="https://github.com/AshishBora/ambient-gan/blob/master/setup.png" width="400">

Here are a few example results:

Measured | Baseline | AmbientGAN (ours)
---------------|-----------------|------------
<img src="https://github.com/AshishBora/ambient-gan/blob/master/images/celebA_drop_patch/measured.png" width="300"> | <img src="https://github.com/AshishBora/ambient-gan/blob/master/images/celebA_drop_patch/inpaint_ns.png" width="300"> | <img src="https://github.com/AshishBora/ambient-gan/blob/master/images/celebA_drop_patch/ambient.png" width="300">
<img src="https://github.com/AshishBora/ambient-gan/blob/master/images/celebA_blur_addnoise/measured.png" width="300"> | <img src="https://github.com/AshishBora/ambient-gan/blob/master/images/celebA_blur_addnoise/wiener_deconv.png" width="300"> | <img src="https://github.com/AshishBora/ambient-gan/blob/master/images/celebA_blur_addnoise/ambient.png" width="300">
<img src="https://github.com/AshishBora/ambient-gan/blob/master/images/cifar_drop_independent/measured.png" width="300"> | <img src="https://github.com/AshishBora/ambient-gan/blob/master/images/cifar_drop_independent/unmeasure_blur.png" width="300"> | <img src="https://github.com/AshishBora/ambient-gan/blob/master/images/cifar_drop_independent/ambient.png" width="300">

Few more samples from AmbientGAN models trained with 1-D projections:

Pad-Rotate-Project | Pad-Rotate-Project-theta
---------------|-----------------
<img src="https://github.com/AshishBora/ambient-gan/blob/master/images/mnist_pad_rotate_project/ambient.png" width="200"> | <img src="https://github.com/AshishBora/ambient-gan/blob/master/images/mnist_pad_rotate_project_with_theta/ambient.png" width="200">


The rest of the README describes how to reproduce the results.

Requirements
---
- Python 2.7
- Tensorflow >= 1.4.0
- matplotlib
- scipy
- numpy
- cvxpy
- scikit-learn
- tqdm
- opencv-python
- pandas

For `pip` installation, use ```$ pip install -r requirements.txt```


Get the data
---
- MNIST data is automatically downloaded
- Get the celebA dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and put the jpeg files in `./data/celebA/`
- Get the CIFAR-10 python data from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and put it in `./data/cifar10/cifar-10-batches-py/*`


Get inference models
---
We need inference models for computing the inception score. 

- For MNIST, you can train your own by
    ```
    cd ./src/mnist/inf
    python train.py
    ```

    [TODO]: Provide a pretrained model.
    <!-- Alternatively, get a pretrained inference model for MNIST from <> and put the checkpoint in `./src/mnist/inf/ckpt/*`
    -->

- Inception model for use with CIFAR-10 is automatically downloaded.


Create experiment scripts
---

Run `./create_scripts/create_scripts.sh`

This will create scripts for all the experiments in the paper.

[Optional] If you want to run only a subset of experiments you can define the grid in `./create_scripts/DATASET_NAME/grid_*.sh` or if you wish to tweak a lot of parameters, you can change `./create_scripts/DATASET_NAME/base_script.sh`. Then run `./create_scripts/create_scripts.sh` as above to create the corresponding scripts (remember to remove any previous files from `./scripts/`)


Run experiments
---

We provide scripts to train on multiple GPUs in parallel. For example, if you wish to use 4 GPUs, you can run:
`./run_scripts/run_sequentially_parallel.sh "0 1 2 3"`

This will start 4 GNU screens. Each program within the screen will attempt to acquire and run experiments from `./scripts/`, one at a time. Each experiment run will save samples, checkpoints, etc. to `./results/`.


See results as you train
---

### Samples

You can see samples for each experiment in `./results/samples/EXPT_DIR/`

`EXPT_DIR` is defined based on the hyperparameters of the experiment. See `./src/commons/dir_def.py` to see how this is done.

### Quantitative plots

Run
```
python src/aggregator_mnist.py
python src/aggregator_cifar.py
```

This will create pickle files in `./results/` with the relevant data in a Pandas dataframe.

Now use the ipython notebooks `./plotting_mnist.ipynb` and `./plotting_cifar.ipynb` to get the relevant plots. The generated plots are also saved to `./results/plots/` (make sure this directory exists)
