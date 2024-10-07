# ImFCSNet PyTorch

This repository hosts the code used in the paper [“FCS videos: Fluorescence correlation spectroscopy in space and time”](https://www.sciencedirect.com/science/article/pii/S0304416524001594).

This repository also serves as a reproduction of the [original ImFCSNet code](https://github.com/ImagingFCS/ImFCS_FCSNet_ImFCSNet) that was released
alongside the paper ["Deep learning reduces data requirements and allows real-time measurements in imaging FCS."](https://www.cell.com/biophysj/fulltext/S0006-3495(23)04119-X).

This is not a fork, but rather a reimplementation in PyTorch to allow for
further experimentation in a different framework that might be more familiar for
deep learning practitioners.

# Setup

## Conda Environment Manager

To setup the Python environment, we recommend using Conda for environment
management.

If you do not have Conda installed, we recommend the lightweight Miniconda
installer: https://docs.anaconda.com/miniconda/.

```shell
# On Linux
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/$ Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm -rf ~/miniconda3/miniconda.sh
```

## Python Environment Creation

Then, you can create a Python environment to install the required libraries for
this codebase.

### Automatic creation

If you have conda installed, you can use the following commands to directly
install the environment. If you have a GPU with CUDA support, use
`environment.yml`.

If you do not have a GPU, you can use `environment-cpu.yml`. Note that training
is not supported in this configuration.

#### CUDA support (Training and Inference)

```shell
$ conda env create -f environment.yml
$ conda activate torch
```

#### CPU (Inference only)

```shell
$ conda env create -f environment-cpu.yml
$ conda activate torch
```

By running the `conda env create -f <file.yml>` command, you will create an
environment called `torch` with all the required dependencies. Remember to
activate the environment whenever you want to use this codebase.

If you use this method, you can skip the
[Installing Libraries](#installing-libraries) section and jump to
[Using the codabase](#using-the-codebase).

### Manual creation

```shell
$ conda create -n torch python=3.10
$ conda activate torch
```

By running `conda activate torch`, you are now within the `torch` environment.
Remember to run the command whenever you want to use this codebase, as the
installed libraries will be isolated to this install location.

## Installing Libraries

You can now install the required libraries for model training and inference.

1. Start by installing PyTorch as per the instructions on the PyTorch website
   (https://pytorch.org/get-started/locally/).

   ```shell
   $ # Your command may differ depending on your system.
   $ conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

2. Next, install `numba`, which is used to accelerate simulation code via the
   GPU.

   ```shell
   $ # Numba for simulations
   $ conda install numba::numba
   ```

3. Next, install the utility libraries that are used throughout the codebase.
   ```shell
   $ # Utility libraries
   $ # timm for image-focused utilities
   $ # yacs for reproducible config files.
   $ # matplotlib/seaborn for plotting purposes.
   $ # torcheval for convenient processing of training and validation metrics.
   $ # Tensorboard for experiment logging.
   $ pip install timm yacs matplotlib seaborn torcheval tensorboard torchinfo tifffile scipy
   ```

# Using the codebase

The codebase is primarily used for **model training** and **model inference**.

## Model Training

To train a model, use the `train.py` script, which handles the process of
building the model, starting the simulations, and training the model.

We adopt the YACS (https://github.com/rbgirshick/yacs) configuration format,
following libraries like Detectron2. This means your entire "experiment" is
described within a single YAML config file, which makes reproducibility easier.

To run model training, execute the following command while in the right
environment (run `conda activate torch` to ensure that all required libraries
are available).

```shell
$ python train.py --cfg path/to/config.yaml
```

We provide the config files used to train the models from the paper in
`imfcsnet-pytorch/configs`.

For example, to train the model used for evaluation of DOPC data, run the
following command.

```shell
$ python train.py --cfg configs/d_networks/model_dopc_2d.yml
```

Currently, you need to have an Nvidia GPU to train models, as the simulation
code is written for CUDA.

As the model trains, a "workdir" will be created, which holds all the artifacts
related to the training run. This includes:

- Logs of training loss
- Model checkpoints
- A dump of the YAML config file.

## Model Inference

Once a model is trained to completion, you can execute inference by running the
following command:

```shell
python inference.py \
    --cfg "configs/d_networks/model_dopc_2d.yml" \
    --ckpt "./workdir/model_dopc_2d/last.ckpt" \
    --files "path/to/tiffs/image_01.tif" \
    --output-folder "model_outputs"
```

This will use your trained model to produce a D or N map of the selected file,
which will be written to the `model_outputs` folder.

By default, inference is executed on the CPU, which can be slow. If you have an
Nvidia GPU, you can accelerate inference by adding in the `--device` argument.

```shell
python inference.py \
    --cfg "configs/d_networks/model_dopc_2d.yml" \
    --ckpt "./workdir/model_dopc_2d/last.ckpt" \
    --files "path/to/tiffs/image_01.tif" \
    --output-folder "model_outputs" \
    --device "cuda"
```

If you are evaluating on experimental data, it is very likely that
photobleaching is present. We also include a vectorized implementation of
Polynomial bleach correction, which can be activated by passing in the
`--bc-order` argument with your corresponding polynomial degree.

```shell
python inference.py \
    --cfg "configs/d_networks/model_dopc_2d.yml" \
    --ckpt "./workdir/model_dopc_2d/last.ckpt" \
    --files "path/to/tiffs/image_01.tif" \
    --output-folder "model_outputs" \
    --device "cuda" \
    --bc-order 4
```

We also provide some convenience functionality via globbing. If you have a
folder of TIFF files, you do not need to run the same command multiple times,
instead, use the glob syntax to run inference on all TIFF files by running:

```shell
python inference.py \
    --cfg "configs/d_networks/model_dopc_2d.yml" \
    --ckpt "./workdir/model_dopc_2d/last.ckpt" \
    --files "path/to/tiffs/*.tif" \
    --output-folder "model_outputs" \
    --device "cuda" \
    --bc-order 4
```

## Reproducing paper results

We include the model checkpoints used for the FCS videos in the paper. To use
them, simply run the inference command while pointing to the right checkpoint.

Note that you should also use the corresponding config file that matches the
name of the checkpoint. This avoids any odd mismatches that may happen.

```shell
python inference.py \
    --cfg "configs/d_networks/model_dopc_2d.yml" \
    --ckpt "./model_checkpoints/model_chok1_2d.ckpt" \
    --files "path/to/tiffs/*.tif" \
    --output-folder "model_outputs" \
    --device "cuda" \
    --bc-order 4
```

## Configuration

If you want to train your own models for your own measurements, you can create
your own custom config file.

Start by creating a copy of one of the model configurations (choose between a 2D
model or a 3D model).

Then, assuming you want to use the same training recipe that we used, all you
need to change is the settings under the `SIMULATION` key.

```yaml
SIMULATION:
  SIM_2D_PHYSICAL:
    # Modify the constants to match your acquisition setup.
    CONSTANTS:
      FRAME_TIME: 0.00206
      MAGNIFICATION: 100
      NA: 1.49
      PIXEL_SIZE: 24
      WAVELENGTH: 507
    # For the PSF, we use a range of 0.5
    # Set the min and max according to your calibrated value.
    VARIABLES:
      MAX:
        PSF_SIGMA_0: 0.96
      MIN:
        PSF_SIGMA_0: 1.06
```

You can also modify the rest of the experimental settings to further explore the
hyperparameter space.

The settings used in our provided configs are designed to match the original
Tensorflow implementation, while also simplifying the training process (i.e.
removing rounds).

To get a better understanding of all the different settings and what they do,
you can reference the documentation under
`imfcsnet-pytorch/imfcs_pytorch/config/defaults.py`, which explains all of the
different key/value pairs.

# References

Wohland, T., Sim, S. R., Demoustier, M., Pandey, S., Kulkarni, R., & Aik, D. (2024). FCS videos: Fluorescence correlation spectroscopy in space and time. Biochimica et Biophysica Acta (BBA)-General Subjects, 130716.

Tang, W. H., Sim, S. R., Aik, D. Y. K., Nelanuthala, A. V. S., Athilingam, T., Röllin, A., & Wohland, T. (2024). Deep learning reduces data requirements and allows real-time measurements in imaging FCS. Biophysical Journal, 123(6), 655-666.
