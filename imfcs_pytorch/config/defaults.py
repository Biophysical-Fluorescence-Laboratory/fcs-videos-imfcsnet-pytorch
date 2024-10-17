"""Definition of the defaults for the YACS configuration.

As per the documentation from https://github.com/rbgirshick/yacs: "This file is the one-stop reference point for all configurable options. It should be very well documented and provide sensible defaults for all options."

Defaults here are based on the simplified no-round implementation, using physical simulations as opposed to the legacy ported-over dimensionless simulations.

The default configuration should not be imported using `from imfcs_pytorch.config.defaults import _C`, but rather accessed from the `from imfcs_pytorch.config.initialization import get_default_cfg` function.

We also use the dictionary-structure of YACS to extract parameter indices as well. See the simulation code in `imfcsnet-pytorch/imfcs_pytorch/data/simulation/simulator` for implementation details. This utilizes Python dictionaries' property of being insert-ordered.
"""

from yacs.config import CfgNode as CN

_C = CN()

# Simulation settings
_C.SIMULATION = CN()
_C.SIMULATION.TYPE = "SIM_2D_PHYSICAL"  # Selects the simulation method to use.
_C.SIMULATION.PARAMETER_SAMPLING_STRATEGY = "minmax"

# Simulation settings that do not change throughout the simulation
# These values are also shared across 2D and 3D cases (and even the dimensionless simulations).
_C.SIMULATION.CONSTANTS = CN()
_C.SIMULATION.CONSTANTS.NUM_PIXELS = 3
_C.SIMULATION.CONSTANTS.STEPS_PER_FRAME = 10
_C.SIMULATION.CONSTANTS.MARGIN = 6.0
_C.SIMULATION.CONSTANTS.FRAMES = 2500

# Physical parameter simulations based on the original Java codebase.
# 2D Simulation Base - Based on the original Java codebase from the ImagingFCS Fiji plugin
_C.SIMULATION.SIM_2D_PHYSICAL = CN()
_C.SIMULATION.SIM_2D_PHYSICAL.DIMENSIONALITY = 4  # [x, y, bleach, in_dark_state]
# CONSTANTS - Simulation parameters that should not change over every sample.
_C.SIMULATION.SIM_2D_PHYSICAL.CONSTANTS = CN()
_C.SIMULATION.SIM_2D_PHYSICAL.CONSTANTS.FRAME_TIME = 0.00106
_C.SIMULATION.SIM_2D_PHYSICAL.CONSTANTS.MAGNIFICATION = 100
_C.SIMULATION.SIM_2D_PHYSICAL.CONSTANTS.NA = 1.45
_C.SIMULATION.SIM_2D_PHYSICAL.CONSTANTS.PIXEL_SIZE = 24
_C.SIMULATION.SIM_2D_PHYSICAL.CONSTANTS.WAVELENGTH = 583  # in nm
_C.SIMULATION.SIM_2D_PHYSICAL.CONSTANTS.DO_BLEACHING = 0  # 0 for False, 1 for True
_C.SIMULATION.SIM_2D_PHYSICAL.CONSTANTS.DO_BLINKING = 0  # 0 for False, 1 for True

# VARIABLES - Simulation parameters that are sampled from a uniform distribution. These can be used as targets for regression by assigning their names to TASK.REGRESSION.TARGETS
# The variables are sampled from the range [MIN.VARIABLE_NAME, MAX.VARIABLE_NAME] in a uniform (natural or log) distribution.
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES = CN()
# MIN - minimal values parameters are allowed to take, on natural scale
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN = CN()
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.CPS = 1000
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.PSF_SIGMA_0 = 0.75
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.NO_OF_PARTICLES = 24
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.TAU_BLEACH = 100000
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.BLEACH_RADIUS = 3.0
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.BLEACH_FRAME = 10000000
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.D1 = 0.02  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.D2 = 0.0  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.D3 = 0.0  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.F2 = 0.0
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.F3 = 0.0
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.TRIPLET_ON_RATE = 1.0
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MIN.TRIPLET_OFF_RATE = 0.0
# MAX - maximal values parameters are allowed to take, on natural scale
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX = CN()
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.CPS = 10000
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.PSF_SIGMA_0 = 0.85
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.NO_OF_PARTICLES = 486
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.TAU_BLEACH = 100000
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.BLEACH_RADIUS = 3.0
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.BLEACH_FRAME = 10000000
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.D1 = 50.0  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.D2 = 0.0  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.D3 = 0.0  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.F2 = 0.0
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.F3 = 0.0
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.TRIPLET_ON_RATE = 1.0
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.MAX.TRIPLET_OFF_RATE = 0.0
# TRANSFORM - Whether to sample from a natural (None) or log-uniform distribution
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM = CN()
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.CPS = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.PSF_SIGMA_0 = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.NO_OF_PARTICLES = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.TAU_BLEACH = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.BLEACH_RADIUS = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.BLEACH_FRAME = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.D1 = "log"  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.D2 = None  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.D3 = None  # in um2/s
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.F2 = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.F3 = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.TRIPLET_ON_RATE = None
_C.SIMULATION.SIM_2D_PHYSICAL.VARIABLES.TRANSFORM.TRIPLET_OFF_RATE = None

# 3D Simulation Base - Based on the original Java codebase from the ImagingFCS Fiji plugin
# 3D Simulation Base - Based on the original Java codebase from the ImagingFCS Fiji plugin
_C.SIMULATION.SIM_3D_PHYSICAL = CN()
_C.SIMULATION.SIM_3D_PHYSICAL.DIMENSIONALITY = 5  # [x, y, z, bleach, in_dark_state]
# CONSTANTS - Simulation parameters that should not change over every sample.
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS = CN()
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.FRAME_TIME = 0.00204
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.MAGNIFICATION = 60
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.NA = 1.0
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.PIXEL_SIZE = 24
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.WAVELENGTH = 515  # in nm
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.Z_DIM_EXT_FACTOR = 10.0  # defining the extension of the z-axis simulation area as a function of the lightsheet thickness.
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.REFRACTIVE_INDEX = (
    1.333  # Used to calculate the z-factor. 1.333 is the refractive index of water.
)
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.DO_BLEACHING = 0  # 0 for False, 1 for True
_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS.DO_BLINKING = 0  # 0 for False, 1 for True

# VARIABLES - Simulation parameters that are sampled from a uniform distribution. These can be used as targets for regression by assigning their names to TASK.REGRESSION.TARGETS
# The variables are sampled from the range [MIN.VARIABLE_NAME, MAX.VARIABLE_NAME] in a uniform (natural or log) distribution.
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES = CN()
# MIN - minimal values parameters are allowed to take, on natural scale
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN = CN()
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.CPS = 5000
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.PSF_SIGMA_0 = 1.05
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.PSF_SIGMA_Z = 2.2
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.NO_OF_PARTICLES = 121
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.TAU_BLEACH = 100000
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.BLEACH_RADIUS = 3.0
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.BLEACH_FRAME = 10000000
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.D1 = 0.02  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.D2 = 0.0  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.D3 = 0.0  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.F2 = 0.0
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.F3 = 0.0
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.TRIPLET_ON_RATE = 1.0
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN.TRIPLET_OFF_RATE = 0.0
# MAX - maximal values parameters are allowed to take, on natural scale
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX = CN()
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.CPS = 15000
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.PSF_SIGMA_0 = 1.15
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.PSF_SIGMA_Z = 2.2
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.NO_OF_PARTICLES = 608
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.TAU_BLEACH = 100000
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.BLEACH_RADIUS = 3.0
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.BLEACH_FRAME = 10000000
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.D1 = 50.0  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.D2 = 0.0  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.D3 = 0.0  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.F2 = 0.0
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.F3 = 0.0
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.TRIPLET_ON_RATE = 1.0
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MAX.TRIPLET_OFF_RATE = 0.0
# TRANSFORM - Whether to sample from a natural (None) or log-uniform distribution
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM = CN()
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.CPS = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.PSF_SIGMA_0 = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.PSF_SIGMA_Z = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.NO_OF_PARTICLES = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.TAU_BLEACH = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.BLEACH_RADIUS = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.BLEACH_FRAME = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.D1 = "log"  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.D2 = None  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.D3 = None  # in um2/s
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.F2 = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.F3 = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.TRIPLET_ON_RATE = None
_C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.TRANSFORM.TRIPLET_OFF_RATE = None

# 2D Simulation with modular margins and particle density
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY = CN()
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.DIMENSIONALITY = (
    4  # [x, y, bleach, in_dark_state]
)
# CONSTANTS - Simulation parameters that should not change over every sample.
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.CONSTANTS = CN()
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.CONSTANTS.FRAME_TIME = 0.00106
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.CONSTANTS.MAGNIFICATION = 100
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.CONSTANTS.NA = 1.45
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.CONSTANTS.PIXEL_SIZE = 24
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.CONSTANTS.WAVELENGTH = 583  # in nm
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.CONSTANTS.DO_BLEACHING = (
    0  # 0 for False, 1 for True
)
_C.SIMULATION.SIM_2D_PHYSICAL_PARTICLE_DENSITY.CONSTANTS.DO_BLINKING = (
    0  # 0 for False, 1 for True
)

# Some basic backend settings that control the behaviour of the simulation queue.
# These default values are arbitrary values, but they can be changed according to the capabilities of the training system.
_C.SIMULATION.BACKEND = CN()
# The maximum number of simulations to hold in the queue. Blocks if the queue is saturated.
# Can be set to 0 for an infinite-length unbounded queue, but this is strongly discouraged as memory use through RAM might explode.
_C.SIMULATION.BACKEND.MAX_QUEUE_SIZE = 12000
# the time to wait before retrying the addition or popping from the simulation queue.
_C.SIMULATION.BACKEND.QUEUE_RETRY_TIME = 3  # In seconds
# Number of simulations to produce on the GPU per thread count.
# 4096 was selected as 32 sub-batches of 128 (2500 x 3 x 3) image stacks.
# This can be scaled up/down depending on your GPU VRAM availability. 4096 is a rather conservative number that uses <1GB of VRAM, just remember that the model training also happens on the same GPU, so VRAM and GPU compute is shared.
_C.SIMULATION.BACKEND.SIM_COUNT = 4096


# General experiment settings.
_C.EXPERIMENT = CN()
_C.EXPERIMENT.WORKDIR = (
    "./workdir"  # A folder that saves the checkpoints, logs and a dumped config file.
)
_C.EXPERIMENT.DEVICE = "cuda:0"  # Device to use for training and simulations. Should be 'cuda:N', where N is the GPU index (which you can obtain using `nvidia-smi`)
# https://arxiv.org/abs/2109.08203 Torch.manual_seed(3407) is all you need
_C.EXPERIMENT.SEED = 3407  # The seed can be set to None to randomize the seed.
_C.EXPERIMENT.TASK = "REGRESSION"

# A collection of checkpointing-related functionality.
_C.EXPERIMENT.CHECKPOINTING = CN()
# Interval checkpointing can be used to save the progression of the model.
# However, saving too frequently will result in a lot of extraneous checkpoints which could take up a lot of disk space.
_C.EXPERIMENT.CHECKPOINTING.DO_INTERVAL_CHECKPOINTING = False
_C.EXPERIMENT.CHECKPOINTING.CHECKPOINTING_INTERVAL = 100000

# Task-specific settings.
# This is controlled by the task defined in _C.EXPERIMENT.TASK.
# Each EXPERIMENT.TASK should correspond to a key under _C.TASK.
# For example, _C.EXPERIMENT.TASK == "REGRESSION" will read values from _C.TASK.REGRESSION.
_C.TASK = CN()
_C.TASK.REGRESSION = CN()
# Since Regression is not arbitrarily limited to a single target, the targets and the corresponding transforms are defined as lists.
# Any value from `SIMULATION.SIM_3D_PHYSICAL.VARIABLES` can be a valid regression target.
# The index of the variable name in TARGETS will correspond to a transform in TARGET_TRANSFORM
# The code is designed such that any target transforms will be reversed during the inference process.
_C.TASK.REGRESSION.TARGETS = ["D1"]
_C.TASK.REGRESSION.TARGET_TRANSFORM = [
    "log"
]  # Apply None, 'log' or 'log10' transform to corresponding target.

# Model settings.
# These will generally be accessed in the model builder from `imfcsnet-pytorch/imfcs_pytorch/builders/model.py`
# New models can be added using similar semantics.
_C.MODEL = CN()
_C.MODEL.NAME = "imfcsnet"  # Model name identifier.
_C.MODEL.WEIGHTS = None  # Pretrained weights to use. Primarily used to do 'round'-based training as per the original codebase.

# ImFCSNet-specific model settings
# Defaults are based on the definition from the original paper (https://www.sciencedirect.com/science/article/pii/S000634952304119X)
_C.MODEL.IMFCSNET = CN()
# The number of filter channels to use across the whole model.
_C.MODEL.IMFCSNET.FILTER_CHANNELS = 45
# The kernel size to use for the initial spatial aggregation phase. The 2nd and 3rd dimensions (height/width) should be equivalent to _C.SIMULATION.CONSTANTS.NUM_PIXELS.
_C.MODEL.IMFCSNET.SPATIAL_AGG_BLOCK_KERNEL_SIZE = (
    200,
    3,
    3,
)
# The kernel size to use for the strided conv1d layer, performing large-scale temporal downsampling.
_C.MODEL.IMFCSNET.STRIDED_CONV_LAYER_KERNEL_SIZE = 100
# Stride of the strided conv1d layer.
_C.MODEL.IMFCSNET.STRIDED_CONV_LAYER_FILTER_STRIDE = 8
_C.MODEL.IMFCSNET.CONV1D_GROUP_STAGES = 2  # Number of stages to use for the conv1d group, performing temporal aggregation. Each stage forms a residual block.
# Number of conv1d/batchnorm blocks per stage.
_C.MODEL.IMFCSNET.CONV1D_GROUP_BLOCKS_PER_STAGE = 2
# Filter size for the conv1d group.
_C.MODEL.IMFCSNET.CONV1D_GROUP_FILTER_SIZE = 50
# Number of stages to use for the Dense Mixing group.
_C.MODEL.IMFCSNET.DENSE_MIXING_NUM_STAGES = 6
# Number of 1x1 Conv1d/batchnorm blocks to use per Dense Mixing stage.
_C.MODEL.IMFCSNET.DENSE_MIXING_BLOCKS_PER_STAGE = 2
# Whether to use the original paper's codebase's weight initialization scheme.
# This basically attempts to reproduce the Tensorflow starting point.
_C.MODEL.IMFCSNET.USE_ORIGINAL_WEIGHT_INIT = False

# Data transformations
# At this stage, transformations are only applied to the image stacks.
_C.TRANSFORMS = CN()
# Universal transforms
# These include any transform that needs to be applied universally.
# For example, normalization.
_C.TRANSFORMS.UNIVERSAL = CN()
# New normalization methods can be added in imfcsnet-pytorch/imfcs_pytorch/builders/transforms.py
_C.TRANSFORMS.UNIVERSAL.NORMALIZATION = (
    "zscore"  # Can be "zscore", "minmax" or "zerofloor"
)
# Training-specific augmentations.
# At this stage, only includes noise, which has been moved out of the simulations to be a data augmentation.
_C.TRANSFORMS.TRAIN = CN()
_C.TRANSFORMS.TRAIN.DISABLE = False  # Debug flag to disable the training augmentations.
_C.TRANSFORMS.TRAIN.NOISE = CN()
_C.TRANSFORMS.TRAIN.NOISE.TYPE = "random"  # Supported noise types include "emccd", "gaussian", "mix" (gaussian+emccd) and "random" (either Gaussian or EMCCD noise)

# Gaussian noise specific settings.
# Gaussian noise is sampled from a normal distribution.
# We sample a `scale` parameter using `scale = np.random.uniform(SCALE_MIN, SCALE_MAX)`
# Then, Gaussian noise is added to each pixel using `np.random.normal(0, scale=scale)`
# See imfcs_pytorch/data/transforms/augmentations/noise.py for implementation details.
_C.TRANSFORMS.TRAIN.NOISE.GAUSSIAN = CN()
_C.TRANSFORMS.TRAIN.NOISE.GAUSSIAN.SCALE_MIN = 1.0
_C.TRANSFORMS.TRAIN.NOISE.GAUSSIAN.SCALE_MAX = 9.0

# EMCCD-specific settings.
# EMCCD noise is sampled using the inverse transform method.
# This requires a pre-generated probability mass function.
# As per the original paper, the EMCCD noise is scaled by a random number sampled from `np.random.uniform(SCALE_MIN, SCALE_MAX)`
# See imfcs_pytorch/data/transforms/augmentations/noise.py for implementation details.
_C.TRANSFORMS.TRAIN.NOISE.EMCCD = CN()
_C.TRANSFORMS.TRAIN.NOISE.EMCCD.PMF_FILE = (
    "imfcs_pytorch/data/transforms/augmentations/emccd_pmf_2d.npy"
)
_C.TRANSFORMS.TRAIN.NOISE.EMCCD.DO_SCALING = True
_C.TRANSFORMS.TRAIN.NOISE.EMCCD.SCALE_MIN = 0.1
_C.TRANSFORMS.TRAIN.NOISE.EMCCD.SCALE_MAX = 1.0

# 'mix' and 'random' noise specific settings.
# For 'random' noise, this defines the that Gaussian noise is chosen.
# Defaults to 0.5, where there is a 50/50 chance for Gaussian/EMCCD noise.
# See imfcs_pytorch/data/transforms/augmentations/noise.py for implementation details.
_C.TRANSFORMS.TRAIN.NOISE.MIX = CN()
_C.TRANSFORMS.TRAIN.NOISE.MIX.GAUSSIAN_PROB = 0.5

# Evaluation augmentations.
# Currently unused, but might include test-time augmentations in the future.
_C.TRANSFORMS.EVAL = CN()

# Dataloader specific settings.
_C.DATALOADER = CN()
_C.DATALOADER.PER_STEP_BATCH_SIZE = 32
_C.DATALOADER.NUM_WORKERS = 0  # Tentatively working, but seems slower than single-process dataloading. Likely needs a bit more work.

# Optimizer settings.
# Generally accessed from the builder functions.
# See imfcsnet-pytorch/imfcs_pytorch/builders/optimizers.py for implementation details.
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = "AdamW"
_C.OPTIMIZER.BASE_LEARNING_RATE = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 0.01
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.BETAS = (0.9, 0.999)
_C.OPTIMIZER.EPS = 1e-7

# # Learning rate scheduler settings.
# Generally accessed from the builder functions.
# See imfcsnet-pytorch/imfcs_pytorch/builders/schedulers.py for implementation details.
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.TYPE = "ExponentialLR"
_C.LR_SCHEDULER.WARMUP_STEPS = 2000
_C.LR_SCHEDULER.MIN_LR = 1e-6
# The following settings only apply to step-based or multiplicative learning rates, not the cosine schedulers we useby default.
_C.LR_SCHEDULER.UPDATE_PERIOD = 5
_C.LR_SCHEDULER.MILESTONES = [60, 80]
_C.LR_SCHEDULER.LR_MULTIPLICATIVE_FACTOR = 0.9999760150262588

# Training-specific settings.
_C.TRAINING = CN()
# Defines the maximum number of iterations to train for before terminating training.
_C.TRAINING.ITERATIONS = 384000  # Iterations here corresponds to the number of batches.
# _C.TRAINING.USE_AMP = True

_C.LOGGING = CN()
_C.LOGGING.DO_TENSORBOARD_LOGGING = True
_C.LOGGING.LOGGING_ITER_INTERVAL = 10

# Low-level settings.
# These primarily include conditional optimizations that only apply to specific requirements. These likely will not be changed.
_C.BACKEND = CN()
_C.BACKEND.CUDNN_BENCHMARK = False  # Might help if input sizes are kept constant, and model does not contain conditionals.
_C.BACKEND.DATASET_DOWNLOAD_FOLDER = "./dataset"  # Only relevant if the datasets used are torchvision datasets with download support.
_C.BACKEND.DATALOADER_PIN_MEMORY = False
# As per most modern publications, the learning rate is scaled based on the batch size. absolute_lr = base_lr * total_batch_size / 256
_C.BACKEND.SCALE_LEARNING_RATE = False
_C.BACKEND.BREAK_WHEN_LOSS_NAN = True


# The following simulations are the dimensionless simulations ported from the original codebase.
# These were modified to work within our new framework.
# These are left here for legacy purposes.
# If you wanted to try reproducing the original paper's results, these would be used.
_C.SIMULATION.SIM_2D_1P_DIMLESS = CN()
# The number of dimensions of the simulation
# Used to initialize the particle positions array.
# In the 2D case, this will have 2 position vectors, x and y
# In the 3D case, this will have 3: x, y and z
# Potentially allows for more in case additional particle-specific parameters need to be tracked
_C.SIMULATION.SIM_2D_1P_DIMLESS.DIMENSIONALITY = 2

# Constants: Generally related to the acquisition setup.
# For dimensionless simulations, these are used to cast the dimensionless parameters back into physical form.
_C.SIMULATION.SIM_2D_1P_DIMLESS.CONSTANTS = CN()
_C.SIMULATION.SIM_2D_1P_DIMLESS.CONSTANTS.FRAME_TIME = 0.00106  # in seconds
_C.SIMULATION.SIM_2D_1P_DIMLESS.CONSTANTS.PIXEL_SIZE = 24  # in μm
_C.SIMULATION.SIM_2D_1P_DIMLESS.CONSTANTS.MAGNIFICATION = 100
_C.SIMULATION.SIM_2D_1P_DIMLESS.CONSTANTS.NA = 1.45
_C.SIMULATION.SIM_2D_1P_DIMLESS.CONSTANTS.WAVELENGTH = 583  # in nm

_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES = CN()
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MEAN = CN()
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MEAN.EMISSION_RATE = 0.583
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MEAN.PARTICLE_DENSITY = 0.48382
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MEAN.PARTICLE_SIG = 0.06074
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MEAN.PHOTON_SIG = 0.67010

_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.SIGMA = CN()
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.SIGMA.EMISSION_RATE = (
    0.477  # # e.g. CPS 1k to 10k, frame time 1.06 ms
)
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.SIGMA.PARTICLE_DENSITY = 4.47152
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.SIGMA.PARTICLE_SIG = (
    7.06284  # e.g. range of D between 0.02 to 50 um^2/s
)
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.SIGMA.PHOTON_SIG = 0.04190  # e.g. PSF 0.75 to 0.85, with mean of 0.8, with WL 583 nm and NA 1.45, , magnification 100X. It's uniformly sampled.

# minimal values parameters are allowed to take, on natural scale
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN = CN()
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN.EMISSION_RATE = 0.106
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN.PARTICLE_DENSITY = 0.1082
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN.PARTICLE_SIG = 0.0086
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN.PHOTON_SIG = 0.6282

# maximal values parameters are allowed to take, on natural scale
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MAX = CN()
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MAX.EMISSION_RATE = 1.060
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MAX.PARTICLE_DENSITY = 2.1634
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MAX.PARTICLE_SIG = 0.4290
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MAX.PHOTON_SIG = 0.7120

# This covers whether each of these values are transformed.
# Should be between [None, "log", "logit", "logitsample"]
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.TRANSFORM = CN()
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.TRANSFORM.EMISSION_RATE = None
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.TRANSFORM.PARTICLE_DENSITY = "log"
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.TRANSFORM.PARTICLE_SIG = "log"
_C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.TRANSFORM.PHOTON_SIG = None

_C.SIMULATION.SIM_3D_1P_DIMLESS = CN()
# The number of dimensions of the simulation
# Used to initialize the particle positions array.
# In the 2D case, this will have 2 position vectors, x and y
# In the 3D case, this will have 3: x, y and z
# Potentially allows for more in case additional particle-specific parameters need to be tracked
_C.SIMULATION.SIM_3D_1P_DIMLESS.DIMENSIONALITY = 3

# Constants: Generally related to the acquisition setup.
# For dimensionless simulations, these are used to cast the dimensionless parameters back into physical form.
# Note that here, we don't define ZFac and LightSheetThickness. Instead, we derive them at runtime based on the parameters defined below.
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS = CN()
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS.FRAME_TIME = 0.00204  # in seconds
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS.PIXEL_SIZE = 24  # in μm
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS.MAGNIFICATION = 60
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS.SIGMA_0 = 1.1
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS.SIGMA_Z = 2.2
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS.NA = 1
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS.WAVELENGTH = 515  # in nm
_C.SIMULATION.SIM_3D_1P_DIMLESS.CONSTANTS.Z_DIM_FACTOR = 10.0

_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES = CN()
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MEAN = CN()
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MEAN.EMISSION_RATE = 1.122
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MEAN.PARTICLE_DENSITY = 0.48382
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MEAN.PARTICLE_SIG = 0.05035
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MEAN.PHOTON_SIG = 0.70810

_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.SIGMA = CN()
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.SIGMA.EMISSION_RATE = (
    0.91800  # # e.g. CPS 1k to 10k, frame time 2.04 ms
)
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.SIGMA.PARTICLE_DENSITY = 4.47152
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.SIGMA.PARTICLE_SIG = (
    7.09195  # e.g. range of D between 0.02 to 50 um^2/s
)
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.SIGMA.PHOTON_SIG = 0.03220  # e.g. PSF 0.75 to 0.85, with mean of 0.8, with WL 583 nm and NA 1.45, , magnification 100X. It's uniformly sampled.

# minimal values parameters are allowed to take, on natural scale
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MIN = CN()
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MIN.EMISSION_RATE = 0.204
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MIN.PARTICLE_DENSITY = 0.1082
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MIN.PARTICLE_SIG = 0.0071
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MIN.PHOTON_SIG = 0.6759

# maximal values parameters are allowed to take, on natural scale
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MAX = CN()
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MAX.EMISSION_RATE = 2.040
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MAX.PARTICLE_DENSITY = 2.1634
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MAX.PARTICLE_SIG = 0.3571
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.MAX.PHOTON_SIG = 0.7403

# This covers whether each of these values are transformed.
# Should be between [None, "log", "logit", "logitsample"]
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.TRANSFORM = CN()
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.TRANSFORM.EMISSION_RATE = None
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.TRANSFORM.PARTICLE_DENSITY = "log"
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.TRANSFORM.PARTICLE_SIG = "log"
_C.SIMULATION.SIM_3D_1P_DIMLESS.VARIABLES.TRANSFORM.PHOTON_SIG = None
