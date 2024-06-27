"""2d simulation code based on the imfcsnet.simulator.fd1t_2d_datagen from the original codebase (https://github.com/ImagingFCS/ImFCS_FCSNet_ImFCSNet/tree/master/imfcsnet/simulator).

Refactoring was done where necessary to increase code readability and modularity. Major changes are documented below:
- Since Sequences is a feature of Tensorflow (https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence), we need to reimplement this in a different way for PyTorch.
- For the `simulate` function (from the original `simulateScanFunc`), the grouping of all parameters into a single `sim_param_store` array is broken up to remove abstraction. This does result in a lot more parameter names, and correspondingly required a modification of the `compile` JIT-er to allow for compilation to CUDA.

References:
- https://pytorch.org/docs/stable/data.html#iterable-style-datasets - A similar option for PyTorch to Sequences is iterable datasets, since that inherently handles the state of the current batch, rather than relying on the __getitem__ and corresponding random sampled indices.
- https://numba.readthedocs.io/en/stable/cuda/index.html - Numba guide to better understand the functions defined in the original imfcsnet.Utilities.ufunc folder
- https://github.com/openai/triton/issues/160 - Discussion about numba vs Triton.
"""

import numpy as np
from enum import Enum, auto
from numba import float32, uint64, uint32
from imfcs_pytorch.data.simulation.utils.numba_funcs import (
    xoroshiro128p_uniform_float32,
    xoroshiro128p_poisson_uint32,
)
from numba.cuda.random import xoroshiro128p_normal_float32

# Replacing the original enums, we can pull the necessary indices from the YACS configuration file.
from imfcs_pytorch.config.defaults import _C
from imfcs_pytorch.utils.config import extract_key_index_from_dict

from numba.cuda.cudadrv.devicearray import DeviceNDArray


# Instead of using enums, we can make a few safe-ish assumptions
# 1) These specific indices will not be called outside of the 2D case (3D will have its own dedicated code file). This means we can keep our Enums here, which prevents code-sharing from complicating the addition of new parameters.
# 2) Why Enums? This is a holdover from the original codebase, but it does make sense, as values can be added and the corresponding index values can be updated as necessary.
# 3) Note that certain values were stripped here as discussed in point (1). For example, 3D specific items like z-factor are removed from the 2D code. These enums are code-specific, and we will not be using them externally from this file.
# class EnumModelParams(Enum):
#     EMISSION_RATE = 0
#     PARTICLE_DENSITY = auto()
#     PARTICLE_SIG = auto()
#     PHOTON_SIG = auto()
#     CCD_NOISE_RATE = auto()


# class EnumSimParams(Enum):
#     NUM_PIXELS = 0
#     STEPS_PER_FRAME = auto()
#     MARGIN = auto()
#     FRAMES = auto()
#     # NOISE_TYPE = auto()
#     # EMCCD_MIN = auto()
#     # EMCCD_FACTOR_MIN = auto()
#     # EMCCD_FACTOR_MAX = auto()
#     # Z_DIM_FACTOR = auto()
#     # Z_FAC = auto()
#     # LIGHT_SHEET_THICKNESS = auto()


class EnumTrans(Enum):
    TRANSNONE = 0
    TRANSLOG = auto()
    TRANSLOGIT = auto()
    TRANSLOGITANDSAMPLE = auto()


# class EnumNoiseType(Enum):
#     GAUSSIAN = 0  # Gaussian noise
#     EXPERIMENTAL_EMCCD = 1  # experimental EMCCD camera probability mass function
#     MIXNOISE = 2  # 50% Gaussian 50% experimental EMCCD


# idx_sim_num_pixels = EnumSimParams.NUM_PIXELS.value
# idx_sim_steps_per_frame = EnumSimParams.STEPS_PER_FRAME.value
# idx_sim_margin = EnumSimParams.MARGIN.value
# idx_sim_frames = EnumSimParams.FRAMES.value

# idx_mod_particle_density = EnumModelParams.PARTICLE_DENSITY.value
# idx_mod_emission_rate = EnumModelParams.EMISSION_RATE.value
# idx_mod_ccd_noise_rate = EnumModelParams.CCD_NOISE_RATE.value
# idx_mod_particle_sig = EnumModelParams.PARTICLE_SIG.value
# idx_mod_photon_sig = EnumModelParams.PHOTON_SIG.value

idx_sim_num_pixels = extract_key_index_from_dict(_C.SIMULATION.CONSTANTS, "NUM_PIXELS")
idx_sim_steps_per_frame = extract_key_index_from_dict(
    _C.SIMULATION.CONSTANTS, "STEPS_PER_FRAME"
)
idx_sim_margin = extract_key_index_from_dict(_C.SIMULATION.CONSTANTS, "MARGIN")
idx_sim_frames = extract_key_index_from_dict(_C.SIMULATION.CONSTANTS, "FRAMES")

idx_mod_particle_density = extract_key_index_from_dict(
    _C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN, "PARTICLE_DENSITY"
)
idx_mod_emission_rate = extract_key_index_from_dict(
    _C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN, "EMISSION_RATE"
)
idx_mod_particle_sig = extract_key_index_from_dict(
    _C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN, "PARTICLE_SIG"
)
idx_mod_photon_sig = extract_key_index_from_dict(
    _C.SIMULATION.SIM_2D_1P_DIMLESS.VARIABLES.MIN, "PHOTON_SIG"
)

# set transformation values
trans_none = EnumTrans.TRANSNONE
trans_log = EnumTrans.TRANSLOG
trans_logit = EnumTrans.TRANSLOGIT
trans_logit_and_sample = EnumTrans.TRANSLOGITANDSAMPLE


def simulate_scan_func(
    rng: DeviceNDArray,
    from_seq_idx: int,
    to_seq_idx: int,
    seq_id: int,
    scan: np.ndarray,
    model_param_store: np.ndarray,
    sim_param_store: np.ndarray,
    model_specific_param_store: np.ndarray,
    position: np.ndarray,
    # emccd_cdf: np.ndarray,
    # numFrames: np.ndarray,
):
    """Function to simulate a 2D image stack based on the defined simulation parameters.

    TODO: Surely there's a better way to handle the parameters than passing in matrices. The only real matrices that require cross-device communication (GPU and CPU) are the `scan` arrays for the image stacks.

    TODO: Even if the variable `sim_param_store` array that relies on randomly sampled values can't be broken into individual values, the model_param_store array is constant, and should perhaps be considered to be locked due to the lack of variability.

    Args:
        rng (DeviceNDArray): RNG state array created by `create_xoroshiro128p_states`. 1D array, dtype=xoroshiro128p_dtype
        from_seq_idx (int): The initial 'from' index for the CUDA kernel execution. Unused in code, but is a necessary parameter for interop.
        to_seq_idx (int): The ending 'to' index for CUDA kernel execution. Unused in code, but is a necessary parameter for interop.
        seq_id (int): The index of the sequence being simulated. This is generally a value between 0 and batches*batch_size, letting the CUDA kernel know what parameter row to use from the model and simulation parameter arrays.
        scan (np.ndarray): The simulated image stack array. This needs to be initialized on-machine as CUDA kernels do not have return values, but rather need to write to provided in/out kernels.
        model_param_store (np.ndarray): Parameter array of variable simulation-specific parameters as dimensionless parameters. These are randomly sampled through the random_par_model method in the IterableDataset class.
        sim_param_store (np.ndarray): Parameter array of constant model-specific parameters. These are constant throughout the whole simulation process, over different runs.
        position (np.ndarray): Position array storing x and y positions of each simulated parameter. Technically not required, but initializing an array on the CPU and then passing it to the GPU is more convenient than manually initializing this array on the GPU.
        emccd_cdf (np.ndarray): Extracted cumulative density function of the EMCCD, used for generating simulated dark noise.
        # numFrames (np.ndarray): Number of frames to simulate for. Tentatively passed in as a full array with constant values across the full length, enabling the potential for variable length simulations.
    """
    # simulation parameters
    num_pixels = sim_param_store[seq_id, idx_sim_num_pixels]
    num_steps = sim_param_store[seq_id, idx_sim_steps_per_frame]
    num_margin = sim_param_store[seq_id, idx_sim_margin]
    num_frames = sim_param_store[seq_id, idx_sim_frames]
    # num_frames = numFrames[seq_id]  # sim_param_store[seq_id, idxSimFrames]
    # noise_type = int64(sim_param_store[seq_id, idx_sim_noise_type])

    # derived simulation parameters
    num_pixelsf32 = float32(num_pixels)
    left_bound = float32(-num_margin)
    right_bound = float32(num_pixels + num_margin)
    total_width = right_bound - left_bound

    # model parameters (step adjusted)
    num_particles = uint64(
        model_param_store[seq_id, idx_mod_particle_density] * total_width * total_width
    )
    emission_rate_per_step = float32(model_param_store[seq_id, idx_mod_emission_rate])
    # ccd_noise_rate = float32(model_param_store[seq_id, idx_mod_ccd_noise_rate])
    particle_sig_per_step = float32(model_param_store[seq_id, idx_mod_particle_sig])
    photon_sig = float32(model_param_store[seq_id, idx_mod_photon_sig])
    # emccd_cdf = emccd_cdf[seq_id, :]
    # emccd_n_min = float32(sim_param_store[seq_id, idx_sim_emccd_min])
    # emccd_factor_min = float32(sim_param_store[seq_id, idx_sim_emccd_factor_min])
    # emccd_factor_max = float32(sim_param_store[seq_id, idx_sim_emccd_factor_max])

    # EMCCD_scale_factor = float32(
    #     xoroshiro128p_uniform_float32(rng, seq_id)
    #     * (emccd_factor_max - emccd_factor_min)
    #     + emccd_factor_min
    # )

    # Controlling the noise type.
    # if noise_type == int64(idx_gaussian_noise):
    #     use_gaussian_noise = True
    # elif noise_type == int64(idx_experimental_emccd_noise):
    #     use_gaussian_noise = False
    # else:  # Mixed noise case
    #     use_gaussian_noise = float32(xoroshiro128p_uniform_float32(rng, seq_id)) >= 0.5
    # NOTE: Cannot use else: raise SystemExit ... with numba. Error encountered.
    #    else:
    #        raise SystemExit("Invalid Noise Type in simulationScanFunc")

    # initialize particle positions
    for particle_id in range(num_particles):
        position[seq_id, particle_id, 0] = (
            xoroshiro128p_uniform_float32(rng, seq_id) * total_width + left_bound
        )
        position[seq_id, particle_id, 1] = (
            xoroshiro128p_uniform_float32(rng, seq_id) * total_width + left_bound
        )

    # start main loop
    for frame in range(num_frames):
        # for pixelX in range(num_pixels):
        #     for pixelY in range(num_pixels):
        #         if use_gaussian_noise:
        #             scan[seq_id, frame, pixelX, pixelY, 0] = math.sqrt(
        #                 ccd_noise_rate
        #             ) * xoroshiro128p_normal_float32(rng, seq_id)
        #         else:
        #             # using inverse transform sampling iso rejection sampling.
        #             # See https://en.wikipedia.org/wiki/Inverse_transform_sampling.
        #             unifval = float32(xoroshiro128p_uniform_float32(rng, seq_id))
        #             counter = uint32(0)
        #             while emccd_cdf[counter] < unifval:
        #                 counter += uint32(1)
        #             scan[seq_id, frame, pixelX, pixelY, 0] = EMCCD_scale_factor * (
        #                 float32(counter) + emccd_n_min
        #             )

        for particle_id in range(num_particles):
            posx = position[seq_id, particle_id, 0]
            posy = position[seq_id, particle_id, 1]
            for step in range(num_steps):
                posx += particle_sig_per_step * xoroshiro128p_normal_float32(
                    rng, seq_id
                )
                posy += particle_sig_per_step * xoroshiro128p_normal_float32(
                    rng, seq_id
                )

                # check if particle is outside the simulation area
                if (
                    posx < left_bound
                    or posx > right_bound
                    or posy < left_bound
                    or posy > right_bound
                ):
                    newpos = (
                        xoroshiro128p_uniform_float32(rng, seq_id) * total_width
                        + left_bound
                    )
                    side = xoroshiro128p_uniform_float32(rng, seq_id)

                    if side < 0.25:
                        posx = newpos
                        posy = left_bound
                    elif side < 0.50:
                        posx = newpos
                        posy = right_bound
                    elif side < 0.75:
                        posx = left_bound
                        posy = newpos
                    else:
                        posx = right_bound
                        posy = newpos

                # sample number of photons emitted during this time step
                photons = xoroshiro128p_poisson_uint32(
                    rng, seq_id, emission_rate_per_step
                )
                for _ in range(photons):
                    # calculate location of photon
                    locx = posx + photon_sig * xoroshiro128p_normal_float32(rng, seq_id)
                    locy = posy + photon_sig * xoroshiro128p_normal_float32(rng, seq_id)
                    # if photon left detector field
                    if (
                        locx < 0.0
                        or locx >= num_pixelsf32
                        or locy < 0.0
                        or locy >= num_pixelsf32
                    ):
                        continue
                    # photon within detector field, so calculate corresponding pixel
                    pixel_x = uint32(locx)
                    pixel_y = uint32(locy)
                    # update scan count
                    scan[seq_id, frame, pixel_x, pixel_y] += 1
            # store positions back to main array
            position[seq_id, particle_id, 0] = posx
            position[seq_id, particle_id, 1] = posy
