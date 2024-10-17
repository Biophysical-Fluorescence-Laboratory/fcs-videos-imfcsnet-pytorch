"""Simulation code based on the original Java code. Instead of working with the dimensionless code from the original paper, we opt to dump the idea of working with dimensionless parameters entirely, as those introduce too many layers of abstraction and potential bugs which might result in difficulties down the line."""

import numpy as np
import math
from numba import float32, uint32, boolean
from imfcs_pytorch.data.simulation.utils.numba_funcs import (
    xoroshiro128p_uniform_float32,
    xoroshiro128p_poisson_uint32,
)
from numba.cuda.random import xoroshiro128p_normal_float32

# Replacing the original enums, we can pull the necessary indices from the YACS configuration file.
from imfcs_pytorch.config.defaults import _C
from imfcs_pytorch.utils.config import extract_key_index_from_dict

# Typing-specific imports.
from numba.cuda.cudadrv.devicearray import DeviceNDArray

# Extract global constants for all simulation types.
idx_sim_num_pixels = extract_key_index_from_dict(_C.SIMULATION.CONSTANTS, "NUM_PIXELS")
idx_sim_steps_per_frame = extract_key_index_from_dict(
    _C.SIMULATION.CONSTANTS, "STEPS_PER_FRAME"
)
idx_sim_margin = extract_key_index_from_dict(_C.SIMULATION.CONSTANTS, "MARGIN")
idx_sim_frames = extract_key_index_from_dict(_C.SIMULATION.CONSTANTS, "FRAMES")

# Extract indices of the simulation-specific constants
idx_sim_frame_time = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "FRAME_TIME"
)
idx_sim_pixel_size = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "PIXEL_SIZE"
)
idx_sim_magnification = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "MAGNIFICATION"
)
idx_sim_na = extract_key_index_from_dict(_C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "NA")
idx_sim_wavelength = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "WAVELENGTH"
)
idx_sim_z_dim_ext_factor = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "Z_DIM_EXT_FACTOR"
)
idx_sim_refractive_index = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "REFRACTIVE_INDEX"
)
idx_sim_do_bleaching = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "DO_BLEACHING"
)
idx_sim_do_blinking = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.CONSTANTS, "DO_BLINKING"
)

# Extract indices of the simulation-specific variables.
# These will be used to reference the randomized parameters obtained through sampling.
idx_mod_cps = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "CPS"
)
idx_mod_psf_sigma_0 = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "PSF_SIGMA_0"
)
idx_mod_psf_sigma_z = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "PSF_SIGMA_Z"
)
idx_mod_no_of_particles = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "NO_OF_PARTICLES"
)
idx_mod_tau_bleach = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "TAU_BLEACH"
)
idx_mod_bleach_radius = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "BLEACH_RADIUS"
)
idx_mod_bleach_frame = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "BLEACH_FRAME"
)
idx_mod_d1 = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "D1"
)
idx_mod_d2 = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "D2"
)
idx_mod_d3 = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "D3"
)
idx_mod_f2 = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "F2"
)
idx_mod_f3 = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "F3"
)
idx_mod_triplet_on_rate = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "TRIPLET_ON_RATE"
)
idx_mod_triplet_off_rate = extract_key_index_from_dict(
    _C.SIMULATION.SIM_3D_PHYSICAL.VARIABLES.MIN, "TRIPLET_OFF_RATE"
)


def simulate(
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
    # Global constant simulation parameters
    num_pixels = sim_param_store[seq_id, idx_sim_num_pixels]
    steps_per_frame = sim_param_store[seq_id, idx_sim_steps_per_frame]
    sim_margin = sim_param_store[seq_id, idx_sim_margin]
    no_of_frames = sim_param_store[seq_id, idx_sim_frames]

    # Simulation specific constants
    frame_time = model_specific_param_store[seq_id, idx_sim_frame_time]
    pixel_size = (
        model_specific_param_store[seq_id, idx_sim_pixel_size] * 1e-6
    )  # cast from m to um
    magnification = model_specific_param_store[seq_id, idx_sim_magnification]
    numerical_aperture = model_specific_param_store[seq_id, idx_sim_na]
    wavelength = (
        model_specific_param_store[seq_id, idx_sim_wavelength] * 1e-9
    )  # Cast from m to nm
    z_dim_ext_factor = model_specific_param_store[seq_id, idx_sim_z_dim_ext_factor]
    refractive_index = model_specific_param_store[seq_id, idx_sim_refractive_index]

    # Flags to toggle bleaching and blinking
    do_bleaching = model_specific_param_store[seq_id, idx_sim_do_bleaching]
    do_blinking = model_specific_param_store[seq_id, idx_sim_do_blinking]

    # Simulation-specific variables
    cps = model_param_store[seq_id, idx_mod_cps]
    psf_sigma_0 = model_param_store[seq_id, idx_mod_psf_sigma_0]
    psf_sigma_z = model_param_store[seq_id, idx_mod_psf_sigma_z]
    no_of_particles = model_param_store[seq_id, idx_mod_no_of_particles]
    tau_bleach = model_param_store[seq_id, idx_mod_tau_bleach]
    # bleach_radius = model_param_store[seq_id, idx_mod_bleach_radius]
    # bleach_frame = model_param_store[seq_id, idx_mod_bleach_frame]
    d1 = model_param_store[seq_id, idx_mod_d1] * 1e-12  # Cast to m2/s from um2
    d2 = model_param_store[seq_id, idx_mod_d2] * 1e-12  # Cast to m2/s from um2
    d3 = model_param_store[seq_id, idx_mod_d3] * 1e-12  # Cast to m2/s from um2
    f2 = model_param_store[seq_id, idx_mod_f2]
    f3 = model_param_store[seq_id, idx_mod_f3]
    triplet_on_rate = model_param_store[seq_id, idx_mod_triplet_on_rate]
    triplet_off_rate = model_param_store[seq_id, idx_mod_triplet_off_rate]

    # Derived parameters
    f1 = 1.0 - f2 - f3
    time_per_step = frame_time / steps_per_frame
    photons_per_step = float32(cps * time_per_step)
    pixel_size = pixel_size / (magnification)
    psf_size = 0.5 * psf_sigma_0 * wavelength / numerical_aperture

    # Bleaching-specific
    bleach_factor = 2.0
    if do_bleaching:
        bleach_factor = math.exp(-time_per_step / tau_bleach)

    # Triplet dark-state specific
    dark_state_fraction = triplet_off_rate / (triplet_off_rate + triplet_on_rate)
    blink_on_factor = math.exp(-time_per_step * triplet_on_rate)
    blink_off_factor = math.exp(-time_per_step * triplet_off_rate)

    # Detection specific
    sim_grid_size = num_pixels * pixel_size
    sim_mid_pos = sim_grid_size / 2.0
    extension_factor = (sim_margin * 2 + num_pixels) / 2 / num_pixels
    sim_size_lower_limit = -extension_factor * sim_grid_size
    sim_size_upper_limit = extension_factor * sim_grid_size
    range_lower_to_upper = (
        sim_size_lower_limit - sim_size_upper_limit
    )  # Used for sampling a ranged-uniform
    sim_detector_size = (
        sim_grid_size / 2
    )  # Detector extends from -sim_detector_size to +sim_detector_size

    # 3D specific derivations
    light_sheet_thickness = (
        psf_sigma_z * wavelength / numerical_aperture / 2.0
    )  # Division by 2 to obtail the 1/sqrt(e) radius
    # Find the z-axis upper and lower bounds.
    sim_size_z_lower_limit = -z_dim_ext_factor * light_sheet_thickness
    sim_size_z_upper_limit = z_dim_ext_factor * light_sheet_thickness
    range_z_lower_to_upper = (
        sim_size_z_lower_limit - sim_size_z_upper_limit
    )  # Used for sampling a ranged-uniform

    z_factor = numerical_aperture / math.sqrt(
        (refractive_index**2) - (numerical_aperture**2)
    )

    # Grouping particles by their diffusion coefficient
    no_particles_group_1 = uint32(round(no_of_particles * f1))
    no_particles_group_2 = uint32(round(no_of_particles * f2))
    no_particles_group_3 = uint32(round(no_of_particles * f3))
    # Also, pre-define the step sizes.
    # This will be used to diffuse the particles.
    step_size_group_1 = math.sqrt(2 * d1 * time_per_step)
    step_size_group_2 = math.sqrt(2 * d2 * time_per_step)
    step_size_group_3 = math.sqrt(2 * d3 * time_per_step)

    # initialize particle positions, bleach state and triplet_on state
    for particle_ind in range(no_of_particles):
        # x[0], y[1] and z[2] positions based on a uniform distribution
        # (r1 - r2) * U + r2
        position[seq_id, particle_ind, 0] = (
            range_lower_to_upper * xoroshiro128p_uniform_float32(rng, seq_id)
            + sim_size_upper_limit
        )
        position[seq_id, particle_ind, 1] = (
            range_lower_to_upper * xoroshiro128p_uniform_float32(rng, seq_id)
            + sim_size_upper_limit
        )
        position[seq_id, particle_ind, 2] = (
            range_z_lower_to_upper * xoroshiro128p_uniform_float32(rng, seq_id)
            + sim_size_z_upper_limit
        )

        # bleach state is always on at the start
        position[seq_id, particle_ind, 3] = 1.0

        # Triplet on/off state is based on the defined dark_state fraction
        if do_blinking:
            if uint32((particle_ind + 1) * dark_state_fraction) > uint32(
                particle_ind * dark_state_fraction
            ):
                position[seq_id, particle_ind, 4] = 0.0
            else:
                position[seq_id, particle_ind, 4] = 1.0
        else:
            position[seq_id, particle_ind, 4] = 1.0

    # Note that the noise addition is stripped out of the simulations, and is instead implemented as a data augmentation.

    # start main loop
    for frame in range(no_of_frames):
        # Bleach Frame was not included in the original Java 3D simulations, leaving this section disabled
        # # If the bleach frame is reached, bleach all particles within the bleach region.
        # if do_bleaching and frame == bleach_frame:
        #     for particle_ind in range(no_of_particles):
        #         if (
        #             math.sqrt(
        #                 math.pow(position[seq_id, particle_ind, 0], 2)
        #                 + math.pow(position[seq_id, particle_ind, 1], 2)
        #             )
        #             < bleach_radius
        #         ):
        #             position[seq_id, particle_ind, 2] = 0.0

        for step in range(steps_per_frame):
            for particle_ind in range(no_of_particles):
                pos_x = position[seq_id, particle_ind, 0]
                pos_y = position[seq_id, particle_ind, 1]
                pos_z = position[seq_id, particle_ind, 2]
                bleach_state = position[seq_id, particle_ind, 3]
                blink_state = position[seq_id, particle_ind, 4]

                # Diffuse particle depending on which D group it belongs to
                # Diffuse in the first group
                if particle_ind < no_particles_group_1:
                    pos_x += step_size_group_1 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )
                    pos_y += step_size_group_1 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )
                    pos_z += step_size_group_1 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )
                # Diffuse in the 2nd group
                elif (particle_ind >= no_particles_group_1) and (
                    particle_ind < (no_particles_group_1 + no_particles_group_2)
                ):
                    pos_x += step_size_group_2 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )
                    pos_y += step_size_group_2 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )
                    pos_z += step_size_group_2 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )
                # Diffuse in the 3rd group
                else:
                    pos_x += step_size_group_3 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )
                    pos_y += step_size_group_3 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )
                    pos_z += step_size_group_3 * xoroshiro128p_normal_float32(
                        rng, seq_id
                    )

                # Bleach particle based on defined probability
                if xoroshiro128p_uniform_float32(rng, seq_id) > bleach_factor:
                    bleach_state = 0.0

                # Blinking of particles.
                # Triplet on/off state is based on the defined dark_state fraction
                if do_blinking:
                    if boolean(blink_state):
                        if xoroshiro128p_uniform_float32(rng, seq_id) > blink_on_factor:
                            blink_state = abs(1.0 - blink_state)
                    else:
                        if (
                            xoroshiro128p_uniform_float32(rng, seq_id)
                            > blink_off_factor
                        ):
                            blink_state = abs(1.0 - blink_state)

                # check if particle is outside the simulation area
                if (
                    pos_x < sim_size_lower_limit
                    or pos_x > sim_size_upper_limit
                    or pos_y < sim_size_lower_limit
                    or pos_y > sim_size_upper_limit
                    or pos_z < sim_size_z_lower_limit
                    or pos_z > sim_size_z_upper_limit
                ):
                    side = xoroshiro128p_uniform_float32(rng, seq_id)
                    side_on_z = xoroshiro128p_uniform_float32(
                        rng, seq_id
                    )  # choose if we place the particle on z boundaries, else it is on either x or y boundaries.

                    if side_on_z < 0.5:  # Resample at random z position
                        if side < 0.25:
                            pos_x = (
                                range_lower_to_upper
                                * xoroshiro128p_uniform_float32(rng, seq_id)
                                + sim_size_upper_limit
                            )
                            pos_y = sim_size_lower_limit
                        elif side < 0.50:
                            pos_x = (
                                range_lower_to_upper
                                * xoroshiro128p_uniform_float32(rng, seq_id)
                                + sim_size_upper_limit
                            )
                            pos_y = sim_size_upper_limit
                        elif side < 0.75:
                            pos_x = sim_size_lower_limit
                            pos_y = (
                                range_lower_to_upper
                                * xoroshiro128p_uniform_float32(rng, seq_id)
                                + sim_size_upper_limit
                            )
                        else:
                            pos_x = sim_size_upper_limit
                            pos_y = (
                                range_lower_to_upper
                                * xoroshiro128p_uniform_float32(rng, seq_id)
                                + sim_size_upper_limit
                            )
                        pos_z = (
                            range_z_lower_to_upper
                            * xoroshiro128p_uniform_float32(rng, seq_id)
                            + sim_size_z_upper_limit
                        )
                    else:  # Resample at z-boundary
                        if side < 0.5:
                            pos_z = sim_size_z_lower_limit
                        else:
                            pos_z = sim_size_z_upper_limit
                        pos_x = (
                            range_lower_to_upper
                            * xoroshiro128p_uniform_float32(rng, seq_id)
                            + sim_size_upper_limit
                        )
                        pos_y = (
                            range_lower_to_upper
                            * xoroshiro128p_uniform_float32(rng, seq_id)
                            + sim_size_upper_limit
                        )

                    bleach_state = 1.0

                # Calculate zcor
                # factor describing the increase of the PSF at the focal plane for a particle not situated in the focal plane
                z_cor = psf_size + math.fabs(pos_z) * (z_factor / 2)

                # Simulate photon emission
                # This is affected by the bleach and blink state.
                no_of_photons = (
                    xoroshiro128p_poisson_uint32(rng, seq_id, photons_per_step)
                    * bleach_state
                    * blink_state
                )
                # Next, affect the photon emission by the z-axis position and the light sheet thickness
                no_of_photons = uint32(
                    round(
                        no_of_photons
                        * math.exp(-0.5 * math.pow((pos_z / light_sheet_thickness), 2))
                    )
                )

                # # Ensure that there are no negative photon counts.
                if no_of_photons < 0:
                    no_of_photons = 0

                # Simulate detection
                for photon_ind in range(no_of_photons):
                    # calculate location of photon
                    # Unlike the source code, no reason to use Box-Muller transform, as the Numba random code already implements normal distribution sampling through an underlying Box Muller transform.
                    locx = pos_x + z_cor * xoroshiro128p_normal_float32(rng, seq_id)
                    locy = pos_y + z_cor * xoroshiro128p_normal_float32(rng, seq_id)

                    if (
                        locx < sim_detector_size
                        and locy < sim_detector_size
                        and locx > -sim_detector_size
                        and locy > -sim_detector_size
                    ):
                        pix_x = uint32(math.floor((locx + sim_mid_pos) / pixel_size))
                        pix_y = uint32(math.floor((locy + sim_mid_pos) / pixel_size))

                        # update scan count
                        scan[seq_id, frame, pix_x, pix_y] += 1
                # store positions back to main array
                position[seq_id, particle_ind, 0] = pos_x
                position[seq_id, particle_ind, 1] = pos_y
                position[seq_id, particle_ind, 2] = pos_z
                position[seq_id, particle_ind, 3] = bleach_state
                position[seq_id, particle_ind, 4] = blink_state
