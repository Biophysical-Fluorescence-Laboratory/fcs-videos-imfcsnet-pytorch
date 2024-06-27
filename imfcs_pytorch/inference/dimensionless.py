import numpy as np

from imfcs_pytorch.utils.parameter_conversions import (
    convert_emission_rate_to_cps,
    # convert_particle_density_to_n,
    convert_particle_sig_to_d,
    convert_photon_sig_to_psf,
)

# Typing-specific imports
from typing import List
from yacs.config import CfgNode


def reverse_dimensionless_conversion(
    cfg: CfgNode,
    model_outs: np.ndarray,
    regression_targets: List[str],
) -> np.ndarray:
    """Converts the dimensionless parameters into their physical parameter forms.

    This should generally only be accessed when using the ported over dimensionless simulations from the original codebase (i.e.  SIM_2D_1P_DIMLESS and SIM_3D_1P_DIMLESS).

    Args:
        cfg (CfgNode): YACS configuration object to extract constants for parameter conversions.
        model_outs (np.ndarray): Model output map as a NumPy array. Generally produced from `infer_fast` or `infer_on_stack`. Expected to be in shape of (TEMPORAL_WINDOWS, REGRESSION_TARGETS, WIDTH, HEIGHT)
        regression_targets (List[str]): Regression target list. Generally defined from the YACS config file as TASK.REGRESSION.TARGETS.

    Raises:
        ValueError: Invalid regression target in `regression_targets`.

    Returns:
        np.ndarray: Output map of shape (TEMPORAL_WINDOWS, REGRESSION_TARGETS, WIDTH, HEIGHT) with regression outputs casted to their corresponding physical representations. For example: PARTICLE_SIG -> D, PARTICLE_DENSITY -> N etc.
    """

    # Extracting task
    simulation_type = cfg.SIMULATION.TYPE

    output_map = np.empty_like(model_outs)

    # Populate the output map based on the designated regression targets.
    # The reverse dimensionless conversions will be called accordingly.
    for class_ind, regression_target in enumerate(regression_targets):
        if regression_target == "EMISSION_RATE":
            output_map[:, class_ind, :, :] = convert_emission_rate_to_cps(
                model_outs[:, class_ind, :, :],
                frame_time=cfg.SIMULATION[simulation_type]["CONSTANTS"]["FRAME_TIME"],
                steps_per_frame=cfg.SIMULATION.CONSTANTS.STEPS_PER_FRAME,
            )
        # PARTICLE_DENSITY should not be converted.
        # Partially, this is because PARTICLE_DENSITY should be the output, as N is not a true estimator.
        # At the same time, the MARGIN parameter could be a CONSTANT or VARIABLE depending on the simulation type.
        # elif regression_target == "PARTICLE_DENSITY":
        #     output_map[:, class_ind, :, :] = convert_particle_density_to_n(
        #         model_outs[:, class_ind, :, :],
        #         pixels_per_side=cfg.SIMULATION.CONSTANTS.NUM_PIXELS,
        #     )
        elif regression_target == "PARTICLE_SIG":
            output_map[:, class_ind, :, :] = convert_particle_sig_to_d(
                model_outs[:, class_ind, :, :],
                frame_time=cfg.SIMULATION[simulation_type]["CONSTANTS"]["FRAME_TIME"],
                steps_per_frame=cfg.SIMULATION.CONSTANTS.STEPS_PER_FRAME,
                pixel_size=cfg.SIMULATION[simulation_type]["CONSTANTS"]["PIXEL_SIZE"],
                magnification=cfg.SIMULATION[simulation_type]["CONSTANTS"][
                    "MAGNIFICATION"
                ],
            )
        elif regression_target == "PHOTON_SIG":
            output_map[:, class_ind, :, :] = convert_photon_sig_to_psf(
                model_outs[:, class_ind, :, :],
                emission_wavelength=cfg.SIMULATION[simulation_type]["CONSTANTS"][
                    "WAVELENGTH"
                ],
                numerical_aperture=cfg.SIMULATION[simulation_type]["CONSTANTS"]["NA"],
                pixel_size=cfg.SIMULATION[simulation_type]["CONSTANTS"]["PIXEL_SIZE"],
                magnification=cfg.SIMULATION[simulation_type]["CONSTANTS"][
                    "MAGNIFICATION"
                ],
            )
        else:
            raise ValueError(f"Invalid regression target {regression_target}.")

    return output_map
