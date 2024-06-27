from imfcs_pytorch.config.defaults import _C
from yacs.config import CfgNode


def get_default_cfg() -> CfgNode:
    """Helper function to get the default configuration file defined in classification/config/defaults.

    Returns:
        CfgNode: Initialized config file with defaults. Should be merged with a defined input file via `cfg.merge_from_file()` to add experiment-specific configurations.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def validate_cfg(cfg: CfgNode):
    """Validation function to ensure that specific keys defined in the CfgNode are valid.

    This is very slow, but a necessary evil to ensure the training process is less fragile. Any new features that have certain expectations should also be validated here.

    YACS may already provide some type safety, but specific assertions can be made to ensure that all parameters fall within expectations.

    Args:
        cfg (CfgNode): YACS config object to validate.

    Raises:
        ValueError: A value defined in the YACS config file is invalid.
    """
    # Validate simulation parameters
    # Ensure that the defined SIMULATION.TYPE is valid.
    # Extract the valid keys (popping 'TYPE', 'BACKEND' and 'CONSTANTS', as these are simulation independent.)
    _valid_simulation_types = list(_C.SIMULATION.keys())
    _valid_simulation_types.remove("TYPE")
    _valid_simulation_types.remove("CONSTANTS")
    _valid_simulation_types.remove("BACKEND")
    _valid_simulation_types.remove("PARAMETER_SAMPLING_STRATEGY")
    if cfg.SIMULATION.TYPE not in _valid_simulation_types:
        raise ValueError(
            f"SIMULATION.TYPE {cfg.SIMULATION.TYPE} is not a valid simulation type. Valid simulation types are {_valid_simulation_types}"
        )

    # Check if the parameter sampling strategy is valid
    if cfg.SIMULATION.PARAMETER_SAMPLING_STRATEGY not in ["modsig", "minmax"]:
        raise ValueError(
            f'SIMULATION.PARAMETER_SAMPLING_STRATEGY must be in ["modsig", "minmax"]. Got {cfg.SIMULATION.PARAMETER_SAMPLING_STRATEGY}.'
        )

    # Ensure that the simulation has a corresponding DIMENSIONALITY key.
    if "DIMENSIONALITY" not in list(cfg.SIMULATION[cfg.SIMULATION.TYPE].keys()):
        raise ValueError(
            f"SIMULATION.{cfg.SIMULATION.TYPE}.DIMENSIONALITY does not exist. This is required to define the dimensions of the particle positions array."
        )
    if not isinstance(cfg.SIMULATION[cfg.SIMULATION.TYPE]["DIMENSIONALITY"], int):
        raise ValueError(
            f'SIMULATION.{cfg.SIMULATION.TYPE}.DIMENSIONALITY must be declared as an integer, and cannot be left as a null (None) value. Got {type(cfg.SIMULATION[cfg.SIMULATION.TYPE]["DIMENSIONALITY"])}'
        )

    # Specific for the dimensionless simulations
    if cfg.SIMULATION.TYPE in ["SIM_2D_1P_DIMLESS", "SIM_3D_1P_DIMLESS"]:
        for key, value in cfg.SIMULATION[cfg.SIMULATION.TYPE]["VARIABLES"][
            "TRANSFORM"
        ].items():
            if value not in [None, "log", "logit", "logitsample"]:
                raise ValueError(
                    f"Error when validating {key}. VAR_TRANSFORM must be [None, log, logit, logitsample], got {value}."
                )

    # Validate that SIMULATION.PARAMETER_SAMPLING_STRATEGY "modsig" only applies to the right SIMULATION.TYPEs.
    if cfg.SIMULATION.PARAMETER_SAMPLING_STRATEGY == "modsig":
        if cfg.SIMULATION.TYPE not in ["SIM_2D_1P_DIMLESS", "SIM_3D_1P_DIMLESS"]:
            raise ValueError(
                '"modsig" PARAMETER_SAMPLING_STRATEGY is only supported for legacy support on the old SIMULATION.TYPEs "SIM_2D_1P_DIMLESS", "SIM_3D_1P_DIMLESS". Use "minmax" instead for other simulations (and honestly, even for the old simulations too, as its much more readable.)'
            )

    # Validate experiment settings
    # Ensure that the defined EXPERIMENT.TASK is valid.
    _valid_experiment_tasks = list(_C.TASK.keys())
    if cfg.EXPERIMENT.TASK not in _valid_experiment_tasks:
        raise ValueError(
            f"{cfg.EXPERIMENT.TASK} is not a valid EXPERIMENT.TASK. Valid tasks are {_valid_experiment_tasks}."
        )

    # Regression task-specific settings.
    if cfg.EXPERIMENT.TASK == "REGRESSION":
        # Check if the targets are valid.
        # We have already validated that the simulation type is valid, so we can directly query the simulation VARIABLES.
        # We can assume that the MIN and MAX keys always exist.
        # For dimensionless sims (as per original paper), these serve as the clipping values.
        # For physical parameters, these are the uniform sampling upper- and lower-bounds.
        _valid_target_names = list(
            cfg.SIMULATION[cfg.SIMULATION.TYPE]["VARIABLES"]["MIN"].keys()
        )
        for target_name in cfg.TASK[cfg.EXPERIMENT.TASK]["TARGETS"]:
            if target_name not in _valid_target_names:
                raise ValueError(
                    f"TASK.REGRESSION.TARGETS includes invalid target names. Valid target names are {_valid_target_names}. Got invalid {target_name}."
                )

        # Check if the target transformations are valid.
        # Check 1: Number of target transforms should match number of targets
        if len(cfg.TASK.REGRESSION.TARGETS) != len(
            cfg.TASK.REGRESSION.TARGET_TRANSFORM
        ):
            raise ValueError(
                f"Length of TASK.REGRESSION.TARGET_TRANSFORM ({len(cfg.TASK.REGRESSION.TARGET_TRANSFORM)}) must be equal to legnth of TASK.REGRESSION.TARGETS ({len(cfg.TASK.REGRESSION.TARGETS)})."
            )
        # Check 2: Target transforms should be valid.
        _valid_target_transforms = [None, "log", "log10"]
        for target_transform_name in cfg.TASK.REGRESSION.TARGET_TRANSFORM:
            if target_transform_name not in _valid_target_transforms:
                raise ValueError(
                    f"TARGET_TRANSFORM {target_transform_name} is invalid. Must be in {_valid_target_transforms}."
                )

        # Check 3: Since log() cannot be applied to negative values, check the MIN keys of the sampling to ensure that negative values are not possible before permitting the transformations.
        for ind, target_name in enumerate(cfg.TASK.REGRESSION.TARGETS):
            if (
                cfg.TASK.REGRESSION.TARGET_TRANSFORM[ind] in ["log", "log10"]
                and cfg.SIMULATION[cfg.SIMULATION.TYPE]["VARIABLES"]["MIN"][target_name]
                <= 0
            ):
                raise ValueError(
                    f"Trying to apply TARGET_TRANSFORM {cfg.TASK.REGRESSION.TARGET_TRANSFORM[ind]} on TARGET {cfg.TASK.REGRESSION.TARGETS[ind]}, which has a MIN of {cfg.SIMULATION[cfg.SIMULATION.TYPE]['VARIABLES']['MIN'][target_name]}. log() operations cannot be performed on values less than or equal to 0."
                )

    # Check if interval checkpointing is feasible.
    # Basically, if there are less iterations than the checkpointing interval, block execution.
    if cfg.EXPERIMENT.CHECKPOINTING.DO_INTERVAL_CHECKPOINTING:
        if (
            cfg.EXPERIMENT.CHECKPOINTING.CHECKPOINTING_INTERVAL
            > cfg.TRAINING.ITERATIONS
        ):
            raise ValueError(
                f"Interval checkpointing is enabled, but the checkpointing interval {cfg.EXPERIMENT.CHECKPOINTING.CHECKPOINTING_INTERVAL} is larger than the amount of training iterations {cfg.TRAINING.ITERATIONS}. To execute training, deactivate interval checkpointing by setting EXPERIMENT.CHECKPOINTING.DO_INTERVAL_CHECKPOINTING = False, or choosing a lower interval in EXPERIMENT.CHECKPOINTING.CHECKPOINTING_INTERVAL."
            )

    # Validate model definitions
    # ImFCSNet
    if cfg.MODEL.NAME == "imfcsnet":
        # The initial spatial aggregation needs to have a kernel size that is compatible with the simulation pixels.
        # Default is (200, 3, 3), and simulation defaults to 3 pixels per-size.
        if cfg.MODEL.IMFCSNET.SPATIAL_AGG_BLOCK_KERNEL_SIZE[1:] != (
            cfg.SIMULATION.CONSTANTS.NUM_PIXELS,
            cfg.SIMULATION.CONSTANTS.NUM_PIXELS,
        ):
            raise ValueError(
                f"MODEL.IMFCSNET.SPATIAL_AGG_BLOCK_KERNEL_SIZE must have a kernel size that is compatible with the simulation pixel count defined in SIMULATION.CONSTANTS.NUM_PIXELS. Got {cfg.MODEL.IMFCSNET.SPATIAL_AGG_BLOCK_KERNEL_SIZE} with simulation pixel size {cfg.SIMULATION.CONSTANTS.NUM_PIXELS}. Perhaps you meant {(cfg.MODEL.IMFCSNET.SPATIAL_AGG_BLOCK_KERNEL_SIZE[0], cfg.SIMULATION.CONSTANTS.NUM_PIXELS, cfg.SIMULATION.CONSTANTS.NUM_PIXELS)}"
            )

    # Validate transform settings
    # Noise augmentations
    if cfg.TRANSFORMS.TRAIN.NOISE.TYPE not in [
        None,
        "gaussian",
        "emccd",
        "mix",
        "random",
    ]:
        raise ValueError(
            f'{cfg.TRANSFORMS.TRAIN.NOISE.TYPE}. Valid noise types (TRANSFORMS.TRAIN.NOISE.TYPE) are [None, "gaussian", "emccd", "mix", "random"]'
        )
    if (cfg.TRANSFORMS.TRAIN.NOISE.MIX.GAUSSIAN_PROB >= 1.0) or (
        cfg.TRANSFORMS.TRAIN.NOISE.MIX.GAUSSIAN_PROB <= 0.0
    ):
        raise ValueError(
            f"TRANSFORMS.TRAIN.NOISE.MIX.GAUSSIAN_PROB must be a float between 0.0 and 1.0. If you only want to use a single noise type, set TRANSFORMS.TRAIN.NOISE.TYPE to 'gaussian' or 'emccd'. Got TRANSFORMS.TRAIN.NOISE.MIX.GAUSSIAN_PROB={cfg.TRANSFORMS.TRAIN.NOISE.MIX.GAUSSIAN_PROB}."
        )
