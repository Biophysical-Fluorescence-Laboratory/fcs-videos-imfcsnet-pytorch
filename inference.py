"""This script performs the duty of inference, whether for a single-file or in-bulk. For convenience, glob-based file selection is also supported. Model predictions can either be output as CSVs or pixel-maps."""

import os
import glob
import argparse
import datetime
import warnings

import tifffile

import torch
import numpy as np
import torch.backends.cudnn as cudnn

from imfcs_pytorch.config.initialization import get_default_cfg, validate_cfg
from imfcs_pytorch.builders.transforms import (
    build_eval_transforms,
    wrapper_batch_transforms,
    build_reverse_target_transforms,
)
from imfcs_pytorch.inference.sliding_window import (
    infer_safe,
    infer_fast,
)
from imfcs_pytorch.inference.io import (
    process_output_map,
)
from imfcs_pytorch.inference.dimensionless import reverse_dimensionless_conversion

from imfcs_pytorch.builders.model import build_model
from imfcs_pytorch.data.transforms.preprocessing.bleach_correction.polynomial import (
    PolynomialBleachCorrectionModule,
)

# Typing-only imports, only used for typehints
from yacs.config import CfgNode
from typing import List


def get_parser() -> argparse.ArgumentParser:
    """Creates the parser to initialize an inference run.

    This should follow the YACS convention of 'There is only one way to configure the same thing.' Preferably, no additional CLI arguments should be added here. Instead, add them to the YACS configuration file, such that they can be overriden using the --config-overrides option.

    Returns:
        argparse.ArgumentParser: CLI Argument Parser.
    """
    parser = argparse.ArgumentParser(description="Image classification inference.")
    parser.add_argument(
        "--cfg",
        required=True,
        metavar="FILE",
        help="Path to YACS config file in .yaml format.",
    )
    parser.add_argument(
        "--overrides",
        metavar="STRING",
        default=[],
        type=str,
        help=(
            "Modify experimental configuration from the command line. See https://github.com/rbgirshick/yacs#command-line-overrides for details. Inputs should be comma-separated: `python inference.py --config-overrides EXPERIMENT.NAME modified_exp`."
        ),
        nargs=argparse.ONE_OR_MORE,
    )
    parser.add_argument(
        "--ckpt",
        required=False,
        default=None,
        metavar="FILE",
        help="Path to an existing checkpoint, if not provided, will load the best_model.ckpt file from the workdir defined in the provided config file.",
    )
    parser.add_argument(
        "--files",
        required=True,
        default=None,
        metavar="FILE",
        nargs="+",
        help="Paths to the files to perform inference on. To run inference on multiple files, provide a space-separated list of files. For example, `--files ./1.tif ./2.tif` will perform inference on both tiff files. Similarly, glob-based extensions are supported. For example, to run inference on all files with the `.tif` extension in a folder, you can use `--files ./folder/*.tif`",
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        default=None,
        metavar="FOLDER",
        help="The folder to save the outputs to. Defaults to None, where the outputs will be written to a folder in the current working directory with an appended datetime string.",
    )
    parser.add_argument(
        "--output-format",
        required=False,
        default="tif",
        type=str,
        metavar="OUTPUT",
        help="The format to save the outputs as. Can be ['csv', 'tif', or 'all']. 'csv' mode should be the standard for further analysis, but 'tif' mode can produce videos. 'all' produces outputs in csv and tif formats simultaneously.",
    )
    parser.add_argument(
        "--device",
        required=False,
        default="cpu",
        type=str,
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--chunking-strategy",
        required=False,
        default="safe",
        type=str,
        help="The strategy to use for chunk generation, can be ['fast' or 'safe']. 'fast' mode attempts to prebatch all images together in a single tensor, which prioritizes execution speed at the cost of potentially having heavy RAM use (not VRAM, which is bounded by the --bs argument). 'safe' mode instead relies on lazy loading from the TIFF file, which heavily reduces the RAM requirements for inference at the cost of slower execution. Defaults to 'safe'.",
    )
    parser.add_argument(
        "--temporal-windowing-strategy",
        required=False,
        default="chunking",
        type=str,
        help="Temporal windowing method, can be ['strided', 'chunking', ' averaging', or 'feature_averaging']. Chunking is what was used in the original paper, where the model's train-time temporal size is used as the stride during inference. 'strided' uses the defined --temporal-window-stride argument to define a specific stride. 'feature_averaging' assumes that the model can internally handle variable-length inputs, and will fail if it does not. 'averaging' performs the same functionality as 'chunking', but takes the mean of all chunks as the prediction for each pixel. Defaults to 'chunking'.",
    )
    parser.add_argument(
        "--temporal-stride",
        required=False,
        type=int,
        default=None,
        help="The temporal stride to use when --temporal-windowing-strategy == 'strided'. This allows the user to define a custom stride value that allows for overlapping.",
    )
    parser.add_argument(
        "--spatial-windowing-strategy",
        required=False,
        default="overlap",
        type=str,
        help="Spatial windwoing method. Can be ['no_overlap', 'overlap' or 'pad']. 'overlap' windows the inputs without adding any padding, which will always result in a shrunken output map unless then `model_input_pixels` is set to 1. 'pad' will attempt to pad the model with 'dead' 0-pixels to obtain an output map of equal size to the input map, though this might not be ideal as there is rarely situations where the model has seen 'dead' 0-pixels. Finally, 'no_overlap' sets the spatial stride to be the model input size, sacrificing spatial resolution for faster execution speed. Defaults to 'overlap'.",
    )
    parser.add_argument(
        "--bc-order",
        required=False,
        default=0,
        type=int,
        help="Order of polynomial bleach correction to use. This might be relevant if the source data undergoes bleaching (which most physical systems do). Defaults to 0, where no bleach correction is applied.",
    )
    parser.add_argument(
        "--bc-mode",
        required=False,
        default="full",
        type=str,
        help="Whether bleach correction should be conducted on the full available frame time <full> or on individual slices <part>. Note that <part> mode is only recommended when --chunking-strategy == 'safe' due to the high RAM requirements. Defaults to 'full', where the full frame count is used during bleach correction.",
    )
    parser.add_argument(
        "--bs",
        required=False,
        default=None,
        type=int,
        metavar="BATCH_SIZE",
        help="This defines the batch-size when performing inference. This was deemed necessary as passing in arbitrary batch sizes would easily hit CUDA OOM errors, By default, this is set to None, where the batch size used for training is used, but this is generally not efficient, as inference is done without gradient tracking, and there is a lot more VRAM to work with. This can be set to -1 to use the full batch size of a chunked image, though this will likely cause CUDA out-of-memory issues (a 50k frame 21x21 input to a 2.5k 3x3 model has a batch size of 8820, which requires more than 10GB of VRAM).",
    )

    return parser


def parse_list_of_files(filepath_args: str) -> List[str]:
    list_of_files = []

    for filepath in filepath_args:
        print(filepath)
        if "*" in filepath:
            list_of_files.extend(glob.glob(filepath))
        else:
            list_of_files.append(filepath)

    if len(list_of_files) == 0:
        raise ValueError(
            "Parsed list of files has length of 0. There are no valid files to execute inference on."
        )
    return list_of_files


def setup(config_path: str, cli_overrides: List[str]) -> CfgNode:
    """Initialize the experimental configuration, and return an immutable configuration as per the YACS convention.

    Args:
        config_path (str): Path to the YACS config file.
        cli_overrides (List[str]): CLI overrides for experimental parameters. Should be a list of

    Returns:
        CfgNode: _description_
    """
    cfg = get_default_cfg()

    # Merge overrides from the configuration file and command-line overrides.
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(cli_overrides)

    # Validate the config file settings through assertions defined in classification/config/initialization.py
    validate_cfg(cfg)

    # In this case, ignore the weights specified in the config file.
    # The weights will be loaded based on the specification from the CLI `--ckpt` argument
    cfg.MODEL.WEIGHTS = None

    # Freeze the config file to make it immutable.
    cfg.freeze()

    return cfg


def main(
    cfg: CfgNode,
    ckpt_path: str,
    file_paths: List[str],
    output_folder: str,
    output_format: str,
    device: str,
    chunking_strategy: str,
    temporal_windowing_strategy: str,
    temporal_stride: int,
    spatial_windowing_strategy: str,
    batch_size: int,
    bc_order: int,
    bc_mode: str,
):
    # If active, turn on CUDNN benchmarking.
    if cfg.BACKEND.CUDNN_BENCHMARK:
        cudnn.benchmark = True

    # Loading the checkpoint.
    if ckpt_path is None:
        # If no ckpt_path is provided, default to using the best.ckpt checkpoint.
        # This does assume that the workdir was not moved though.
        ckpt_path = os.path.join(
            cfg.EXPERIMENT.WORKDIR, "checkpoints", "best_model.ckpt"
        )
    ckpt_dict = torch.load(ckpt_path, map_location="cpu")

    # Building the model.
    model = build_model(
        cfg=cfg, output_list=cfg.TASK[cfg.EXPERIMENT.TASK]["TARGETS"], device="cpu"
    )

    # Load the checkpoint state dict.
    model.load_state_dict(ckpt_dict["model_state_dict"], strict=True)
    model.to(device)

    transforms = build_eval_transforms(
        cfg=cfg,
        device=device,
    )

    # The model can be set to eval mode.
    model.eval()

    # Loop through all of the provided files.
    file_paths = parse_list_of_files(file_paths)

    # Create the output folder.
    os.makedirs(output_folder, exist_ok=True)

    for fp in file_paths:
        # Read the tiff file into memory.
        image_stack = tifffile.imread(fp)

        # If requested, initialize the bleach correction module
        # This will be passed into the different inference functions.
        if bc_order > 0:
            poly_bleach_correction = PolynomialBleachCorrectionModule(
                bc_order, cfg.SIMULATION[cfg.SIMULATION.TYPE]["CONSTANTS"]["FRAME_TIME"]
            )
        else:
            poly_bleach_correction = None

        # Forward pass through the model.
        with torch.no_grad():
            # Since our models don't necessarily support the loading of images with arbitrary x- and y-axis lengths, we use our helper functions to perform the slicing and batching.
            if chunking_strategy == "fast":
                # Since the transforms are applied to individual inputs, we need to add support for batching.
                transforms = wrapper_batch_transforms(transforms)

                # Cast the image to a PyTorch-compatible format
                # uint16 of TIFFs are not directly cast-able to PyTorch Tensors.
                # While using bleach correction can help, best to have this safety measure in place.
                # int32 is the best middle ground we have,
                image_stack = image_stack.astype(np.int32)

                if temporal_windowing_strategy in ["chunking", "averaging", "strided"]:
                    output_map = infer_fast(
                        input_arr=image_stack,
                        model=model,
                        batch_size=batch_size,
                        preprocessing_chain=transforms,
                        model_input_pixels=cfg.SIMULATION.CONSTANTS.NUM_PIXELS,
                        temporal_window_size=cfg.SIMULATION.CONSTANTS.FRAMES,
                        temporal_stride=temporal_stride,
                        temporal_windowing_strategy=temporal_windowing_strategy,
                        spatial_windowing_strategy=spatial_windowing_strategy,
                        bleach_correction_module=poly_bleach_correction,
                        bleach_correction_mode=bc_mode,
                    )
                elif temporal_windowing_strategy == "feature_averaging":
                    # For feature averaging, we face some problems where a large image would cause CUDA OOM errors.
                    # Instead of forcing the same batching logic, we can use the same fast inference code which includes batching logic.
                    # The difference here is that we use the number of frames of the image as the temporal window size.
                    # This allows supported models to handle the variable length input, which does not need to match the training size.
                    output_map = infer_fast(
                        input_arr=image_stack,
                        model=model,
                        batch_size=batch_size,
                        preprocessing_chain=transforms,
                        model_input_pixels=cfg.SIMULATION.CONSTANTS.NUM_PIXELS,
                        temporal_window_size=image_stack.shape[0],
                        temporal_stride=temporal_stride,
                        temporal_windowing_strategy=temporal_windowing_strategy,
                        spatial_windowing_strategy=spatial_windowing_strategy,
                        bleach_correction_module=poly_bleach_correction,
                        bleach_correction_mode=bc_mode,
                    )
                else:
                    raise ValueError(
                        f"temporal_windowing_strategy must be in ['strided', 'chunking', 'averaging', or 'feature_averaging']. Got {temporal_windowing_strategy}"
                    )
            elif chunking_strategy == "safe":
                if temporal_windowing_strategy in ["chunking", "averaging", "strided"]:
                    output_map = infer_safe(
                        input_arr=image_stack,
                        model=model,
                        batch_size=batch_size,
                        preprocessing_chain=transforms,
                        model_input_pixels=cfg.SIMULATION.CONSTANTS.NUM_PIXELS,
                        temporal_window_size=cfg.SIMULATION.CONSTANTS.FRAMES,
                        temporal_stride=temporal_stride,
                        temporal_windowing_strategy=temporal_windowing_strategy,
                        spatial_windowing_strategy=spatial_windowing_strategy,
                        bleach_correction_module=poly_bleach_correction,
                        bleach_correction_mode=bc_mode,
                    )
                elif temporal_windowing_strategy == "feature_averaging":
                    # For feature averaging, we face some problems where a large image would cause CUDA OOM errors.
                    # Instead of forcing the same batching logic, we can use the same fast inference code which includes batching logic.
                    # The difference here is that we use the number of frames of the image as the temporal window size.
                    # This allows supported models to handle the variable length input, which does not need to match the training size.
                    output_map = infer_safe(
                        input_arr=image_stack,
                        model=model,
                        batch_size=batch_size,
                        preprocessing_chain=transforms,
                        model_input_pixels=cfg.SIMULATION.CONSTANTS.NUM_PIXELS,
                        temporal_window_size=image_stack.shape[0],
                        temporal_stride=temporal_stride,
                        temporal_windowing_strategy=temporal_windowing_strategy,
                        spatial_windowing_strategy=spatial_windowing_strategy,
                        bleach_correction_module=poly_bleach_correction,
                        bleach_correction_mode=bc_mode,
                    )
                else:
                    raise ValueError(
                        f"temporal_windowing_strategy must be in ['strided', 'chunking', 'averaging', or 'feature_averaging']. Got {temporal_windowing_strategy}"
                    )
            else:
                raise ValueError(
                    f"--chunking-strategy must be in ['fast' or 'safe'], got '{chunking_strategy}'"
                )

            # In the case of averaging, average over the temporal axis.
            if temporal_windowing_strategy == "averaging":
                output_map = np.mean(output_map, axis=0)

            # If target transformations were applied during training (i.e. log, log10 etc), reverse those transformations here.
            reverse_target_transforms = build_reverse_target_transforms(
                cfg.TASK.REGRESSION.TARGET_TRANSFORM
            )
            for class_ind, reverse_target_transform in enumerate(
                reverse_target_transforms
            ):
                output_map[:, class_ind] = reverse_target_transform(
                    output_map[:, class_ind]
                )

            # Reverse the dimensionless parameters if that is the defined parameter space.
            # This makes evaluation a lot easier, as physical parameters are the expected output format.
            if cfg.SIMULATION.TYPE in ["SIM_2D_1P_DIMLESS", "SIM_3D_1P_DIMLESS"]:
                output_map = reverse_dimensionless_conversion(
                    cfg,
                    output_map,
                    regression_targets=cfg.TASK[cfg.EXPERIMENT.TASK]["TARGETS"],
                )

        # Save the output in the desired format.
        process_output_map(
            output_path=os.path.join(
                output_folder, f"{os.path.basename(fp)}.{output_format}"
            ),
            output_map=output_map,
            output_format=output_format,
            taget_params_list=cfg.TASK[cfg.EXPERIMENT.TASK]["TARGETS"],
        )


if __name__ == "__main__":
    # Parse arguments from the CLI.
    args = get_parser().parse_args()
    config_file = args.cfg
    cli_overrides = args.overrides
    ckpt = args.ckpt
    files_args = args.files
    output_folder = args.output_folder
    output_format = args.output_format
    device = args.device
    chunking_strategy = args.chunking_strategy
    temporal_windowing_strategy = args.temporal_windowing_strategy
    temporal_stride = args.temporal_stride
    spatial_windowing_strategy = args.spatial_windowing_strategy

    batch_size = args.bs
    bc_order = args.bc_order
    bc_mode = args.bc_mode

    # Generates the experimental configuration node.
    cfg = setup(config_path=config_file, cli_overrides=cli_overrides)

    if output_format not in ["csv", "tif", "all"]:
        raise ValueError(
            f"--output-format must be in ['csv', 'tif', 'all'], got {output_format}."
        )

    # If the --fast-chunking-batchsize is not defined, default to the training batchsize defined in the config file.
    if not batch_size:
        warnings.warn(
            "A batch size was not specified in --bs. This will default to your training batch size, but note that this is likely inefficient, as inference does not need gradient tracking, meaning you likely have more VRAM headroom than you would during training."
        )
        batch_size = cfg.DATALOADER.PER_STEP_BATCH_SIZE
    # If the value is set to a negative value, set it to -1 to indicate full-batch inference.
    if batch_size < -1:
        batch_size = -1

    # Validate polynomial bleach correction order
    if not isinstance(bc_order, int):
        raise ValueError(
            f"--bc-order must be set to a positive integer. Got {bc_order} of type {type(bc_order)}"
        )
    if bc_order < 0:
        raise ValueError(
            f"--bc-order must be set to a positive integer. Got {bc_order}"
        )
    if bc_mode not in ["full", "part"]:
        raise ValueError(f'--bc-mode must be in ["full", "part"]. Got {bc_mode}')

    # Validate potential conflicts, or raise warnings.
    if bc_mode == "part":
        if chunking_strategy != "safe":
            warnings.warn(
                "--bc-mode is defined as 'part', but --chunking-strategy was selected as 'fast'. This might cause heavy use of system memory due to the vectorized implementation of bleach correction. It is strongly recommended that 'safe' chunking is used instead."
            )
    if temporal_stride is not None:
        if temporal_windowing_strategy != "strided":
            raise ValueError(
                "A custom temporal stride --temporal-stride is only supported when --temporal-windowing-strategy == 'strided'."
            )
    else:
        if temporal_windowing_strategy == "strided":
            raise ValueError(
                "temporal_stride is None, but temporal_windowing_strategy is 'strided'. Define a temporal_stride using --temporal-stride N, where N is your desired stride value."
            )

    if temporal_windowing_strategy == "strided" and chunking_strategy == "fast":
        warnings.warn(
            "--chunking-strategy is set to 'fast', and --temporal-windowing-strategy is set to 'strided'. This could cause issues with high RAM use on large image stacks with high frame/pixel counts. Using --chunking-strategy == 'safe' is strongly recommended for inference with custom strides."
        )

    # Default output_folder to the cwd with an appended datetime string.
    if output_folder is None:
        output_folder = f"./outputs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Conduct the inference loop
    main(
        cfg=cfg,
        ckpt_path=ckpt,
        file_paths=files_args,
        output_folder=output_folder,
        output_format=output_format,
        device=device,
        chunking_strategy=chunking_strategy,
        temporal_windowing_strategy=temporal_windowing_strategy,
        temporal_stride=temporal_stride,
        spatial_windowing_strategy=spatial_windowing_strategy,
        batch_size=batch_size,
        bc_order=bc_order,
        bc_mode=bc_mode,
    )
