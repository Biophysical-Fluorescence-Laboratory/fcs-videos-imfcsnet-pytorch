"""Collection of transformations and data augmentation techniques.

In the context of ImFCSNet and its derivatives, input transforms and augmentations generally include simulated noise and normalization techniques.

Helper functionality is also included:
- Wrapper for batch inputs.
- Target transformations (log targets) and their reverse transforms.
"""

import torch
import numpy as np
from torchvision.transforms import v2 as T

from imfcs_pytorch.data.transforms.preprocessing.tensors import (
    ToTorchTensor,
    SendToDevice,
)
from imfcs_pytorch.data.transforms.preprocessing.normalization import (
    PerStackZScoreNormalization,
    PerStackMinMaxNormalization,
    PerStackZeroFloorNormalization,
)
from imfcs_pytorch.data.transforms.augmentations.noise import (
    AddGaussianNoise,
    AddEMCCDNoise,
)

# Typing-specific imports
from typing import List
from yacs.config import CfgNode
from imfcs_pytorch.data.transforms.base import Transform


def build_default_transforms(device: str) -> List[Transform]:
    """Basic function to generate the default transforms. These are the minimal requirements to conduct forward passes.

    The default chain casts the image stack into a PyTorch float tensor, and sends it to the correct device.

    Args:
        device (str): Device to send the PyTorch Tensor to. This should be the same device that the network model is loaded on.

    Returns:
        List[Transform]: List of transforms. Note that this will need to be wrapped in T.Compose() to properly work.
    """
    return [
        ToTorchTensor(),
        SendToDevice(device=device),
    ]


def build_noise_augmentations(cfg: CfgNode, device: str) -> List[Transform]:
    """Helper function to create the noise augmentations.

    Args:
        cfg (CfgNode): YACS config object.
        device (str): Device to use for random noise sampling.

    Returns:
        List[Transform]: List of noise augmentations. Note that this will need to be wrapped in T.Compose() to properly work. Lists are returned here since the 'mix' case includes 2-sequential transforms.
    """
    # ["gaussian", "emccd", "mix", "random"]
    if cfg.TRANSFORMS.TRAIN.NOISE.TYPE == "gaussian":
        return [
            AddGaussianNoise(
                cfg.TRANSFORMS.TRAIN.NOISE.GAUSSIAN.SCALE_MIN,
                cfg.TRANSFORMS.TRAIN.NOISE.GAUSSIAN.SCALE_MAX,
                device=device,
            )
        ]
    elif cfg.TRANSFORMS.TRAIN.NOISE.TYPE == "emccd":
        # Parse the EMCCD PMF file
        # This is basically just the probability mass function, with the minimum value appended to the end.
        emccd_arr = np.load(cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.PMF_FILE)
        emccd_pmf = emccd_arr[:-1]
        emccd_min = emccd_arr[-1]

        return [
            AddEMCCDNoise(
                emccd_pdf=emccd_pmf,
                emccd_min=emccd_min,
                scale_emccd_noise=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.DO_SCALING,
                emccd_noise_scale_min=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.SCALE_MIN,
                emccd_noise_scale_max=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.SCALE_MAX,
                device=device,
            )
        ]
    elif cfg.TRANSFORMS.TRAIN.NOISE.TYPE == "mix":
        # In the mix case, use both Gaussian AND simulated EMCCD Noise.

        # Parse the EMCCD PMF file
        # This is basically just the probability mass function, with the minimum value appended to the end.
        emccd_arr = np.load(cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.PMF_FILE)
        emccd_pmf = emccd_arr[:-1]
        emccd_min = emccd_arr[-1]

        return [
            AddGaussianNoise(
                cfg.TRANSFORMS.TRAIN.NOISE.GAUSSIAN.SCALE_MIN,
                cfg.TRANSFORMS.TRAIN.NOISE.GAUSSIAN.SCALE_MAX,
                device=device,
            ),
            AddEMCCDNoise(
                emccd_pdf=emccd_pmf,
                emccd_min=emccd_min,
                scale_emccd_noise=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.DO_SCALING,
                emccd_noise_scale_min=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.SCALE_MIN,
                emccd_noise_scale_max=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.SCALE_MAX,
                device=device,
            ),
        ]
    elif cfg.TRANSFORMS.TRAIN.NOISE.TYPE == "random":
        # In the random case, randomly apply Gaussian OR simulated EMCCD noise.
        # We can achieve this using random-choice

        # Parse the EMCCD PMF file
        # This is basically just the probability mass function, with the minimum value appended to the end.
        emccd_arr = np.load(cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.PMF_FILE)
        emccd_pmf = emccd_arr[:-1]
        emccd_min = emccd_arr[-1]

        return [
            T.RandomChoice(
                [
                    AddGaussianNoise(
                        cfg.TRANSFORMS.TRAIN.NOISE.GAUSSIAN.SCALE_MIN,
                        cfg.TRANSFORMS.TRAIN.NOISE.GAUSSIAN.SCALE_MAX,
                        device=device,
                    ),
                    AddEMCCDNoise(
                        emccd_pdf=emccd_pmf,
                        emccd_min=emccd_min,
                        scale_emccd_noise=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.DO_SCALING,
                        emccd_noise_scale_min=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.SCALE_MIN,
                        emccd_noise_scale_max=cfg.TRANSFORMS.TRAIN.NOISE.EMCCD.SCALE_MAX,
                        device=device,
                    ),
                ],
                p=[
                    cfg.TRANSFORMS.TRAIN.NOISE.MIX.GAUSSIAN_PROB,
                    1.0 - cfg.TRANSFORMS.TRAIN.NOISE.MIX.GAUSSIAN_PROB,
                ],
            )
        ]
    else:
        raise ValueError(
            f'Invalid TRANSFORMS.TRAIN.NOISE.TYPE {cfg.TRANSFORMS.TRAIN.NOISE.TYPE}. Should be in ["gaussian", "emccd", "mix" or "random"].'
        )


def build_normalization_transform(cfg: CfgNode, method_identifier: str) -> Transform:
    """Helper function to build the normalization transformations.

    Any new normalization techniques should be added into this function. This ensures that the same normalization will be applied to both training and inference regimes.

    At this stage, all normalization techniques do not take in any inputs, though this could be extended to access parameters defined in the `cfg` input argument, which is not currently in-use. For example, applying a pre-defined mean/std normalization method, as opposed to normalizing per-input.

    Args:
        method_identifier (str): Normalization method name string.

    Raises:
        ValueError: Invalid normalization method identifier string.

    Returns:
        Transform: The normalization transform. Should be incorporated into a T.Compose chain for propert functionality.
    """
    # Dictionary associating namestrings with the normalization methods.
    # Any new normalization methods should be added here.
    method_identifier_dict = {
        "zscore": PerStackZScoreNormalization(),
        "minmax": PerStackMinMaxNormalization(),
        "zerofloor": PerStackZeroFloorNormalization(),
    }

    if method_identifier in method_identifier_dict.keys():
        return method_identifier_dict[method_identifier]
    else:
        raise ValueError(
            f"{method_identifier} is not a valid normalization method. Valid normalization methods are {method_identifier_dict.keys()}"
        )


def build_train_transforms(
    cfg: CfgNode,
    device: str,
    disable_flag: bool,
) -> T.Compose:
    """Builder function to create the train-time transform chain.

    This includes casting functions, simulated noise (if requested) and normalization (if requested).

    Args:
        cfg (CfgNode): YACS config object holding the transform setting parameters.
        device (str): Device to use for tensor operations.
        disable_flag (bool): Debug option to deactivate all augmentations except the default required casting operations.

    Returns:
        T.Compose: Composed list of augmentations to be applied on each element of a batch (C, T, H, W) (for example as a Dataset transform).
    """
    # Add default transforms.
    transform_chain = [
        ToTorchTensor(),
        SendToDevice(device=device),
    ]

    # For debugging purposes, we can deactivate the non-default augmentations
    if not disable_flag:
        # Add data augmentations.
        # Noise augmentations.
        if cfg.TRANSFORMS.TRAIN.NOISE.TYPE is not None:
            transform_chain.extend(build_noise_augmentations(cfg=cfg, device=device))

        # Normalization will be applied at the end.
        if cfg.TRANSFORMS.UNIVERSAL.NORMALIZATION is not None:
            transform_chain.append(
                build_normalization_transform(
                    cfg, cfg.TRANSFORMS.UNIVERSAL.NORMALIZATION
                )
            )

        transform_chain = T.Compose(transform_chain)

    return transform_chain


def build_eval_transforms(cfg: CfgNode, device: str) -> T.Compose:
    """Builder function to create the test-time transform chain. Note that this is designed to be applied to each element, and not over entire batches. If this is intended to be applied elementwise on a batch, wrap it with wrapper_batch_transforms() to ensure that normalizations are not applied over the entire batch.

    This differs from the train-time transforms in that no data augmentations are added.

    This includes casting functions and normalization (if requested).

    Args:
        cfg (CfgNode): YACS config object holding the transform setting parameters.
        device (str): Device to use for tensor operations.

    Returns:
        T.Compose: Composed list of augmentations to be applied on each input (C, T, H, W) (i.e. as a Dataset transform). Wrap in wrapper_batch_transforms() if this should be applied elementwise over a batch.
    """
    # Add default transforms.
    transform_chain = [
        ToTorchTensor(),
        SendToDevice(device=device),
    ]

    # Normalization will be applied at the end.
    if cfg.TRANSFORMS.UNIVERSAL.NORMALIZATION is not None:
        transform_chain.append(
            build_normalization_transform(cfg, cfg.TRANSFORMS.UNIVERSAL.NORMALIZATION)
        )

    transform_chain = T.Compose(transform_chain)

    return transform_chain


def wrapper_batch_transforms(transform_chain: T) -> T:
    """_summaHelper function to wrap the transform chain (generally written for single-inputs) for the batch-case. In other words, converts a transform chain designed for (C, T, W, H) to work on (BATCH_SIZE, C, T, W, H).

    The transformation chains built by `build_eval_transforms` and `build_train_transforms` follow Torchvision's convention, where they are designed for application over individual samples, rather than over batches. This breaks if you are trying to apply the transforms element-wise over a pre-built batch. This wrapper function ensures that the transformations are not applied over the entire batch, but rather applied per-sample.

    For example, if Min-Max (0-1) normalization was applied to a full-batch, while the batch is guaranteed to be in [0, 1], each sample does not share the same guarantee.

    This will likely be used during inference, where inputs are batched for accelerated inference throughput.ry_

    Args:
        transform_chain (T): Transformation chain wrapped by T.Compose. Generally built through `build_eval_transforms` or `build_train_transforms`.

    Returns:
        T: Modified transform chain to work on batches, ensuring that transforms are applied element-wise to each input sample.
    """
    return T.Lambda(
        lambda input_batch: torch.stack(
            [transform_chain(sample) for sample in input_batch]
        )
    )


def build_target_transforms(transform_name_list: List[str]) -> List[callable]:
    """Helper function to produce a list of transforms for targets.

    Initally written for the case of physical simulations and D-regression, as the large range of targets might be a bit un-tenable for the MSELoss, particularly when coupled with log-uniform sampling.

    Args:
        transform_name_list (List[str]): List of transforms, should be in [None, "log" and "log10"].

    Returns:
        List[callable]: List of callable functions, should include [lambda x:x, torch.log, torch.log10]
    """
    # Otherwise, construct the callable transforms list.
    callable_transforms = []

    for transform_name_string in transform_name_list:
        if transform_name_string == None:
            callable_transforms.append(lambda x: x)
        elif transform_name_string == "log":
            callable_transforms.append(torch.log)
        elif transform_name_string == "log10":
            callable_transforms.append(torch.log10)
        else:
            raise ValueError(
                f"Unknown transform_name_string {transform_name_string}. Must be [None, 'log' or 'log10']"
            )
    return callable_transforms


def build_reverse_target_transforms(transform_name_list: List[str]) -> List[callable]:
    """Helper function to produce a list of transforms for targets, reversing the transformations made during training time into values in the natural scale they were defined in.

    Initally written for the case of physical simulations and D-regression, as the large range of targets might be a bit un-tenable for the MSELoss, particularly when coupled with log-uniform sampling.

    Args:
        transform_name_list (List[str]): List of transforms, should be in [None, "log" and "log10"].

    Returns:
        List[callable]: List of callable functions, should include [lambda x:x, torch.log, torch.log10]
    """
    # Construct the callable transforms list.
    callable_reverse_transforms = []

    for transform_name_string in transform_name_list:
        if transform_name_string == None:
            callable_reverse_transforms.append(lambda x: x)
        elif transform_name_string == "log":
            callable_reverse_transforms.append(np.exp)
        elif transform_name_string == "log10":
            callable_reverse_transforms.append(lambda x: 10**x)
        else:
            raise ValueError(
                f"Unknown transform_name_string {transform_name_string}. Must be [None, 'log' or 'log10']"
            )
    return callable_reverse_transforms
