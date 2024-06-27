import torch
from imfcs_pytorch.model.imfcsnet.imfcsnet import build_imfcsnet
from imfcs_pytorch.model.basic.histogram import build_histogram_of_features
from imfcs_pytorch.model.basic.linreg import build_linear_regression

# Typing-specific imports
from torch import nn
from typing import List
from yacs.config import CfgNode


def build_model(cfg: CfgNode, output_list: List[str], device: str) -> nn.Module:
    """Wrapper function that defines the model based on the provided config file. Sends the model to the correct device, and loads pretrained weights if defined under MODEL.WEIGHTS.

    Any new models should be defined in cfg.MODEL, and their definitions should be added in this function, ensuring that interaction of the config file is only limited to the scope of the builder function. This ensures that models cannot arbitrarily change during training or inference.

    Args:
        cfg (CfgNode): YACS config object defining the experiment state.
        output_list (List[str]): List of output names, generally used to initialize the number of output neurons.
        device (str): Device to load model on. Model is sent to the correct device early on, as resuming checkpoints with optimizer states might fail if the model is not on matching device types.

    Returns:
        nn.Module: Built PyTorch model object.
    """
    if cfg.MODEL.NAME.lower() == "imfcsnet":
        model = build_imfcsnet(
            filter_channels=cfg.MODEL.IMFCSNET.FILTER_CHANNELS,
            spatial_agg_block_kernel_size=cfg.MODEL.IMFCSNET.SPATIAL_AGG_BLOCK_KERNEL_SIZE,
            strided_conv_layer_kernel_size=cfg.MODEL.IMFCSNET.STRIDED_CONV_LAYER_KERNEL_SIZE,
            strided_conv_layer_filter_stride=cfg.MODEL.IMFCSNET.STRIDED_CONV_LAYER_FILTER_STRIDE,
            conv1d_group_stages=cfg.MODEL.IMFCSNET.CONV1D_GROUP_STAGES,
            conv1d_group_blocks_per_stage=cfg.MODEL.IMFCSNET.CONV1D_GROUP_BLOCKS_PER_STAGE,
            conv1d_group_filter_size=cfg.MODEL.IMFCSNET.CONV1D_GROUP_FILTER_SIZE,
            dense_mixing_num_stages=cfg.MODEL.IMFCSNET.DENSE_MIXING_NUM_STAGES,
            dense_mixing_blocks_per_stage=cfg.MODEL.IMFCSNET.DENSE_MIXING_BLOCKS_PER_STAGE,
            output_neurons=len(output_list),
            use_original_init=cfg.MODEL.IMFCSNET.USE_ORIGINAL_WEIGHT_INIT,
        )
    elif cfg.MODEL.NAME.lower() == "linreg":
        model = build_linear_regression(output_neurons=len(output_list))
    elif cfg.MODEL.NAME.lower() == "histogram":
        model = build_histogram_of_features(
            histogram_bins=cfg.MODEL.HISTOGRAM.BINS, output_neurons=len(output_list)
        )
    else:
        raise ValueError(f"Invalid MODEL.NAME {cfg.MODEL.NAME}.")

    # Send model to device.
    # This needs to happen before loading the state dictionaries.
    # There are some oddities when following the general convention of map_location="cpu".
    # See: https://discuss.pytorch.org/t/loading-a-model-runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least-two-devices-cuda-0-and-cpu/143897/9
    model.to(device)

    # After building the model, attempt to load weights if specified.
    if cfg.MODEL.WEIGHTS is not None:
        # Load the checkpoint.
        ckpt = torch.load(cfg.MODEL.WEIGHTS, map_location=device)

        # We assume that the `model_state_dict` key exists in the checkpoint, and that convention should be obeyed.
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    return model
