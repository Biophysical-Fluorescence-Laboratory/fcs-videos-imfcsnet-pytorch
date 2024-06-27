"""Helper functions for inference.

Since image-stack based models might not be directly applicable to images with arbitrary x- and y-axis lengths (10x21 image stacks), or even support arbitrary temporal dimension lengths (50k frames for a model trained on 2.5k frames), we need these helper functions to help simplify the inference process.

Ideally, any new models should be able to work within this framework.
"""

import torch
from torch import nn
import numpy as np

from torchvision.transforms import v2 as T
from torch.utils.data import Dataset, DataLoader

# Typing-specific imports
from typing import List
from imfcs_pytorch.data.transforms.preprocessing.bleach_correction.polynomial import (
    PolynomialBleachCorrectionModule,
)


def pad_input(
    input_arr: np.ndarray,
    model_input_pixels: int = 3,
    padding_constant_vals: int = 0,
) -> np.ndarray:
    """Pad the input such that the sliding window inference does not change in resolution.

    Since the original ImFCSNet was designed for 3x3 inputs, this meant we would always lose resolution when performing sliding window inference (called 'overlap' in the original paper).

    This function performs padding to retain the resolution, though we likely need to accept that boundary values will be inherently wrong due to the constant padding that is performed by default.

    Would be interesting to see how other padding methods might help (https://arxiv.org/abs/2010.02178)

    Args:
        input_arr (np.ndarray): Input array to perform padding on.
        model_input_pixels (int, optional): The number of input pixels-per-side the model expects. Defaults to 3.
        padding_constant_vals (int, optional): The constant value to use for padding. Defaults to 0.

    Returns:
        np.ndarray: Padded variant of the input array.
    """
    # After extracting the original shape to be used as the output, pad the input to retain this shape.
    padding_width = (model_input_pixels - 1) // 2
    return np.pad(
        input_arr,
        pad_width=(
            (0, 0),
            (padding_width, padding_width),
            (padding_width, padding_width),
        ),
        mode="constant",
        constant_values=padding_constant_vals,
    )


def infer_fast(
    input_arr: np.ndarray,
    model: nn.Module,
    batch_size: int,
    temporal_windowing_strategy: str,
    temporal_stride: int = None,
    preprocessing_chain: callable = None,
    model_input_pixels: int = 3,
    spatial_windowing_strategy: str = "overlap",
    temporal_window_size: int = 2500,
    bleach_correction_module: PolynomialBleachCorrectionModule = None,
    bleach_correction_mode: str = "full",
) -> np.ndarray:
    """A faster implementation of the inference logic, relying on batching to perform faster inference with minimal blocking operations.

    The original `infer_on_stack` method was a bit slow since model inference was done on an input-by-input basis, which is far from efficient. Instead, we can leverage batching to parallize the inference workloads.

    This implementation is triggered when `fast_chunking` is requested as the spatial windowing strategy, and batches the inputs effectively for faster inference. We observed a 25-35% speedup in inference by applying this approach, with no change in model predictions.

    Args:
        input_arr (np.ndarray): Input array with arbitrary temporal and spatial dimensions. This function will handle the spatial and temporal windowing dynamically to produce a valid output map.
        model (nn.Module): Model to use for inference. Note that the temporal windowing stride will need to be set according to the model specifications.
        batch_size (int): Batch size to use during inference. Passing the whole batch is generally not viable for large inputs (whether over the temporal or spatial axis). Note that the use of batches during inference incurs a corresponding VRAM cost if GPUs are used. In this case, we default to using the training batch size during inference, but this can be scaled up or down depending on your available CUDA compute resources.
        temporal_windowing_strategy (str): Temporal windowing method, can be ['fast_chunking' or 'feature_averaging'].'feature_averaging' assumes that the model can internally handle variable-length inputs. 'fast_chunking' represents the case where we pre-batch the chunks into a large payload, which is faster at the cost of more GPU VRAM use. Defaults to 'chunking'.
        temporal_stride (int, optional): The stride to use over the temporal FRAMES dimension. Only used when temporal_windowing_strategy == "strided", which allows for custom strides to be assigned. Defaults to None.
        preprocessing_chain (callable, optional): Transforms chain, generally includes casting to float(), normalization, and sending to specified PyTorch device. Defaults to None.
        model_input_pixels (int, optional): The model input size in pixels-per-side. Used for handling the windowing process. Defaults to 3.
        spatial_windowing_strategy (str, optional):Spatial windwoing method. Can be ['overlap' or 'pad']. 'overlap' windows the inputs without adding any padding, which will always result in a shrunken output map unless then `model_input_pixels` is set to 1. 'pad' will attempt to pad the model with 'dead' 0-pixels to obtain an output map of equal size to the input map, though this might not be ideal as there is rarely situations where the model has seen 'dead' 0-pixels. Defaults to 'overlap'.
        temporal_window_size (int, optional): Size of the temporal window over the FRAMES dimension. Should be equivalent to the model's training frame count, unless the model inherently allows for variable input lengths. Defaults to 2500.
        temporal_windowing_stride (int, optional): The temporal windowing stride, used in the case where `temproal_windowing_strategy == "chunking"`. This should be set to the temporal size (or number of frames) used during train-time. Defaults to 2500.

    Raises:
        ValueError: Invalid input variables.

    Returns:
        np.ndarray: Output map.
    """
    # Validating input parameters
    # For temporal windowing, the Java code takes every 2500 chunk, computes the prediction, then averages all predictions.
    # Meanwhile, the original code *might* just use the full 50,000 frames and compute the D directly due to the averaging layer.
    # For now, we allow this to be toggled using an input argument.
    # This also allows for comparisons in the future.
    if temporal_windowing_strategy not in [
        "strided",
        "chunking",
        "averaging",
        "feature_averaging",
    ]:
        raise ValueError(
            f"temporal_windowing_streategy must be 'fast_chunking' or 'feature_averaging'. Got {temporal_windowing_strategy}."
        )

    # Check temporal stride settings.
    # A temporal stride is only allowed in 'strided' mode.
    if temporal_stride is not None:
        if temporal_windowing_strategy != "strided":
            raise ValueError(
                f"temporal_stride is set to {temporal_stride}, but custom strides are only supported when temporal_windowing_strategy is 'strided'. Got temporal_windowing_strategy == '{temporal_windowing_strategy}'."
            )
    else:
        if temporal_windowing_strategy == "strided":
            raise ValueError(
                "temporal_stride is None, but temporal_windowing_strategy is 'strided'. Define a temporal_stride using --temporal-stride N, where N is your desired stride value."
            )
        # If 'strided' mode is not selected, default to usingt he model's FRAME dimension length during training.
        temporal_stride = temporal_window_size

    # For spatial windowing, 'overlap' is the default because using a 3x3 input and assuming the central prediction means we need to slice off the outer edges by 1 pixel each. This appears to be the strategy used in the paper, as the scale bars for NLS are shorter than those for ImFCSNet, and the pixel counts are lowered by 2 (1 for each side).
    # However, if we wanted to ensure the same values, we could pad the outer pixels with constant zeros. This likely won't work as the model has never been trained to deal with 'dead' pixels, but we can try.
    if spatial_windowing_strategy not in ["no_overlap", "overlap", "pad"]:
        raise ValueError(
            f"spatial windowing strategy must be in ['no_overlap', 'overlap', 'pad']. Got {spatial_windowing_strategy}."
        )

    # For full bleach correction, apply bleach correction on the full stack.
    if (bleach_correction_mode == "full") and (bleach_correction_module is not None):
        input_arr = bleach_correction_module.correct_full_stack(input_arr)
    # Cast to float32, which is the format required
    input_arr = input_arr.astype(np.float32)

    # This inference code is inspired by the code used for the Java ONNX implementation, as the original code never shared its inference pipeline, only the training pipeline. (https://github.com/ImagingFCS/Imaging_FCS_1_62/blob/main/java/imfcs/Imaging_FCS_1_62.java#L21634)
    # At the very least, the Java implementation was endorsed by the original implementer as being correct, so it is as good a starting point as we're going to get.
    # For fast_chunking, we hand-code a lot of the logic to allow for faster inference.
    # Start by doing padding if requested
    if spatial_windowing_strategy == "pad":
        input_arr = pad_input(input_arr, model_input_pixels=model_input_pixels)
        spatial_stride = 1
    elif spatial_windowing_strategy == "overlap":
        spatial_stride = 1
    elif spatial_windowing_strategy == "no_overlap":
        spatial_stride = model_input_pixels

    input_arr = torch.from_numpy(input_arr)  # [10000, 20, 10]

    # Start by doing temporal chunking.
    input_arr = input_arr.unfold(0, temporal_window_size, temporal_stride)
    # Rearrange so the temporal batch dimension is at the first dim.
    input_arr = torch.permute(input_arr, (0, -1, 1, 2))  # [4, 2500, 20, 10]

    # Next, perform spatial chunking.
    input_arr = input_arr.unfold(-2, model_input_pixels, spatial_stride)
    input_arr = torch.permute(input_arr, (0, 2, 1, 4, 3))
    input_arr = input_arr.unfold(-1, model_input_pixels, spatial_stride)
    input_arr = torch.permute(input_arr, (0, 1, 4, 2, 3, 5))

    # shape: torch.Size([20, 19, 19, 2500, 3, 3])

    # At this stage, should have an array of shape: (temporal_chunk, x_chunk, y_chunk, 2500, 3, 3)
    # Now, just need to perfom inference, and remap to the output map.
    # Cache the shape to be used later for reversing the reshaping.
    _shape = input_arr.shape
    input_arr = input_arr.reshape(
        -1, temporal_window_size, model_input_pixels, model_input_pixels
    )  # torch.Size([7220, 2500, 3, 3])

    # For part-based bleach correction, do bleach correction on each slice.
    # Abuse notation to apply the inputs to each pixel in each batch.
    # Since the bleach correction module expects an input in shape (frames, width, height)
    # Given our inputs in the shape (total_batches, frames, width, height)
    # We can force cast our batched inputs into the shape (frames, total_batches, width*height)
    # Then, we can reverse this to cast the bleach corrected stacks to (total_batches, frames, width, height)
    if (bleach_correction_mode == "part") and (bleach_correction_module is not None):
        # Bleach correction can then be applied to each pixel in each batch.
        # This works since the only requirement is for the frames-dimension to remain unchanged.
        # This has unconstrained RAM use, which might be an issue when doing the polynomial fits.
        # Instead, prefer batching, which is safer.
        if batch_size == -1:
            # Do full-batch bleach correction, has unconstrained RAM use, and hung a compute node with 252 GB RAM
            # reshape model_input_batch from (total_batches, frames, width, height) to (frames, total_batches, width*height)
            _total_batches, _frames, _width, _height = input_arr.size()
            input_arr = input_arr.reshape((_frames, _total_batches, _width * _height))

            input_arr = torch.from_numpy(
                bleach_correction_module.correct_full_stack(input_arr.numpy())
            )

            # Finally, reverse the reshaping to obtain the original dimensions.
            input_arr = input_arr.reshape(_total_batches, _frames, _width, _height)
        else:
            # Do bleach correction batch-by-batch. Definitely slower, but at least has constrained RAM use.
            input_arr = torch.split(input_arr, split_size_or_sections=batch_size, dim=0)
            for sub_batch in input_arr:
                _total_batches, _frames, _width, _height = sub_batch.size()
                sub_batch = sub_batch.view((_frames, _total_batches, _width * _height))
                # Apply transform chain to the model input.
                sub_batch = torch.from_numpy(
                    bleach_correction_module.correct_full_stack(sub_batch.numpy())
                )
                # Finally, reverse the reshaping to obtain the original dimensions.
                sub_batch = sub_batch.view(_total_batches, _frames, _width, _height)
            # Re-cast the bleach corrected parts into a single batch
            input_arr = torch.cat(input_arr, dim=0)

    # Perform inference and write to our output_map
    with torch.no_grad():
        # If the fast_chunking_bs value is -1, do full-batch inference.
        if batch_size == -1:
            # Apply transform chain to the model input.
            input_arr = preprocessing_chain(input_arr)  # torch.Size([-1, 2500, 3, 3])

            output_map = model(input_arr.float().unsqueeze(1))

            # Since the code expects a numpy array, we cast accordingly.
            output_map = output_map.detach().cpu().numpy()
        else:
            # Otherwise, this means a valid chunking batch size is defined.
            # As such, we split the full-batch into smaller sub-batches.
            # This is primarily motivated by VRAM limits.
            output_map = []  # Track outputs in a list to concat
            sub_batches = torch.split(
                input_arr, split_size_or_sections=batch_size, dim=0
            )
            for sub_batch in sub_batches:
                # Apply transform chain to the model input.
                sub_batch = preprocessing_chain(
                    sub_batch
                )  # torch.Size([N, 2500, 3, 3])

                # Append the sub-batch outputs into the tracking list.
                # Since VRAM is the concern, we cast these sub-batches into detached CPU-tensors.
                output_map.append(model(sub_batch.float().unsqueeze(1)).detach().cpu())

            # Now, merge the sub-output-batches into a full output_map
            output_map = torch.cat(output_map, dim=0)

            # Cast to NumPy array
            output_map = output_map.numpy()

    # Reshape the output map to the original shape
    _output_shape = (*_shape[:3], -1)
    output_map = output_map.reshape(_output_shape)  # torch.Size([20, 19, 19, 1])

    # Finally, to match the expectations of the original code
    # Dimensions need to be (T, C, H, W), where T is the temporal chunk number, and C is the class index.
    output_map = np.moveaxis(output_map, [3, 1, 2], [1, 2, 3])

    return output_map


class ChunkLazyLoaderDataset(Dataset):
    def __init__(
        self,
        input_arr: np.ndarray,
        temporal_window_size: int,
        temporal_stride: int,
        spatial_x_window_size: int,
        spatial_x_window_stride: int,
        spatial_y_window_size: int,
        spatial_y_window_stride: int,
        bleach_correction_module: PolynomialBleachCorrectionModule,
        transforms: T.Compose,
    ):
        # For lazy loading purposes, calculate the total number of chunks mathematically.
        # This allows us to create the output map by using the indices of each chunk-input/chunk-prediction pair.
        frames, height, width = input_arr.shape
        self.temporal_chunk_count = (
            frames - temporal_window_size
        ) // temporal_stride + 1
        self.height_chunk_count = (
            height - spatial_x_window_size
        ) // spatial_x_window_stride + 1
        self.width_chunk_count = (
            width - spatial_y_window_size
        ) // spatial_y_window_stride + 1

        self.total_chunk_count = (
            self.temporal_chunk_count * self.height_chunk_count * self.width_chunk_count
        )

        # Once the calculations are completed, store a reference to the image stack.
        self.image_stack = input_arr

        # Also store the parameters required to implement sliding window.
        self.temporal_window_size = temporal_window_size
        self.temporal_stride = temporal_stride
        self.spatial_x_window_size = spatial_x_window_size
        self.spatial_x_window_stride = spatial_x_window_stride
        self.spatial_y_window_size = spatial_y_window_size
        self.spatial_y_window_stride = spatial_y_window_stride

        self.transforms = transforms
        self.bleach_correction_module = bleach_correction_module

    def __len__(self):
        return self.total_chunk_count

    def __getitem__(self, chunk_idx: int):
        # Compute the bounds of each chunk.
        # The chunks are indexed in the order: t, x, y.
        # For example, if the input array is 1x3x3:
        # [[[1, 2, 3]
        #   [4, 5, 6]
        #   [7, 8, 9]]]

        # Compute the indices for temporal, spatial x, and spatial y dimensions
        temporal_idx = chunk_idx // (self.height_chunk_count * self.width_chunk_count)
        spatial_idx = chunk_idx % (self.height_chunk_count * self.width_chunk_count)
        spatial_x_idx = spatial_idx // self.width_chunk_count
        spatial_y_idx = spatial_idx % self.width_chunk_count

        # Compute the start and end indices for each dimension
        t_start = temporal_idx * self.temporal_stride
        t_end = t_start + self.temporal_window_size
        x_start = spatial_x_idx * self.spatial_x_window_stride
        x_end = x_start + self.spatial_x_window_size
        y_start = spatial_y_idx * self.spatial_y_window_stride
        y_end = y_start + self.spatial_y_window_size

        # Extract the chunk from the input array
        chunk = self.image_stack[t_start:t_end, x_start:x_end, y_start:y_end]

        # For --bc-mode == 'part', run bleach correction on each chunk.
        if self.bleach_correction_module is not None:
            chunk = self.bleach_correction_module.correct_full_stack(chunk)

        if self.transforms is not None:
            chunk = self.transforms(chunk)

        return chunk

    def predictions_to_output_map(self, predictions: List[torch.Tensor]):
        num_classes = predictions[0].shape[0]

        # Initialize an empty array to store the 3D image
        output_map = np.zeros(
            (
                self.temporal_chunk_count,
                num_classes,
                self.height_chunk_count,
                self.width_chunk_count,
            )
        )

        # Iterate over each chunk index
        for chunk_idx, pred in enumerate(predictions):
            # Compute the indices for temporal, spatial x, and spatial y dimensions
            temporal_idx = chunk_idx // (
                self.height_chunk_count * self.width_chunk_count
            )
            spatial_idx = chunk_idx % (self.height_chunk_count * self.width_chunk_count)
            spatial_x_idx = spatial_idx // self.width_chunk_count
            spatial_y_idx = spatial_idx % self.width_chunk_count

            # Assign the prediction to the corresponding position in the 3D image
            output_map[temporal_idx, :, spatial_x_idx, spatial_y_idx] = pred

        return output_map

    def predictions_to_attention_maps(
        self, predicted_attention_list: List[torch.Tensor]
    ):
        num_heads = predicted_attention_list[0].shape[0]

        # Initialize an empty array to store the 3D image
        output_map = np.zeros(
            (
                self.temporal_chunk_count,
                num_heads,
                self.height_chunk_count,
                self.width_chunk_count,
            )
        )

        # Iterate over each chunk index
        for chunk_idx, pred in enumerate(predicted_attention_list):
            # Compute the indices for temporal, spatial x, and spatial y dimensions
            temporal_idx = chunk_idx // (
                self.height_chunk_count * self.width_chunk_count
            )
            spatial_idx = chunk_idx % (self.height_chunk_count * self.width_chunk_count)
            spatial_x_idx = spatial_idx // self.width_chunk_count
            spatial_y_idx = spatial_idx % self.width_chunk_count

            # Assign the prediction to the corresponding position in the 3D image
            output_map[temporal_idx, :, spatial_x_idx, spatial_y_idx] = np.mean(
                pred, axis=-1
            )

        return output_map


def build_safe_chunking_dataset(
    input_arr: np.ndarray,
    temporal_stride: int = None,
    preprocessing_chain: callable = None,
    model_input_pixels: int = 3,
    spatial_windowing_strategy: str = "overlap",
    temporal_window_size: int = 2500,
    bleach_correction_module: PolynomialBleachCorrectionModule = None,
    bleach_correction_mode: str = "full",
) -> ChunkLazyLoaderDataset:
    """Build the lazy loader dataset for safe inference.

    This function was split out as it can be reused for different features, such as attention map visualizations.

    Args:
        input_arr (np.ndarray): Input array with arbitrary temporal and spatial dimensions. This function will handle the spatial and temporal windowing dynamically to produce a valid output map.
        temporal_stride (int, optional): The stride to use over the temporal FRAMES dimension. Only used when temporal_windowing_strategy == "strided", which allows for custom strides to be assigned. Defaults to None.
        preprocessing_chain (callable, optional): Transforms chain, generally includes casting to float(), normalization, and sending to specified PyTorch device. Defaults to None.
        model_input_pixels (int, optional): The model input size in pixels-per-side. Used for handling the windowing process. Defaults to 3.
        spatial_windowing_strategy (str, optional):Spatial windwoing method. Can be ['overlap' or 'pad']. 'overlap' windows the inputs without adding any padding, which will always result in a shrunken output map unless then `model_input_pixels` is set to 1. 'pad' will attempt to pad the model with 'dead' 0-pixels to obtain an output map of equal size to the input map, though this might not be ideal as there is rarely situations where the model has seen 'dead' 0-pixels. Defaults to 'overlap'.
        temporal_window_size (int, optional): Size of the temporal window over the FRAMES dimension. Should be equivalent to the model's training frame count, unless the model inherently allows for variable input lengths. Defaults to 2500.
        temporal_windowing_stride (int, optional): The temporal windowing stride, used in the case where `temproal_windowing_strategy == "chunking"`. This should be set to the temporal size (or number of frames) used during train-time. Defaults to 2500.

    Raises:
        ValueError: Invalid input variables.

    Returns:
        np.ndarray: Output map.
    """
    part_bleach_correction = None
    # For full bleach correction, apply bleach correction on the full stack.
    if (bleach_correction_mode == "full") and (bleach_correction_module is not None):
        input_arr = bleach_correction_module.correct_full_stack(input_arr)
    elif (bleach_correction_mode == "part") and (bleach_correction_module is not None):
        part_bleach_correction = bleach_correction_module
    # Cast to float32, which is the format required
    input_arr = input_arr.astype(np.float32)

    # This inference code is inspired by the code used for the Java ONNX implementation, as the original code never shared its inference pipeline, only the training pipeline. (https://github.com/ImagingFCS/Imaging_FCS_1_62/blob/main/java/imfcs/Imaging_FCS_1_62.java#L21634)
    # At the very least, the Java implementation was endorsed by the original implementer as being correct, so it is as good a starting point as we're going to get.
    # For fast_chunking, we hand-code a lot of the logic to allow for faster inference.
    # Start by doing padding if requested
    if spatial_windowing_strategy == "pad":
        input_arr = pad_input(input_arr, model_input_pixels=model_input_pixels)
        spatial_stride = 1
    elif spatial_windowing_strategy == "overlap":
        spatial_stride = 1
    elif spatial_windowing_strategy == "no_overlap":
        spatial_stride = model_input_pixels

    return ChunkLazyLoaderDataset(
        input_arr=input_arr,
        temporal_window_size=temporal_window_size,
        temporal_stride=temporal_stride,
        spatial_x_window_size=model_input_pixels,
        spatial_x_window_stride=spatial_stride,
        spatial_y_window_size=model_input_pixels,
        spatial_y_window_stride=spatial_stride,
        transforms=preprocessing_chain,
        bleach_correction_module=part_bleach_correction,  # This only applies when bc-mode == part
    )


def infer_safe(
    input_arr: np.ndarray,
    model: nn.Module,
    batch_size: int,
    temporal_windowing_strategy: str,
    temporal_stride: int = None,
    preprocessing_chain: callable = None,
    model_input_pixels: int = 3,
    spatial_windowing_strategy: str = "overlap",
    temporal_window_size: int = 2500,
    bleach_correction_module: PolynomialBleachCorrectionModule = None,
    bleach_correction_mode: str = "full",
) -> np.ndarray:
    """A slower, but more memory-safe implementation of the inference logic.

    This function leverages lazy loading to prevent heavy RAM use, but in turn sacrifices execution speed. This is the preferred mode of execution if you have an input image stack that has a high frame or pixel count.

    Args:
        input_arr (np.ndarray): Input array with arbitrary temporal and spatial dimensions. This function will handle the spatial and temporal windowing dynamically to produce a valid output map.
        model (nn.Module): Model to use for inference. Note that the temporal windowing stride will need to be set according to the model specifications.
        batch_size (int): Batch size to use during inference. Passing the whole batch is generally not viable for large inputs (whether over the temporal or spatial axis). Note that the use of batches during inference incurs a corresponding VRAM cost if GPUs are used. In this case, we default to using the training batch size during inference, but this can be scaled up or down depending on your available CUDA compute resources.
        temporal_windowing_strategy (str): Temporal windowing method, can be ['fast_chunking' or 'feature_averaging'].'feature_averaging' assumes that the model can internally handle variable-length inputs. 'fast_chunking' represents the case where we pre-batch the chunks into a large payload, which is faster at the cost of more GPU VRAM use. Defaults to 'chunking'.
        temporal_stride (int, optional): The stride to use over the temporal FRAMES dimension. Only used when temporal_windowing_strategy == "strided", which allows for custom strides to be assigned. Defaults to None.
        preprocessing_chain (callable, optional): Transforms chain, generally includes casting to float(), normalization, and sending to specified PyTorch device. Defaults to None.
        model_input_pixels (int, optional): The model input size in pixels-per-side. Used for handling the windowing process. Defaults to 3.
        spatial_windowing_strategy (str, optional):Spatial windwoing method. Can be ['overlap' or 'pad']. 'overlap' windows the inputs without adding any padding, which will always result in a shrunken output map unless then `model_input_pixels` is set to 1. 'pad' will attempt to pad the model with 'dead' 0-pixels to obtain an output map of equal size to the input map, though this might not be ideal as there is rarely situations where the model has seen 'dead' 0-pixels. Defaults to 'overlap'.
        temporal_window_size (int, optional): Size of the temporal window over the FRAMES dimension. Should be equivalent to the model's training frame count, unless the model inherently allows for variable input lengths. Defaults to 2500.
        temporal_windowing_stride (int, optional): The temporal windowing stride, used in the case where `temproal_windowing_strategy == "chunking"`. This should be set to the temporal size (or number of frames) used during train-time. Defaults to 2500.

    Raises:
        ValueError: Invalid input variables.

    Returns:
        np.ndarray: Output map.
    """
    # Validating input parameters
    # For temporal windowing, the Java code takes every 2500 chunk, computes the prediction, then averages all predictions.
    # Meanwhile, the original code *might* just use the full 50,000 frames and compute the D directly due to the averaging layer.
    # For now, we allow this to be toggled using an input argument.
    # This also allows for comparisons in the future.
    if temporal_windowing_strategy not in [
        "strided",
        "chunking",
        "averaging",
        "feature_averaging",
    ]:
        raise ValueError(
            f"temporal_windowing_streategy must be 'fast_chunking' or 'feature_averaging'. Got {temporal_windowing_strategy}."
        )

    # Check temporal stride settings.
    # A temporal stride is only allowed in 'strided' mode.
    if temporal_stride is not None:
        if temporal_windowing_strategy != "strided":
            raise ValueError(
                f"temporal_stride is set to {temporal_stride}, but custom strides are only supported when temporal_windowing_strategy is 'strided'. Got temporal_windowing_strategy == '{temporal_windowing_strategy}'."
            )
    else:
        if temporal_windowing_strategy == "strided":
            raise ValueError(
                "temporal_stride is None, but temporal_windowing_strategy is 'strided'. Define a temporal_stride using --temporal-stride N, where N is your desired stride value."
            )
        # If 'strided' mode is not selected, default to usingt he model's FRAME dimension length during training.
        temporal_stride = temporal_window_size

    # For spatial windowing, 'overlap' is the default because using a 3x3 input and assuming the central prediction means we need to slice off the outer edges by 1 pixel each. This appears to be the strategy used in the paper, as the scale bars for NLS are shorter than those for ImFCSNet, and the pixel counts are lowered by 2 (1 for each side).
    # However, if we wanted to ensure the same values, we could pad the outer pixels with constant zeros. This likely won't work as the model has never been trained to deal with 'dead' pixels, but we can try.
    if spatial_windowing_strategy not in ["no_overlap", "overlap", "pad"]:
        raise ValueError(
            f"spatial windowing strategy must be in ['no_overlap', 'overlap', 'pad']. Got {spatial_windowing_strategy}."
        )

    chunk_lazy_laoder_dataest = build_safe_chunking_dataset(
        input_arr,
        temporal_stride,
        preprocessing_chain,
        model_input_pixels,
        spatial_windowing_strategy,
        temporal_window_size,
        bleach_correction_module,
        bleach_correction_mode,
    )

    lazy_loader_dataloader = DataLoader(
        chunk_lazy_laoder_dataest,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Store predictions as a list
    preds_list = []
    for batch_of_chunks in lazy_loader_dataloader:
        # Perform inference and write to our output_map
        with torch.no_grad():
            output_map = model(batch_of_chunks.float().unsqueeze(1))

            # Since the code expects a numpy array, we cast accordingly.
            # Use extend to store by indices.
            preds_list.extend(output_map.detach().cpu().numpy())

    # Reshape the output map to the original shape
    output_map = chunk_lazy_laoder_dataest.predictions_to_output_map(preds_list)

    return output_map
