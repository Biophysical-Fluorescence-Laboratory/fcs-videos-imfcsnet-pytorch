"""Collection of Tensor-related operations.

These are generally default transforms that should always be in use (as TIFF files and our simulations are both ingested as NumPy arrays, which are not directly applicable with PyTorch). At this stage, only contains simple operations such as casting to float and sending to devices.
"""

import torch
import numpy as np

# Typing-specific imports
from typing import Tuple


class ToTorchTensor:
    """Cast a NumPy ndarray to a PyTorch Tensor in the float() format."""

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Cast a NumPy ndarray to a PyTorch Tensor in the float() format.

        Args:
            image (np.ndarray): Input image stack.

        Returns:
            torch.Tensor: Input stack as a PyTorch float() tensor.
        """
        return torch.Tensor(image).float()


class SendToDevice:
    def __init__(self, device: Tuple[str, torch.device]):
        """Helper class to send PyTorch Tensors to the correct device.

        Seems trivial, but is necessary as some of our noise augmentations rely on GPU code to generate noise efficiently. By sending our images to the right device early, this allows us to save on unnecessary transfers.

        Args:
            device (Tuple[str, torch.device]): PyTorch device to send Tensors to. Generally "cpu" of "cuda:N", where N is the CUDA device index accessible from `nvidia-smi`.
        """
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Send PyTorch Tensor to specified device.

        Args:
            image (torch.Tensor): PyTorch tensor to send.

        Returns:
            torch.Tensor: Reference to PyTorch tensor on specific device.
        """
        return image.to(self.device)
