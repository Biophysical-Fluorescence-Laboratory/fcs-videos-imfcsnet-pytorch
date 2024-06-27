"""Collection of normalization transforms.

Any new transforms should be included here, and added to the normalization builders under `imfcsnet-pytorch/imfcs_pytorch/builders/transforms.py` for easy integration into both training and inference pipelines.

As a rule, any transformations here should be designed for **single intensity traces**, and not applied over entire batches. This follows the Torchvision transform convention, which makes for easier integration and modularity.
"""

import torch

# Typing-specific imports


class PerStackZScoreNormalization:
    """Z-score normalization to normalize each input to have a mean of 0 and a standard deviation of 1.

    This is the original method implemented in the paper.
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Perform Z-score standardization over an input image.

        Args:
            image (torch.Tensor): Input image stack.

        Returns:
            torch.Tensor: Normalized input.
        """
        # Normalize over the whole array
        # Note that this happens prior to batching, so it is safe to just normalize over the whole mean and stdev.
        avg = torch.mean(image)
        std = torch.std(image)
        image = (image - avg) / std

        return image


class PerStackMinMaxNormalization:
    """Min-max normalization, which normalizes an input to have a minimum of 0.0 and a maximum of 1.0."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Perform min-max normalization on an input image stack.

        Args:
            image (torch.Tensor): Input image stack.

        Raises:
            ValueError: Invalid range.

        Returns:
            torch.Tensor: Normalized input.
        """
        # Find the minimum and maximum values within the image
        min_val = torch.min(image)
        max_val = torch.max(image)

        # Ensure a non-zero range to avoid division by zero
        range_val = max_val - min_val
        if range_val == 0:
            raise ValueError(
                "Cannot scale when the difference between the min and max are 0."
            )

        # Normalize the image using min-max scaling
        image = (image - min_val) / range_val

        return image

class PerStackZeroFloorNormalization:
    """Shifts the inputs such that their minimum pixel value is always 0.

    This was the initial proposed normalization scheme for the N-based networks. Unlike D-only networks, N networks cannot use scale-independent normalization schemes, as the input intensities are a function of the particle densities and the noise model.

    Similarly, while our simulations are always centred around 0, real-life data has a camera offset built into the hardware, which might lead to domain shifts if we do not handle the difference in the background values.

    This normalization scheme is inspired by the background subtraction routine in the ImagingFCS Fiji plugin, where the background value is assumed to be the minimum pixel value, and this is subtracted from the input.
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Shifts the minimum value of the input to always be 0.

        Args:
            image (torch.Tensor): Input image stack.

        Returns:
            torch.Tensor: Normalized input.
        """
        # Extract the minimum of the image
        min_val = torch.min(image)

        # Shift the image such that the minimum is always 0.
        # If min_val is negative, shifts upwards.
        # If min_val is positive, shifts downwards.
        image = image - min_val

        return image


if __name__ == "__main__":
    from torchvision.transforms import v2 as T

    def wrapper_batch_transforms(transform_chain: T) -> T:
        """Helper function to wrap the transform chain (generally written for single-inputs) for the batch-case. This will likely be used during inference, where inputs are batched for accelerated inference throughput."""
        return T.Lambda(
            lambda images: torch.stack([transform_chain(image) for image in images])
        )

    normalization_chain = T.Compose([PerStackMinMaxNormalization()])
    # batch_normalization = T.Lambda(
    #     lambda images: torch.stack([normalization_chain(image) for image in images])
    # )
    batch_normalization = wrapper_batch_transforms(normalization_chain)

    print("Testing transform chain applied on full batch.")
    # Generate a batch of random images
    images = torch.randn(4, 3, 224, 224)

    # Apply the transformations
    normalized_images = batch_normalization(images)

    # Check if mean and standard deviation are different for each image
    for i in range(4):
        image_mean = normalized_images[i].mean()
        image_std = normalized_images[i].std()
        print(f"Image {i+1}: Mean = {image_mean:.4f}, Std = {image_std:.4f}")

        image_min = normalized_images[i].min()
        image_max = normalized_images[i].max()
        print(f"Min {i+1}: {image_min}, Max {i+1}: {image_max}")

    # Check if pixel values are roughly between -1 and 1 after normalization
    normalized_images = (
        normalized_images.numpy()
    )  # Convert to numpy for easier inspection
    print(f"Minimum value: {normalized_images.min():.4f}")
    print(f"Maximum value: {normalized_images.max():.4f}")

    print("=" * 12)
    print("Testing transform chain applied on each element individually.")
    # Generate a batch of random images
    images = torch.randn(4, 3, 224, 224)

    normalized_images = torch.clone(images)

    # Check if mean and standard deviation are different for each image
    for i in range(4):
        normalized_images[i] = normalization_chain(normalized_images[i])

        image_mean = normalized_images[i].mean()
        image_std = normalized_images[i].std()
        print(f"Image {i+1}: Mean = {image_mean:.4f}, Std = {image_std:.4f}")

        image_min = normalized_images[i].min()
        image_max = normalized_images[i].max()
        print(f"Min {i+1}: {image_min}, Max {i+1}: {image_max}")

    # Check if pixel values are roughly between -1 and 1 after normalization
    normalized_images = (
        normalized_images.numpy()
    )  # Convert to numpy for easier inspection
    print(f"Minimum value: {normalized_images.min():.4f}")
    print(f"Maximum value: {normalized_images.max():.4f}")
