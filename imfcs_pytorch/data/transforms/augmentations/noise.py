"""A collection of noise-related data augmentations.

Generally applied during training to simulate the noisy photon detection process of EMCCD cameras.

Note that we deliberately try to speed up the noise generation process by relying on PyTorch. If you are experiencing slow noise generation, that might be due to the use of device="cpu", which does not benefit from the GPU-random number generation."""

import torch
import numpy as np

# Typing-specific imports
from typing import Tuple


class AddGaussianNoise:
    def __init__(
        self,
        gaussian_noise_scale_min: float,
        gaussian_noise_scale_max: float,
        device: Tuple[str, torch.device],
    ):
        """Data augmentation callable class to add Gaussian noise to an input.

        Args:
            gaussian_noise_scale_min (float): The minimum standard deviation to use for sampling from the Gaussian distribution.
            gaussian_noise_scale_max (float): The maximum standard deviation to use for sampling from the Gaussian distribution.
            device (Tuple[str, torch.device]): Device to use for noise generation.
        """
        self.gaussian_noise_scale_min = gaussian_noise_scale_min
        self.gaussian_noise_scale_max = gaussian_noise_scale_max

        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input.

        Args:
            image (torch.Tensor): Input image stack.

        Returns:
            torch.Tensor: Input image augmented with Gaussian noise.
        """
        # Add gaussian noise to image
        image += torch.randn(image.size(), device=self.device) * np.random.uniform(
            self.gaussian_noise_scale_min, self.gaussian_noise_scale_max
        )

        return image


class AddEMCCDNoise:
    def __init__(
        self,
        emccd_pdf: np.ndarray,
        emccd_min: int,
        emccd_noise_scale_min: float,
        emccd_noise_scale_max: float,
        scale_emccd_noise: bool = True,
        device: Tuple[str, torch.device] = "cpu",
    ):
        """Data augmentation callable class to add simulated EMCCD noise to an input.

        Args:
            emccd_pdf (np.ndarray): The probability density function extracted from a dark image.
            emccd_min (int): The minimum value required to shift the EMCCD PDF to have a mean of 0.
            emccd_noise_scale_min (float): The minimum scaling factor to apply to the EMCCD noise.
            emccd_noise_scale_max (float): Tme maximum scaling factor to apply to the EMCCD noise.
            scale_emccd_noise (bool, optional): Whether or not to apply EMCCD noise scaling. Original paper quotes "ased on our observation, using a “full" EMCCD noise, i.e. without multiplying a factor, results in a high training loss that is hard to decrease and the model may not perform well during validation. ". Defaults to True.
            device (Tuple[str, torch.device], optional): Device to use for noise generation. Defaults to "cpu".

        Raises:
            ValueError: Invalid input parameters.
        """
        # Verify values
        if not isinstance(emccd_pdf, np.ndarray):
            raise ValueError(
                f"emccd_pmf must be a NumPy ndarray. Got {type(emccd_pdf)}. Perhaps you forgot to call np.load() on the path?"
            )
        # Checking for valid ints is surprisingly tough
        # See https://stackoverflow.com/a/48940855
        if not int(emccd_min) == emccd_min:
            raise ValueError(
                f"emccd_min must be an int or similar. Got {type(emccd_min)}."
            )
        if scale_emccd_noise:
            if not isinstance(emccd_noise_scale_max, (int, float)):
                raise ValueError(
                    f"emccd_noise_scale_max must be a numeric int or float. Got {type(emccd_noise_scale_max)}."
                )
            if not isinstance(emccd_noise_scale_min, (int, float)):
                raise ValueError(
                    f"emccd_noise_scale_min must be a numeric int or float. Got {type(emccd_noise_scale_min)}."
                )

            # Enable scaling beyond 0.0 and 1.0
            # if emccd_noise_scale_max > 1.0 or emccd_noise_scale_max < 0.0:
            #     raise ValueError(
            #         f"emccd_noise_scale_max must be between 0.0 and 1.0, got {emccd_noise_scale_max}"
            #     )
            # if emccd_noise_scale_max > 1.0 or emccd_noise_scale_max < 0.0:
            #     raise ValueError(
            #         f"emccd_noise_scale_max must be between 0.0 and 1.0, got {emccd_noise_scale_max}"
            #     )

            if emccd_noise_scale_max <= emccd_noise_scale_min:
                raise ValueError(
                    f"For scaling EMCCD noise with scale_emccd_noise. emccd_noise_scale_max ({emccd_noise_scale_max}) must be greater than emccd_noise_scale_min ({emccd_noise_scale_min})."
                )

        # Derive cumulative density function (CDF) from the probability density function (PDF)
        self.emccd_cdf = torch.from_numpy(np.cumsum(emccd_pdf / np.sum(emccd_pdf)))
        self.emccd_min = emccd_min

        # Parameters for noise scaling.
        self.scale_emccd_noise = scale_emccd_noise
        # self.emccd_noise_scale_min = emccd_noise_scale_min
        # self.emccd_noise_scale_max = emccd_noise_scale_max

        # # Create a uniform distribution generator.
        # # This will be used for sampling during noise generation.
        # self.uniform_generator = torch.distributions.uniform.Uniform(low=0.0, high=1.0)

        # # Do the same for the noise scaling if requested.
        if self.scale_emccd_noise:
            self.scale_uniform_generator = torch.distributions.uniform.Uniform(
                low=emccd_noise_scale_min, high=emccd_noise_scale_max
            )

        # Save device
        self.device = device

    def generate_emccd_noise(
        self, image_shape: torch.Size, device: Tuple[str, torch.device] = "cpu"
    ) -> torch.Tensor:
        """Convenience function to generate EMCCD noise from the pre-defined probability mass function.

        Args:
            image_shape (torch.Size): Size of the input image stack. EMCCD noise will be generated in this specific shape.
            device (Tuple[str, torch.device], optional): Device to use for EMCCD noise generation. Defaults to "cpu".

        Returns:
            torch.Tensor: Simulated EMCCD noise in the same shape as the input image.
        """
        # Since this is time consuming, we use a complex 1-liner that avoids unnecessary write-to-memory calls.
        # The same code in multi-line is left as comments below for easier readability.
        # # Step 1: Generate a set of random numbers from the pre-defined uniform generator. This matches the input image shape.
        # random_numbers = self.uniform_generator.sample(image.shape)

        # # Step 2: Sample the argmax based on the CDF. We use broadcasting to a dummy dimension to do this without using a loop.
        # emccd_noise_counts = torch.argmax((self.emccd_cdf.reshape(1, 1, 1, -1) > random_numbers.unsqueeze(-1)).float(), dim=-1)

        # # Step 3: Add the minimum value to have the generate noise centred correctly around the distribution.
        # emccd_noise_counts += self.emccd_min

        return (
            torch.argmax(
                (
                    self.emccd_cdf.reshape(1, 1, 1, -1).to(device)
                    > torch.rand(image_shape, device=device).unsqueeze(-1)
                ).float(),
                dim=-1,
            )
            + self.emccd_min
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Add simulated EMCCD noise to an input image stack.

        Args:
            image (torch.Tensor): Input image stack.

        Returns:
            torch.Tensor: Input image augmented with simulated EMCCD noise.
        """
        # Add gaussian noise to image
        emccd_noise = self.generate_emccd_noise(image.shape, device=self.device).float()

        # Scale the EMCCD noise if requested.
        if self.scale_emccd_noise:
            emccd_noise *= self.scale_uniform_generator.sample()

        return image + emccd_noise
