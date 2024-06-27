"""This is a set of learning rate schedulers that can be built. Most of PyTorch's built-in schedulers are implemented, as well as the CosineAnnealingWithWarmups which is commonly used for self-supervised learning workloads."""

import torch
import math
from torch.optim.optimizer import Optimizer

from typing import List
from torch.optim.lr_scheduler import LRScheduler


def build_lr_scheduler(
    optimizer: Optimizer,
    lr_scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    update_period: int,
    milestones: List[int],
    lr_multiplicative_factor: float,
) -> LRScheduler:
    """Builder function for learning rate schedulers.

    Args:
        optimizer (Optimizer): The base optimizer, likely initialized via `build_optimizer()`.
        lr_scheduler_type (str): Learning rate scheduler identifier string.
        total_steps (int): Maximum number of steps that can be taken. This generally will be the maximum epoch count.
        warmup_steps (int): Number of warmup steps to take.
        min_lr (float): Minimum learning rate. This ensures that the learning rate does not hit 0.0 during the decay process.
        update_period (int): Step interval between updates.
        milestones (List[int]): Milestones. This is only used by the MultiStepLR, where the learning rate is decayed at specific milestones.
        lr_multiplicative_factor (float): The multiplicative factor applied to the learning rate depending on the scheduler type.

    Raises:
        ValueError: Invalid identifier. You might need to implement it manually if it is not already implemented.

    Returns:
        LRScheduler: Initialized learning rate scheduler of your specified type.
    """
    if (
        lr_scheduler_type == "None"
        or lr_scheduler_type == None
        or lr_scheduler_type == "ConstantLR"
    ):
        # This is a dummy method that maintains the same learning rate throughout.
        # This allows us to use scheduler.step() in a similar method to all of the other schedulers.
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif lr_scheduler_type == "CosineAnnealingWithWarmup":
        # See https://stackoverflow.com/a/75089936 for a simple implementation.
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps, eta_min=0, last_epoch=-1, verbose=False
        )

        def warmup(current_step: int):
            # return 1 / (2 ** (float(warmup_steps - current_step)))
            return 1 / float(warmup_steps - current_step)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup
        )

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, train_scheduler], [warmup_steps]
        )
    elif lr_scheduler_type == "HalfCycleCosineWithWarmup":
        # This was the original implementation from the ConvNeXt-V2 repo, modified to work in the context of PyTorch's native LambdaLR scheduler.
        def halfcycle_cosinewithwarmup(current_step: int):
            """Decay the learning rate with half-cycle cosine after warmup"""
            if current_step < warmup_steps:
                return float(current_step / warmup_steps)
            else:
                return max(
                    min_lr,
                    0.5
                    * (
                        1.0
                        + math.cos(
                            math.pi
                            * (current_step - warmup_steps)
                            / (total_steps - warmup_steps)
                        )
                    ),
                )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=halfcycle_cosinewithwarmup
        )
    elif lr_scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, update_period, lr_multiplicative_factor
        )
    elif lr_scheduler_type == "MultiplicativeLR":
        return torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_multiplicative_factor
        )
    elif lr_scheduler_type == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, lr_multiplicative_factor
        )
    elif lr_scheduler_type == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, lr_multiplicative_factor
        )
    elif lr_scheduler_type == "PolynomialLR":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=update_period, power=lr_multiplicative_factor
        )
    elif lr_scheduler_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    elif lr_scheduler_type == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=warmup_steps, T_mult=lr_multiplicative_factor
        )
    else:
        raise ValueError(f"Invalid lr_scheduler identifier - {lr_scheduler_type}")
