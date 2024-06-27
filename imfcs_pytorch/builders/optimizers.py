from torch import optim

# Typing-specific imports.
from typing import Tuple


def build_optimizer(
    param_groups: dict,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    betas: Tuple,
    eps: float,
) -> optim.Optimizer:
    """Builder function for optimizers.

    Args:
        param_groups (dict): Parameter groups for updating. Generally accessed via `model.parameters()`, but can also be a dict to assign different learning rates to different parameter groups (https://pytorch.org/docs/stable/optim.html#per-parameter-options).
        optimizer_type (str): Optimizer identifier string, generally will be the optimizer class name as in `torch.optim`.
        learning_rate (float): Learning rate. If no per-group settings are set, this applies uniformly to all parameters.
        weight_decay (float): The weight decay setting for the optimizer.
        momentum (float): Momentum setting for supported optimizers, such as SGD.
        betas (Tuple): Betas for specific optimizers. Generally applies to Adam derivatives.
        eps (float): Numerical stability term for optimizers.

    Raises:
        ValueError: An invalid optimizer identifier string was provided. You might need to implement it if it is not included in PyTorch.

    Returns:
        optim.Optimizer: Initialized optimizer object.
    """
    if optimizer_type == "Adam":
        return optim.Adam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type == "AdamW":
        return optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type == "RAdam":
        return optim.RAdam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type == "NAdam":
        return optim.NAdam(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type == "Adamax":
        return optim.Adamax(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type == "Adagrad":
        return optim.Adagrad(
            param_groups, lr=learning_rate, weight_decay=weight_decay, eps=eps
        )
    elif optimizer_type == "SGD":
        return optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "RMSprop":
        return optim.RMSprop(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
        )
    else:
        raise ValueError(f"Invalid optimizer_type - {optimizer_type}")
