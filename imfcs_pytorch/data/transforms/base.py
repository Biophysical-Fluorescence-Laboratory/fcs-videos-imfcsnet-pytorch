"""This includes some abstract base classes for transformations.

The initial motivation was to ensure that we could use type hints, but it also serves as a safety net for existing implementated transforms.

Note that the use of abstract base classes are never actually enforced in code (i.e., there is never a case where we enforce an isinstance() check).
"""

from abc import ABCMeta, abstractmethod

# Typing-specific imports
import torch
import numpy as np
from typing import Union


class Transform(ABCMeta):
    @abstractmethod
    def __call__(self, x: Union[torch.Tensor, np.ndarray]):
        return x
