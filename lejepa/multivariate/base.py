import torch
from typing import Union, Iterable
import numpy as np


class MultivariateTest(torch.nn.Module):

    def prepare_data(self, x):
        # Convert numpy to torch if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Ensure it's a torch tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected numpy array or torch tensor, got {type(x)}")

        # Ensure 2D shape
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (N, D), got shape {x.shape}")

        return x
