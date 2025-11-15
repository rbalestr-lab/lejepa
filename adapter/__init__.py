"""
LeJEPA Adapter - Linear Warmup + Cosine Annealing Learning Rate Scheduler

This module provides an adapter class that enables the use of LeJEPA training functionality:

- AdamW optimizer with lr=5e-4, weight_decay=5e-2
- Linear warmup + Cosine annealing decay (final_lr = initial_lr / 1000)
- SIGReg loss for embedding regularization to standard normal

Available adapters can be imported directly or accessed through this module.
"""

from .lejepa_adapter import *

__all__ = [
    # Export all public classes/functions from adapter modules
    # Will be populated by the imported modules' __all__ if defined
]
