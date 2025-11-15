"""
Shared utility functions for LeJEPA.

Contains helper functions used across multiple modules.
"""

import torch
from torch import distributed as dist


def all_reduce(x, op="AVG"):
    """
    Perform all-reduce operation in distributed training, or return input unchanged.
    
    This function wraps PyTorch's distributed all_reduce to handle both distributed
    and non-distributed contexts gracefully.
    
    Args:
        x: Tensor to reduce across all processes
        op: Reduction operation, one of:
            - "AVG" or "avg": Average across processes
            - "SUM" or "sum": Sum across processes
            - "MAX" or "max": Maximum across processes
            - "MIN" or "min": Minimum across processes
            
    Returns:
        Reduced tensor if distributed training is active, otherwise returns x unchanged
        
    Example:
        >>> # In distributed training
        >>> loss = compute_loss(data)
        >>> loss = all_reduce(loss, op="AVG")  # Average loss across all GPUs
        
        >>> # In single-GPU training
        >>> loss = compute_loss(data)
        >>> loss = all_reduce(loss, op="AVG")  # Returns loss unchanged
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x
    
    try:
        # Try newer functional API first (PyTorch 2.0+)
        from torch.distributed._functional_collectives import (
            all_reduce as functional_all_reduce,
        )
        return functional_all_reduce(x, op.lower(), dist.group.WORLD)
    except ImportError:
        # Fall back to older API (PyTorch 1.x)
        from torch.distributed.nn import all_reduce as functional_all_reduce
        from torch.distributed.nn import ReduceOp
        
        reduce_op = ReduceOp.__dict__[op.upper()]
        return functional_all_reduce(x, reduce_op)
