"""
LeJEPA Loss Wrapper with Proper Training Configuration

This implements LeJEPA's recommended training setup:
- AdamW optimizer with lr=5e-4, weight_decay=5e-2
- Linear warmup + Cosine annealing decay (final_lr = initial_lr / 1000)
- SIGReg loss for embedding regularization

LeJEPA tests whether embeddings follow a target distribution (typically standard normal)
using statistical tests. This wrapper provides both the loss function and proper optimizer config.

Usage:
    from lejepa_noise_adapter import LeJEPALossWrapper
    
    # Create model
    model = ...
    
    # LeJEPA creates its own optimizer with proper config
    lejepa_wrapper = LeJEPALossWrapper(
        model.parameters(),
        lr=5e-4,
        weight_decay=5e-2,
        total_steps=10000,
        warmup_steps=1000,
        univariate_test='epps_pulley',
        num_slices=512
    )
    
    # Training loop
    for step in range(steps):
        lejepa_wrapper.zero_grad()
        embeddings = model(x)
        
        # Compute LeJEPA loss
        loss = lejepa_wrapper.compute_loss(embeddings)
        
        loss.backward()
        lejepa_wrapper.step()
"""

import torch
from typing import Literal, Dict, Any, Iterable
import warnings
import math

# LeJEPA Configuration Constants
# These values come from the official LeJEPA paper recommendations

# Learning Rate Schedule
DEFAULT_INITIAL_LR = 5e-4  # Recommended starting learning rate for most architectures
DEFAULT_WEIGHT_DECAY_VIT = 5e-2  # Weight decay for Vision Transformers
DEFAULT_WEIGHT_DECAY_RESNET = 5e-4  # Weight decay for ResNets and CNNs
LR_DECAY_FACTOR = 1000.0  # Final LR = Initial LR / 1000 (cosine annealing endpoint)

# Statistical Test Configuration
DEFAULT_NUM_SLICES = 512  # Number of random projections for multivariate test
DEFAULT_N_POINTS = 17  # Integration points for Epps-Pulley test
DEFAULT_REDUCTION = 'mean'  # How to aggregate statistics across slices

# Learning Rate Schedule Parameters
COSINE_DECAY_MULTIPLIER = 0.5  # Coefficient for cosine annealing computation
COSINE_DECAY_OFFSET = 0.5  # Offset for cosine annealing computation

try:
    from lejepa.univariate.epps_pulley import EppsPulley
    from lejepa.univariate.anderson_darling import AndersonDarling
    from lejepa.univariate.cramer_von_mises import CramerVonMises
    from lejepa.multivariate.slicing import SlicingUnivariateTest
    LEJEPA_AVAILABLE = True
except ImportError:
    LEJEPA_AVAILABLE = False
    warnings.warn("LeJEPA not available. Install lejepa package for SIGReg loss.")


class LeJEPAAdapter:
    """
    LeJEPA training wrapper with proper configuration from official recommendations.
    
    Implements:
    - AdamW optimizer with lr=5e-4, weight_decay=5e-2
    - Linear warmup + Cosine annealing decay (final_lr = initial_lr / 1000)
    - SIGReg loss for embedding regularization to standard normal
    
    Args:
        params: Model parameters to optimize
        lr: Initial learning rate (default: 5e-4, per LeJEPA recommendations)
        weight_decay: Weight decay coefficient (default: 5e-2 for ViT, 5e-4 for ResNets)
        total_steps: Total number of training steps for cosine schedule
        warmup_steps: Number of linear warmup steps (default: 0)
        univariate_test: Which univariate test to use:
            - 'epps_pulley': Fast Epps-Pulley test (default)
            - 'anderson_darling': Anderson-Darling test
            - 'cramer_von_mises': Cramer-von-Mises test
        num_slices: Number of random projections for multivariate test (default: 512)
        reduction: How to aggregate slice statistics ('mean', 'sum', 'none')
        n_points: Number of integration points for univariate test (default: 17)
        target_loss: Optional loss threshold for early stopping. If provided and loss falls
            below this value, sets `should_stop` flag to True. Check via `should_stop_training()`
            to exit training loop gracefully. (default: None - no early stopping)
        
    Example:
        >>> model = MyModel()
        >>> lejepa = LeJEPAAdapter(
        ...     model.parameters(),
        ...     lr=5e-4,
        ...     weight_decay=5e-2,
        ...     total_steps=10000,
        ...     warmup_steps=1000,
        ...     target_loss=0.01  # Stop when loss < 0.01
        ... )
        >>> 
        >>> for step in range(10000):
        ...     lejepa.zero_grad()
        ...     embeddings = model(x)  # (batch, dim)
        ...     loss = lejepa.compute_loss(embeddings)
        ...     loss.backward()
        ...     lejepa.step()
        ...     
        ...     if lejepa.should_stop_training():
        ...         print(f"Target loss {lejepa.target_loss} reached!")
        ...         break
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = DEFAULT_INITIAL_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY_VIT,
        total_steps: int = 10000,
        warmup_steps: int = 0,
        univariate_test: Literal['epps_pulley', 'anderson_darling', 'cramer_von_mises'] = 'epps_pulley',
        num_slices: int = DEFAULT_NUM_SLICES,
        reduction: str = DEFAULT_REDUCTION,
        n_points: int = DEFAULT_N_POINTS,
        target_loss: float = None,
    ):
        if not LEJEPA_AVAILABLE:
            raise ImportError(
                "LeJEPA is not installed. Install it with: pip install lejepa\n"
                "Or install from source: https://github.com/facebookresearch/LeJEPA"
            )
        
        # Create AdamW optimizer with LeJEPA's recommended config
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.initial_lr = lr
        self.final_lr = lr / LR_DECAY_FACTOR
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        self.num_slices = num_slices
        self.reduction = reduction
        
        # Create univariate test
        if univariate_test == 'epps_pulley':
            univariate = EppsPulley(n_points=n_points)
        elif univariate_test == 'anderson_darling':
            univariate = AndersonDarling()
        elif univariate_test == 'cramer_von_mises':
            univariate = CramerVonMises()
        else:
            raise ValueError(f"Unknown univariate test: {univariate_test}")
        
        # Create multivariate slicing test
        self.loss_fn = SlicingUnivariateTest(
            univariate_test=univariate,
            num_slices=num_slices,
            reduction=reduction,
        )
        
        self.univariate_test_name = univariate_test
        
        # Early stopping configuration
        self.target_loss = target_loss
        self.should_stop = False
        self.last_loss = None
    
    def compute_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute LeJEPA SIGReg loss for embeddings.
        
        This tests whether embeddings follow a standard normal distribution
        using the configured statistical test. If target_loss is set and the
        loss falls below the threshold, sets internal flag for early stopping.
        
        Args:
            embeddings: Embeddings tensor of shape (batch_size, embedding_dim)
                       or (num_samples, embedding_dim)
        
        Returns:
            Scalar loss tensor (higher = embeddings deviate more from standard normal)
        """
        loss = self.loss_fn(embeddings)
        
        # Track loss and check early stopping condition
        self.last_loss = loss.item()
        if self.target_loss is not None and self.last_loss < self.target_loss:
            self.should_stop = True
        
        return loss
    
    def should_stop_training(self) -> bool:
        """
        Check if training should stop due to reaching target loss.
        
        Returns:
            True if target_loss was set and has been reached, False otherwise
        """
        return self.should_stop
    
    def _get_lr(self) -> float:
        """
        Compute current learning rate with linear warmup + cosine annealing.
        
        Following LeJEPA's schedule:
        - Linear warmup from 0 to initial_lr over warmup_steps
        - Cosine annealing from initial_lr to final_lr over remaining steps
        """
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = COSINE_DECAY_MULTIPLIER * (1 + math.cos(math.pi * progress))
            return self.final_lr + (self.initial_lr - self.final_lr) * cosine_decay
    
    def step(self, closure=None):
        """
        Perform optimization step with learning rate scheduling.
        
        Updates learning rate according to linear warmup + cosine annealing schedule.
        """
        # Update learning rate
        current_lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Perform optimization step
        result = self.optimizer.step(closure)
        self.step_count += 1
        return result
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients in wrapped optimizer."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'initial_lr': self.initial_lr,
            'final_lr': self.final_lr,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'target_loss': self.target_loss,
            'should_stop': self.should_stop,
            'last_loss': self.last_loss,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.step_count = state_dict['step_count']
        self.initial_lr = state_dict.get('initial_lr', self.initial_lr)
        self.final_lr = state_dict.get('final_lr', self.final_lr)
        self.total_steps = state_dict.get('total_steps', self.total_steps)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.target_loss = state_dict.get('target_loss', self.target_loss)
        self.should_stop = state_dict.get('should_stop', False)
        self.last_loss = state_dict.get('last_loss', None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about training progress."""
        current_lr = self._get_lr()
        return {
            'step_count': self.step_count,
            'current_lr': current_lr,
            'initial_lr': self.initial_lr,
            'final_lr': self.final_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'univariate_test': self.univariate_test_name,
            'num_slices': self.num_slices,
            'reduction': self.reduction,
            'target_loss': self.target_loss,
            'last_loss': self.last_loss,
            'should_stop': self.should_stop,
        }
    
    @property
    def param_groups(self):
        """Access parameter groups of wrapped optimizer."""
        return self.optimizer.param_groups
    
    def __repr__(self) -> str:
        """Return concise string representation."""
        return (
            f"{self.__class__.__name__}("
            f"lr={self.initial_lr:.1e}, "
            f"wd={self.optimizer.param_groups[0]['weight_decay']:.1e}, "
            f"warmup={self.warmup_steps}, "
            f"test={self.univariate_test_name}, "
            f"step={self.step_count}/{self.total_steps})"
        )