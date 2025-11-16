"""
LeJEPA Optimizer Adapter with SIGReg Loss

This implements the LeJEPA training configuration as described in the paper
"LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
(arXiv:2511.08544)

LeJEPA's key contribution is SIGReg (Sketched Isotropic Gaussian Regularization),
which constrains embeddings to follow an isotropic Gaussian distribution N(0, I)
using statistical tests. Unlike other methods (I-JEPA, VICReg), LeJEPA uses:
- NO variance/covariance regularization heuristics
- NO stop-gradients or teacher-student networks  
- NO complex schedulers or hyperparameters
- ONLY statistical tests for Gaussian regularization (single hyperparameter)

This adapter provides:
- AdamW optimizer with lr=5e-4, weight_decay=5e-2
- Linear warmup + Cosine annealing decay (final_lr = initial_lr / 1000)
- SIGReg loss: statistical tests for multivariate normality

Usage:
    from adapter import LeJEPAAdapter
    
    # Create model
    model = MyEncoder()
    
    # Initialize adapter
    lejepa = LeJEPAAdapter(
        model.parameters(),
        lr=5e-4,
        weight_decay=5e-2,
        total_steps=10000,
        warmup_steps=1000
    )
    
    # Training loop
    for batch in dataloader:
        lejepa.zero_grad()
        
        # Get embeddings from your model
        embeddings = model(batch)
        
        # Compute SIGReg loss (tests if embeddings ~ N(0, I))
        loss = lejepa.compute_loss(embeddings)
        
        loss.backward()
        lejepa.step()
"""

import torch
import torch.nn.functional as F
from typing import Literal, Dict, Any, Iterable, Optional
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
    from lejepa.univariate.jarque_bera import ExtendedJarqueBera, VCReg
    from lejepa.univariate.watson import Watson
    from lejepa.univariate.shapiro_wilk import ShapiroWilk
    from lejepa.univariate.entropy import Entropy
    from lejepa.univariate.likelihood import NLL
    from lejepa.univariate.moments import Moments
    from lejepa.multivariate.slicing import SlicingUnivariateTest
    from lejepa.multivariate.bhep import BHEP
    from lejepa.multivariate.hz import HZ
    from lejepa.multivariate.hv import HV
    from lejepa.multivariate.comb import COMB
    LEJEPA_AVAILABLE = True
except ImportError:
    LEJEPA_AVAILABLE = False
    warnings.warn("LeJEPA not available. Install lejepa package for SIGReg loss.")


class LeJEPAAdapter:
    """
    LeJEPA training wrapper implementing SIGReg (Sketched Isotropic Gaussian Regularization).
    
    As described in "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics",
    this adapter uses statistical tests to constrain embeddings to follow an isotropic Gaussian
    distribution N(0, I). This is the core LeJEPA loss - no additional heuristics needed.
    
    Unlike I-JEPA or VICReg, LeJEPA:
    - Uses ONLY statistical regularization (single hyperparameter)
    - No variance/covariance regularization heuristics
    - No stop-gradients or teacher-student networks
    - No complex schedulers beyond standard cosine annealing
    
    Plus proper training configuration:
    - AdamW optimizer with lr=5e-4, weight_decay=5e-2
    - Linear warmup + Cosine annealing decay (final_lr = initial_lr / 1000)
    
    Args:
        params: Model parameters to optimize
        lr: Initial learning rate (default: 5e-4, per LeJEPA recommendations)
        weight_decay: Weight decay coefficient (default: 5e-2 for ViT, 5e-4 for ResNets)
        total_steps: Total number of training steps for cosine schedule
        warmup_steps: Number of linear warmup steps (default: 0)
        test: Shorthand for test selection (default: 'epps_pulley')
            Use this for quick selection, or use univariate_test/multivariate_test for explicit choice
        univariate_test: Univariate test (wrapped with slicing for multivariate):
            - 'epps_pulley': Fast Epps-Pulley test (default, characteristic function)
            - 'anderson_darling': Anderson-Darling test (tail-sensitive EDF)
            - 'cramer_von_mises': Cramer-von-Mises test (balanced EDF)
            - 'extended_jarque_bera': Extended Jarque-Bera test (4-moment omnibus)
            - 'watson': Watson test (circular CVM variant)
            - 'shapiro_wilk': Shapiro-Wilk test (correlation-based)
            - 'vcreg': VCReg test (2-moment: mean & variance only)
            - 'entropy': Entropy-based test (information-theoretic)
            - 'nll': Negative log-likelihood test
            - 'moments': Direct moment matching test
        num_slices: Number of random projections for slicing (default: 512, only for univariate tests)
        multivariate_test: Direct multivariate test (no slicing, O(N²) complexity):
            - 'bhep': Beta-Henze Energy-based Projection test
            - 'hz': Henze-Zirkler test (adaptive bandwidth)
            - 'hv': Henze-Visagie test
            - 'comb': Combined test
            Note: These are more accurate but slower. Use for N < 1000 or when slicing isn't sufficient.
        beta: Bandwidth parameter for BHEP test (default: 0.1, only used if multivariate_test='bhep')
        gamma: Bandwidth parameter for HV test (default: 1.0, only used if multivariate_test='hv')
        reduction: How to aggregate slice statistics ('mean', 'sum', 'none')
        n_points: Number of integration points for univariate test (default: 17)
        clip_value: Minimum threshold for test statistics - values below are clipped
            to zero for noise reduction (default: None, no clipping)
        sampler: Random sampling method for projection directions (default: 'gaussian')
        target_loss: Optional loss threshold for early stopping (default: None)
        lambda_: Trade-off between center loss and SIGReg (default: 0.5, paper's recommendation)
            - 0.0: Pure center loss (prediction only)
            - 0.5: Balanced (recommended)
            - 1.0: Pure SIGReg (regularization only)
        
    Example:
        >>> # Generate multi-view augmentations (2 global + 6 local)
        >>> global_views = [global_transform(img) for img in batch for _ in range(2)]
        >>> local_views = [local_transform(img) for img in batch for _ in range(6)]
        >>> 
        >>> model = MyEncoder()
        >>> lejepa = LeJEPAAdapter(
        ...     model.parameters(),
        ...     lr=5e-4,
        ...     weight_decay=5e-2,
        ...     total_steps=10000,
        ...     warmup_steps=1000,
        ...     lambda_=0.5  # Balanced prediction + regularization
        ... )
        >>> 
        >>> for step in range(10000):
        ...     lejepa.zero_grad()
        ...     
        ...     # Get embeddings for all views
        ...     g_emb = torch.stack([model(v) for v in global_views])  # (2, B, D)
        ...     l_emb = torch.stack([model(v) for v in local_views])   # (6, B, D)
        ...     all_emb = torch.cat([g_emb, l_emb], dim=0)             # (8, B, D)
        ...     
        ...     # Compute LeJEPA loss: (1-λ)×center + λ×SIGReg
        ...     loss = lejepa.compute_loss(g_emb, all_emb)
        ...     
        ...     loss.backward()
        ...     lejepa.step()
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = DEFAULT_INITIAL_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY_VIT,
        total_steps: int = 10000,
        warmup_steps: int = 0,
        test: str = 'epps_pulley',
        # Univariate test options (these get wrapped with slicing)
        univariate_test: Optional[Literal['epps_pulley', 'anderson_darling', 'cramer_von_mises', 
                                 'extended_jarque_bera', 'watson', 'shapiro_wilk',
                                 'vcreg', 'entropy', 'nll', 'moments']] = None,
        num_slices: int = DEFAULT_NUM_SLICES,
        # Multivariate test options (direct multivariate, no slicing)
        multivariate_test: Optional[Literal['bhep', 'hz', 'hv', 'comb']] = None,
        beta: float = 0.1,  # For BHEP
        gamma: float = 1.0,  # For HV
        # Common options
        reduction: str = DEFAULT_REDUCTION,
        n_points: int = DEFAULT_N_POINTS,
        clip_value: float = None,
        sampler: str = 'gaussian',
        target_loss: float = None,
        # LeJEPA hyperparameter
        lambda_: float = 0.5,  # Trade-off: (1-λ)×center_loss + λ×SIGReg (paper default: 0.5)
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
        
        # Determine which test to use (priority: multivariate_test > univariate_test > test)
        if multivariate_test is not None:
            # Direct multivariate test (no slicing)
            if multivariate_test == 'bhep':
                self.sigreg_test = BHEP(beta=beta)
            elif multivariate_test == 'hz':
                self.sigreg_test = HZ()
            elif multivariate_test == 'hv':
                self.sigreg_test = HV(gamma=gamma)
            elif multivariate_test == 'comb':
                self.sigreg_test = COMB()
            else:
                raise ValueError(f"Unknown multivariate test: {multivariate_test}")
            self.test_name = multivariate_test
            self.is_multivariate = True
        else:
            # Univariate test with slicing (default)
            test_to_use = univariate_test if univariate_test is not None else test
            
            if test_to_use == 'epps_pulley':
                univariate = EppsPulley(n_points=n_points)
            elif test_to_use == 'anderson_darling':
                univariate = AndersonDarling()
            elif test_to_use == 'cramer_von_mises':
                univariate = CramerVonMises()
            elif test_to_use == 'extended_jarque_bera':
                univariate = ExtendedJarqueBera()
            elif test_to_use == 'watson':
                univariate = Watson()
            elif test_to_use == 'shapiro_wilk':
                univariate = ShapiroWilk()
            elif test_to_use == 'vcreg':
                univariate = VCReg()
            elif test_to_use == 'entropy':
                univariate = Entropy()
            elif test_to_use == 'nll':
                univariate = NLL()
            elif test_to_use == 'moments':
                univariate = Moments()
            else:
                raise ValueError(f"Unknown univariate test: {test_to_use}")
            
            # Create multivariate slicing test for SIGReg
            self.sigreg_test = SlicingUnivariateTest(
                univariate_test=univariate,
                num_slices=num_slices,
                reduction=reduction,
                sampler=sampler,
                clip_value=clip_value,
            )
            self.test_name = test_to_use
            self.is_multivariate = False
        
        # Early stopping configuration
        self.target_loss = target_loss
        self.should_stop = False
        self.last_loss = None
        
        # LeJEPA hyperparameter
        self.lambda_ = lambda_
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
    
    def compute_loss(
        self,
        global_embeddings: torch.Tensor,
        all_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LeJEPA loss: (1-λ)×center_loss + λ×SIGReg
        
        This implements the complete LeJEPA framework with both prediction and
        regularization terms, as described in Algorithm 2 of the paper.
        
        The center loss measures how well all views predict the mean of global views:
            center_loss = ||centers - all_embeddings||²
            where centers = mean(global_embeddings, dim=0)
        
        The SIGReg loss tests if embeddings follow N(0, I) distribution.
        
        Args:
            global_embeddings: Global view embeddings, shape (num_global_views, batch_size, embed_dim)
                              These are the "target" views (typically 2 large crops)
            all_embeddings: All view embeddings, shape (num_all_views, batch_size, embed_dim)
                           Includes both global and local views (e.g., 2 global + 6 local = 8 total)
        
        Returns:
            Scalar loss tensor combining center loss and SIGReg
            
        Example:
            >>> # Generate 2 global + 6 local views
            >>> global_views = [transform_global(img) for _ in range(2)]
            >>> local_views = [transform_local(img) for _ in range(6)]
            >>> 
            >>> # Get embeddings
            >>> g_emb = torch.stack([model(v) for v in global_views])  # (2, B, D)
            >>> l_emb = torch.stack([model(v) for v in local_views])   # (6, B, D)
            >>> all_emb = torch.cat([g_emb, l_emb], dim=0)             # (8, B, D)
            >>> 
            >>> # Compute LeJEPA loss
            >>> loss = adapter.compute_loss(g_emb, all_emb)
        """
        # Validate shapes
        if global_embeddings.ndim != 3 or all_embeddings.ndim != 3:
            raise ValueError(
                f"Expected 3D tensors (num_views, batch_size, embed_dim), "
                f"got global: {global_embeddings.shape}, all: {all_embeddings.shape}"
            )
        
        num_global = global_embeddings.shape[0]
        num_all = all_embeddings.shape[0]
        batch_size = global_embeddings.shape[1]
        embed_dim = global_embeddings.shape[2]
        
        if all_embeddings.shape[1] != batch_size or all_embeddings.shape[2] != embed_dim:
            raise ValueError(
                f"Shape mismatch: global {global_embeddings.shape} vs all {all_embeddings.shape}"
            )
        
        if num_global > num_all:
            raise ValueError(
                f"num_global_views ({num_global}) cannot exceed num_all_views ({num_all})"
            )
        
        # Compute center loss: MSE from all views to mean of global views
        # centers = g_emb.view(-1, bs, K).mean(0)  # Paper's algorithm
        centers = global_embeddings.mean(dim=0)  # Shape: (batch_size, embed_dim)
        
        # Center loss: squared distance from each view to centers
        # sim = (centers - a_emb).square().mean()  # Paper's algorithm
        center_loss = (all_embeddings - centers.unsqueeze(0)).square().mean()
        
        # Compute SIGReg loss: average over all views
        # sigreg = mean(SIGReg(emb, global_step) for emb in a_emb)  # Paper's algorithm
        sigreg_losses = []
        for view_idx in range(num_all):
            view_emb = all_embeddings[view_idx]  # Shape: (batch_size, embed_dim)
            sigreg_loss = self.sigreg_test(view_emb)
            sigreg_losses.append(sigreg_loss)
        
        sigreg = torch.stack(sigreg_losses).mean()
        
        # Combine losses: (1-λ)×center_loss + λ×SIGReg
        # return (1-lambd)*sim + lambd*sigreg  # Paper's algorithm
        total_loss = (1 - self.lambda_) * center_loss + self.lambda_ * sigreg
        
        # Track loss and check early stopping condition
        self.last_loss = total_loss.item()
        if self.target_loss is not None and self.last_loss < self.target_loss:
            self.should_stop = True
        
        return total_loss
    
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
            'test': self.test_name,
            'is_multivariate': self.is_multivariate,
            'num_slices': self.num_slices if not self.is_multivariate else None,
            'reduction': self.reduction if not self.is_multivariate else None,
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