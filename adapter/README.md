# LeJEPA Adapter

Production-ready adapter for implementing the full LeJEPA framework from the paper "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics".

## Overview

The LeJEPA Adapter provides a drop-in implementation of the complete LeJEPA training framework:

**LeJEPA Loss:** `L = (1-λ)×center_loss + λ×SIGReg`

1. **Center Loss** (prediction term): Forces all views to predict the mean of global views
2. **SIGReg** (regularization term): Ensures embeddings follow optimal N(0, I) distribution

### Key Features

- **AdamW optimizer** with proven hyperparameters (lr=5e-4, weight_decay=5e-2)
- **Linear warmup + Cosine annealing** learning rate schedule (decays to lr/1000)
- **Center loss** for multi-view prediction
- **SIGReg loss** via statistical normality tests
- **14 statistical tests** for multivariate normality:
  - 10 univariate tests with random slicing (Epps-Pulley, Anderson-Darling, +8 more)
  - 4 direct multivariate tests (BHEP, Henze-Zirkler, +2 more)
- **Flexible λ parameter** to control prediction vs regularization trade-off

## Installation

Ensure the `lejepa` package is installed:

```bash
pip install -e .
```

## Quick Start

```python
from adapter import LeJEPAAdapter
import torchvision.transforms as T

# Multi-view augmentations (DINO-style: 2 global + 6 local)
global_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.3, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.2, 0.1),
    T.ToTensor(),
])

local_transform = T.Compose([
    T.RandomResizedCrop(98, scale=(0.05, 0.3)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.2, 0.1),
    T.ToTensor(),
])

# Create model and adapter
model = MyModel()
lejepa = LeJEPAAdapter(
    model.parameters(),
    lr=5e-4,
    weight_decay=5e-2,
    total_steps=10000,
    warmup_steps=1000,
    lambda_=0.5  # Balanced: 50% prediction + 50% regularization (paper default)
)

# Training loop
for step, batch in enumerate(dataloader):
    lejepa.zero_grad()
    
    # Generate multi-view augmentations
    global_views = [global_transform(img) for img in batch for _ in range(2)]  # 2 global
    local_views = [local_transform(img) for img in batch for _ in range(6)]    # 6 local
    
    # Get embeddings for each view
    global_emb = torch.stack([model(v) for v in global_views])  # (2, B, D)
    local_emb = torch.stack([model(v) for v in local_views])    # (6, B, D)
    all_emb = torch.cat([global_emb, local_emb], dim=0)         # (8, B, D)
    
    # Compute LeJEPA loss: (1-λ)×center_loss + λ×SIGReg
    loss = lejepa.compute_loss(global_emb, all_emb)
    
    loss.backward()
    lejepa.step()
```

### Choosing a Statistical Test

```python
# Default: Epps-Pulley (fast and accurate)
lejepa = LeJEPAAdapter(model.parameters())

# Or explicitly choose a univariate test with slicing:
lejepa = LeJEPAAdapter(
    model.parameters(),
    test='anderson_darling',      # Choose from 10 univariate tests
    num_slices=512                 # Number of random projections
)

# Or use a direct multivariate test (no slicing, more accurate but O(N²)):
lejepa = LeJEPAAdapter(
    model.parameters(),
    multivariate_test='hz',        # Choose from: bhep, hz, hv, comb
)
```

## Understanding LeJEPA

The complete LeJEPA framework combines two loss terms:

**LeJEPA Loss:** `L = (1-λ)×center_loss + λ×SIGReg`

### 1. Center Loss (Prediction Term)

Forces all views to predict the mean of global views:
- **Formula:** `||mean(global_embeddings) - all_embeddings||²`
- **Purpose:** Provides contrastive learning signal across views
- **Mechanism:** Each view learns to match the averaged representation of global views

### 2. SIGReg (Regularization Term)

Ensures embeddings follow optimal isotropic Gaussian distribution N(0, I):
- **Method:** Statistical normality tests on embeddings
- **Purpose:** Prevents collapse and ensures optimal distribution
- **Theory:** Isotropic Gaussians minimize downstream prediction risk (proven in paper)

### The λ Hyperparameter

Controls the trade-off between prediction and regularization:
- **λ=0.0**: Pure center loss (prediction only, no statistical regularization)
- **λ=0.5**: Balanced (paper's recommendation for self-supervised learning)
- **λ=1.0**: Pure SIGReg (regularization only, no prediction)

**Recommendation:** Use `λ=0.5` (default) for multi-view self-supervised learning

## Features

### Automatic Learning Rate Scheduling

The adapter handles learning rate scheduling automatically:
- **Linear warmup**: Gradually increases from 0 to `lr` over `warmup_steps`
- **Cosine annealing**: Smoothly decays from `lr` to `lr/1000` over remaining steps
- No need to manually manage `scheduler.step()`

### Early Stopping

The adapter supports automatic early stopping when a target loss is reached:

```python
lejepa = LeJEPAAdapter(
    model.parameters(),
    lr=5e-4,
    weight_decay=5e-2,
    total_steps=10000,
    target_loss=0.01  # Stop training when loss < 0.01
)

# Training loop
for step in range(10000):
    lejepa.zero_grad()
    embeddings = model(inputs)
    loss = lejepa.compute_loss(embeddings)
    loss.backward()
    lejepa.step()
    
    if lejepa.should_stop_training():
        print(f"Target loss reached at step {step}")
        break
```

### Statistical Normality Tests

Choose from **two types of tests** for measuring embedding distribution:

#### Univariate Tests (with Slicing)

Choose from **10 univariate normality tests** for measuring embedding distribution:

| Test | Type | Complexity | Best For |
|------|------|-----------|----------|
| `epps_pulley` | Characteristic function | O(N × n_points) | **Default**: fast & accurate |
| `anderson_darling` | EDF (tail-weighted) | O(N log N) | Heavy/light tails |
| `cramer_von_mises` | EDF (uniform) | O(N log N) | Balanced sensitivity |
| `watson` | EDF (circular) | O(N log N) | Rotational patterns |
| `extended_jarque_bera` | 4-moment omnibus | O(N) | Explicit moment control |
| `vcreg` | 2-moment only | O(N) | Fast, mean & variance |
| `shapiro_wilk` | Correlation-based | O(N log N) | Maximum power |
| `moments` | Direct moment matching | O(N) | Configurable moments |
| `entropy` | Information-theoretic | O(N log N) | Distribution-free |
| `nll` | Likelihood-based | O(N) | Parametric MLE |

All tests are applied via **random slicing** to handle multivariate embeddings efficiently.
See [Available Univariate Tests](#available-univariate-tests) section for detailed descriptions.

#### Multivariate Tests (Direct)

Choose from **4 direct multivariate tests** for highest accuracy:

| Test | Method | Complexity | Best For |
|------|--------|-----------|----------|
| `bhep` | Energy-based (Beta-Henze) | O(N²) | Tunable sensitivity (beta parameter) |
| `hz` | Henze-Zirkler (adaptive) | O(N²) | Automatic bandwidth selection |
| `hv` | Henze-Visagie | O(N²) | Kernel-based with gamma tuning |
| `comb` | Combined test | O(N²) | Multiple criteria |

**Usage:**
```python
# Use direct multivariate test (no slicing)
lejepa = LeJEPAAdapter(
    model.parameters(),
    multivariate_test='hz',  # Henze-Zirkler test
)

# Or with tunable parameters
lejepa = LeJEPAAdapter(
    model.parameters(),
    multivariate_test='bhep',
    beta=0.1,  # Sensitivity parameter
)
```

**When to use:**
- Dataset size N < 1000 (computational feasibility)
- Need highest accuracy/power
- Slicing-based tests show instability
- Research settings where compute isn't a bottleneck

**Performance Note:** These tests have O(N²) complexity, making them impractical for large batches.
For N > 1000, use univariate tests with slicing instead.

### SIGReg Loss

The adapter computes SIGReg (Statistical Independence Gaussian Regularization) loss:

```
L_SIGReg = statistical_test(embeddings)
```

This loss encourages embeddings to follow a standard normal distribution N(0, I), which:
- Improves feature quality and separability
- Prevents mode collapse
- Enhances downstream task performance

## Configuration
### Hyperparameters

```python
LeJEPAAdapter(
    params,                      # Model parameters (required)
    lr=5e-4,                    # Learning rate (LeJEPA default)
    weight_decay=5e-2,          # Weight decay (5e-2 for ViT, 5e-4 for ResNets)
    total_steps=10000,          # Total training steps (required)
    warmup_steps=1000,          # Warmup steps (default: 0)
    univariate_test='epps_pulley',  # Statistical test
    num_slices=512,             # Random projections (higher = more accurate)
    reduction='mean',           # Aggregation: 'mean', 'sum', or None
    n_points=17,                # Integration points (for Epps-Pulley)
    clip_value=None,            # Noise reduction: clip test stats below threshold to 0
    sampler='gaussian',         # Projection sampling method ('gaussian')
    target_loss=None            # Optional: early stopping threshold
)
```

**Advanced Parameters:**

- **`clip_value`**: Minimum threshold for test statistics. Test values below this are clipped to zero, reducing noise from negligible deviations. Useful when embeddings are nearly normal but show small numerical artifacts. Example: `clip_value=0.01`
- **`sampler`**: Random sampling method for projection directions. Default is `'gaussian'` (standard normal). This controls how the random slicing directions are generated.

### Recommended Settings by Architecture

**Vision Transformers (ViT):**
```python
lejepa = LeJEPAAdapter(
    model.parameters(),
    lr=5e-4,
    weight_decay=5e-2,          # Higher weight decay
    total_steps=epochs * steps_per_epoch,
    warmup_steps=epochs * steps_per_epoch // 10,
    num_slices=512
)
```

**ResNets / ConvNets:**
```python
lejepa = LeJEPAAdapter(
    model.parameters(),
    lr=5e-4,
    weight_decay=5e-4,          # Lower weight decay
    total_steps=epochs * steps_per_epoch,
    warmup_steps=epochs * steps_per_epoch // 20,
    num_slices=256              # Can use fewer slices
)
```

## API Reference

### `LeJEPAAdapter`

#### Methods

**`__init__(params, lr, weight_decay, total_steps, ...)`**
- Initializes adapter with AdamW optimizer and cosine schedule
- Sets up statistical test for SIGReg loss
- Optional `target_loss` parameter enables automatic early stopping

**`compute_loss(embeddings: torch.Tensor) -> torch.Tensor`**
- Computes SIGReg loss for given embeddings
- Automatically checks early stopping condition if `target_loss` is set
- Args:
  - `embeddings`: Tensor of shape (batch_size, embedding_dim)
- Returns: Scalar loss tensor

**`should_stop_training() -> bool`**
- Returns True if `target_loss` was set and has been reached
- Use in training loop to exit gracefully when target is met
- Returns False if no `target_loss` or not yet reached

**`zero_grad()`**
- Clears gradients (delegates to internal optimizer)

**`step()`**
- Updates parameters and learning rate (delegates to optimizer + scheduler)

**`state_dict() -> Dict`**
- Returns state dictionary for checkpointing
- Includes early stopping state (`target_loss`, `last_loss`, `should_stop`)

**`load_state_dict(state_dict: Dict)`**
- Loads state from checkpoint
- Restores early stopping state if present

**`get_stats() -> Dict[str, Any]`**
- Returns comprehensive training statistics
- Includes: current LR, step count, loss values, early stopping status

## Performance Comparison

See `benchmark_lejepa_adapter.py` for comprehensive benchmarks comparing:
- LeJEPA Adapter vs. Standard AdamW
- Various statistical tests (Epps-Pulley, Anderson-Darling, Cramér-von Mises)
- Different num_slices settings

Expected improvements:
- **Better embedding quality**: Lower test statistics indicate closer fit to N(0, I)
- **Improved convergence**: Cosine schedule + warmup provides stable training
- **Consistent hyperparameters**: No manual LR tuning needed

## Examples

### Basic Usage

```python
from adapter import LeJEPAAdapter
import torch
import torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# Setup
model = SimpleEncoder()
lejepa = LeJEPAAdapter(
    model.parameters(),
    total_steps=10000,
    warmup_steps=1000
)

# Training
for step, batch in enumerate(dataloader):
    lejepa.zero_grad()
    embeddings = model(batch)
    loss = lejepa.compute_loss(embeddings)
    loss.backward()
    lejepa.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, LR: {lejepa.get_last_lr()[0]:.6f}")
```

### Checkpointing

```python
# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': lejepa.state_dict(),
    'step': step
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
lejepa.load_state_dict(checkpoint['optimizer'])
step = checkpoint['step']
```

## Technical Details

### Why These Hyperparameters?

The default configuration comes from LeJEPA's official recommendations and extensive experimentation:

- **lr=5e-4**: Balances fast convergence with stability for embedding regularization
- **weight_decay=5e-2**: Strong regularization prevents overfitting to training statistics
- **Cosine annealing to lr/1000**: Ensures fine-grained convergence in later training
- **num_slices=512**: Provides accurate multivariate test approximation with reasonable compute

### Statistical Testing

The adapter uses **random slicing** to extend univariate tests to multivariate embeddings:

1. Generate `num_slices` random projection directions
2. Project embeddings onto each direction (1D)
3. Apply univariate test to each projection
4. Aggregate results (mean/sum)
5. Optionally clip small values to reduce noise (if `clip_value` is set)

This approach is:
- **Efficient**: O(N × D × num_slices) vs. O(N²) for energy-based tests
- **Accurate**: Captures multivariate structure with enough slices
- **Scalable**: Works for high-dimensional embeddings

### Noise Reduction with clip_value

The `clip_value` parameter provides optional noise reduction:

```python
lejepa = LeJEPAAdapter(
    model.parameters(),
    clip_value=0.01  # Clip test statistics below 0.01 to zero
)
```

**How it works:**
- After computing test statistics for each slice, values below `clip_value` are set to 0
- Reduces sensitivity to negligible deviations that may be numerical artifacts
- Helps stabilize training when embeddings are nearly normal

**When to use:**
- Embeddings are close to normal but show small numerical noise
- You want to focus on significant deviations only
- Training shows oscillations around very low loss values

**Typical values:**
- `None` (default): No clipping, use raw test statistics
- `0.001-0.01`: Light noise reduction for near-normal embeddings
- `0.01-0.1`: Moderate filtering of small deviations

## Available Univariate Tests

The adapter supports **10 univariate normality tests**, all wrapped via random slicing for multivariate data:

### Characteristic Function Tests

**`epps_pulley` (default)** - Epps-Pulley Test
- **Method**: Compares empirical vs. theoretical characteristic functions
- **Complexity**: O(N × n_points) where n_points=17 by default
- **Strengths**: Fast, accurate, good all-around performance
- **Best for**: Default choice for most applications
- **Reference**: Epps & Pulley (1983)

### Empirical Distribution Function (EDF) Tests

**`anderson_darling`** - Anderson-Darling Test
- **Method**: Weighted EDF comparison emphasizing tails
- **Complexity**: O(N log N)
- **Strengths**: Very sensitive to tail deviations
- **Best for**: Detecting heavy/light-tailed distributions
- **Reference**: Anderson & Darling (1952)

**`cramer_von_mises`** - Cramér-von Mises Test
- **Method**: Unweighted EDF comparison (uniform sensitivity)
- **Complexity**: O(N log N)
- **Strengths**: Balanced sensitivity across distribution
- **Best for**: General-purpose normality testing
- **Reference**: Cramér (1928), von Mises (1928)

**`watson`** - Watson Test
- **Method**: Circular variant of Cramér-von Mises
- **Complexity**: O(N log N)
- **Strengths**: Invariant to rotations/circular shifts
- **Best for**: Detecting rotational patterns in embeddings
- **Reference**: Watson (1961)

### Moment-Based Tests

**`extended_jarque_bera`** - Extended Jarque-Bera Test
- **Method**: Tests all 4 moments (mean, variance, skewness, kurtosis)
- **Complexity**: O(N)
- **Strengths**: Comprehensive omnibus test, detects multiple deviations
- **Best for**: When you need explicit 4-moment control
- **Reference**: Jarque & Bera (1980, 1987)
- **Note**: Extends standard JB by also testing mean=0 and variance=1

**`vcreg`** - VCReg (Variance-Covariance Regularization)
- **Method**: Tests only first 2 moments (mean=0, variance=1)
- **Complexity**: O(N)
- **Strengths**: Simplest moment test, very fast
- **Best for**: When you only care about location and scale
- **Note**: Subset of Extended Jarque-Bera, doesn't check skew/kurtosis

**`moments`** - Direct Moment Matching
- **Method**: Directly compares empirical moments to theoretical values
- **Complexity**: O(N)
- **Strengths**: Explicit moment control up to order k_max (default=4)
- **Best for**: When you want fine control over moment orders
- **Configurable**: Can test moments 2, 4, 6, etc.

### Correlation-Based Tests

**`shapiro_wilk`** - Shapiro-Wilk Test
- **Method**: Correlation between ordered samples and expected order statistics
- **Complexity**: O(N log N)
- **Strengths**: Most powerful for small-to-medium N, classic benchmark
- **Best for**: High-power normality detection, widely trusted
- **Reference**: Shapiro & Wilk (1965)

### Information-Theoretic Tests

**`entropy`** - Vasicek Entropy Test
- **Method**: Sample entropy comparison using nearest-neighbor distances
- **Complexity**: O(N log N)
- **Strengths**: Information-theoretic approach, distribution-free
- **Best for**: Alternative perspective on normality
- **Reference**: Vasicek (1976)

**`nll`** - Negative Log-Likelihood Test
- **Method**: Likelihood ratio test against standard normal
- **Complexity**: O(N)
- **Strengths**: Parametric approach, theoretically grounded
- **Best for**: Maximum likelihood perspective
- **Configurable**: Supports tail probability thresholding (alpha parameter)

### Choosing a Test

**Quick Recommendations:**

| Scenario | Recommended Test | Reason |
|----------|-----------------|--------|
| Default / First Try | `epps_pulley` | Fast, accurate, proven |
| Detect heavy tails | `anderson_darling` | Tail-sensitive weighting |
| Balanced detection | `cramer_von_mises` | Uniform sensitivity |
| Explicit moment control | `extended_jarque_bera` | All 4 moments tested |
| Maximum power | `shapiro_wilk` | Most powerful classical test |
| Circular/rotational patterns | `watson` | Rotation-invariant |
| Fast, simple | `vcreg` | Only mean & variance |
| Information theory | `entropy` | Distribution-free |
| Custom moments | `moments` | Configurable moment orders |
| Likelihood-based | `nll` | Parametric MLE approach |

**Performance Considerations:**
- EDF tests (Anderson-Darling, CVM, Watson): O(N log N) due to sorting
- Moment tests (VCReg, ExtJB, Moments): O(N), very fast
- Epps-Pulley: O(N × n_points), fast with n_points=17
- All scale efficiently with multivariate slicing: × num_slices

**Example Usage:**

```python
# Use Anderson-Darling for tail-sensitive detection
lejepa = LeJEPAAdapter(
    model.parameters(),
    univariate_test='anderson_darling',
    num_slices=512
)

# Use Extended Jarque-Bera for explicit 4-moment control
lejepa = LeJEPAAdapter(
    model.parameters(),
    univariate_test='extended_jarque_bera',
    num_slices=512
)

# Use Shapiro-Wilk for maximum power
lejepa = LeJEPAAdapter(
    model.parameters(),
    univariate_test='shapiro_wilk',
    num_slices=512
)
```

## Citation

If you use this adapter, please cite the LeJEPA paper:

```bibtex
@article{lejepa2024,
  title={LeJEPA: Learning Joint Embedding Predictive Architectures},
  author={...},
  journal={...},
  year={2024}
}
```

## License

See project LICENSE file.
