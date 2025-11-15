# LeJEPA Adapter

Production-ready adapter for training models with LeJEPA's statistical normality testing as a loss function.

## Overview

The LeJEPA Adapter provides a drop-in replacement for standard PyTorch optimizers that incorporates LeJEPA's recommended training configuration:

- **AdamW optimizer** with proven hyperparameters (lr=5e-4, weight_decay=5e-2)
- **Linear warmup + Cosine annealing** learning rate schedule (decays to lr/1000)
- **SIGReg loss** for embedding regularization to standard normal distribution
- **Statistical tests** for multivariate normality (Epps-Pulley, Anderson-Darling, Cramér-von Mises)

## Installation

Ensure the `lejepa` package is installed:

```bash
pip install -e .
```

## Quick Start

```python
from adapter import LeJEPAAdapter

# Create your model
model = MyModel()

# Initialize LeJEPA adapter (replaces optimizer creation)
lejepa = LeJEPAAdapter(
    model.parameters(),
    lr=5e-4,                    # Initial learning rate
    weight_decay=5e-2,          # Weight decay (5e-2 for ViT, 5e-4 for ResNets)
    total_steps=10000,          # Total training steps
    warmup_steps=1000,          # Linear warmup steps
    univariate_test='epps_pulley',  # Statistical test to use
    num_slices=512              # Number of random projections
)

# Training loop
for step in range(10000):
    lejepa.zero_grad()
    
    # Forward pass - get embeddings
    embeddings = model(inputs)  # Shape: (batch_size, embedding_dim)
    
    # Compute LeJEPA loss (tests if embeddings ~ N(0, I))
    loss = lejepa.compute_loss(embeddings)
    
    # Backward pass and optimization
    loss.backward()
    lejepa.step()
    
    # Learning rate automatically scheduled
```

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

for step in range(10000):
    lejepa.zero_grad()
    embeddings = model(inputs)
    loss = lejepa.compute_loss(embeddings)
    loss.backward()
    lejepa.step()
    
    # Check if target reached
    if lejepa.should_stop_training():
        print(f"Target loss {lejepa.target_loss} reached at step {step}!")
        print(f"Final loss: {lejepa.last_loss:.6f}")
        break
```

**Benefits:**
- Graceful exit when embeddings reach desired normality
- Saves compute by avoiding unnecessary training
- Prevents overfitting to the normality objective
- Last loss and stopping flag saved in checkpoints

### Statistical Normality Tests

Choose from multiple univariate tests for measuring embedding distribution:

| Test | Complexity | Best For |
|------|-----------|----------|
| `epps_pulley` | O(N × n_points) | Default, fast and accurate |
| `anderson_darling` | O(N log N) | Tail-sensitive detection |
| `cramer_von_mises` | O(N log N) | Balanced sensitivity |

All tests are applied via random slicing to handle multivariate embeddings efficiently.

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
    target_loss=None            # Optional: early stopping threshold
)
```

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

This approach is:
- **Efficient**: O(N × D × num_slices) vs. O(N²) for energy-based tests
- **Accurate**: Captures multivariate structure with enough slices
- **Scalable**: Works for high-dimensional embeddings

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
