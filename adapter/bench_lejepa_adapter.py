"""
Benchmark: LeJEPA Adapter vs Standard AdamW

Compares LeJEPA Adapter against standard AdamW optimizer across multiple
statistical tests to evaluate embedding quality improvements.

Tests evaluated:
- Epps-Pulley (characteristic function)
- Anderson-Darling (tail-sensitive EDF)
- Cramér-von Mises (balanced EDF)
- BHEP (energy-based, multivariate)

Metrics:
- Final loss values (lower = better normality)
- Training stability (loss variance)
- Learning dynamics (convergence speed)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from pathlib import Path

# Import LeJEPA tests
import lejepa as ds
from lejepa.univariate.epps_pulley import EppsPulley
from lejepa.univariate.anderson_darling import AndersonDarling
from lejepa.univariate.cramer_von_mises import CramerVonMises
from lejepa.multivariate.slicing import SlicingUnivariateTest
from lejepa.multivariate.bhep import BHEP

# Import adapter
from adapter import LeJEPAAdapter


class SimpleEncoder(nn.Module):
    """Simple encoder for testing embedding quality."""
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def generate_synthetic_data(n_samples: int = 1000, input_dim: int = 784) -> torch.Tensor:
    """Generate synthetic data for training."""
    return torch.randn(n_samples, input_dim)


def create_tests() -> Dict[str, nn.Module]:
    """Create all statistical tests for evaluation."""
    return {
        'epps_pulley': SlicingUnivariateTest(
            univariate_test=EppsPulley(),
            num_slices=512,
            reduction='mean'
        ),
        'anderson_darling': SlicingUnivariateTest(
            univariate_test=AndersonDarling(),
            num_slices=512,
            reduction='mean'
        ),
        'cramer_von_mises': SlicingUnivariateTest(
            univariate_test=CramerVonMises(),
            num_slices=512,
            reduction='mean'
        ),
        'bhep': BHEP(beta=0.1)
    }


def train_with_lejepa(
    model: nn.Module,
    data: torch.Tensor,
    univariate_test: str,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    batch_size: int = 128,
    num_slices: int = 512
) -> Dict[str, List[float]]:
    """Train model with LeJEPA adapter."""
    
    # Map test names to adapter format
    test_mapping = {
        'epps_pulley': 'epps_pulley',
        'anderson_darling': 'anderson_darling',
        'cramer_von_mises': 'cramer_von_mises',
        'bhep': 'epps_pulley'  # BHEP test uses epps_pulley for SIGReg
    }
    
    adapter_test = test_mapping.get(univariate_test, 'epps_pulley')
    
    # Create adapter
    adapter = LeJEPAAdapter(
        model.parameters(),
        lr=5e-4,
        weight_decay=5e-2,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        univariate_test=adapter_test,
        num_slices=num_slices,
        reduction='mean'
    )
    
    # Track metrics
    losses = []
    lrs = []
    
    # Training loop
    for step in range(total_steps):
        # Sample batch
        idx = torch.randint(0, data.size(0), (batch_size,))
        batch = data[idx]
        
        # Forward pass
        adapter.zero_grad()
        embeddings = model(batch)
        loss = adapter.compute_loss(embeddings)
        
        # Backward pass
        loss.backward()
        adapter.step()
        
        # Track metrics
        losses.append(loss.item())
        lrs.append(adapter.get_stats()['current_lr'])
    
    return {
        'losses': losses,
        'lrs': lrs,
        'final_loss': np.mean(losses[-100:])  # Average last 100 steps
    }


def train_with_adamw(
    model: nn.Module,
    data: torch.Tensor,
    test: nn.Module,
    total_steps: int = 1000,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 5e-2
) -> Dict[str, List[float]]:
    """Train model with standard AdamW (no SIGReg loss)."""
    
    # Create optimizer (no scheduler for fair comparison)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Track metrics
    losses = []
    
    # Training loop
    for step in range(total_steps):
        # Sample batch
        idx = torch.randint(0, data.size(0), (batch_size,))
        batch = data[idx]
        
        # Forward pass
        optimizer.zero_grad()
        embeddings = model(batch)
        
        # Compute loss using same test
        with torch.no_grad():
            loss = test(embeddings)
        
        # For AdamW, we use MSE to 0 (simple regularization)
        # This is NOT the same as SIGReg but provides a baseline
        train_loss = embeddings.pow(2).mean()
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Track test statistic
        losses.append(loss.item())
    
    return {
        'losses': losses,
        'lrs': [lr] * total_steps,  # Constant LR
        'final_loss': np.mean(losses[-100:])
    }


def run_benchmark(
    test_name: str,
    test: nn.Module,
    data: torch.Tensor,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    num_runs: int = 5
) -> Dict[str, any]:
    """Run benchmark comparing LeJEPA vs AdamW."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}\n")
    
    lejepa_results = []
    adamw_results = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...")
        
        # LeJEPA Adapter
        model_lejepa = SimpleEncoder()
        torch.manual_seed(run)
        np.random.seed(run)
        
        start_time = time.time()
        result_lejepa = train_with_lejepa(
            model_lejepa,
            data,
            univariate_test=test_name,  # Pass the test name directly
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )
        lejepa_time = time.time() - start_time
        lejepa_results.append(result_lejepa)
        
        # Standard AdamW
        model_adamw = SimpleEncoder()
        torch.manual_seed(run)
        np.random.seed(run)
        
        start_time = time.time()
        result_adamw = train_with_adamw(
            model_adamw,
            data,
            test,
            total_steps=total_steps
        )
        adamw_time = time.time() - start_time
        adamw_results.append(result_adamw)
        
        print(f"  LeJEPA final loss: {result_lejepa['final_loss']:.6f} ({lejepa_time:.2f}s)")
        print(f"  AdamW final loss: {result_adamw['final_loss']:.6f} ({adamw_time:.2f}s)")
    
    # Aggregate results
    lejepa_losses = [r['final_loss'] for r in lejepa_results]
    adamw_losses = [r['final_loss'] for r in adamw_results]
    
    improvement = (np.mean(adamw_losses) - np.mean(lejepa_losses)) / np.mean(adamw_losses) * 100
    
    print(f"\n{'-'*60}")
    print(f"Summary ({num_runs} runs):")
    print(f"  LeJEPA: {np.mean(lejepa_losses):.6f} ± {np.std(lejepa_losses):.6f}")
    print(f"  AdamW:  {np.mean(adamw_losses):.6f} ± {np.std(adamw_losses):.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    print(f"{'-'*60}")
    
    return {
        'test_name': test_name,
        'lejepa_results': lejepa_results,
        'adamw_results': adamw_results,
        'lejepa_mean': np.mean(lejepa_losses),
        'lejepa_std': np.std(lejepa_losses),
        'adamw_mean': np.mean(adamw_losses),
        'adamw_std': np.std(adamw_losses),
        'improvement_pct': improvement
    }


def plot_results(all_results: List[Dict], output_dir: Path = Path('adapter')):
    """Create visualization of benchmark results."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Extract data
    test_names = [r['test_name'].replace('_', ' ').title() for r in all_results]
    lejepa_means = [r['lejepa_mean'] for r in all_results]
    lejepa_stds = [r['lejepa_std'] for r in all_results]
    adamw_means = [r['adamw_mean'] for r in all_results]
    adamw_stds = [r['adamw_std'] for r in all_results]
    improvements = [r['improvement_pct'] for r in all_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LeJEPA Adapter Benchmark Results', fontsize=16, fontweight='bold')
    
    # 1. Bar chart: Mean final losses
    ax1 = axes[0, 0]
    x = np.arange(len(test_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, lejepa_means, width, yerr=lejepa_stds, 
                    label='LeJEPA Adapter', capsize=5, color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, adamw_means, width, yerr=adamw_stds,
                    label='Standard AdamW', capsize=5, color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Final Loss (lower = better)', fontweight='bold')
    ax1.set_title('Final Loss Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Improvement percentage
    ax2 = axes[0, 1]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax2.barh(test_names, improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Improvement (%)', fontweight='bold')
    ax2.set_title('LeJEPA vs AdamW Improvement', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax2.text(val, i, f' {val:.1f}%', va='center', 
                ha='left' if val > 0 else 'right', fontweight='bold')
    
    # 3. Training curves for first test (example)
    ax3 = axes[1, 0]
    example_result = all_results[0]
    lejepa_losses = example_result['lejepa_results'][0]['losses']
    adamw_losses = example_result['adamw_results'][0]['losses']
    
    ax3.plot(lejepa_losses, label='LeJEPA Adapter', color='#2ecc71', linewidth=2)
    ax3.plot(adamw_losses, label='Standard AdamW', color='#e74c3c', linewidth=2)
    ax3.set_xlabel('Training Step', fontweight='bold')
    ax3.set_ylabel('Loss', fontweight='bold')
    ax3.set_title(f'Training Dynamics ({test_names[0]})', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Learning rate schedule
    ax4 = axes[1, 1]
    lrs = example_result['lejepa_results'][0]['lrs']
    ax4.plot(lrs, color='#3498db', linewidth=2)
    ax4.set_xlabel('Training Step', fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontweight='bold')
    ax4.set_title('LeJEPA Learning Rate Schedule', fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'benchmark_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved benchmark plot to: {output_path}")
    
    plt.close()


def print_summary_table(all_results: List[Dict]):
    """Print formatted summary table."""
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY TABLE")
    print("="*80)
    print(f"{'Test':<20} {'LeJEPA Mean':<15} {'AdamW Mean':<15} {'Improvement':<15}")
    print("-"*80)
    
    for r in all_results:
        test_name = r['test_name'].replace('_', ' ').title()
        lejepa_str = f"{r['lejepa_mean']:.6f} ± {r['lejepa_std']:.6f}"
        adamw_str = f"{r['adamw_mean']:.6f} ± {r['adamw_std']:.6f}"
        imp_str = f"{r['improvement_pct']:+.2f}%"
        
        print(f"{test_name:<20} {lejepa_str:<15} {adamw_str:<15} {imp_str:<15}")
    
    print("="*80)
    print("\nInterpretation:")
    print("  • Lower loss = embeddings closer to standard normal N(0, I)")
    print("  • Positive improvement = LeJEPA Adapter outperforms AdamW")
    print("  • SIGReg loss provides regularization beyond simple MSE")
    print("="*80 + "\n")


def main():
    """Run full benchmark suite."""
    
    print("\n" + "="*80)
    print("LeJEPA Adapter Benchmark")
    print("="*80)
    print("\nConfiguration:")
    print("  • Model: 3-layer MLP (784 → 512 → 512 → 256)")
    print("  • Training: 1000 steps, batch size 128")
    print("  • LeJEPA: lr=5e-4, weight_decay=5e-2, warmup=100 steps")
    print("  • AdamW: lr=5e-4, weight_decay=5e-2, constant LR")
    print("  • Tests: Epps-Pulley, Anderson-Darling, Cramér-von Mises, BHEP")
    print("  • Runs per test: 5")
    print("="*80)
    
    # Generate data
    print("\nGenerating synthetic data...")
    data = generate_synthetic_data(n_samples=5000, input_dim=784)
    
    # Create tests
    tests = create_tests()
    
    # Run benchmarks
    all_results = []
    for test_name, test in tests.items():
        result = run_benchmark(
            test_name=test_name,
            test=test,
            data=data,
            total_steps=1000,
            warmup_steps=100,
            num_runs=5
        )
        all_results.append(result)
    
    # Print summary
    print_summary_table(all_results)
    
    # Plot results
    plot_results(all_results)
    
    print("\n✓ Benchmark complete!")


if __name__ == '__main__':
    main()
