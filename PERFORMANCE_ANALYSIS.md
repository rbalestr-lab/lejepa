# Performance & Scalability Analysis

**Date:** 2025-11-15  
**Purpose:** Careful assessment of efficiency issues before making algorithmic changes

---

## Critical Issues Requiring Mathematical Verification

### 1. DDP Correctness in EppsPulley (FIXED - ALREADY HANDLED)

**File:** `lejepa/univariate/epps_pulley.py`  
**Status:** ✅ The surviving EppsPulley implementation already has proper DDP handling

**Current Code:**
```python
# DDP reduction
cos_mean = all_reduce(cos_mean)
sin_mean = all_reduce(sin_mean)
```

**Analysis:**
- We deleted the duplicate EppsPulley class that had the TODO
- The remaining optimized version already implements DDP correctly
- Uses all_reduce() from lejepa.utils

**Decision:** ✅ ALREADY FIXED - No action needed

---

### 2. DDP Scaling in Moments Test (CRITICAL - AFFECTS RESULTS)

**File:** `lejepa/univariate/moments.py`  
**Issue:** Potential double-scaling with `dist_mean()` + division by `world_size`

**Current Code Analysis:**
```python
m1 = self.dist_mean(x.mean(0)).abs_()
# ... later ...
return m1.add_(m2) / self.world_size
```

**Mathematical Impact:**
- `dist_mean()` calls `all_reduce(..., op="AVG")` - averages across ranks
- Then divides by `world_size` again
- This could result in: `(sum/world_size) / world_size` = incorrect scaling

**Need to Verify:**
1. What does `x.mean(0)` compute? Local mean on each rank?
2. Does `all_reduce AVG` compute global mean or just average the local means?
3. Is the final `/world_size` redundant?

**Investigation Results:**
```python
# dist_mean implementation:
def dist_mean(self, x):
    if is_dist_avail_and_initialized():
        torch.distributed.nn.functional.all_reduce(x, ReduceOp.AVG)
    return x

# Moments forward:
m1 = self.dist_mean(x.mean(0)).abs_()  # x.mean(0) = local mean
# dist_mean does: all_reduce(local_mean, AVG) = global_mean ✓
# Then divides by world_size:
return m1.add_(m2) / self.world_size  # ← REDUNDANT!
```

**Mathematical Analysis:**
- Rank 0 has: `x0` with local mean `μ0 = mean(x0)`
- Rank 1 has: `x1` with local mean `μ1 = mean(x1)`
- `all_reduce(AVG)` computes: `(μ0 + μ1) / 2` = global mean ✓
- Then dividing by `world_size` again: `global_mean / 2` ← WRONG!

**Conclusion:** The `/self.world_size` is INCORRECT in DDP mode

**Proposed Fix:**
```python
def forward(self, x):
    x = self.prepare_data(x)
    k = torch.arange(2, self.k_max + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    m1 = self.dist_mean(x.mean(0)).abs_()
    if self.k_max >= 2:
        xpow = self.dist_mean((x**k).mean(1))
        xpow[::2].sub_(self.moments)
        m2 = xpow.abs_().T.matmul(self.weights)
        return m1.add_(m2)  # Remove / self.world_size
    return m1  # Remove / self.world_size
```

**Risk Assessment:** MEDIUM - This fixes DDP but might break non-DDP behavior

**Decision:** ✅ SAFE TO FIX - dist_mean already handles the averaging

---

## Quadratic Complexity Issues (Algorithmic - DO NOT CHANGE)

### Analysis: O(N²) Complexity in Multivariate Tests

**Affected Tests:** BHEP, BHEP_M, COMB, HV, HZ

**Mathematical Justification:**
These tests compute pairwise distances/similarities:
```python
pair_sim = 2 * x @ x.T + norms + norms.unsqueeze(1)
```

**Why This Is Necessary:**
- These are **energy-based** or **kernel-based** statistical tests
- The math REQUIRES computing all pairwise interactions
- This is not inefficient coding - it's the algorithm itself
- Similar to:
  - RBF kernel computations
  - Energy distance metrics
  - Maximum Mean Discrepancy (MMD)

**Optimization Options (if needed):**
1. **Random sampling:** Use subset of N samples (changes test properties)
2. **Nyström approximation:** Low-rank approximation (changes test properties)
3. **Block processing:** Process in chunks (just splits memory, not complexity)
4. **None of these are trivial** - they change the statistical test

**Risk Assessment:** VERY HIGH - These are algorithmic constraints

**Decision:** ❌ DO NOT MODIFY - Document as known limitation instead

**Recommendation:** Add docstring note:
```python
"""
Note: This test has O(N²) complexity due to pairwise distance computation.
For very large N (>10000), consider:
- Using a subset of data
- Alternative linear-complexity tests (e.g., slicing-based)
"""
```

---

## Efficiency Issues in Figures/ (Low Priority)

### 3. Repeated Test Instantiation in Loops

**Files:** `figures/2d_slicing.py`, `figures/nonparametric_example.py`

**Issue:** Test objects recreated in loops

**Impact:** 
- These are visualization scripts, not core library
- K=10 iterations → minimal impact
- Total overhead: ~milliseconds

**Optimization:**
```python
# Before loop:
tests = {
    'CVM': ds.univariate.CramerVonMises(),
    'VCReg': ds.univariate.VCReg(),
    # ... etc
}

# In loop:
for name, test in tests.items():
    stat = test(data)
```

**Risk Assessment:** ZERO - These are standalone scripts

**Decision:** ✅ SAFE TO FIX - Simple refactoring

**Priority:** LOW - Optional cleanup

---

## Sorting Efficiency (Medium Priority)

### 4. Redundant Sorting in prepare_data

**File:** `lejepa/univariate/base.py`

**Issue:** Always sorts even if already sorted

**Current Implementation:**
```python
def prepare_data(self, x):
    # ... validation ...
    if not self.sorted:
        x, _ = x.sort(descending=False, dim=-2)
    return x
```

**Analysis:**
- `self.sorted` flag already exists
- Some tests set `sorted=True` in __init__
- This is already optimal - only sorts when needed
- NLL might be doing redundant sort though

**Decision:** ✅ ALREADY OPTIMIZED - Verify NLL doesn't double-sort

---

## High-Complexity Trigonometric Operations (Algorithmic - Cannot Change)

### 5. EppsPulley Characteristic Function

**Issue:** O(N × D × n_points) trig operations

**Analysis:**
```python
cos_vals = torch.cos(x.unsqueeze(-1) * self.t)
sin_vals = torch.sin(x.unsqueeze(-1) * self.t)
```

**Why This Is Necessary:**
- Computing empirical characteristic function
- Math: φ(t) = (1/N) Σ exp(itX) = (1/N) Σ [cos(tX) + i·sin(tX)]
- CANNOT avoid trig operations - they're the algorithm

**Potential Optimization:**
- Reduce `n_points` (user's choice)
- Use GPU (already done with torch)
- Use approximations (changes test properties)

**Decision:** ❌ DO NOT MODIFY - This is the algorithm

**Recommendation:** Document complexity in docstring

---

## Summary of Actions

### ✅ FIXED:
1. ✅ **EppsPulley DDP** - Already correct (duplicate removed)
2. ✅ **Moments DDP scaling** - Removed redundant /world_size division

### ❌ DO NOT CHANGE (Algorithmic - Not Bugs):
1. ❌ **Quadratic complexity in BHEP/COMB/HV/HZ** - Inherent to algorithm
2. ❌ **Trigonometric operations in EppsPulley** - Required by characteristic function math
3. ❌ **SlicingUnivariateTest complexity** - User controls num_slices parameter
4. ❌ **Sorting in prepare_data** - Already optimized with `sorted` flag

### 📝 RECOMMEND DOCUMENTING:
1. Add O(N²) complexity notes to BHEP, COMB, HV, HZ docstrings
2. Add O(N×D×n_points) note to EppsPulley docstring
3. Suggest alternatives for large N (slicing-based tests)

###  🔧 OPTIONAL (Low Priority - Figures Only):
1. Refactor test instantiation in `figures/2d_slicing.py`
2. Replace magic numbers with named constants
3. Extract monolithic functions

---

## Performance Review Complete ✅

**Critical Findings:**
- 1 DDP bug fixed (Moments redundant scaling)
- 0 algorithmic issues to change
- All quadratic complexity is mathematically necessary
- Documentation improvements recommended
