# CGoFed: Paper vs Implementation Audit

**Date**: 2026-02-12
**Paper**: "CGoFed: Constrained Gradient Optimization Strategy for Federated Class Incremental Learning", IEEE TKDE, Vol. 37, No. 5, May 2025
**Authors**: Jiyuan Feng, Xu Yang, Liwen Liang, Weihong Han, Binxing Fang, Qing Liao

---

## Changes Made in This Audit

**FIX applied**: Removed `_weighted_aggregate_with_history()` server-side model blending from
`CGoFedAggregator.aggregate()`. This method blended the aggregated model with historical
task models at the server level — a mechanism **NOT described in the paper**. The paper's
Eq. 11 defines `A(Θ)` as a **local training loss term** only, not a server-side interpolation.

**Files changed**:
- `fed_learning/strategies/incremental/cgofed.py`:
  - Removed `_weighted_aggregate_with_history()` method (was ~80 lines)
  - Removed its call from `aggregate()` — server now returns pure FedAvg result
  - Updated `CGoFedAggregator` docstring to clarify paper-correct behavior
  - Fixed duplicate NaN check in `_compute_similarity()`
  - Removed unused `math` import
  - Fixed misleading docstring in `build_space_from_client_data()` that incorrectly said
    "Uses gradient vectors from samples (not activations)" — the code uses activations via
    forward hooks, matching Eq. 2
  - Fixed misleading docstring in `_store_client_representations()` that said
    "gradient representations" — corrected to "activation representations"

**What was preserved**:
- Similarity computation (Eq. 10) and storage at end of task — needed for next task's local regularization
- `_current_similarity_weights` / `_current_historical_models` — used by `get_local_regularization_info()` for base server path
- `cross_task_weight` constructor parameter — kept for config/test backward compatibility (now unused internally)

**Tests**: All 52 tests pass after the fix.

---

## Equation-by-Equation Analysis

### Eq. 2 — Representation Space: `R^t = F(Θ^t, X^t)`

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| What is R? | Representation matrix from forward propagation, `R^t = [z_1, ..., z_{n_s}]` | Input activations collected via forward hooks (`_collect_activations`, `cgofed.py:257-415`) |
| Scope | Single representation for the whole model | **Per-layer** representation — builds separate SVD basis for each Linear/Conv layer |
| Layer types | Not specified | Linear + Conv1d + Conv2d (with unfold for Conv to match weight dimensions) |

**Activations vs Gradients clarification**:

The paper's Section IV-A is titled "Strong-Constrainted Gradient Update" and uses the term
"gradient space," but Eq. 2 explicitly states the representation is obtained through
**"forward propagation"** — meaning **activations, NOT gradients**. The term "gradient space"
is used conceptually: for a linear layer `y = Wx`, the gradient `∂L/∂W = (∂L/∂y) · x^T`
lies in the space spanned by the input activations `x`. So collecting input activations via
forward propagation characterizes the gradient space indirectly. This is the same principle
used by GPM [41] (Saha et al., "Gradient Projection Memory for Continual Learning") which
the paper cites as its foundation.

The implementation correctly uses **input activations** via forward hooks (`_collect_activations`).

**[FIXED]** A misleading docstring in `build_space_from_client_data()` previously said
"Uses gradient vectors from samples (not activations)" — this was incorrect. The underlying
code always called `_collect_activations` which uses forward hooks. The docstring has been
corrected.

**Verdict**: **Correct**. The implementation uses activations from forward propagation, matching
Eq. 2. The per-layer approach is a practical engineering extension common in GPM-family methods.

---

### Eq. 3-4 — SVD & Rank Selection

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| SVD | `R^t = U^t Σ^t (V^t)^T` | `A = R.T; U, S, Vh = torch.linalg.svd(A)` (`cgofed.py:549-550`) |
| Basis | `M^t = [u_1, ..., u_κ]` from left singular vectors U | `basis = U[:, :k]` — same left singular vectors |
| Rank κ | Energy threshold ε: `‖(R^t)_κ‖²_F ≥ ε‖R^t‖²_F` | Cumulative energy ratio < `energy_threshold` (`cgofed.py:552-561`) |

**Verdict**: **Correct**. The transpose `R.T` ensures U columns span the feature dimension (matching gradient weight dimensions), which is the right choice for projection.

---

### Eq. 5 — Importance Weights

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| Formula | `Λ^t = 1/(1 + e^{-Σ^t})` | `torch.sigmoid(self.beta * S[:k])` (`cgofed.py:567`) |
| Weighted basis | `M^t = [λ_1·u_1, ..., λ_κ·u_κ]` | `weighted_basis = basis * importance` (`cgofed.py:859`) |

**Verdict**: **Correct** when `beta=1.0` (default). The `beta` parameter is an additional tuning knob not in the paper.

---

### Eq. 7-8 — Relaxation Coefficient μ_t

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| Decay | `f(α, t) = α^t` | `self.lambda_decay ** (task_id - self.t_reset)` (`cgofed.py:163`) |
| Reset | `μ^t = μ^init · f(α, t-t_τ) if AF ≥ τ` | `self.t_reset = current_task; self.mu_coefficient = 1.0` (`cgofed.py:203-204`) |
| μ_init | "typically set to 1" | `mu_projection` parameter (default falls back to `mu`) |

**Verdict**: **Correct**. The decay logic properly implements both cases of Eq. 8.

---

### Eq. 9 — Gradient Projection (core equation)

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| Formula | `∇L' = ∇L - μ_t(∇L)M^t(M^t)^T` | `grad_new = grad_2d - mu_t * torch.mm(grad_2d, Uf)` (`cgofed.py:771-782`) |

**IMPORTANT NOTE** on how `Uf` (the projection matrix) is built:

**Paper**: `P = M^t @ (M^t)^T` where `M^t` is the importance-weighted basis of task t. For multiple old tasks, the paper stores bases in memory `M = {M^1, M^2, ...}` but doesn't specify how to combine them.

**Implementation** (`_cache_projection_matrices`, `cgofed.py:823-915`):

1. Concatenates importance-weighted bases from **all** old tasks
2. Re-orthogonalizes via SVD to get union of subspaces
3. Builds projector: `Uf = U_orth @ diag(S_normalized²) @ U_orth^T`

This ensures eigenvalues of the projector are bounded in [0, 1], preventing gradient amplification when multiple tasks overlap. Without re-orthogonalization, naively summing per-task projectors could produce eigenvalues > 1.

**Verdict**: The projection formula matches Eq. 9. The **multi-task projector construction** is an engineering improvement over the paper's ambiguous multi-task description; it handles overlapping subspaces robustly.

---

### Eq. 10 — Cross-Task Similarity

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| Metric | `ϕ^t_i = D(R^t_i, R^t_k)`, L2-norm distance | Negative Frobenius norm: `-‖R1 - R2‖_F` (`cgofed.py:1085-1121`) |
| Selection | Choose π tasks with **lowest** ϕ (closest) | Sort by similarity descending, take top-K (`cgofed.py:1190-1193`) |
| Scope | Per-client similarity matrix Ψ_k across all other clients' tasks | Aggregator: task-level. CGoFedServer: per-client level (`cgofed_server.py:317-379`) |

**Verdict**: **Correct**. Frobenius norm = L2 norm for vectorized matrices. Selecting by highest `-distance` = lowest distance.

---

### Eq. 11 — Cross-Task Cooperation Loss A(Θ)

**Paper**: `A(Θ^t_k, Θ^old) = Σ_j Σ_i (ϕ^j_i / Σϕ) ‖Θ^t_k - Θ^j_i‖²`

This is a **loss term** during local training.

**Implementation** — Local loss term (`compute_loss`, `cgofed.py:612-657`):

```python
total_loss = ce_loss + (lambda_cross_task / 2) * Σ(w_i * ‖Θ - Θ_hist‖²)
```

- Matches the paper's intent (regularization toward historical models during local training)
- Uses softmax weights instead of ϕ/Σϕ (see Deviation #4 below)
- Adds `lambda_cross_task / 2` scaling (see Deviation #5 below)

**[FIXED]** Previously, the implementation also had server-side model blending via
`_weighted_aggregate_with_history()` which blended `(1-λ)*θ_curr + λ*Σ(w_i*θ_hist)`
at aggregation time. This was **NOT in the paper** and has been removed. The server
now returns pure FedAvg parameters. Cross-task regularization happens exclusively
through the local loss term, matching Algorithm 1.

**Verdict**: **Correct** after fix. Eq. 11 is now applied only as a local training loss term.

---

### Eq. 12 — Personalized Aggregation

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| Formula | `Θ^{t,g}_k = Θ^t_k + Σ_{i≠k} (ϕ^t_i/Σϕ) Θ^t_i` | `β * Θ^t_k + (1-β) * Σ w_i Θ^t_i` (`cgofed_server.py:509-513`) |
| Combination | **Additive** (Θ_k + weighted others) | **Convex combination** with `eq12_self_weight` (default 0.5) |
| Weights | `ϕ/Σϕ` (direct proportion to distance) | softmax(-distance) |
| Timing | After task completion | After each round, used to initialize next round |

**Verdict**: **Deviation**. The paper's additive formula `Θ_k + Σ w_i Θ_i` is not a proper weighted average (total weight > 1). The implementation's convex combination is more numerically sound. The paper says "The ratio between the model of client k and the sum of the models from other clients can be set empirically," suggesting the authors intended a tunable blend.

---

### Eq. 13-14 — Complete Objective

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| Formula | `min [1/K Σ L_k(Θ^t_k) + A(Θ^t_k, Θ^old)]` | `CE + (λ/2) * A(Θ)` with separate mu_projection for gradient projection |
| Proximal term | NOT included | `mu=0.0` default (disabled) — correct |
| CE uses | Personalized model Θ^{t,g}_k (Eq. 13) | Model initialized from Eq.12 personalized params (`cgofed_worker.py:59-63`) |

**Verdict**: **Correct**. The initialization from personalized params effectively implements Eq. 13's intent. No FedProx proximal term is added (matching the paper).

---

### Eq. 16 — Average Forgetting

| Aspect | Paper | Implementation |
|--------|-------|---------------|
| Formula | `AF = 1/(T-1) Σ max_t(a_{t,i} - a_{T,i})` | `Σ max(0, best_acc[j] - current_acc[j]) / count` (`cgofed.py:189-199`) |

**Verdict**: **Correct**. `best_acc_per_task[tid]` stores `max_t a_{t,i}`, and `current_acc_per_task[tid]` stores `a_{T,i}`.

---

## Summary of Differences

### Correctly Implemented

- **Eq. 2**: Representation via forward propagation
- **Eq. 3-4**: SVD decomposition and rank selection
- **Eq. 5**: Sigmoid importance weights
- **Eq. 7-8**: Relaxation coefficient with AF-triggered reset
- **Eq. 9**: Gradient projection formula
- **Eq. 10**: L2-distance similarity + top-K selection
- **Eq. 11**: Cross-task cooperation loss as local training loss term **(fixed in this audit)**
- **Eq. 13-14**: Complete objective with personalized model initialization
- **Eq. 16**: Average Forgetting metric

### Remaining Deviations / Extensions

| # | What | Paper | Implementation | Impact |
|---|------|-------|---------------|--------|
| 1 | **Per-layer processing** | Single global R^t | Per-layer SVD + per-layer projection | Practical necessity; paper's notation is simplified |
| 2 | **Multi-task projector** | Ambiguous on combining M^1, M^2, ... | Re-orthogonalize union of subspaces, bounded eigenvalues | Engineering improvement; prevents gradient amplification |
| 3 | **Eq. 12 formula** | Additive: `Θ_k + Σ w_i Θ_i` | Convex: `β*Θ_k + (1-β)*Σ w_i Θ_i` | More numerically sound; paper acknowledges ratio is empirical |
| 4 | **Weight normalization** | Direct proportion `ϕ/Σϕ` | softmax(-distance) | More intuitive (higher weight to more similar tasks) |
| 5 | **lambda_cross_task scaling** | No explicit scaling in Eq. 14 | `(λ/2) * A(Θ)` | Additional hyperparameter for controlling regularization strength |
| 6 | **Timing of Eq. 12** | End of task | Every round (personalized init for next round) | More frequent personalization than paper describes |

### Fixed Issues

| # | What | Before | After |
|---|------|--------|-------|
| 1 | **Server-side model blending** | `_weighted_aggregate_with_history()` blended `(1-λ)*θ_curr + λ*Σ(w_i*θ_hist)` at aggregation — NOT in paper, double-dipped on cross-task regularization | **Removed**. Server returns pure FedAvg. Eq. 11 applied only as local loss term. |
| 2 | **Duplicate NaN check** | `_compute_similarity()` had identical NaN/Inf guard duplicated | **Fixed**. Single check retained. |
| 3 | **Unused import** | `import math` was only used by the removed blending method | **Removed**. |
| 4 | **Misleading docstrings (activations vs gradients)** | `build_space_from_client_data()` said "Uses gradient vectors from samples (not activations)"; `_store_client_representations()` said "gradient representations" | **Fixed**. Docstrings corrected to "activation-based" / "activation representations", matching Eq. 2's "forward propagation". |

---

## Algorithm Flow Comparison (After Fix)

### Paper Algorithm 1

```
ON CLIENTS (parallel for each k):
  if t = 1:
    Train Θ^t_k via Eq. 1 (standard CE)
    Update gradient via Eq. 6 (strict constraint)
  else:
    Train Θ^t_k via Eq. 14 (CE + A(Θ) cross-task loss)
    Compute μ_t via Eq. 8
    Update gradient via Eq. 9 (relaxed constraint)
  Construct R^t_k via Eq. 2
  Update basis M^t_k via Eq. 3, 4
  Compute weights Λ^t via Eq. 5
  Send Θ^t_k and R^t_k to server

ON SERVER:
  if t = 1:
    Standard aggregation Θ^{t,g}_k
  Calculate similarity Ψ_k via Eq. 10
  Personalized aggregation Θ^{t,g}_k via Eq. 12
```

### Implementation Flow (After Fix)

```
SERVER set_task(t):
  Sync historical data from aggregator
  if t > 0:
    Compute per-client similarities (Eq. 10) using client train data
    Prepare per-client regularization info for Eq. 14

CLIENT train (parallel per GPU):
  Initialize model from Eq. 12 personalized params (or global if t=0)
  For each epoch/batch:
    Forward pass → CE loss + A(Θ) cross-task loss (Eq. 14)
    Backward pass
    Gradient projection via Eq. 9 (pre_step)
    Optimizer step
  Compute representation R^t_k (Eq. 2)

SERVER aggregate:
  FedAvg of client updates                    ← pure FedAvg, no blending
  Store representations (Eq. 2)
  Store client/global models for history
  At end of task: compute similarity (Eq. 10) for future use
  Compute Eq. 12 personalized models for next round
```

**Key alignment with paper**: The server now performs pure FedAvg + similarity computation. All cross-task regularization happens locally via compute_loss (Eq. 14). Personalized aggregation (Eq. 12) is handled by CGoFedServer.

---

## Overall Assessment

The implementation now faithfully captures all core algorithmic ideas of CGoFed as described in the paper:

- **Gradient projection** (Eq. 2-9): Correct with robust multi-task projector
- **Relaxation mechanism** (Eq. 7-8): Correct power decay with AF-triggered reset
- **Cross-task regularization** (Eq. 11, 14): Now correctly applied only as local loss term
- **Personalized aggregation** (Eq. 12): Implemented with convex combination variant
- **Similarity computation** (Eq. 10): Correct L2-based task selection

The remaining deviations (#1-6 in the table above) are engineering decisions that improve numerical stability or adapt the paper's simplified notation to real neural network architectures. None of them contradict the paper's core algorithm.
