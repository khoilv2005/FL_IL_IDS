# CGoFed Implementation Verification Document
## Mapping: Paper Equations ↔ Pseudocode ↔ Actual Code

**Paper**: "CGoFed: Constrained Gradient Optimization Strategy for Federated Class Incremental Learning" (IEEE TKDE 2025)

---

## 1. REPRESENTATION SPACE (Paper Section 5.1, Eq. 2-4)

### Paper Eq. 2: Representation from Forward Propagation
**Formula**: $R^t = F(\Theta^t, X^t) = [z_1, ..., z_n]$

**Pseudocode**:
```
for each batch (X, y) in data_loader:
    z = forward(X)  # Get layer activations
    R = concatenate([R, z])
```

**Actual Code**: `cgofed.py:247-405`
```python
def _collect_activations(self, model, data_loader, device, num_samples):
    # Line 280-370: Hook functions capture input activations
    def make_hook(layer_name):
        def hook_fn(module, inp, out):
            activation = inp[0]  # Capture input activation (Eq. 2)
            # Line 288-367: Handle Linear/Conv activation shapes
            captured[layer_name] = features.cpu()
    # Line 377-400: Forward pass triggers hooks
    for X, y in data_loader:
        _ = model(X)  # R^t = F(Θ^t, X^t)
```

### Paper Eq. 3: SVD Decomposition
**Formula**: $R^t = U^t \Sigma^t (V^t)^T$

**Pseudocode**:
```
U, S, Vh = SVD(R.T)  # Note: R.T for correct dimensions
```

**Actual Code**: `cgofed.py:538-542`
```python
# Line 538-542
A = R.T  # [d, N] - Paper uses R^T for SVD
U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # Eq. 3
```

### Paper Eq. 4: Energy-Based Rank Selection
**Formula**: Select $\kappa$ s.t. $\frac{\sum_{i=1}^{\kappa} \sigma_i^2}{\sum_{i=1}^{d} \sigma_i^2} \geq \text{threshold}$

**Pseudocode**:
```
cum_energy = cumsum(S^2)
k = min(i where cum_energy[i] / total_energy >= threshold)
```

**Actual Code**: `cgofed.py:544-550`
```python
# Line 544-550
energy = S**2
cum_energy = torch.cumsum(energy, dim=0)
total_energy = cum_energy[-1] + 1e-10
ratio = cum_energy / total_energy
k = (ratio < self.energy_threshold).sum().item() + 1  # Eq. 4
k = min(k, len(S), self.max_rank)
```

---

## 2. IMPORTANCE WEIGHTS (Paper Eq. 5)

### Paper Eq. 5: Sigmoid Importance
**Formula**: $\Lambda^t = \text{sigmoid}(\Sigma^t) = \frac{1}{1 + \exp(-\beta \cdot \sigma_i)}$

**Pseudocode**:
```
importance = sigmoid(beta * S[:k])
```

**Actual Code**: `cgofed.py:554`
```python
# Line 554
importance = torch.sigmoid(self.beta * S[:k])  # Eq. 5
```

---

## 3. RELAXATION COEFFICIENT (Paper Eq. 7-8)

### Paper Eq. 7: Power Decay
**Formula**: $f(\alpha, t) = \alpha^t$

**Pseudocode**:
```
mu_coefficient = alpha ^ (current_task - t_reset)
```

**Actual Code**: `cgofed.py:152-154`
```python
# Line 152-154
if task_id > 0:
    # Paper: f(α, t) = α^t
    self.mu_coefficient = self.lambda_decay ** (task_id - self.t_reset)  # Eq. 7
```

### Paper Eq. 8: Reset Condition
**Formula**: If $AF \geq \theta$, then $\alpha = 1.0$ and $t_{reset} = t$

**Pseudocode**:
```
if AF > theta_threshold:
    t_reset = current_task
    mu_coefficient = 1.0
```

**Actual Code**: `cgofed.py:191-197`
```python
# Line 191-197 (in update_forgetting)
if self.last_af > self.theta_threshold:  # Eq. 8 condition
    self.t_reset = self.current_task
    self.mu_coefficient = 1.0  # Reset
    print(f"⚠️ AF={self.last_af:.4f} > θ={self.theta_threshold}, reset μ to 1.0")
```

### Paper Eq. 8 (continued): Effective μ
**Formula**: $\mu_t = \mu_{init} \cdot f(\alpha, t - t_{reset})$

**Pseudocode**:
```
mu_t = mu_projection * mu_coefficient
```

**Actual Code**: `cgofed.py:721-722`
```python
# Line 721-722
mu_t = self.mu_projection * self.mu_coefficient  # Eq. 8
```

---

## 4. GRADIENT PROJECTION (Paper Eq. 9)

### Paper Eq. 9: Main Projection Formula
**Formula**: $\nabla L \leftarrow \nabla L - \mu_t \cdot (\nabla L) \cdot M^t \cdot (M^t)^T$

Where $M^t = U[:, :\kappa]$ (basis) and projection matrix $P = M \cdot \text{diag}(\Lambda) \cdot M^T$

**Pseudocode**:
```
for each layer:
    P = M @ diag(importance) @ M.T  # Build projection matrix
    grad_new = grad - mu_t * (grad @ P)
    grad = grad_new
```

**Actual Code**: `cgofed.py:713-736`
```python
# Line 713-723: Build projection matrix (called in _cache_projection_matrices)
Uf = torch.mm(basis * importance, basis.T)  # [d, d] - Eq. 9 projection matrix

# Line 714-736 (in pre_step): Apply projection
projected = torch.mm(grad_2d, Uf)  # grad @ M @ M^T
mu_t = self.mu_projection * self.mu_coefficient  # μ_t
grad_new = grad_2d - mu_t * projected  # Eq. 9: g' = g - μ_t * (g @ P)
```

---

## 5. CROSS-TASK REGULARIZATION (Paper Section 5.2, Eq. 10-14)

### Paper Eq. 9 (Section 5.2): Similarity Computation
**Formula**: $\text{sim}(R_1, R_2) = \frac{R_1 \cdot R_2}{||R_1|| \cdot ||R_2||}$ (Cosine similarity)

**Pseudocode**:
```
similarity = dot(R1, R2) / (norm(R1) * norm(R2))
```

**Actual Code**: `cgofed.py:902-928`
```python
# Line 902-928
def _compute_similarity(self, R1: torch.Tensor, R2: torch.Tensor) -> float:
    norm1 = torch.norm(R1)
    norm2 = torch.norm(R2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    cosine_sim = torch.dot(R1.flatten(), R2.flatten()) / (norm1 * norm2)  # Eq. 9
    return float(torch.clamp(cosine_sim, -1.0, 1.0).item())
```

### Paper Eq. 10: TOP-K Selection
**Formula**: Select $K$ models with highest similarity

**Pseudocode**:
```
similarities = []
for each historical task:
    sim = compute_similarity(current_rep, hist_rep)
    similarities.append((task_id, sim))
similarities.sort(reverse=True)
selected = similarities[:K]
```

**Actual Code**: `cgofed.py:930-964`
```python
# Line 930-964
def _select_top_k_similar(self) -> List[Dict]:
    similarities = []
    for tid in range(self.current_task):
        if tid in self.task_representations and tid in self.task_global_models:
            hist_rep = self.task_representations[tid]
            sim = self._compute_similarity(current_rep, hist_rep)  # Eq. 9
            similarities.append({"task_id": tid, "similarity": sim, ...})
    
    similarities.sort(key=lambda x: x["similarity"], reverse=True)  # Sort
    selected = similarities[:self.top_k]  # Eq. 10: Select TOP-K
    return selected
```

### Paper Eq. 11: Weighted Aggregation with History
**Formula**: $\theta_{final} = (1 - \lambda) \cdot \theta_{current} + \lambda \cdot \sum_i (w_i \cdot \theta_{hist,i})$

Where $w_i = \text{softmax}(similarity_i)$

**Pseudocode**:
```
weights = softmax(similarities)
hist_agg = sum(weights[i] * hist_models[i])
result = (1 - lambda) * current + lambda * hist_agg
```

**Actual Code**: `cgofed.py:967-1043`
```python
# Line 967-1043
def _weighted_aggregate_with_history(self, current_params, selected_models):
    # Line 996-1000: Compute softmax weights
    sim_scores = torch.tensor([s["similarity"] for s in selected_models])
    sim_scores = torch.clamp(sim_scores, min=-10.0, max=10.0)
    weights = F.softmax(sim_scores, dim=0)  # Eq. 11: w_i = softmax(sim_i)
    
    # Line 1005-1020: Weighted aggregation of historical models
    hist_agg = OrderedDict()
    for i, model_info in enumerate(selected_models):
        w = weights[i].item()
        for k, v in model_info["params"].items():
            if k not in hist_agg:
                hist_agg[k] = w * v.float()
            else:
                hist_agg[k] += w * v.float()
    
    # Line 1023-1043: Blend with current
    λ = self.cross_task_weight
    result = OrderedDict()
    for k in current_params:
        hist_v = hist_agg[k].to(current_params[k].device)
        result[k] = (1 - λ) * current_params[k].float() + λ * hist_v  # Eq. 11
```

---

## 6. AVERAGE FORGETTING (Paper Eq. 16)

### Paper Eq. 16: AF Calculation
**Formula**: $AF = \frac{1}{t} \sum_{j<t} \max(0, a_j^{best} - a_j^{current})$

**Pseudocode**:
```
forgetting = []
for each previous task j:
    f = best_acc[j] - current_acc[j]
    forgetting.append(max(0, f))
AF = mean(forgetting)
```

**Actual Code**: `cgofed.py:179-189`
```python
# Line 179-189
def update_forgetting(self, task_accuracies: Dict[int, float]):
    # Line 181-187
    forgetting = []
    for tid in range(self.current_task):
        if tid in self.best_acc_per_task and tid in self.current_acc_per_task:
            f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
            forgetting.append(max(0, f))  # Eq. 16
    
    if forgetting:
        self.last_af = sum(forgetting) / len(forgetting)  # Eq. 16: AF
```

---

## 7. DATA FLOW VERIFICATION

### Task Transition Flow

```
Task 0 (No Projection):
  ├─ collect_activations() → SVD → cache basis
  └─ No projection (task_id == 0)

Task 1+ (With Projection):
  ├─ set_task() → update μ_coefficient (Eq. 7-8)
  ├─ pre_step() → project gradients (Eq. 9)
  │   ├─ Load cached basis from previous tasks
  │   ├─ Build projection matrix: Uf = M @ diag(Λ) @ M^T
  │   └─ Apply: grad = grad - μ_t * (grad @ Uf)
  ├─ Aggregation
  └─ Cross-task blend (if task > 0) (Eq. 10-11)
```

---

## 8. KEY IMPLEMENTATION DIFFERENCES FROM PAPER

| Aspect | Paper | Implementation | Note |
|--------|-------|----------------|------|
| **Basis Storage** | In-memory | File-based (lazy loading) | `temp_dir` for scalability |
| **Activation Collection** | Full forward pass | Hook-based | More efficient |
| **Projection Target** | All layers | Linear + Conv only | See `_get_projection_target_modules()` |
| **Device Handling** | Single GPU | Multi-GPU | Per-device cache for thread safety |
| **SVD Energy** | Fixed 95% | Configurable (`energy_threshold`) | Tunable |

---

## 9. VERIFICATION CHECKLIST

- [x] Eq. 2: Forward activation collection → `_collect_activations()`
- [x] Eq. 3: SVD decomposition → `torch.linalg.svd()`
- [x] Eq. 4: Energy-based rank → `cumsum(energy)`
- [x] Eq. 5: Sigmoid importance → `torch.sigmoid()`
- [x] Eq. 7: Power decay → `lambda_decay ** (task_id - t_reset)`
- [x] Eq. 8: AF reset mechanism → `update_forgetting()`
- [x] Eq. 9: Gradient projection → `pre_step()`
- [x] Eq. 10: TOP-K selection → `_select_top_k_similar()`
- [x] Eq. 11: Weighted aggregation → `_weighted_aggregate_with_history()`
- [x] Eq. 16: AF calculation → `update_forgetting()`

---

## 10. CRITICAL CODE SECTIONS FOR DEBUGGING

### 10.1 Verify SVD Basis (Eq. 3-4)
**File**: `cgofed.py:538-550`
```python
# Check if rank is reasonable
print(f"Layer {layer_name}: R=[{n_samples}, {d}], k={k}")
# Expected: k should be ~20-50% of d for energy_threshold=0.85
```

### 10.2 Verify Gradient Projection (Eq. 9)
**File**: `cgofed.py:714-736`
```python
# Check reduction percentage
reduction = (orig_norm - new_norm) / orig_norm
print(f"{layer_name}: reduction={reduction*100:.1f}%")
# Expected: 10-70% depending on μ_t
```

### 10.3 Verify μ Calculation (Eq. 7-8)
**File**: `cgofed.py:152-166`
```python
print(f"μ_projection = {self.mu_projection} * {self.mu_coefficient:.4f} = {self.mu_projection * self.mu_coefficient:.4f}")
# Expected: Task 1=0.9, Task 2=0.54, Task 3=0.32, etc.
```

---

**Document Version**: 1.0  
**Generated**: For CGoFed Verification  
**Purpose**: Map implementation to paper equations for correctness validation
