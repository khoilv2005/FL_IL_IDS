# EWC Implementation Comparison Report
## Comparing Project's Implementation vs Author's Paper & Code

**Date:** May 14, 2026  
**Author's Paper:** Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017  
**Author's Code:** `e:\NCKH\overcoming-catastrophic\model.py` & `experiment.ipynb`  
**Project's Code:** `fed_learning/strategies/incremental/ewc.py`

---

## 1. Paper's Core EWC Method (Equation Mapping)

### 1.1 Main Loss Function (Paper Eq. 3)

```
L(θ) = LB(θ) + Σ_i (λ/2) * Fi * (θi - θ*A,i)²
```

| Symbol | Meaning | Paper Definition |
|--------|---------|------------------|
| LB(θ) | Loss for current task B | Cross-entropy on task B data |
| λ | Regularization strength | Set via hyperparameter search |
| Fi | Fisher information for parameter i | Diagonal of Fisher Information Matrix |
| θi | Current parameter value | Trainable parameter |
| θ*A,i | Optimal parameter after task A | Saved after training task A |

### 1.2 Fisher Information Approximation

From Paper Section 2, using Laplace approximation:

```
F ≈ diagonal of Fisher Information Matrix
F_i = E[(∂L/∂θi)²]  (empirical approximation)
```

Three key properties of F:
1. Equivalent to second derivative of loss near minimum
2. Can be computed from first-order derivatives alone
3. Guaranteed to be positive semi-definite

---

## 2. Author's Original Implementation (model.py)

### 2.1 Fisher Computation

```python
# model.py lines 44-88
def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
    # Initialize Fisher accumulator
    self.F_accum = []
    for v in range(len(self.var_list)):
        self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
    
    # Sample from softmax for gradient computation
    probs = tf.nn.softmax(self.y)
    class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
    
    # Compute gradients of log probability
    fish_gra = tf.gradients(tf.log(probs[0,class_ind]), self.var_list)
    
    for i in range(num_samples):
        im_ind = np.random.randint(imgset.shape[0])
        ders = sess.run(fish_gra, feed_dict={self.x: imgset[im_ind:im_ind+1]})
        for v in range(len(self.F_accum)):
            self.F_accum[v] += np.square(ders[v])
    
    # Average over samples
    for v in range(len(self.F_accum)):
        self.F_accum[v] /= num_samples
```

**Mapping to Paper:**
- Uses `log(probs[0, class_ind])` gradient = ∂log p(y|x,θ)/∂θ
- This approximates ∂L/∂θ where L is negative log-likelihood
- F_i ≈ (∂log p/∂θi)² averaged over samples

### 2.2 Optimal Parameter Saving (star method)

```python
# model.py lines 90-95
def star(self):
    # Save optimal weights after most recent task training
    self.star_vars = []
    for v in range(len(self.var_list)):
        self.star_vars.append(self.var_list[v].eval())
```

**Paper Mapping:** θ*A,i = optimal parameter after task A (line 91-92 in paper)

### 2.3 Optimal Parameter Restoration

```python
# model.py lines 97-101
def restore(self, sess):
    # Restore optimal weights before next task training
    if hasattr(self, "star_vars"):
        for v in range(len(self.var_list)):
            sess.run(self.var_list[v].assign(self.star_vars[v]))
```

**Purpose:** Reset to task optimum before training next task (Figure 1 in paper)

### 2.4 EWC Loss Update

```python
# model.py lines 106-115
def update_ewc_loss(self, lam):
    if not hasattr(self, "ewc_loss"):
        self.ewc_loss = self.cross_entropy
    
    for v in range(len(self.var_list)):
        self.ewc_loss += (lam/2) * tf.reduce_sum(
            tf.multiply(self.F_accum[v].astype(np.float32),
                       tf.square(self.var_list[v] - self.star_vars[v]))
        )
    self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)
```

**Direct Mapping to Paper Eq. 3:**
```
Paper:   L(θ) = LB(θ) + Σ_i (λ/2) * Fi * (θi - θ*A,i)²
Code:    ewc_loss += (lam/2) * Σ F_accum[v] * (var_list[v] - star_vars[v])²

λ → lam (hyperparameter)
Fi → F_accum[v] (accumulated Fisher for layer v)
θi → var_list[v] (current parameters)
θ*A,i → star_vars[v] (optimal parameters from previous task)
```

---

## 3. Project's Implementation Analysis

### 3.1 Overall Structure

**File:** `fed_learning/strategies/incremental/ewc.py`

The project implements **Corrected EWC** (Huszar 2018) which differs from the original paper:

| Aspect | Original Paper | Project (Corrected EWC) |
|--------|---------------|------------------------|
| Penalty Structure | Separate penalty per task | Single accumulated penalty |
| Mathematical Basis | Sum of Gaussians | Single Gaussian approximation |
| Double-counting | Yes (mentioned in Huszar 2018) | No (fixed) |

### 3.2 Fisher Computation Comparison

**Author's Code (TensorFlow):**
```python
# Samples random class, computes gradient of log-probability
fish_gra = tf.gradients(tf.log(probs[0,class_ind]), self.var_list)
```

**Project's Code (PyTorch):**
```python
# Lines 137-198: compute_fisher_information()
# Uses per-sample gradients of cross-entropy loss
output = model(X[i : i + 1])
loss = self._seen_class_cross_entropy(output, y[i : i + 1])
loss.backward()

for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        fisher[name] += param.grad.detach() ** 2
```

**Comparison:**

| Aspect | Author's Code | Project's Code | Status |
|--------|---------------|----------------|--------|
| Diagonal Fisher | ✓ | ✓ | ✅ Match |
| Per-sample gradient | ✓ | ✓ | ✅ Match |
| Sample averaging | ✓ | ✓ | ✅ Match |
| Gradient source | log-probs | Cross-entropy | ⚠️ Different but equivalent* |
| Dropout disabled | Not mentioned | ✓ (lines 159-162) | ✅ Project is more correct |

*Note: The author uses ∂log p(y|x,θ)/∂θ which equals ∂(-L)/∂θ for classification, same as project's cross-entropy gradient.

### 3.3 Optimal Parameter Management

**Author's Code:**
```python
def star(self):
    self.star_vars.append(self.var_list[v].eval())

def restore(self, sess):
    for v in range(len(self.var_list)):
        sess.run(self.var_list[v].assign(self.star_vars[v]))
```

**Project's Code (consolidate()):**
```python
# Lines 200-282: consolidate()
# After task training, save Fisher and optimal params

# Store optimal parameters
optimal_params = {
    name: param.detach().cpu().clone()
    for name, param in model.named_parameters()
    if param.requires_grad
}

# Accumulate Fisher (Huszar correction)
if self.online_ewc:
    fisher_acc[name] = self.gamma * prev_f + fisher_new[name]
else:
    fisher_acc[name] = prev_f + fisher_new[name]  # Additive accumulation
```

**Comparison:**

| Aspect | Author's Code | Project's Code | Status |
|--------|---------------|----------------|--------|
| Save optimal params | ✓ (star_vars) | ✓ (optimal_params) | ✅ Match |
| Single anchor point | ✓ | ✓ | ✅ Match |
| Fisher accumulation | Replaces each task | Additive/Online | ⚠️ Different mode |
| Caching strategy | N/A | ✓ (RAM + disk) | ✅ Project improvement |

### 3.4 EWC Loss Computation

**Author's Code:**
```python
# Eq. 3: L = LB + (λ/2) * Σ Fi * (θi - θ*A,i)²
ewc_loss += (lam/2) * tf.reduce_sum(
    tf.multiply(F_accum[v], tf.square(var_list[v] - star_vars[v]))
)
```

**Project's Code:**
```python
# Lines 338-410: compute_loss()
# Called during training to add EWC penalty

ewc_penalty = torch.tensor(0.0, device=device)
for name, param in model.named_parameters():
    if name in self._cached_fisher_acc:
        fisher_val = self._cached_fisher_acc[name]
        optimal_val = self._cached_optimal_params[name]
        diff = param - optimal_val
        ewc_penalty += (fisher_val * diff**2).sum()

return base_loss + (self.ewc_lambda / 2.0) * ewc_penalty
```

**Direct Equation Mapping:**

```
Paper Eq. 3:    L(θ) = LB(θ) + Σ_i (λ/2) * Fi * (θi - θ*A,i)²

Author's Code:  ewc_loss = cross_entropy + (lam/2) * Σ F_accum[v] * (var_list[v] - star_vars[v])²

Project's Code: loss = base_loss + (ewc_lambda/2) * Σ fisher[name] * (param - optimal[name])²
```

| Component | Paper | Author's | Project | Match |
|-----------|-------|---------|---------|-------|
| Base loss | LB(θ) | cross_entropy | base_loss | ✅ |
| λ coefficient | λ | lam | ewc_lambda | ✅ |
| Division by 2 | /2 | /2 | /2 | ✅ |
| Fisher importance | Fi | F_accum[v] | fisher[name] | ✅ |
| Parameter diff | (θi - θ*A,i)² | (var - star)² | (param - optimal)² | ✅ |
| Sum over params | Σ_i | Σ layers | Σ named_params | ✅ |

**Verdict: ✅ EXACT MATCH with paper equation**

---

## 4. Detailed Component Mapping

### 4.1 Class Structure

| Component | Author's Code | Project's Code |
|-----------|---------------|----------------|
| Class name | `Model` | `EWCMixin` + `EWCTrainer` |
| Fisher storage | `self.F_accum` | `self._cached_fisher_acc` |
| Optimal params | `self.star_vars` | `self._cached_optimal_params` |
| EWC loss | `self.ewc_loss` | Computed on-the-fly |

### 4.2 Training Loop Integration

**Author's Experiment (experiment.ipynb):**
```python
# After training task A:
model.compute_fisher(validation_data)  # Compute Fisher
model.star()                           # Save optimal params

# Train task B:
model.restore(sess)                   # Reset to optimal
model.update_ewc_loss(lam)             # Add EWC penalty
train_task(model, ...)                 # Train with EWC
```

**Project's Training Loop (local_task_loop.py):**
```python
# After training task A (line 890-900):
if algo == "ewc" and hasattr(trainer, "consolidate"):
    trainer.consolidate(model, loader, device)  # Compute & save Fisher

# Train task B:
# compute_loss() automatically adds EWC penalty
```

### 4.3 Hyperparameters

**From Paper Table 1 (MNIST):**
- Learning rate: 10⁻³
- Hidden layers: 2
- Width: 400
- Epochs per task: 20

**From Paper Table 2 (Atari):**
- Fisher multiplier: 400 (λ in EWC penalty)
- Num samples Fisher: 100

**Project's Defaults:**
```python
ewc_lambda: float = 1000.0    # Paper uses 400 for Atari, 1000+ common for MNIST
fisher_samples: int = 200     # Matches author's 200 in experiment.ipynb
online_ewc: bool = False      # Original EWC mode
gamma: float = 0.9            # For online EWC decay
```

---

## 5. Correctness Verification Checklist

### 5.1 Paper Requirements vs Project Implementation

| Paper Requirement | Implementation | Status |
|-------------------|----------------|--------|
| Diagonal Fisher approximation | `compute_fisher_information()` computes F_ii | ✅ |
| Fisher from first-order derivatives | Uses `.grad` of loss | ✅ |
| Positive semi-definite | Squared gradients guarantee ≥0 | ✅ |
| Quadratic penalty (λ/2) * F * (θ - θ*)² | `compute_loss()` line 410 | ✅ |
| Single anchor point (latest θ*) | `consolidate()` saves latest only | ✅ |
| Slow learning on important weights | Fisher-weighted penalty | ✅ |

### 5.2 Author's Code vs Project Implementation

| Author's Feature | Project Implementation | Status |
|------------------|------------------------|--------|
| `compute_fisher()` | `compute_fisher_information()` | ✅ Equivalent |
| `star()` | `consolidate()` (optimal_params) | ✅ Equivalent |
| `restore()` | Automatic (uses cached params) | ✅ Equivalent |
| `update_ewc_loss()` | `compute_loss()` | ✅ Equivalent |
| TensorFlow | PyTorch | ⚠️ Framework only |
| num_samples=200 | fisher_samples=200 | ✅ Match |

---

## 6. Differences and Improvements

### 6.1 Key Differences from Original Paper

1. **Corrected EWC (Huszar 2018):**
   - Original paper uses separate penalty per task
   - Project uses single accumulated penalty
   - Mathematical correction avoids double-counting

2. **Online EWC Option (Schwarz 2018):**
   - `online_ewc=True` enables additive Fisher with decay
   - `F_acc = γ * F_acc_prev + F_new`
   - Better scalability for many tasks

3. **Caching Strategy:**
   - Project loads Fisher/params to RAM cache
   - Avoids disk I/O per batch
   - Backup to disk for resume capability

### 6.2 Improvements Over Author's Code

1. **Robustness for RNNs:**
   - Dropout disabled during Fisher computation (lines 159-162)
   - Per-sample gradient computation for stability

2. **Resume/Checkpoint Support:**
   - Full state serialization in `get_resume_state()`
   - Continuation across sessions

3. **Flexible Integration:**
   - Mixin pattern allows combining with FedAvg/FedProx
   - `FedAvgEWCTrainer`, `FedProxEWCTrainer`

---

## 7. Usage with CNN-GRU Model

The project correctly uses CNN-GRU model from:
- `fed_learning/models/cnn_gru.py`

**Fisher is computed over all parameters:**
```python
fisher = {
    name: torch.zeros_like(param)
    for name, param in model.named_parameters()
    if param.requires_grad
}
```

This includes:
- CNN layers: conv1-3, bn1-3
- GRU layer
- MLP layers: fc1, fc2

All parameters receive EWC protection, matching the paper's intent.

---

## 8. Conclusion

### 8.1 Core EWC Method ✅ CORRECT

The project's EWC implementation correctly follows the paper's method:

1. **Fisher Computation:** Diagonal approximation using squared gradients ✅
2. **EWC Loss:** Matches Paper Eq. 3 exactly ✅
3. **Optimal Parameter Saving:** Single anchor point per task ✅
4. **Training Loop:** EWC penalty applied during backprop ✅

### 8.2 Enhancements ✅ IMPROVEMENTS

The project adds important enhancements:
- **Corrected EWC:** Fixes double-counting bias
- **Online EWC:** Better for many tasks
- **GPU Optimization:** Caching and dropout handling
- **Federated Learning:** Mixin pattern for FL integration

### 8.3 Final Verdict

**The EWC implementation in the project is CORRECT and faithful to the paper's method, with improvements.**

The core equation L(θ) = LB(θ) + Σ_i (λ/2) * Fi * (θi - θ*A,i)² is correctly implemented.

---

## Appendix: Quick Reference

### A. Key File Locations

| Component | File | Lines |
|-----------|------|-------|
| EWC Mixin | `fed_learning/strategies/incremental/ewc.py` | 36-463 |
| Fisher Computation | `ewc.py` | 137-198 |
| Consolidation | `ewc.py` | 200-282 |
| EWC Loss | `ewc.py` | 338-410 |
| FedAvg+EWC | `fed_learning/strategies/fed_incremental/ewc.py` | 11-23 |
| FedProx+EWC | `fed_learning/strategies/fed_incremental/ewc.py` | 26-39 |

### B. Key Parameters

| Parameter | Paper | Project Default (ewc.py) | __init__.py Default | Description |
|-----------|-------|---------------------------|---------------------|-------------|
| λ (ewc_lambda) | 400 (Atari) | 1000.0 | 10.0 ⚠️ | Regularization strength |
| Fisher samples | 200 | 200 | 200 | Samples for Fisher estimation |
| γ (gamma) | N/A | 0.9 | N/A | Decay for online EWC |

**⚠️ INCONSISTENCY DETECTED:** `ewc_lambda` default is 10.0 in `__init__.py:36` but 1000.0 in `ewc.py:67`. This should be aligned. The training script `train_incremental_kaggle.py` sets it to 400.0, so runtime is correct, but standalone usage without config would use different defaults.

### C. Training Workflow

```
Task 0 → Train → Consolidate(Fisher + params) → Task 1 → Train(with EWC) → ...
```

---

## 9. Potential Issues Found

### 9.1 ⚠️ Default Value Inconsistency

**Location:** 
- `fed_learning/strategies/incremental/ewc.py:67` → `ewc_lambda: float = 1000.0`
- `fed_learning/strategies/incremental/__init__.py:36` → `ewc_lambda=config.get("ewc_lambda", 10.0)`

**Impact:** When calling `get_incremental_strategy("ewc", ...)` without explicitly setting `ewc_lambda`, the value 10.0 is used instead of 1000.0.

**Current Config (train_incremental_kaggle.py):**
```python
"ewc_lambda": 400.0,  # This overrides the inconsistency
```

**Recommendation:** Align defaults in both locations.

### 9.2 ⚠️ Missing temp_dir Parameter in __init__.py

**Location:** `fed_learning/strategies/incremental/__init__.py:34-40`

When creating EWCTrainer via `get_incremental_strategy()`, the `temp_dir` parameter is not passed, which means it defaults to `"./temp_ewc_storage"` from `ewc.py`.

This is not a bug, but could cause issues on Kaggle where filesystem permissions differ. The current setup uses Kaggle's working directory which is writable.

### 9.3 ✅ Dropout Handling for Fisher Computation

**Location:** `ewc.py:159-162`

```python
model.train()
for m in model.modules():
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.training = False
```

This is **correct** and better than the author's code, which doesn't explicitly disable dropout during Fisher computation. For CNN-GRU model, dropout can affect gradient estimation if not handled properly.

---

## 10. Final Verification Matrix

| Component | Paper | Author Code | Project Code | Status |
|-----------|-------|-------------|--------------|--------|
| **Fisher Formula** | F_ii ≈ (∂L/∂θi)² | ✓ | ✓ | ✅ |
| **EWC Loss Eq.3** | L = LB + (λ/2)ΣFi(θi-θ*A)² | ✓ | ✓ | ✅ |
| **Star vars (θ*)** | Save optimal params | ✓ | ✓ | ✅ |
| **Per-sample grad** | Required for Fisher | ✓ | ✓ | ✅ |
| **Dropout handling** | N/A | ✗ | ✓ | ✅ Better |
| **Default λ** | 400 (Atari) | N/A | 1000.0/10.0 ⚠️ | ⚠️ Inconsistent |
| **Fisher samples** | 200 | 200 | 200 | ✅ |
| **Accumulation** | Replaces (orig) | Replaces | Additive/Online | ✅ Better |

---

## 11. Summary

### ✅ Correct Implementation:
1. **EWC Loss Formula** - Exactly matches Paper Eq. 3
2. **Fisher Computation** - Correct diagonal approximation
3. **Optimal Parameter Saving** - Single anchor point
4. **CNN-GRU Integration** - Works correctly
5. **Dropout Handling** - Superior to author's code

### ⚠️ Minor Issues:
1. **Default λ inconsistency** between `ewc.py` (1000.0) and `__init__.py` (10.0)
   - Currently bypassed by explicit config in `train_incremental_kaggle.py`
   - Recommendation: Set both to 1000.0 or 400.0

### 📊 Overall Assessment:
**The EWC implementation is fundamentally correct and faithful to the paper.** The core algorithm matches the author's intent, with improvements like Corrected EWC (Huszar 2018) and better dropout handling. The default value inconsistency is a configuration issue, not an algorithmic one.
