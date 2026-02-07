# AGENTS.md - AI4FIDS Project

## Project Overview

Python/PyTorch research project for Federated Class Incremental Learning (FCIL) applied to Intrusion Detection Systems. Core library: `fed_learning/` (local package, not pip-installable). Entry point: `train_incremental_kaggle.py` (designed for Kaggle).

10 pluggable FL strategies: FedAvg, FedProx, FedAvgM, Fed+, CGoFed, FedCBDR, FedAvg+EWC, FedProx+EWC, FedAvg+LwF, FedProx+LwF.

## Build & Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

No build step. `fed_learning/` is imported directly. On Kaggle, `sys.path` manipulation imports it from a dataset.

Dependencies: torch>=1.12, numpy>=1.21, pandas>=1.3, scikit-learn>=1.0, matplotlib>=3.5, seaborn>=0.11, tqdm>=4.62, pytest>=7.0.

## Test Commands

```bash
pytest tests/test_algorithms.py -v                                    # All tests
pytest tests/test_algorithms.py::TestCGoFed -v                        # Single class
pytest tests/test_algorithms.py::TestCGoFed::test_cgofed_set_task -v  # Single method
pytest tests/test_algorithms.py -k "test_fedavg" -v                   # Keyword match
pytest tests/test_algorithms.py --cov=fed_learning --cov-report=html  # Coverage
```

Tests use `sys.path.insert` for imports (no conftest.py). No pytest config file.

## Linting & Formatting

No linting tools configured. No CI/CD. Follow conventions below.

## Project Structure

```
fed_learning/
  core/                    # BaseTrainer (4 hooks), BaseAggregator
  strategies/
    federated/             # FedAvg, FedProx, FedAvgM, Fed+
    incremental/           # CGoFed, EWC (mixin), FedLwF, FedCBDR
    __init__.py            # STRATEGIES registry + get_strategy() factory
  models/                  # CNN_GRU_Model (DeepFed paper)
  clients/                 # FederatedClient, CGoFedClient, FedLwFClient, FedCBDRClient
  servers/                 # FederatedServer, IncrementalServer, FedLwFServer, FedCBDRServer
  data/                    # loader.py (bulk NPZ), incremental_loader.py (task-aware)
  training/                # runner.py + per-algorithm GPU workers
  visualization/           # IEEE-standard plots
prepare_data/              # CSV -> NPZ preprocessing pipeline
tests/test_algorithms.py   # 47 unit tests (only test file)
train_incremental_kaggle.py  # Main entry point
```

## Architecture & Key Patterns

### Strategy Pattern + Factory
```python
from fed_learning.strategies import get_strategy
trainer, aggregator = get_strategy("fedprox", mu_fedprox=0.5)
```
All strategies registered in `STRATEGIES` dict in `fed_learning/strategies/__init__.py`.

### Hook-based Training Loop
`BaseTrainer` in `core/trainer.py` provides 4 hooks (override to customize):
- `compute_loss()` — Add regularization (FedProx proximal, EWC penalty, LwF distillation)
- `pre_train()` — Before training (CGoFed SVD basis computation)
- `pre_step()` — After backward, before optimizer.step (CGoFed gradient projection)
- `post_step()` — After optimizer.step (Fed+ weight correction)

### Mixin Pattern (EWC)
```python
class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer): pass   # EWC must come first in MRO
class FedProxEWCTrainer(EWCMixin, FedProxTrainer): pass
```

### Critical: Dual-mu Config for CGoFed
CGoFed uses **two separate mu values** — do not confuse them:
- `mu_fedprox` (CONFIG: 1.5) → proximal term `||θ - θ_global||²` in `compute_loss()`
- `mu_cgofed` (CONFIG: 5.0) → gradient projection coefficient (paper Eq. 9) in `pre_step()`

In `CGoFedTrainer`: `self.mu` = proximal, `self.mu_projection` = gradient projection. The factory maps `mu_fedprox` → `mu` and `mu_cgofed` → `mu_projection`.

### Data Format
NPZ files: `client_{cid}_train.npz` (keys: `X_train`, `y_train`), `global_test_data.npz` (keys: `X_test`, `y_test`). All data on CPU as tensors; moved to GPU at train time via `non_blocking=True`.

### Multi-GPU Training
`threading.Thread` distributes clients across GPUs. Mixed precision via `torch.cuda.amp`.

## Code Style Guidelines

### Imports
Stdlib → third-party → local, separated by blank lines. Relative imports within `fed_learning/` (e.g., `from ...core import BaseTrainer`). Absolute imports in tests and entry point.
```python
import os
from typing import Optional, Dict

import torch
import torch.nn as nn

from ..core import BaseTrainer, BaseAggregator
```

### Naming
- Classes: `PascalCase` (`FedProxTrainer`). Exception: `CNN_GRU_Model` (keep as-is)
- Functions/methods: `snake_case` (`compute_loss`, `train_federated_multigpu`)
- Constants: `UPPER_SNAKE_CASE` (`STRATEGIES`, `MODULE_PATH`)
- Private: `_` prefix (`_weighted_average`, `_create_batches`)

### Type Annotations
Use `typing` on public API signatures. Inline hints for class attributes.
```python
def aggregate(self, results: List[Dict], global_params: Optional[OrderedDict] = None) -> OrderedDict:
```

### Docstrings & Comments
Google-style docstrings with Args/Returns/Raises. Module docstrings include paper references. Inline comments are often in **Vietnamese** — preserve them. Use `# === SECTION ===` for major blocks.

### Formatting
- 4-space indent, ~100 char line length, trailing commas in multi-line collections
- One blank line between methods, two between top-level definitions

### Error Handling
- `ValueError` for bad algorithm names/config
- `try/except ImportError` for optional deps with feature flags (`VISUALIZATION_AVAILABLE`)

### Module Exports
Each package declares `__all__`. When adding strategies, update both `STRATEGIES` dict and `__all__` in `fed_learning/strategies/__init__.py`.

## Adding a New Strategy

1. Create trainer (extend `BaseTrainer`) in `strategies/federated/` or `strategies/incremental/`
2. Create aggregator (extend `BaseAggregator`) or reuse existing
3. Register in `STRATEGIES` dict in `strategies/__init__.py`
4. Add constructor logic in `get_strategy()` for custom args
5. Update `list_strategies()` to include the new strategy
6. Add tests in `tests/test_algorithms.py`
7. If needed: add specialized client in `clients/`, server in `servers/`, worker in `training/`
