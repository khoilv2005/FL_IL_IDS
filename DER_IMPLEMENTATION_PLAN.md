# DER (Dynamically Expandable Representation) Implementation Plan

## Context

Thêm DER như một FCIL strategy mới bên cạnh CGoFed, EWC, FedLwF, FedCBDR. DER là phương pháp **architecture-based** CIL (Yan et al., CVPR 2021) — mở rộng feature extractor mỗi task và dùng channel-level mask pruning để giữ model compact.

---

## Nguyên Tắc: Kế Thừa + Isolation

> **Tận dụng kế thừa** từ base classes và utility classes có sẵn.
> **KHÔNG sửa logic** của bất kỳ file hiện tại nào.
> Chỉ thêm dòng import/registry vào `__init__.py` và `train_incremental_kaggle.py`.

### Bảng Kế Thừa

| File DER mới | Kế thừa từ | Reuse được gì |
|-------------|-----------|---------------|
| `models/der_model.py` | `CNN_GRU_Model` | Toàn bộ CNN+GRU backbone via `get_fused_representation()` |
| `servers/der_server.py` | `IncrementalServer` | `evaluate_global()`, `evaluate_per_task()`, `set_task()`, device setup, `get/set_global_params()` |
| `clients/der_client.py` | `FederatedClient` | `setup_for_gpu()`, `_create_batches()`, `_amp_ctx()` |
| `clients/der_client.py` | Import `ReplayBuffer` từ `fedcbdr.py` | Replay buffer logic có sẵn |
| `strategies/incremental/der.py` | `BaseTrainer`, `BaseAggregator` | `compute_loss()` default, hooks, `_weighted_average()` |

### Files KHÔNG chỉnh sửa logic

Tất cả file hiện tại giữ nguyên. Chỉ thêm import/registry vào:
- `strategies/__init__.py`
- `strategies/incremental/__init__.py`
- `servers/__init__.py`
- `clients/__init__.py`
- `train_incremental_kaggle.py`

---

## DER Algorithm Summary (from paper)

### Core Idea
Mỗi task thêm một feature extractor mới. Các extractor cũ bị freeze.
```
Super-feature: u = [F_1(x), F_2(x), ..., F_t(x)]   (concatenation)
```

### Training Pipeline (mỗi task)

**Stage 1 — Representation Learning:**
- Train extractor mới `F_t` + auxiliary classifier `H_t^a`
- Auxiliary classifier: `|Y_t| + 1` classes (new classes + 1 "other")
- Channel-level mask pruning + gradient compensation
- Loss: `L_DER = L_{H_t} + λ_a * L_{H_t^a} + λ_s * L_S`

**Stage 2 — Classifier Learning:**
- Freeze tất cả extractors
- Train unified classifier `H_t` trên balanced data (current + exemplars)
- Temperature scaling: `Softmax(H_t(u) / δ)`

### Key Formulas

| Component | Formula |
|-----------|---------|
| Mask | `m_l = σ(s · e_l)` |
| Annealing | `s = 1/s_max + (s_max - 1/s_max) · (b-1)/(B-1)` |
| Sparsity Loss | `L_S = Σ active_weights / total_weights` |
| Gradient Compensation | `g' = g / (1 - m)` |

### Hyperparameters (paper defaults)
- `λ_a = 1.0`, `λ_s = 0.25~0.75`, `s_max = 15`, `δ = 1~5`

---

## Chi Tiết Từng File Mới

### 1. `fed_learning/models/der_model.py`

```python
from ..models.cnn_gru import CNN_GRU_Model

class CNNGRUBackbone(CNN_GRU_Model):
    """Feature-only extractor. Kế thừa CNN_GRU_Model, bỏ MLP head.

    Reuse hoàn toàn get_fused_representation() — không duplicate code.
    """
    def __init__(self, input_shape):
        super().__init__(input_shape, num_classes=2)  # dummy num_classes
        # Xóa classifier head (không cần cho feature extraction)
        del self.fc1, self.fc2, self.dropout
        self.output_dim = self.cnn_output_size + self.gru_output_size

    def forward(self, x):
        # Kế thừa trực tiếp từ CNN_GRU_Model
        return self.get_fused_representation(x)


class DERModel(nn.Module):
    """Dynamically Expandable Representation model.

    Grows by adding CNNGRUBackbone each task.
    Old backbones frozen, new backbone trainable.
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.extractors = nn.ModuleList()
        self.mask_embeds = nn.ParameterList()
        self.classifier = None
        self.aux_classifier = None
        self.feat_dim = None
        self.current_task = -1
        self.frozen_mask_values = {}

    def add_task(self, new_classes, s_max=15.0):
        """Add new extractor for a new task."""
        self.current_task += 1

        # 1. Freeze old extractor + binarize its mask
        if self.current_task > 0:
            prev = self.current_task - 1
            for p in self.extractors[prev].parameters():
                p.requires_grad = False
            self.frozen_mask_values[prev] = torch.sigmoid(
                self.mask_embeds[prev].data * s_max
            ).detach().clone()
            self.mask_embeds[prev].requires_grad = False

        # 2. Add new extractor (kế thừa CNN_GRU_Model backbone)
        backbone = CNNGRUBackbone(self.input_shape)
        self.extractors.append(backbone)
        if self.feat_dim is None:
            self.feat_dim = backbone.output_dim

        # 3. Mask embedding for new extractor
        self.mask_embeds.append(nn.Parameter(torch.zeros(self.feat_dim)))

        # 4. Expand unified classifier
        super_feat_dim = (self.current_task + 1) * self.feat_dim
        new_classifier = nn.Linear(super_feat_dim, self.num_classes)
        if self.classifier is not None:
            with torch.no_grad():
                old_out, old_in = self.classifier.weight.shape
                new_classifier.weight[:, :old_in] = self.classifier.weight
                new_classifier.bias[:] = self.classifier.bias
        self.classifier = new_classifier

        # 5. Auxiliary classifier: |new_classes| + 1
        self.aux_classifier = nn.Linear(self.feat_dim, len(new_classes) + 1)

    def get_super_feature(self, x, s=None):
        """Forward all extractors → apply masks → concatenate."""
        features = []
        for t, extractor in enumerate(self.extractors):
            feat = extractor(x)  # Gọi get_fused_representation() via inheritance
            if t < self.current_task:
                mask = self.frozen_mask_values[t].to(feat.device)
            elif s is not None:
                mask = torch.sigmoid(s * self.mask_embeds[t])
            else:
                mask = torch.sigmoid(self.mask_embeds[t])
            features.append(feat * mask)
        return torch.cat(features, dim=1)

    def forward(self, x, s=None):
        u = self.get_super_feature(x, s)
        return self.classifier(u)

    def forward_aux(self, x, s=None):
        feat = self.extractors[self.current_task](x)
        if s is not None:
            mask = torch.sigmoid(s * self.mask_embeds[self.current_task])
            feat = feat * mask
        return self.aux_classifier(feat)

    def get_trainable_params(self):
        """Only trainable params (for optimizer — Stage 1)."""
        params = list(p for p in self.extractors[self.current_task].parameters()
                      if p.requires_grad)
        if self.mask_embeds[self.current_task].requires_grad:
            params.append(self.mask_embeds[self.current_task])
        params.extend(self.classifier.parameters())
        if self.aux_classifier is not None:
            params.extend(self.aux_classifier.parameters())
        return params

    def get_classifier_params(self):
        """Only classifier params (for optimizer — Stage 2)."""
        return list(self.classifier.parameters())

    def freeze_all_extractors(self):
        """Freeze ALL extractors + masks (for Stage 2)."""
        for ext in self.extractors:
            for p in ext.parameters():
                p.requires_grad = False
        for emb in self.mask_embeds:
            emb.requires_grad = False

    def unfreeze_current_extractor(self):
        """Unfreeze current extractor (restore after Stage 2)."""
        for p in self.extractors[self.current_task].parameters():
            p.requires_grad = True
        self.mask_embeds[self.current_task].requires_grad = True

    def get_mask_sparsity_loss(self, s):
        mask = torch.sigmoid(s * self.mask_embeds[self.current_task])
        return mask.sum() / mask.numel()
```

### 2. `fed_learning/strategies/incremental/der.py`

```python
from ...core import BaseTrainer, BaseAggregator

class DERTrainer(BaseTrainer):
    """DER Trainer — kế thừa BaseTrainer, override compute_loss + pre_step."""

    def __init__(self, lambda_aux=1.0, lambda_sparsity=0.5, s_max=15.0,
                 temperature=2.0, buffer_size=500):
        self.lambda_aux = lambda_aux
        self.lambda_sparsity = lambda_sparsity
        self.s_max = s_max
        self.temperature = temperature
        self.buffer_size = buffer_size

        self.current_task = 0
        self.seen_classes = set()
        self.old_classes = []
        self.new_classes = []
        self.training_stage = 1

        self.current_batch = 0
        self.total_batches = 1

        # Forgetting tracking (same interface as FedLwF/FedCBDR)
        self.best_acc_per_task = {}
        self.current_acc_per_task = {}
        self.last_af = 0.0

    def set_task(self, task_id, new_classes):
        self.old_classes = list(self.seen_classes)
        self.new_classes = list(new_classes)
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        self.current_batch = 0

    def set_stage(self, stage):
        self.training_stage = stage
        self.current_batch = 0

    def compute_annealing_s(self):
        if self.total_batches <= 1:
            return self.s_max
        ratio = min(1.0, self.current_batch / max(1, self.total_batches - 1))
        return 1.0 / self.s_max + (self.s_max - 1.0 / self.s_max) * ratio

    def compute_loss(self, model, output, target, global_params=None,
                     inputs=None, **kwargs):
        if self.training_stage == 1:
            return self._stage1_loss(model, output, target, inputs)
        else:
            return self._stage2_loss(output, target)

    def _stage1_loss(self, model, output, target, inputs):
        """L_DER = L_CE + λ_a * L_aux + λ_s * L_sparsity"""
        s = self.compute_annealing_s()
        ce_loss = F.cross_entropy(output, target)

        aux_loss = torch.tensor(0.0, device=output.device)
        if hasattr(model, 'forward_aux') and inputs is not None and self.current_task > 0:
            aux_output = model.forward_aux(inputs, s=s)
            aux_target = self._remap_aux_targets(target, output.device)
            aux_loss = F.cross_entropy(aux_output, aux_target)

        sparsity_loss = torch.tensor(0.0, device=output.device)
        if hasattr(model, 'get_mask_sparsity_loss'):
            sparsity_loss = model.get_mask_sparsity_loss(s)

        self.current_batch += 1
        return ce_loss + self.lambda_aux * aux_loss + self.lambda_sparsity * sparsity_loss

    def _stage2_loss(self, output, target):
        return F.cross_entropy(output / self.temperature, target)

    def _remap_aux_targets(self, target, device):
        new_cls_map = {c: i for i, c in enumerate(self.new_classes)}
        other_idx = len(self.new_classes)
        remapped = torch.full_like(target, other_idx)
        for c, i in new_cls_map.items():
            remapped[target == c] = i
        return remapped

    def pre_step(self, model, global_params=None, **kwargs):
        """Gradient compensation (HAT paper): g = g / (1 - mask + eps)"""
        if self.training_stage != 1:
            return
        if not hasattr(model, 'mask_embeds') or model.current_task < 0:
            return
        # Implementation of gradient compensation...

    def update_forgetting(self, task_accuracies):
        """Same interface as FedLwF/FedCBDR."""
        self.current_acc_per_task = task_accuracies.copy()
        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)
        if len(self.best_acc_per_task) > 1:
            forgetting_sum = 0.0
            count = 0
            for tid in self.best_acc_per_task:
                if tid != self.current_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting_sum += max(0, f)
                    count += 1
            self.last_af = forgetting_sum / max(1, count)


class DERAggregator(BaseAggregator):
    """Kế thừa BaseAggregator, reuse _weighted_average()."""

    def __init__(self):
        self.trainable_keys = set()

    def set_trainable_keys(self, keys):
        self.trainable_keys = set(keys)

    def aggregate(self, results, global_params=None, **kwargs):
        agg = self._weighted_average(results)  # Reuse từ BaseAggregator
        # Restore frozen params from global
        if global_params is not None and self.trainable_keys:
            for k in agg:
                if k not in self.trainable_keys:
                    agg[k] = global_params[k].clone()
        return agg
```

### 3. `fed_learning/clients/der_client.py`

```python
from .client import FederatedClient                              # Kế thừa base
from ..strategies.incremental.fedcbdr import ReplayBuffer        # Reuse replay buffer
from ..core import BaseTrainer

class DERClient(FederatedClient):
    """DER Client — kế thừa FederatedClient + import ReplayBuffer từ FedCBDR.

    Reuse từ FederatedClient:
    - setup_for_gpu(model, device)
    - _create_batches(batch_size)
    - _amp_ctx()

    Reuse từ FedCBDR:
    - ReplayBuffer (class-balanced storage, importance sampling, herding)
    """

    def __init__(self, client_id, X_train, y_train, buffer_size=500):
        super().__init__(client_id, X_train, y_train)
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)  # Reuse!
        self.buffer_size = buffer_size
        self.current_task = 0
        self.current_classes = set()
        self.seen_classes = set()

    def set_task_data(self, X_train, y_train, task_id, task_classes):
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)
        self.current_task = task_id
        self.current_classes = set(task_classes)
        self.seen_classes.update(task_classes)

    def train(self, trainer, epochs, batch_size, lr, global_params=None,
              stage=1, **kwargs):
        """Two-stage training."""
        self.model.train()

        # Choose params based on stage
        if stage == 2 and hasattr(self.model, 'get_classifier_params'):
            params = self.model.get_classifier_params()
        elif hasattr(self.model, 'get_trainable_params'):
            params = self.model.get_trainable_params()
        else:
            params = self.model.parameters()

        optimizer = trainer.get_optimizer_class()(params, lr=lr)
        scaler = GradScaler(enabled=self.use_amp)
        trainer.set_stage(stage)
        trainer.pre_train(self.model, global_params, lr=lr)

        total_loss = 0.0
        total_samples = 0

        for ep in range(epochs):
            if stage == 2:
                batch_gen = self._create_balanced_batches(batch_size)
            elif self.replay_buffer.total_samples > 0 and self.current_task > 0:
                batch_gen = self._create_combined_batches(batch_size)
            else:
                batch_gen = self._create_batches(batch_size)  # Reuse từ base!

            for X_batch, y_batch in batch_gen:
                optimizer.zero_grad()
                with self._amp_ctx():  # Reuse từ base!
                    out = self.model(X_batch)
                    loss = trainer.compute_loss(
                        self.model, out, y_batch, global_params, inputs=X_batch)

                # Standard backward + step (same pattern as base FederatedClient)
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0)
                    trainer.pre_step(self.model, global_params)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0)
                    trainer.pre_step(self.model, global_params)
                    optimizer.step()
                trainer.post_step(self.model, global_params)

                total_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)

        trainer.post_train(self.model, global_params)

        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict(
                (k, v.cpu().clone()) for k, v in self.model.state_dict().items()),
        }

    def _create_combined_batches(self, batch_size, replay_ratio=0.5):
        """Current + replay data. Reuse ReplayBuffer.get_all_samples()."""
        X_replay, y_replay = self.replay_buffer.get_all_samples()
        if X_replay is None:
            yield from self._create_batches(batch_size)  # Fallback to base
            return
        # Mix current + replay (same pattern as FedCBDRClient)
        # ...

    def _create_balanced_batches(self, batch_size):
        """Balanced batches for Stage 2. Reuse ReplayBuffer.get_balanced_batch()."""
        # Combine current data + replay buffer, sample balanced per class
        # ReplayBuffer.get_balanced_batch() provides balanced replay samples
        # ...

    def update_exemplars(self, model):
        """Herding selection → add to ReplayBuffer."""
        # Extract features via model.extractors[current_task](x)
        # Herding: iteratively select closest-to-mean
        # self.replay_buffer.add_samples(X_sel, y_sel, imp_sel, class_ids=...)
```

### 4. `fed_learning/training/der_worker.py`

```python
from ..models.der_model import DERModel
from ..strategies.incremental.der import DERTrainer

def train_der_clients_on_gpu(gpu_id, clients, global_params, config,
                              results_dict, trainer, use_cpu, stage=1):
    """Same threading pattern as fedcbdr_worker.py."""
    device = "cpu" if use_cpu else f"cuda:{gpu_id}"

    # DERModel thay vì CNN_GRU_Model
    model = DERModel(config["input_shape"], config["num_classes"]).to(device)

    epochs = config.get("local_epochs", 3)
    batch_size = config.get("batch_size", 128)
    lr = config.get("learning_rate", 0.001)

    for client in clients:
        model.load_state_dict({k: v.to(device) for k, v in global_params.items()})
        client.setup_for_gpu(model, device)  # Reuse từ FederatedClient
        result = client.train(
            trainer=trainer, epochs=epochs, batch_size=batch_size,
            lr=lr, global_params=global_params, stage=stage)
        results_dict[client.client_id] = result

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 5. `fed_learning/servers/der_server.py`

```python
from .incremental_server import IncrementalServer

class DERServer(IncrementalServer):
    """DER Server — KẾ THỪA IncrementalServer.

    Reuse từ IncrementalServer (→ FederatedServer):
    - evaluate_global(seen_classes_only=True)   ← hoạt động vì DERModel.forward() trả logits
    - evaluate_per_task()                        ← không cần override
    - set_task() base logic                      ← override để thêm model expansion
    - get_global_params() / set_global_params()  ← hoạt động với mọi nn.Module
    - Device setup (num_gpus, primary_device, use_cpu)
    - History dict

    Override:
    - __init__(): Tạo DERModel thay vì CNN_GRU_Model
    - set_task(): Thêm model.add_task() + aggregator key update
    - train_round(): Thêm stage parameter + dùng der_worker
    """

    def __init__(self, clients, test_data, config):
        # Gọi parent (tạo CNN_GRU_Model tạm)
        super().__init__(clients, test_data, config)

        # Thay thế global_model bằng DERModel
        from ..models.der_model import DERModel
        del self.global_model
        self.global_model = DERModel(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)

        print(f"📊 Strategy: DER (Dynamically Expandable Representation)")

    def set_task(self, task_id, task_classes, seen_classes=None):
        # Parent xử lý: task tracking, seen_classes, task_classes dict
        super().set_task(task_id, task_classes, seen_classes)

        # DER-specific: expand model
        s_max = self.config.get("s_max", 15.0)
        self.global_model.add_task(task_classes, s_max=s_max)

        # Update aggregator trainable keys
        if hasattr(self.aggregator, 'set_trainable_keys'):
            trainable_keys = [k for k, p in self.global_model.named_parameters()
                              if p.requires_grad]
            self.aggregator.set_trainable_keys(trainable_keys)

    def train_round(self, participating_clients=None, stage=1, verbose=True):
        """Override: thêm stage param, dùng der_worker."""
        from ..training.der_worker import train_der_clients_on_gpu

        clients = participating_clients or self.clients
        global_params = self.get_global_params()  # Reuse từ parent

        # Same multi-GPU threading pattern
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        results_dict = {}
        threads = []
        for gpu_id in range(self.num_gpus):
            if clients_per_gpu[gpu_id]:
                t = Thread(target=train_der_clients_on_gpu, args=(
                    gpu_id, clients_per_gpu[gpu_id], global_params,
                    self.config, results_dict, self.trainer, self.use_cpu, stage))
                threads.append(t)
                t.start()
        for t in threads:
            t.join()

        results = list(results_dict.values())
        new_params = self.aggregator.aggregate(results, global_params)
        self.set_global_params(new_params)  # Reuse từ parent

        avg_loss = float(np.mean([r["loss"] for r in results]))
        if verbose:
            print(f"  → Stage {stage} loss: {avg_loss:.4f}")
        return {"train_loss": avg_loss}

    def coordinate_exemplar_update(self, participating_clients=None):
        clients = participating_clients or self.clients
        for client in clients:
            if hasattr(client, 'update_exemplars'):
                client.update_exemplars(self.global_model)
```

---

## Files Chỉ Thêm Import/Registry

### 6. `fed_learning/strategies/__init__.py`

```python
# THÊM import
from .incremental.der import DERTrainer, DERAggregator

# THÊM vào STRATEGIES dict
"der": {"trainer": DERTrainer, "aggregator": DERAggregator},

# THÊM vào get_strategy()
elif algo_lower == "der":
    trainer = strategy["trainer"](
        lambda_aux=config.get("lambda_aux", 1.0),
        lambda_sparsity=config.get("lambda_sparsity", 0.5),
        s_max=config.get("s_max", 15.0),
        buffer_size=config.get("buffer_size", 500),
    )

# THÊM vào list_strategies()
"der": "DER - Dynamically Expandable Representation for FCIL",
```

### 7. `fed_learning/strategies/incremental/__init__.py`
```python
from .der import DERTrainer, DERAggregator
```

### 8. `fed_learning/servers/__init__.py`
```python
from .der_server import DERServer
```

### 9. `fed_learning/clients/__init__.py`
```python
from .der_client import DERClient
```

### 10. `train_incremental_kaggle.py`

```python
# THÊM imports
from fed_learning.clients.der_client import DERClient
from fed_learning.servers.der_server import DERServer

# THÊM CONFIG params
"lambda_aux": 1.0,
"lambda_sparsity": 0.5,
"s_max": 15.0,
"der_temperature": 2.0,
"der_stage1_rounds": 5,
"der_stage2_rounds": 3,

# create_clients(): THÊM elif
elif algo == "der":
    client = DERClient(cid, X, y, buffer_size=config.get("buffer_size", 500))

# get_algorithm_specific_components(): THÊM elif
elif algo == "der":
    return DERServer(clients, test_data, task_config)

# post_task_processing(): THÊM elif
elif algo == "der":
    if hasattr(server, 'coordinate_exemplar_update'):
        server.coordinate_exemplar_update(participating_clients)

# Task loop: THÊM if/else cho two-stage
if algo == "der":
    for r in range(config.get("der_stage1_rounds", config["rounds_per_task"])):
        server.train_round(participating_clients, stage=1)
    for r in range(config.get("der_stage2_rounds", 3)):
        server.train_round(participating_clients, stage=2)
else:
    train_federated_multigpu(server, task_config)  # Code hiện tại KHÔNG đổi
```

---

## Thứ Tự Implementation

| Step | File | Loại | Kế thừa từ |
|------|------|------|-----------|
| 1 | `models/der_model.py` | **MỚI** | `CNN_GRU_Model` |
| 2 | `strategies/incremental/der.py` | **MỚI** | `BaseTrainer`, `BaseAggregator` |
| 3 | `clients/der_client.py` | **MỚI** | `FederatedClient` + import `ReplayBuffer` |
| 4 | `training/der_worker.py` | **MỚI** | Pattern từ `fedcbdr_worker.py` |
| 5 | `servers/der_server.py` | **MỚI** | `IncrementalServer` |
| 6 | 4× `__init__.py` | **THÊM IMPORT** | — |
| 7 | `train_incremental_kaggle.py` | **THÊM ELIF** | — |

---

## Lưu Ý Quan Trọng

### 1. CNNGRUBackbone kế thừa CNN_GRU_Model
- `del self.fc1, self.fc2, self.dropout` sau `super().__init__()` là an toàn
- `get_fused_representation()` không dùng fc1/fc2/dropout → không bị ảnh hưởng
- State_dict sẽ KHÔNG chứa fc1/fc2 → sạch

### 2. DERServer kế thừa IncrementalServer
- `super().__init__()` tạo CNN_GRU_Model tạm → `del` + replace bằng DERModel
- `evaluate_global()` gọi `self.global_model(X_batch)` → DERModel.forward() trả logits → hoạt động
- `set_task()` gọi `super().set_task()` trước → thêm model expansion sau

### 3. DERClient reuse ReplayBuffer từ fedcbdr.py
- `ReplayBuffer` là utility class độc lập (không phụ thuộc FedCBDRTrainer)
- Có sẵn: `add_samples()`, `get_all_samples()`, `get_balanced_batch()`, rebalancing
- Không cần duplicate replay logic

### 4. Tránh Double set_task() Bug
- `DERServer.set_task()` KHÔNG gọi `trainer.set_task()`
- `train_incremental_kaggle.py` gọi riêng (line 490-491)

### 5. Checkpoint với Dynamic Architecture
Checkpoint cần save thêm:
```python
"der_num_tasks": model.current_task + 1,
"der_task_classes_history": server.task_classes,
```
Khi load: gọi `model.add_task()` đúng số lần trước `load_state_dict()`.

---

## Verification Plan

1. **Unit test DERModel**: `add_task()` 3 lần → verify dims, freezing, weight inheritance
2. **Unit test DERTrainer**: Loss computation cả 2 stages, annealing schedule
3. **Integration**: `algorithm="der"`, 2 tasks → accuracy > random
4. **Isolation**: `algorithm="cgofed"` sau khi thêm DER → kết quả IDENTICAL
