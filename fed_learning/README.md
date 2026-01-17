# Federated Learning Module

Module Federated Learning với hỗ trợ Multi-GPU và Strategy Pattern.

## 📁 Cấu trúc dự án

```
fed_learning/
├── core/                    # Các lớp trừu tượng cơ sở
│   ├── trainer.py           # BaseTrainer
│   └── aggregator.py        # BaseAggregator
│
├── strategies/              # Thuật toán học (plugin)
│   └── federated/           # Các thuật toán FL
│       ├── fedavg.py        # FedAvg
│       ├── fedavgm.py       # FedAvg + Momentum
│       ├── fedprox.py       # FedProx
│       └── fedplus.py       # Fed+
│
├── clients/                 # Client
│   └── client.py            # FederatedClient
│
├── servers/                 # Server
│   └── server.py            # FederatedServer
│
├── models/                  # Mô hình mạng
│   └── cnn_gru.py           # CNN-GRU model
│
├── data/                    # Tiện ích load dữ liệu
│   └── loader.py            # Load data vào RAM
│
├── training/                # Tiện ích training
│   ├── runner.py            # Điều phối training (orchestrator)
│   └── worker.py            # Train clients trên GPU (worker)
│
└── visualization/           # Vẽ biểu đồ
    └── plots.py             # Training curves, confusion matrix
```

---

## 🧩 Mô tả từng module

### `core/` - Các lớp trừu tượng

| Class | Mô tả |
|-------|-------|
| **BaseTrainer** | Lớp trừu tượng với hooks: `compute_loss()`, `post_step()`, `pre_train()`, `post_train()` |
| **BaseAggregator** | Lớp trừu tượng với method `aggregate()` để tổng hợp model |

### `strategies/` - Thuật toán học

| Thuật toán | Trainer | Aggregator | Mô tả |
|------------|---------|------------|-------|
| **FedAvg** | Standard | Weighted Avg | Baseline federated averaging |
| **FedAvgM** | Standard | Weighted Avg + Momentum | Server momentum |
| **FedProx** | + Proximal term | Weighted Avg | Xử lý dữ liệu không đồng nhất |
| **Fed+** | + Correction step | Weighted Avg | Regularization động |

### `training/` - Tiện ích training

| File | Vai trò | Mô tả |
|------|---------|-------|
| `runner.py` | Orchestrator | Vòng lặp chính: điều phối rounds, evaluate, log |
| `worker.py` | Worker | Train clients trên GPU cụ thể (multi-threaded) |

---

## 🚀 Sử dụng nhanh

```python
from fed_learning import (
    FederatedClient,
    FederatedServer,
    load_all_client_data_to_ram,
    train_federated_multigpu,
)

# Load dữ liệu
client_data, test_data, input_shape, num_classes = load_all_client_data_to_ram(
    data_dir="./data", num_clients=100
)

# Tạo clients
clients = [
    FederatedClient(cid, data['X_train'], data['y_train'])
    for cid, data in enumerate(client_data)
]

# Cấu hình
config = {
    "algorithm": "fedprox",  # fedavg, fedavgm, fedprox, fedplus
    "mu": 0.01,
    "num_rounds": 10,
    "local_epochs": 3,
    "batch_size": 128,
    "learning_rate": 0.001,
    "input_shape": input_shape,
    "num_classes": num_classes,
}

# Tạo server và train
server = FederatedServer(clients, test_data, config)
history = train_federated_multigpu(server, config)
```

---

## ➕ Thêm thuật toán mới

1. Tạo trainer/aggregator trong `strategies/federated/` (hoặc `incremental/`, `decentralized/`):

```python
# strategies/incremental/ewc.py
from ...core import BaseTrainer, BaseAggregator

class EWCTrainer(BaseTrainer):
    def compute_loss(self, model, output, target, global_params=None, **kwargs):
        ce_loss = super().compute_loss(model, output, target)
        ewc_penalty = self._compute_ewc_penalty(model)
        return ce_loss + self.lambda_ewc * ewc_penalty
```

2. Đăng ký trong `strategies/__init__.py`:

```python
STRATEGIES["ewc"] = {
    "trainer": EWCTrainer,
    "aggregator": FedAvgAggregator,
}
```

3. Sử dụng:

```python
trainer, aggregator = get_strategy("ewc", lambda_ewc=1000)
```

---

## 📊 Các thuật toán có sẵn

```python
from fed_learning import list_strategies

print(list_strategies())
# {
#     'fedavg': 'Federated Averaging - trung bình có trọng số',
#     'fedavgm': 'FedAvg + Server Momentum - tăng tốc hội tụ',
#     'fedprox': 'Federated Proximal - xử lý dữ liệu không đồng nhất',
#     'fedplus': 'Fed+ - regularization động cho dữ liệu không đồng nhất',
# }
```
