# FedCBDR: Class-wise Balancing Data Replay for Federated Class-Incremental Learning

**Tác giả:** Zhuang Qi, Ying-Peng Tang, Lei Meng, Han Yu, Xiaoxiao Li, Xiangxu Meng  
**Nguồn:** arXiv:2507.07712, 2025

---

## 1. Tổng quan (Abstract)

FedCBDR là phương pháp mới cho **Federated Class-Incremental Learning (FCIL)** giải quyết vấn đề **mất cân bằng lớp** trong replay buffer. Phương pháp đề xuất hai module chính:

1. **Global-perspective Data Replay (GDR)** - Phối hợp toàn cục để chọn mẫu cân bằng, bảo vệ quyền riêng tư
2. **Task-aware Temperature Scaling (TTS)** - Điều chỉnh logits động để xử lý mất cân bằng giữa các task

FedCBDR cải thiện **2%-15% Top-1 accuracy** so với các baseline (FedEWC, FedLwF, TARGET, LANDER, Re-Fed).

---

## 2. Vấn đề giải quyết

### 2.1 Class Imbalance trong Replay Buffer

Trong FCIL, mỗi client lưu một **memory buffer** với kích thước cố định M samples. Vấn đề:

- Nếu chỉ sampling cục bộ → **class imbalance** nghiêm trọng
- Các class hiếm có thể không được đại diện đủ
- Dẫn đến **catastrophic forgetting** cho các class cũ

### 2.2 Task Imbalance

- Số lượng mẫu task mới >> task cũ (từ buffer)
- Model bias về task mới
- Quên kiến thức task cũ

---

## 3. Phương pháp đề xuất

### 3.1 Global-perspective Data Replay (GDR)

**Mục tiêu:** Chọn mẫu đại diện nhất từ góc nhìn toàn cục, bảo vệ quyền riêng tư.

#### Step 1: Feature Extraction & Encryption (Client-side)

```
X_k^(i) = M_g(D_k^(i))     # Extract features using global model
```

Mã hóa ISVD (Inverse SVD):

```
X_k^(i)' = P_k^(i) · X_k^(i) · Q^(i)
```

- `P_k`: Ma trận trực giao ngẫu nhiên (client-specific)
- `Q`: Ma trận trực giao ngẫu nhiên (shared)

#### Step 2: Global SVD (Server-side)

```
X^(i)' = concat{X_k^(i)' | k=1,...,K}
SVD: X^(i)' = U^(i)' Σ^(i)' V^(i)'^T
```

#### Step 3: Leverage Score Computation

```
τ_k^{i,j} = ||e_{i,j}^T U_k^(i)||_2^2
```

- **Leverage score cao** = sample đại diện hơn trong latent space
- Normalize thành phân phối xác suất: `p_k^{i,j} = τ_k^{i,j} / Σ τ`

#### Step 4: Importance-based Sampling

- Sample theo phân phối `p`
- Server gửi indices được chọn về clients
- Clients update replay buffer

### 3.2 Task-aware Temperature Scaling (TTS)

**Loss Function:**

$$\mathcal{L}_{TTS} = \frac{1}{N_{old}} \sum_{i=1}^{N_{old}} \omega_{old} \cdot CE\left(y_i, \text{Softmax}\left(\frac{z^{old}}{\tau_{old}} \| \frac{z^{new}}{\tau_{new}}\right)\right) + \frac{1}{N_{new}} \sum_{j=1}^{N_{new}} \omega_{new} \cdot CE\left(y_j, \ldots\right)$$

Trong đó:

- `τ_old < 1`: Temperature thấp → logits sắc nét hơn cho old classes (bảo toàn)
- `τ_new > 1`: Temperature cao → logits mềm hơn cho new classes
- `ω_old > 1`: Weight cao hơn cho old samples
- `ω_new < 1`: Weight thấp hơn cho new samples

### 3.3 Tích hợp với FedAvg

```
Algorithm: FedCBDR

For each task s = 1 to T:
    For each round r = 1 to R:
        For each client k:
            θ_k ← θ_g                          # Load global model

            if s == 1:
                # Stage 1: Standard CE training
                min_{θ_k} CE(y, f(x))
            else:
                # Stage 2: TTS training with replay
                D_train = D_new ∪ B_replay
                min_{θ_k} L_TTS

            Upload θ_k to server

        θ_g ← (1/K) Σ θ_k                      # FedAvg aggregation

    # GDR: Update replay buffers after task
    For each client k:
        Extract features, encrypt via ISVD
        Upload encrypted features

    Server: Compute leverage scores via global SVD
    Clients: Update buffers with selected samples
```

---

## 4. Hyperparameters

| Parameter | Mô tả                      | Giá trị khuyến nghị |
| --------- | -------------------------- | ------------------- |
| **τ_old** | Temperature cho old logits | 0.8 - 0.9           |
| **τ_new** | Temperature cho new logits | 1.1 - 1.2           |
| **ω_old** | Weight cho old samples     | 1.1 - 1.4           |
| **ω_new** | Weight cho new samples     | 0.7 - 0.9           |
| **M**     | Buffer size per client     | Dataset-dependent   |
| **K**     | Số clients                 | 5 - 10              |
| **E**     | Local epochs               | 2                   |
| **B**     | Batch size                 | 128                 |
| **R**     | Rounds per task            | 100                 |

**Best configuration (từ sensitivity analysis):**

- `τ_old = 0.9`, `τ_new = 1.1`
- `ω_old = 1.1`, `ω_new = 0.9`

---

## 5. Implementation trong dự án

### 5.1 Cấu trúc files

```
fed_learning/
├── strategies/incremental/
│   └── fedcbdr.py          # FedCBDRTrainer, FedCBDRAggregator,
│                           # ReplayBuffer, LeverageScoreCalculator
├── clients/
│   └── fedcbdr_client.py   # FedCBDRClient với replay buffer
├── servers/
│   └── fedcbdr_server.py   # FedCBDRServer với GDR module
└── training/
    └── fedcbdr_worker.py   # Multi-GPU training worker

train_fedcbdr_kaggle.py     # Kaggle training script
```

### 5.2 Sử dụng

```python
from fed_learning.servers.fedcbdr_server import FedCBDRServer
from fed_learning.clients.fedcbdr_client import FedCBDRClient

# Config
config = {
    "tau_old": 0.9,
    "tau_new": 1.1,
    "omega_old": 1.1,
    "omega_new": 0.9,
    "buffer_size": 500,
    "use_replay": True,
    "replay_ratio": 0.5,
    ...
}

# Create clients
clients = [
    FedCBDRClient(cid, X, y, buffer_size=500)
    for cid, (X, y) in enumerate(client_data)
]

# Create server
server = FedCBDRServer(clients, test_data, config)

# Train incrementally
for task_id, task_classes in task_structure.items():
    server.set_task(task_id, task_classes)

    for round_idx in range(rounds_per_task):
        server.train_round()

    # Update replay buffers via GDR
    server.coordinate_gdr()
```

### 5.3 Key Classes

#### `FedCBDRTrainer`

- Implements TTS loss function
- Tracks old/new classes per task
- Applies temperature scaling and sample weighting

#### `ReplayBuffer`

- Class-balanced storage
- Importance-based sample selection
- Herding for representative sampling

#### `FedCBDRClient`

- Extends FederatedClient with replay buffer
- Combined training on new + replay data
- Feature extraction for GDR

#### `FedCBDRServer`

- Coordinates GDR (leverage score computation)
- Task-aware evaluation
- Forgetting metrics computation

---

## 6. So sánh với CGoFed

| Aspect          | FedCBDR                   | CGoFed                |
| --------------- | ------------------------- | --------------------- |
| **Approach**    | Replay-based              | Gradient projection   |
| **Memory**      | Stores old samples        | Stores SVD basis      |
| **Privacy**     | ISVD encryption           | Gradient space        |
| **Loss**        | TTS (temperature scaling) | Proximal + projection |
| **Aggregation** | FedAvg                    | Weighted FedAvg       |
| **Complexity**  | O(nd²) for SVD            | O(nd²) for SVD        |

---

## 7. Kết quả mong đợi

- **Giảm forgetting** nhờ class-balanced replay
- **Cân bằng plasticity-stability** nhờ TTS
- **Privacy-preserving** nhờ ISVD encryption
- **Scalable** với số lượng clients lớn
