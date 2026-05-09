# EWC.md - Map 1:1 giữa EWC gốc và code hiện tại

Tài liệu này map từng bước của EWC với implementation hiện tại trong repo
`FL_IL_IDS`, đồng thời đối chiếu với repo gốc:

- Upstream: https://github.com/ariseff/overcoming-catastrophic
- Code hiện tại:
  - `fed_learning/strategies/incremental/ewc.py`
  - `fed_learning/strategies/fed_incremental/ewc.py`
  - `fed_learning/training/post_task.py`
  - `fed_learning/training/local_task_loop.py`
  - `fed_learning/strategies/__init__.py`
  - `fed_learning/core/trainer.py`

Ghi chú quan trọng:

- Upstream là implementation TensorFlow cho supervised continual learning trên
  một model duy nhất, chủ yếu trong `model.py` và notebook `experiment.ipynb`.
- Repo hiện tại adapt EWC sang PyTorch, fixed-head IDS, local IL, và federated
  variants `fedavg_ewc` / `fedprox_ewc`.
- Upstream dùng EWC chuẩn: lưu Fisher của task gần nhất và `star_vars`, rồi cộng
  quadratic penalty vào loss.
- Repo hiện tại dùng **corrected EWC** theo Huszar: một Fisher tích lũy duy nhất
  và một anchor params mới nhất, tránh double-counting nhiều penalty task cũ.
  Repo cũng có option Online EWC với decay `gamma`.

---

## 1. Mục tiêu của EWC

### Ý nghĩa trong EWC

EWC chống catastrophic forgetting bằng cách phạt model nếu các tham số quan trọng
với task cũ bị thay đổi quá nhiều khi học task mới.

Loss tổng quát:

```text
L(θ) = L_task_mới(θ) + (λ / 2) * Σ_i F_i * (θ_i - θ_i*)²
```

Trong đó:

- `θ_i*`: giá trị tham số sau khi học xong task cũ;
- `F_i`: diagonal Fisher Information, đo mức quan trọng của tham số;
- `λ`: độ mạnh regularization.

### Code hiện tại

File: `fed_learning/strategies/incremental/ewc.py`

`EWCMixin.compute_loss()`:

```python
base_loss = super().compute_loss(model, output, target, global_params, **kwargs)
...
ewc_penalty += (fisher_val * diff**2).sum()
return base_loss + (self.ewc_lambda / 2.0) * ewc_penalty
```

### Code gốc upstream

File upstream: `model.py`

`update_ewc_loss(lam)`:

```python
self.ewc_loss += (lam/2) * tf.reduce_sum(
    tf.multiply(self.F_accum[v].astype(np.float32),
                tf.square(self.var_list[v] - self.star_vars[v]))
)
```

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Base loss | `cross_entropy` | `super().compute_loss(...)` | Repo cho phép CE/FedProx/seen-class CE |
| Quadratic penalty | `F_accum * (var - star_vars)^2` | `fisher_acc * (param - optimal_params)^2` | 1:1 core |
| Lambda | `lam` | `ewc_lambda` | Tương đương |
| Framework | TensorFlow graph | PyTorch eager | Implementation khác |

---

## 2. Model và output head

### Ý nghĩa trong EWC

EWC không yêu cầu kiến trúc đặc biệt. Nó chỉ cần:

1. model có tham số trainable;
2. loss để tính gradient;
3. cơ chế lưu Fisher và optimal params sau task.

### Code hiện tại

Repo hiện tại thường dùng `CNN_GRU_Model` fixed 34-head cho IDS. Vì fixed-head
luôn output đủ 34 logits, base trainer có seen-class CE để class chưa xuất hiện
không tham gia softmax khi train incremental.

File: `fed_learning/core/trainer.py`

`_seen_class_cross_entropy()`:

```python
seen_classes = getattr(self, "seen_classes", None)
...
seen_logits = output.index_select(dim=1, index=class_tensor)
return F.cross_entropy(seen_logits, remapped, reduction=reduction)
```

### Code gốc upstream

File upstream: `model.py`

Upstream dùng MLP đơn giản:

```python
W1 = weight_variable([in_dim, 50])
b1 = bias_variable([50])
W2 = weight_variable([50, out_dim])
b2 = bias_variable([out_dim])
self.var_list = [W1, b1, W2, b2]
```

`out_dim` lấy từ label placeholder, ví dụ MNIST 10 class.

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Kiến trúc | 2-layer MLP TensorFlow | CNN-GRU PyTorch | EWC model-agnostic |
| Output | `out_dim` theo dataset | fixed `total_classes = 34` | Adapt IDS |
| Class incremental guard | Không có trong model.py | seen-class CE | Repo-specific cần cho fixed-head |

---

## 3. Lưu optimal parameters `θ*`

### Ý nghĩa trong EWC

Sau khi học xong một task, EWC lưu snapshot tham số hiện tại làm điểm neo
`θ*`. Khi học task sau, penalty đo khoảng cách từ tham số hiện tại đến điểm neo
này.

### Code hiện tại

File: `fed_learning/strategies/incremental/ewc.py`

Trong `consolidate()`:

```python
optimal_params = {
    name: param.detach().cpu().clone()
    for name, param in model.named_parameters()
    if param.requires_grad
}
```

Sau đó lưu ra disk:

```python
torch.save(optimal_params, params_path)
```

### Code gốc upstream

File upstream: `model.py`

`star()`:

```python
self.star_vars = []
for v in range(len(self.var_list)):
    self.star_vars.append(self.var_list[v].eval())
```

`restore()`:

```python
sess.run(self.var_list[v].assign(self.star_vars[v]))
```

Notebook upstream gọi `model.restore(sess)` trước khi thử train task mới với
vanilla/EWC để so sánh nhiều setting từ cùng điểm xuất phát.

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Lưu params tối ưu | `star_vars` | `optimal_params` dict | 1:1 |
| Restore params | `restore(sess)` | resume/load state qua trainer/server | Repo framework hóa |
| Storage | RAM trong TensorFlow object | RAM cache + `.pt` backup | Repo hỗ trợ resume |

---

## 4. Tính diagonal Fisher

### Ý nghĩa trong EWC

Fisher diagonal đo độ quan trọng của từng tham số. Tham số có Fisher lớn sẽ bị
phạt mạnh hơn nếu thay đổi khi học task mới.

### Code hiện tại

File: `fed_learning/strategies/incremental/ewc.py`

`compute_fisher_information()`:

- chuyển model sang train mode nhưng tắt dropout;
- khởi tạo Fisher zeros theo từng parameter;
- duyệt từng sample;
- forward một sample;
- tính seen-class CE;
- backward;
- cộng bình phương gradient;
- chia trung bình theo số sample.

Đoạn cốt lõi:

```python
output = model(X[i : i + 1])
loss = self._seen_class_cross_entropy(output, y[i : i + 1])
loss.backward()
...
fisher[name] += param.grad.detach() ** 2
...
fisher[name] /= sample_count
```

### Code gốc upstream

File upstream: `model.py`

`compute_fisher()`:

- tạo `F_accum`;
- lấy softmax probabilities;
- sample một class từ distribution;
- tính gradient của `log p(class | x)`;
- cộng bình phương gradient;
- chia cho `num_samples`.

Đoạn cốt lõi:

```python
probs = tf.nn.softmax(self.y)
class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
fish_gra = tf.gradients(tf.log(probs[0,class_ind]), self.var_list)
...
self.F_accum[v] += np.square(ders[v])
self.F_accum[v] /= num_samples
```

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Fisher type | Diagonal Fisher | Diagonal empirical Fisher | 1:1 mục tiêu |
| Gradient source | `grad log p(sampled class)` | `grad CE(true label)` | Khác estimator |
| Per-sample gradient | Có | Có | Tương đương |
| Average over samples | Có | Có | Tương đương |
| Dropout handling | Không relevant MLP nhỏ | Train mode nhưng tắt Dropout | Repo xử lý RNN/cuDNN + ổn định |

Điểm cần chú ý: upstream dùng Fisher estimator theo class sampled từ model
distribution; repo dùng empirical Fisher từ supervised CE với nhãn thật. Đây là
biến thể phổ biến trong PyTorch continual learning, nhưng không giống từng dòng
với upstream.

---

## 5. Consolidate sau mỗi task

### Ý nghĩa trong EWC

Sau khi task kết thúc, EWC phải:

1. tính Fisher của task vừa học;
2. lưu optimal params;
3. chuẩn bị penalty cho task tiếp theo.

### Code hiện tại

File: `fed_learning/training/post_task.py`

Federated path `fedavg_ewc` / `fedprox_ewc` gọi consolidate sau task:

```python
elif "ewc" in algo:
    ...
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    trainer.consolidate(server.global_model, loader, device)
```

Ở đây repo gom train data từ các client tham gia task hiện tại, giới hạn theo
`fisher_samples`, rồi tính Fisher trên global model sau training.

File: `fed_learning/training/local_task_loop.py`

Local IL path `ewc` cũng consolidate sau task:

```python
if algo == "ewc" and hasattr(trainer, "consolidate"):
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    trainer.consolidate(model, loader, device)
```

### Code gốc upstream

Notebook upstream sau khi train task A gọi:

```python
model.compute_fisher(mnist.validation.images, sess, num_samples=200)
model.star()
```

Sau đó khi train task B với EWC, gọi:

```python
model.update_ewc_loss(lam)
```

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Compute Fisher sau task | `model.compute_fisher(...)` | `trainer.consolidate(...)` | Tương đương |
| Lưu `θ*` | `model.star()` | trong `consolidate()` | Tương đương, repo gộp vào một hàm |
| Data dùng cho Fisher | validation images | train samples task hiện tại | Khác nguồn data |
| Federated setting | Không có | gom data client để Fisher global | Repo-specific |

---

## 6. Cộng dồn Fisher: upstream vs corrected EWC

### Ý nghĩa

Đây là điểm khác quan trọng nhất.

Upstream implementation rất gần EWC gốc: mỗi lần gọi `update_ewc_loss(lam)`, nó
cộng penalty mới vào `self.ewc_loss`. Nếu train qua nhiều task, cách này có thể
dẫn đến nhiều penalty riêng.

Repo hiện tại dùng corrected EWC:

```text
L = L_base + (λ/2) * Σ_i F_acc_i * (θ_i - θ*_latest_i)^2
```

Chỉ có một Fisher tích lũy và một anchor latest params.

### Code hiện tại

File: `fed_learning/strategies/incremental/ewc.py`

Trong `consolidate()`:

```python
if self.online_ewc:
    fisher_acc[name] = self.gamma * prev_f + fisher_new[name]
else:
    fisher_acc[name] = prev_f + fisher_new[name]
```

Trong `compute_loss()` chỉ dùng latest accumulated Fisher:

```python
latest_task = max(self.ewc_data.keys())
...
fisher_val = self._cached_fisher_acc[name]
optimal_val = self._cached_optimal_params[name]
```

### Code gốc upstream

File upstream: `model.py`

`update_ewc_loss()`:

```python
if not hasattr(self, "ewc_loss"):
    self.ewc_loss = self.cross_entropy

for v in range(len(self.var_list)):
    self.ewc_loss += (lam/2) * ...
```

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Penalty storage | `self.ewc_loss` cộng dần | single accumulated penalty | Repo dùng corrected EWC |
| Fisher history | `F_accum` gần nhất trong object | `fisher_acc` tích lũy | Khác nhưng có chủ đích |
| Online decay | Không có trong repo gốc này | optional `online_ewc`, `gamma` | Repo mở rộng |
| Double-counting guard | Không rõ/có thể double-count | Có theo Huszar correction | Repo cải tiến |

---

## 7. Local loss trong federated EWC

### Ý nghĩa trong repo hiện tại

EWC không thay đổi thuật toán aggregation. Nó thay đổi local objective của client
khi train. Vì vậy repo dùng mixin để ghép EWC vào FedAvg hoặc FedProx.

### Code hiện tại

File: `fed_learning/strategies/fed_incremental/ewc.py`

```python
class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer):
    ...

class FedProxEWCTrainer(EWCMixin, FedProxTrainer):
    ...
```

File: `fed_learning/strategies/__init__.py`

```python
"fedavg_ewc": {
    "trainer": FedAvgEWCTrainer,
    ...
}
"fedprox_ewc": {
    "trainer": FedProxEWCTrainer,
    ...
}
```

Với `fedprox_ewc`, base loss đã có FedProx term, rồi EWC mixin cộng thêm EWC
penalty:

```text
L = CE + FedProx + EWC
```

Với `fedavg_ewc`:

```text
L = CE + EWC
```

### Code gốc upstream

Upstream không có federated wrapper. Nó train một model bằng:

```python
self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)
```

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| FL aggregation | Không có | FedAvg/FedProx base trainers | Repo-specific |
| EWC injection | `update_ewc_loss()` đổi train op | `EWCMixin.compute_loss()` | Tương đương về điểm can thiệp |
| Aggregator | Không có | FedAvg/FedProx aggregator | EWC không cần aggregator riêng |

---

## 8. Task đầu tiên xử lý thế nào?

### Code hiện tại

Trong `compute_loss()`:

```python
if not self.ewc_data:
    return base_loss
```

Nghĩa là task đầu tiên train bình thường, chưa có EWC penalty vì chưa có Fisher
của task cũ. Sau khi task đầu kết thúc, `consolidate()` mới tạo Fisher/params để
task sau dùng.

### Code gốc upstream

Upstream cũng train task đầu bình thường bằng vanilla loss:

```python
self.set_vanilla_loss()
```

Sau task đầu mới gọi:

```python
model.compute_fisher(...)
model.star()
```

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Task đầu | Vanilla CE | base trainer loss | Tương đương |
| Fisher trước task đầu | Không có | Không có | Tương đương |
| Penalty bắt đầu từ task sau | Có | Có | Tương đương |

---

## 9. Resume/cache/storage

### Ý nghĩa trong repo hiện tại

EWC Fisher có cùng shape với toàn bộ parameters, nên cần cache để không đọc disk
mỗi batch và cần serialize để resume split-run.

### Code hiện tại

File: `fed_learning/strategies/incremental/ewc.py`

State chính:

```python
self.ewc_data
self._cached_fisher_acc
self._cached_optimal_params
self._cache_device
```

`get_resume_state()` lưu:

- `ewc_lambda`;
- `fisher_samples`;
- `online_ewc`;
- `gamma`;
- `latest_fisher_acc`;
- `latest_optimal_params`;
- forgetting stats.

`load_resume_state()` materialize lại `.pt` files:

```python
torch.save(fisher_payload, fisher_path)
torch.save(params_payload, params_path)
```

### Code gốc upstream

Upstream không có resume. `F_accum` và `star_vars` sống trong object/session
TensorFlow của notebook.

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Fisher storage | RAM/session | RAM cache + `.pt` | Repo-specific |
| Resume | Không có | Có | Repo extension |
| Device cache | Không có | CPU/GPU cache per task | Repo extension |

---

## 10. Average Forgetting metric

### Ý nghĩa trong repo hiện tại

Average Forgetting không phải phần core của EWC penalty, nhưng repo có để logging
và thống kê cross-task performance.

### Code hiện tại

File: `fed_learning/strategies/incremental/ewc.py`

`update_forgetting()`:

```python
f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
forgetting.append(max(0, f))
self.last_af = sum(forgetting) / len(forgetting)
```

### Code gốc upstream

Notebook upstream plot accuracy của các task khi train sequentially, nhưng không
có method Average Forgetting tương ứng trong `model.py`.

### Đối chiếu

| Bước | Upstream EWC | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Accuracy tracking | Notebook plot test accuracy | `update_forgetting()` | Repo thêm metric |
| Ảnh hưởng loss | Không | Không | Chỉ logging/thống kê |

---

## 11. Bảng tổng hợp fidelity

| Cơ chế EWC | File repo hiện tại | File upstream | Mức map |
|---|---|---|---|
| Base CE loss | `core/trainer.py`, `EWCMixin.compute_loss` | `model.py:cross_entropy` | 1:1 semantic |
| Diagonal Fisher | `compute_fisher_information` | `compute_fisher` | 1:1 mục tiêu, estimator khác |
| Per-sample gradient square | `param.grad ** 2` | `np.square(ders[v])` | 1:1 |
| Average Fisher | `fisher[name] /= sample_count` | `F_accum[v] /= num_samples` | 1:1 |
| Optimal params | `optimal_params` | `star_vars` | 1:1 |
| EWC penalty | `fisher_acc * diff**2` | `F_accum * square(var-star)` | 1:1 core |
| Task đầu no penalty | `if not ewc_data` | vanilla train before Fisher | 1:1 |
| Consolidate after task | `post_task_processing`, `_post_task_local` | notebook calls `compute_fisher` + `star` | 1:1 semantic |
| Corrected accumulated Fisher | `fisher_acc = prev + new` | Không có trực tiếp | Repo correction/extension |
| Online EWC decay | `gamma * prev + new` | Không có | Repo extension |
| Federated wrappers | `fed_incremental/ewc.py` | Không có | Repo-specific |
| Resume/cache | `get_resume_state`, `.pt` files | Không có | Repo-specific |

---

## 12. Các sai khác/điểm cần chú ý

1. Upstream là TensorFlow single-model; repo hiện tại là PyTorch và có cả local
   IL lẫn federated variants.

2. Upstream Fisher estimator dùng gradient của `log p(class sampled từ softmax)`.
   Repo dùng empirical Fisher từ gradient CE với nhãn thật. Cả hai đều là
   diagonal Fisher-style importance, nhưng không giống từng dòng.

3. Upstream `update_ewc_loss()` cộng penalty vào TensorFlow graph. Repo tính
   penalty runtime trong `compute_loss()` ở mỗi mini-batch.

4. Repo hiện tại dùng corrected EWC theo Huszar: một Fisher tích lũy và một
   latest anchor. Đây là khác biệt có chủ đích so với EWC notebook gốc.

5. Với fixed 34-head IDS, repo dùng seen-class CE. Upstream MNIST example không
   có guard này.

6. Federated EWC trong repo tính Fisher trên global model sau task bằng dữ liệu
   gom từ client task hiện tại. Upstream chỉ dùng validation images trong một
   process local.

7. `EWCMixin` phải đứng trước trainer nền trong MRO:

```python
class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer): pass
```

Điều này đảm bảo `EWCMixin.compute_loss()` gọi được `super().compute_loss(...)`
của FedAvg/FedProx rồi cộng EWC penalty.

---

## 13. Kết luận ngắn

Implementation EWC hiện tại giữ đúng lõi của upstream:

- train task đầu bằng loss thường;
- sau task tính diagonal Fisher;
- lưu optimal params;
- khi học task sau cộng quadratic penalty theo Fisher;
- tham số quan trọng với task cũ bị hạn chế thay đổi.

Các phần khác biệt chính là do repo adapt sang IDS + federated + resume:

- PyTorch thay TensorFlow;
- CNN-GRU fixed 34-head thay MLP MNIST;
- seen-class CE cho class-incremental fixed-head;
- empirical Fisher thay Fisher sampled-class của upstream;
- corrected/online EWC thay vì graph penalty cộng dần;
- FedAvg/FedProx wrappers và post-task Fisher trên global model;
- cache/storage để chạy nhiều task và resume.
