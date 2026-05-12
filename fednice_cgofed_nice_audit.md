# Audit end-to-end: NICE/FedNICE và CGoFed

Phạm vi:

- Paper local:
  - `paper/NICE.txt`
  - `paper/cgofed.txt`
- Upstream NICE:
  - `https://github.com/BurakGurbuz97/NICE/tree/main/Source`
- Upstream CGoFed:
  - `https://github.com/fengjiyuan/cgofed`
- Code hiện tại:
  - `fed_learning/models/nice_model.py`
  - `fed_learning/strategies/incremental/nice.py`
  - `fed_learning/strategies/fed_incremental/nice.py`
  - `fed_learning/clients/nice_client.py`
  - `fed_learning/servers/nice_server.py`
  - `fed_learning/training/nice_worker.py`
  - `fed_learning/strategies/fed_incremental/cgofed.py`
  - `fed_learning/clients/cgofed_client.py`
  - `fed_learning/servers/cgofed_server.py`
  - `fed_learning/training/cgofed_worker.py`
  - `fed_learning/training/task_loop.py`

## Findings

### P0 - NICE loss mất toàn bộ gradient khi mini-batch chỉ có 1 class

File hiện tại: `fed_learning/strategies/incremental/nice.py`

- `NICETrainer.compute_loss()` lấy `batch_classes` từ class xuất hiện trong mini-batch.
- Sau đó slice logits chỉ còn các class đó:

```python
batch_classes = sorted({... torch.unique(target) ...})
...
return F.cross_entropy(output.index_select(dim=1, index=class_tensor), remapped)
```

Với batch chỉ có một class, logits còn shape `[B, 1]`, target luôn `0`, CE = `0`,
gradient = `0`. Trong FL non-IID, client/batch một class là case rất thường gặp.

Đã kiểm tra trực tiếp:

```text
loss 0.0
grad_norm 0.0
nonzero_grad 0
```

Upstream NICE không slice theo class trong batch. File upstream `Source/train_eval.py`:

```python
stream_output = network.forward_output(data)
ce_loss = loss(stream_output, target.long())
```

Tác động:

- NICE/FedNICE có thể không học gì trên client đơn class.
- Neuron selection vẫn chạy, nhưng classifier/backbone không nhận gradient phân biệt.
- Kết quả đặc biệt nguy hiểm với IDS split theo client non-IID.

Hướng sửa:

- Không slice theo `torch.unique(target)` trong batch.
- Nếu cần chặn unseen, slice theo `trainer.seen_classes` hoặc task classes/learner output classes,
  không theo class hiện diện trong batch.
- Ít nhất phải đảm bảo số class trong CE >= 2 khi training.

### P1 - NICE tau-greedy selection không còn là pruning lặp trên learner set trước đó

File hiện tại: `fed_learning/strategies/incremental/nice.py`

Code hiện tại reset mọi neuron non-mature trước khi đo activation:

```python
for name in model.LAYER_NAMES:
    if name == "fc2":
        continue
    ranks = model.unit_ranks[name]
    ranks[ranks < 2] = 0
...
activations = model.get_activations(data)
...
young_mask = ranks == 0
```

Nghĩa là mỗi phase chọn lại từ toàn bộ non-mature pool.

Upstream NICE làm khác. File upstream `Source/nice_operations.py`:

```python
layer_activations = network.get_activation_selection(data)
...
mask[network.current_learner_neurons[index]] = False
scores[mask] = 0.0
...
new_ranks[new_ranks < 2] = 0
new_ranks[layer_selected_units] = 1
```

Upstream đo activation trên `current_learner_neurons` của phase trước, rồi prune dần
theo `activation_perc`. Neuron đã bị loại khỏi learner set không được quay lại trong
phase sau của cùng episode.

Tác động:

- Code hiện tại có thể re-introduce neuron đã bị prune trong phase trước.
- Số neuron learner cuối phase có thể khác NICE gốc.
- Cơ chế tau-greedy từ "lọc dần subnet" thành "chọn lại top activation mỗi phase".

Hướng sửa:

- Khi `tau < 1.0`, đo/chọn từ learner set hiện tại, không từ toàn bộ young pool.
- Reset `ranks < 2` về 0 chỉ sau khi đã xác định `selected`.

### P1 - NICE bỏ L2 masked-weight regularization của upstream

File hiện tại: `fed_learning/strategies/incremental/nice.py`

`NICETrainer.compute_loss()` chỉ trả về CE.

Upstream NICE `Source/train_eval.py` cộng thêm masked L2:

```python
ce_loss = loss(stream_output, target.long())
reg_loss = (args.weight_decay * network.l2_loss())
batch_loss = reg_loss + ce_loss
```

Upstream `network.l2_loss()` tính trên masked weights, không phải weight decay optimizer
thông thường.

Tác động:

- Sparsity/regularization behavior lệch upstream.
- Khi mask vật lý zero weight, L2 theo masked weights giữ loss đúng với sparse network.

Hướng sửa:

- Thêm `model.l2_loss()` cho `NICEModel`.
- `compute_loss = CE + nice_weight_decay * model.l2_loss()`.

### P1 - NICE BatchNorm chỉ freeze khi toàn layer mature, không freeze từng unit như upstream

File hiện tại: `fed_learning/models/nice_model.py`

```python
def freeze_bn_for_mature(self):
    ...
    if np.all(ranks >= 2):
        bn.eval()
```

Upstream NICE update freeze mask cho BN theo unit mature. File upstream
`Source/nice_operations.py`:

```python
frozen_units = network.unit_ranks[layer_index][0] > 1
module.freeze_units(torch.tensor(frozen_units, dtype=torch.bool).to(get_device()))
```

Và `Source/architecture.py` zero gradient BN affine cho frozen units:

```python
module.weight.grad = torch.where(module.frozen_units, torch.zeros_like(...), ...)
module.bias.grad = torch.where(module.frozen_units, torch.zeros_like(...), ...)
```

Tác động:

- Partial mature layer vẫn cho BN affine/running stats đổi.
- Mature channel có thể bị drift qua BN dù conv/linear row đã freeze.

Hướng sửa:

- Cần BN custom hoặc hook để freeze affine gradient theo unit.
- Running stats theo unit khó hơn; tối thiểu freeze affine per mature unit.

### P1 - CGoFed projection basis lệch code gốc: hiện tại dùng activation input, upstream dùng gradient matrix

Upstream CGoFed `client.py`:

```python
def get_grad_matrix(net, device, x, y=None):
    example_out = net(example_data)
    ...
    if 'weight' in m and 'bn' not in m and 'fc3' not in m:
        grad = params.grad.data.view(sz, -1).detach().cpu().numpy()
        activation = grad.transpose()
        grad_list.append(activation)
```

Sau đó SVD trên `grad_list`:

```python
U, S, Vh = np.linalg.svd(activation, full_matrices=False)
...
grad_basis.append(U[:, 0:r])
```

Code hiện tại:

- `fed_learning/clients/cgofed_client.py` dùng forward hook input activation:

```python
features = self._activation_to_features(layer_name, module, activation)
...
grams[layer_name] += chunk.T.double().mm(chunk.double())
```

- `fed_learning/strategies/fed_incremental/cgofed.py` cũng có path collect activation:

```python
def _collect_activations(...)
```

Tác động:

- Không 1:1 với upstream code.
- Projection space hiện tại là activation covariance/input basis, không phải gradient basis như code gốc.
- Có thể vẫn hợp lý theo diễn giải paper, nhưng nếu mục tiêu là reproduce repo gốc thì đây là drift lớn.

Hướng sửa:

- Nếu muốn bám code gốc: build basis từ gradient matrix sau backward hoặc dummy supervised backward,
  exclude classifier head như upstream.
- Nếu giữ activation basis: document rõ đây là adaptation, không phải CGoFed upstream 1:1.

### P1 - CGoFed hiện tại project cả output layer `fc2`, upstream loại classifier head

Upstream CGoFed `client.py` loại `fc3`:

```python
if 'weight' in m and 'bn' not in m and 'fc3' not in m:
```

Model gốc dùng multi-head `fc3`, và head theo task được chọn bằng:

```python
head_idx = model.tid2head[int(task_id)]
logits = output[head_idx]
```

Code hiện tại chọn projection target gồm mọi `nn.Linear`, `nn.Conv*`, `nn.GRU`:

```python
if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.GRU)):
    modules.append((name, module))
```

Với `CNN_GRU_Model`, `fc2` là single fixed-head 34 class. Vậy output layer cũng bị
projection constraint.

Tác động:

- Khác upstream: upstream không project classifier head.
- Với single-head 34, project `fc2` có thể làm new-class rows bị ràng buộc bởi old-task
  subspace, nhất là khi class-incremental cần plasticity ở classifier.

Hướng sửa:

- Audit bằng ablation: project `fc2` vs không project `fc2`.
- Nếu muốn bám upstream: exclude output head khỏi projection target.

### P2 - CGoFed Eq.14/Eq.12 pre-round state mặc định tắt

File hiện tại: `fed_learning/servers/cgofed_server.py`

```python
if self.config.get("cgofed_pre_round_state", False):
    self._prepare_initial_round_state(global_params, verbose=verbose)
```

Comment trong code nói rõ nếu không có pre-round pass thì `rounds_per_task=1` sẽ
disable Eq.14 hoàn toàn cho task mới.

Tác động:

- Với config `rounds_per_task=1`, task mới không nhận regularization/personalized init
  từ đầu task.
- Với `rounds_per_task>1`, round đầu của task mới vẫn train không có Eq.14/Eq.12.

Hướng sửa:

- Bật mặc định `cgofed_pre_round_state=True` cho CGoFed.
- Hoặc bỏ flag, luôn chuẩn bị nếu `current_task > 0` và history có sẵn.

### P2 - CGoFed bỏ mixup và loss scaling đặc thù upstream

Upstream CGoFed `client.py`:

```python
data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.2, device=device)
local_loss_value = mixup_criterion(...)
reg_loss_value = reg_loss(model)
loss_value = local_loss_value + reg_loss_value / (10 ** (b2 - b1 + 2)) * 2
```

Code hiện tại:

- Không có mixup trong `CGoFedClient.train()`.
- Regularization Eq.14 dùng `lambda_cross_task / 2` cố định trong
  `CGoFedTrainer.compute_loss()`.

Tác động:

- Không 1:1 upstream.
- Có thể ổn vì IDS/tabular-sequence không nhất thiết cần mixup image-style, nhưng nên ghi rõ.

## Những phần đang đúng/hợp lý

### NICE output layer

- Upstream NICE fixed full output layer (`SparseOutput(..., output_size)`).
- Code hiện tại fixed `fc2 = nn.Linear(256, num_classes)`.
- Class chưa tới lượt có node sẵn; task mới set output neuron age=1 trong
  `NICEServer.set_task()`.

### NICE training order

Luồng hiện tại đúng khung lớn:

1. `set_task()` set class mới thành learner.
2. Mỗi phase chọn neuron.
3. `drop_young_to_learner()`.
4. `grow_all_to_young()`.
5. Train bằng `forward_output()`.
6. `reset_frozen_gradients()`.
7. Cuối task `increase_unit_ranks()` và `update_freeze_masks()`.

### CGoFed relaxation

Relax constraint thay đổi theo task/AF, không thay trong giữa task theo từng batch.
Code hiện tại tính `mu_coefficient` trong `CGoFedTrainer.set_task()` và reset qua
`update_forgetting()`, đúng hướng với cơ chế relax theo task.

### CGoFed client-local projection

Việc build SVD basis trên client-local data và giữ basis trong client state là hợp lý
cho FL privacy hơn path centralized. Đây là adaptation tốt, nhưng cần ghi rõ khác
upstream code.

## Kết luận

Nếu chỉ xét "chạy được", pipeline NICE/CGoFed khá đầy đủ.

Nếu xét "đúng paper/upstream end-to-end", có 2 vấn đề cần xử lý trước:

1. `NICETrainer.compute_loss()` slice theo mini-batch class làm batch đơn class có
   loss/gradient bằng 0. Đây là lỗi training thật.
2. `select_learner_units()` của NICE đang chọn lại từ toàn bộ non-mature pool mỗi phase,
   không prune dần từ learner set như upstream.

Sau đó mới tới fidelity CGoFed:

1. basis hiện tại activation-based, upstream gradient-matrix-based;
2. projection hiện tại gồm output `fc2`, upstream loại classifier head;
3. pre-round Eq.12/Eq.14 mặc định tắt.

