# NICE.md - Map 1:1 giữa NICE gốc và code hiện tại

Tài liệu này map từng bước của NICE với implementation hiện tại trong repo
`FL_IL_IDS`, đồng thời đối chiếu với repo gốc:

- Upstream: https://github.com/BurakGurbuz97/NICE/tree/main/Source
- Code hiện tại: `fed_learning/models/nice_model.py`, `fed_learning/clients/nice_client.py`,
  `fed_learning/servers/nice_server.py`, `fed_learning/strategies/incremental/nice.py`,
  `fed_learning/strategies/fed_incremental/nice.py`, `fed_learning/training/task_loop.py`,
  `fed_learning/training/local_task_loop.py`.

Ghi chú quan trọng:

- Upstream NICE chạy single-machine continual learning với Avalanche.
- Repo hiện tại adapt NICE sang federated class-incremental IDS, nên có thêm
  `NICEServer`, `NICEClient`, worker đa GPU, FedAvg-style aggregation, checkpoint,
  resume, và local IL mode.
- Upstream dùng episode index bắt đầu từ `1`; repo hiện tại dùng task/episode
  index bắt đầu từ `0`.
- Output head trong repo hiện tại là fixed-head: `num_classes = total_classes = 34`
  ngay từ task đầu. Class chưa thấy bị chặn qua loss/eval/context logic.

---

## 1. Khởi tạo model, output head, và tuổi neuron

### Ý nghĩa trong NICE

Mỗi neuron có tuổi:

- `0`: young/surplus, dự trữ cho episode tương lai.
- `1`: learner, đang học episode hiện tại.
- `>1`: mature, lưu tri thức cũ và bị freeze.
- `999`: input pseudo-layer trong repo gốc.

Output layer vẫn có toàn bộ class ngay từ đầu. Khi episode mới đến, chỉ output
unit của class trong episode đó được set thành learner.

### Code hiện tại

File: `fed_learning/models/nice_model.py`

- `NICEModel.__init__()` tạo backbone CNN-GRU và fixed output:

```python
self.fc2 = nn.Linear(256, num_classes)
self._layer_dims = {
    "conv1": 64,
    "conv2": 128,
    "conv3": 256,
    "gru": 100,
    "fc1": 256,
    "fc2": num_classes,
}
```

- `NICEModel._init_unit_ranks()` set tất cả layer về age `0`.
- `NICEServer.set_task()` set output neuron của class mới sang learner:

```python
for cls_id in task_classes:
    self.global_model.unit_ranks["fc2"][cls_id] = 1
```

### Code gốc upstream

File upstream: `Source/architecture.py`

Trong `CNN_Simple.__init__()`:

```python
self.output_layer = SparseOutput(1024, output_size, layer_name="output")
unit_ranks_list = [([999]*self.input_size, "input")]
```

File upstream: `Source/nice_operations.py`

Trong `select_learner_units(...)`, output units của episode được thêm trực tiếp:

```python
top_unit_indices.append(train_episode.classes_in_this_experience)
```

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Fixed output head | `SparseOutput(..., output_size)` | `fc2 = Linear(256, num_classes)` | Tương đương về ý tưởng fixed-head |
| Tất cả neuron ban đầu young | `unit_ranks` khởi tạo `0` | `unit_ranks` dict khởi tạo `0` | Tương đương |
| Output class mới thành learner | `classes_in_this_experience` trong `select_learner_units` | `NICEServer.set_task()` set `fc2[class]=1` | Cùng logic, vị trí khác do federated server quản lý task |

### Fixed-head hay dynamic-head?

NICE gốc không dynamic expand output layer theo từng episode. Output layer được
tạo cố định ngay khi khởi tạo backbone bằng `output_size` của benchmark/dataset:

```python
self.output_layer = SparseOutput(1024, output_size, layer_name="output")
```

Khi episode mới đến, upstream không thêm node output mới. Thay vào đó, code chỉ
đưa output units của class trong episode hiện tại vào tập learner:

```python
top_unit_indices.append(train_episode.classes_in_this_experience)
```

Vì vậy, cơ chế của upstream và repo hiện tại giống nhau ở điểm cốt lõi:

- tất cả output node tồn tại vật lý từ task đầu;
- class chưa xuất hiện vẫn có output node, nhưng chưa được set learner;
- khi class xuất hiện, output node tương ứng được chuyển sang learner;
- training/evaluation chỉ cho các class hợp lệ của task/context tham gia.

Điểm khác là upstream fixed theo `output_size` truyền từ dataset, còn repo hiện
tại fixed cụ thể theo IDS config:

```python
config["num_classes"] = config["total_classes"]  # total_classes = 34
```

Nói ngắn gọn: NICE gốc cũng là fixed-head, không phải dynamic-head. Repo mình
fixed-head `34` là cùng hướng với NICE gốc, chỉ khác kích thước output do dataset.

---

## 2. Lịch phase trong mỗi episode/task

### Ý nghĩa trong NICE

Mỗi episode gồm nhiều phase. Phase đầu tiên dùng 100% neuron ứng viên; các phase
sau chọn neuron bằng activation threshold `activation_perc` của paper. Episode
cuối dùng 100% để sử dụng hết capacity còn lại.

### Code hiện tại

File: `fed_learning/training/task_loop.py`

- `_resolve_nice_schedule()` đọc `nice_max_phases` và `nice_phase_epochs`.
- Path federated NICE chạy `nice_rounds = max_phases`, mỗi exposed round là một
  NICE phase:

```python
server.train_round(..., phase_offset=_r, max_phases_override=1)
```

File: `fed_learning/clients/nice_client.py`

- `NICEClient.train()` chạy phase loop.
- Nếu `is_last_task=True` thì `phase_tau = 1.0`.
- Nếu `global_phase == 0` thì `phase_tau = 1.0`.
- Các phase sau dùng `tau` từ config.

### Code gốc upstream

File upstream: `Source/learner.py`

`Learner.learn_episode(...)` có vòng lặp:

```python
phase_index = 1
selection_perc = 100.0
...
selection_perc = self.args.activation_perc
```

Trong episode cuối, upstream gán lại `selection_perc = 100.0`.

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Phase đầu giữ 100% | `selection_perc = 100.0` | `global_phase == 0 -> phase_tau = 1.0` | Tương đương |
| Phase sau dùng activation percent | `args.activation_perc` | `trainer.tau`, mặc định `0.95` | Tương đương, khác scale percent vs fraction |
| Episode cuối dùng 100% | `episode_index == number_of_tasks` | `is_last_task -> phase_tau = 1.0` | Tương đương |
| Phase được expose thành round | Không có federated round | Mỗi phase là 1 `server.train_round` | Adapt cho federated tracking/checkpoint |

---

## 3. Chọn learner neuron bằng activation

### Ý nghĩa trong NICE

Sau khi forward một subset data của episode hiện tại, NICE tính activation từng
neuron và chọn tập nhỏ nhất đạt ngưỡng tổng activation. Nếu ngưỡng là 95%, tập
được chọn phải giải thích ít nhất 95% tổng activation của learner candidates.

### Code hiện tại

File: `fed_learning/strategies/incremental/nice.py`

- `pick_top_neurons(scores, tau)` sort activation giảm dần, lấy đến khi
  cumulative sum đạt `tau * total`.
- `select_learner_units(model, tau, data)`:
  - reset non-mature về young cho hidden layers;
  - nếu `tau >= 1.0`, promote tất cả young thành learner;
  - nếu không, gọi `model.get_activations(data)`;
  - chọn young units có activation cao và set age `1`.

File: `fed_learning/models/nice_model.py`

- `get_activations()` trả mean absolute activation cho `conv1`, `conv2`,
  `conv3`, `gru`, `fc1`.
- Đã dùng masked path `_apply_masked_conv`, `_apply_masked_linear` để activation
  nhất quán với connectivity hiện tại.

### Code gốc upstream

File upstream: `Source/nice_operations.py`

`pick_top_neurons(...)` làm greedy theo activation:

```python
sort_indices = torch.argsort(-scores)
...
if accumulate >= total * selection_ratio / 100.0:
    break
```

`select_learner_units(...)` lấy activation qua:

```python
layer_activations = network.get_activation_selection(data)
```

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Greedy top activation | `pick_top_neurons(scores, selection_ratio)` | `pick_top_neurons(scores, tau)` | Tương đương |
| Scale ngưỡng | percent, ví dụ `95.0` | fraction, ví dụ `0.95` | Khác scale config |
| Activation source | `get_activation_selection` với `Let_Learner` filter | `get_activations` trên CNN-GRU masked path | Adapt cho CNN-GRU IDS |
| Output units | Append class trong episode tại `select_learner_units` | Server set `fc2[class]=1` trước training | Cùng ý tưởng, khác nơi thực hiện |

---

## 4. Drop young -> learner/non-young connections

### Ý nghĩa trong NICE

NICE cắt kết nối từ neuron young sang neuron đã được gán cho episode nào đó
(`learner` hoặc `mature`). Mục tiêu là neuron trẻ đang học episode mới không
làm thay đổi input của memory neuron cũ.

### Code hiện tại

File: `fed_learning/strategies/incremental/nice.py`

`drop_young_to_learner(model)` xử lý:

- `conv1 -> conv2`;
- `conv2 -> conv3`;
- `conv3 -> fc1` qua flatten mapping;
- `gru -> fc1`;
- `fc1 -> fc2`.

Sau khi update mask, code gọi:

```python
model.apply_masks_to_weights()
```

File: `fed_learning/models/nice_model.py`

`apply_masks_to_weights()` zero physical `weight.data` tại vị trí mask bằng `0`.

### Code gốc upstream

File upstream: `Source/nice_operations.py`

`drop_young_to_learner(...)` tạo drop mask từ:

```python
all_young_indices = [...]
all_not_young_indices = [...]
```

Sau đó set `weight_mask[...] = 0` cho các kết nối young -> non-young.

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Drop young -> non-young | Dùng sparse layer masks | Dùng `weight_masks` dict theo layer | Tương đương |
| Conv -> linear mapping | Có `conv2lin_mapping_size` | Dùng `cnn_len = cnn_output_size // 256` | Adapt cho CNN-GRU |
| Physical zero | `module.set_mask(...)` trong sparse module | `apply_masks_to_weights()` | Tương đương gần đúng |
| GRU | Upstream CNN image không có GRU | Repo thêm `gru -> fc1` và output mask GRU | Domain adaptation |

---

## 5. Grow all -> young connections

### Ý nghĩa trong NICE

Sau khi selection/drop, NICE mở lại tất cả incoming connections vào neuron young
để neuron dự trữ có thể nhận tín hiệu khi được dùng trong phase/episode sau.

### Code hiện tại

File: `fed_learning/strategies/incremental/nice.py`

`grow_all_to_young(model)`:

- với layer 2D/3D: enable tất cả input connections của target neuron age `0`;
- với GRU: set output mask/bias mask của young units về `1`;
- không zero weight vì đây là bước enable mask.

### Code gốc upstream

File upstream: `Source/nice_operations.py`

`grow_all_to_young(...)` tạo `grow_mask` cho target young units và set mask về `1`.

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Enable incoming to young | `weight_mask[grow_mask] = 1` | `mask[idx, :] = 1` hoặc `mask[idx,:,:] = 1` | Tương đương |
| Bias young | Upstream set qua sparse module mask | Repo set `bias_masks[name][young] = 1` | Tương đương |
| GRU | Không có | Output mask của GRU young = 1 | Adapt cho CNN-GRU |

---

## 6. Forward training: MaskedOutYoung và LetLearner

### Ý nghĩa trong NICE

Trong training:

- `MaskedOutYoung`: zero young units ở penultimate representation để young
  không đóng góp lung tung vào output.
- `LetLearner`: chỉ learner output logits được đi qua và nhận gradient.

### Code hiện tại

File: `fed_learning/models/nice_model.py`

- `MaskedOutYoung.forward/backward()` zero young columns cả forward và backward.
- `LetLearner.forward/backward()` zero non-learner logits/gradients.
- `NICEModel.forward_output()`:
  - forward backbone;
  - apply masked `fc1`;
  - apply `MaskedOutYoung` trên `fc1`;
  - apply masked `fc2`;
  - apply `LetLearner` trên output logits.

File: `fed_learning/clients/nice_client.py`

Trong phase training, client dùng:

```python
output = model.forward_output(X_batch)
loss = trainer.compute_loss(model, output, y_batch, global_params)
```

### Code gốc upstream

File upstream: `Source/architecture.py`

Upstream dùng hai custom autograd function cùng vai trò:

```python
x = MaskedOut_Young.apply(x, self.current_young_neurons[-1])
x = Let_Learner.apply(x, self.current_learner_neurons[-1])
```

File upstream: `Source/train_eval.py`

Training gọi:

```python
stream_output = network.forward_output(data)
ce_loss = loss(stream_output, target.long())
```

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Mask young penultimate | `MaskedOut_Young` | `MaskedOutYoung` | Tương đương |
| Let learner output | `Let_Learner` | `LetLearner` | Tương đương |
| Training forward | `network.forward_output(data)` | `model.forward_output(X_batch)` | Tương đương |
| Loss scope | CE trên output sau LetLearner | CE remap trên class có trong batch | Repo thêm guard để unseen/frozen logits không vào denominator |

---

## 7. Loss trong task hiện tại

### Ý nghĩa trong NICE

Paper và repo gốc tránh để output class không liên quan can thiệp vào learning
của episode hiện tại. Repo hiện tại làm rõ hơn bằng cách chỉ tính CE trên class
xuất hiện trong mini-batch.

### Code hiện tại

File: `fed_learning/strategies/incremental/nice.py`

`NICETrainer.compute_loss()`:

- lấy `torch.unique(target)`;
- tạo `class_tensor`;
- remap target về index cục bộ;
- `F.cross_entropy(output.index_select(...), remapped)`.

### Code gốc upstream

File upstream: `Source/train_eval.py`

Upstream dùng output đã qua `Let_Learner`, rồi CE:

```python
stream_output = network.forward_output(data)
ce_loss = loss(stream_output, target.long())
```

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Output non-learner bị zero | Có | Có | Tương đương |
| CE denominator | CE trên full output đã zero non-learner | CE trên batch classes | Repo chuyển thành stricter class-sliced CE để ổn định fixed 34-head IDS |
| Unseen class gradient | Giảm qua LetLearner | Bị loại khỏi CE | Repo bảo vệ mạnh hơn |

---

## 8. Freeze mature neurons

### Ý nghĩa trong NICE

Neuron mature (`age > 1`) không được cập nhật nữa. NICE freeze incoming weights
của mature neurons và freeze BN units liên quan.

### Code hiện tại

File: `fed_learning/strategies/incremental/nice.py`

`update_freeze_masks(model)` tạo:

```python
model.freeze_masks[name] = ranks > 1
```

File: `fed_learning/models/nice_model.py`

`reset_frozen_gradients()` zero gradient rows của mature neurons cho conv/linear.
GRU không freeze theo row internal weights; repo mask GRU output thay vì can thiệp
vào weight matrices phức tạp của GRU.

File: `fed_learning/clients/nice_client.py`

Sau `loss.backward()`:

```python
model.reset_frozen_gradients()
```

File: `fed_learning/models/nice_model.py`

`freeze_bn_for_mature()` set BN eval nếu cả layer đã mature.

### Code gốc upstream

File upstream: `Source/nice_operations.py`

`update_freeze_masks(...)` tạo freeze masks dựa trên mature units:

```python
mature_neurons = network.get_frozen_units()
```

File upstream: `Source/architecture.py`

`reset_frozen_gradients()` zero gradient theo `freeze_masks`.

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Mature definition | `unit_layer > 1` | `ranks > 1` | Tương đương |
| Freeze gradient | `module.weight.grad[freeze_mask] = 0` | row-wise zero grad trong `reset_frozen_gradients` | Tương đương cho conv/linear |
| BN freeze | `freeze_bn_layers()`/freeze BN units | `freeze_bn_for_mature()` layer-level | Gần đúng, đơn giản hơn |
| GRU freeze | Không có GRU | output mask GRU, skip internal GRU grad row freeze | Adapt, không 1:1 |

---

## 9. End episode/task: context memory, age increment, freeze mask

### Ý nghĩa trong NICE

Khi episode kết thúc:

1. Push activation memory cho context detector.
2. Tăng age của neuron đã dùng.
3. Update freeze masks.
4. Freeze BN.

### Code hiện tại

File: `fed_learning/servers/nice_server.py`

`NICEServer.end_task()`:

- gọi `update_context_detector_memory(verbose=True)`;
- gọi `increase_unit_ranks(self.global_model)`;
- gọi `update_freeze_masks(self.global_model)`;
- gọi `freeze_bn_for_mature()`;
- đồng bộ `frozen_keys` và `freeze_masks` vào `NICEAggregator`.

File: `fed_learning/training/post_task.py`

Post-task dispatcher gọi hook `server.end_task()` nếu server có method này.

### Code gốc upstream

File upstream: `Source/learner.py`

`Learner.end_episode(...)` có thứ tự:

```python
self.context_detector.push_activations(...)
self.network = increase_unit_ranks(self.network)
self.network = update_freeze_masks(self.network)
self.network.freeze_bn_layers()
```

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Push context memory | `context_detector.push_activations` | `update_context_detector_memory` -> `push_activations` | Tương đương, data lấy từ simulated clients |
| Age increment | `increase_unit_ranks` | `increase_unit_ranks` | Tương đương |
| Freeze masks | `update_freeze_masks` | `update_freeze_masks` + aggregator sync | Repo thêm federated protection |
| BN freeze | `freeze_bn_layers` | `freeze_bn_for_mature` | Adapt |

---

## 10. Context detector: memory và train logistic chain

### Ý nghĩa trong NICE

Context detector lưu binary activation fingerprints của từng episode. Khi đã có
episode mới, nó train chuỗi binary classifiers:

- classifier cho episode k: positive = samples episode k;
- negative = samples của các episode sau k;
- khi predict, tính chain probability và chọn episode có xác suất lớn nhất.

### Code hiện tại

File: `fed_learning/servers/nice_server.py`

`ContextDetector` hiện tại có:

- `activation_memory: Dict[int, np.ndarray]`;
- `context_masks: Dict[int, np.ndarray]`;
- `binarize_thresholds` per layer;
- `context_learners`;
- `episode_classes`.

`push_activations()`:

- episode `0` fit threshold `mean + std`;
- lưu per-sample binary vectors;
- lưu context mask `unit_ranks > 0`.

`train_models(current_episode)`:

- loop `k in range(current_episode)`;
- positive = `activation_memory[k][:, mask]`;
- negative = concat `activation_memory[j][:, mask]` với `j > k`;
- fit `LogisticRegression(max_iter=1000, solver="lbfgs")`.

`predict_episodes_batch()`:

- lấy `predict_proba`;
- tính `pos_probs`, `neg_probs`;
- tính chain probability;
- return `argmax`.

### Code gốc upstream

File upstream: `Source/context_detector.py`

Upstream có các thành phần tương ứng:

- `quantized_context_representations`;
- `context_layers_masks`;
- `layer_binarizers`;
- `train_models(...)`;
- `tree_preds(...)`;
- `predict_context(...)`.

Trích yếu logic chain:

```python
prev_neg_prob = np.prod(neg_probs[:, :episode_index], axis=1)
chain_probs[:, episode_index] = prev_neg_prob * pos_probs[:, episode_index]
```

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Quantize activation | `Binarizer.fit(mean+std)`, `quantize` | `binarize_thresholds[name]=mean+std` | Tương đương |
| Context masks | `network.unit_ranks[index][0] > 0` | concat `unit_ranks[layer] > 0` | Tương đương |
| Logistic learners | prototype `args.context_learner` | sklearn `LogisticRegression` fixed | Tương đương nếu upstream config là LogisticRegression |
| Episode indexing | 1-based | 0-based | Cần chú ý khi đọc code |
| Prediction | `tree_preds` chain probabilities | `predict_episodes_batch` chain probabilities | Tương đương |

Cần lưu ý trong code hiện tại: `ContextDetector` đang có cặp method
`predict_episode/predict_episodes_batch` bị định nghĩa hai lần trong file; Python
sẽ dùng bản sau. Bản sau mới là bản chain-probability argmax gần với upstream
`tree_preds`.

---

## 11. Inference/test: boost class của episode dự đoán

### Ý nghĩa trong NICE

Khi test, NICE:

1. Forward input để lấy logits và activations.
2. Context detector dự đoán episode.
3. Tăng rất lớn logits của class thuộc episode đó.
4. `argmax` chọn class trong context dự đoán.

### Code hiện tại

File: `fed_learning/models/nice_model.py`

`get_output_and_context_activations(X_batch)` forward một lần để lấy cả:

- `logits`;
- per-sample context activations `conv1`, `conv2`, `conv3`, `gru`.

File: `fed_learning/servers/nice_server.py`

`evaluate_global()`:

- tính `loss_out` với global unseen mask để loss ổn định;
- gọi `_apply_context_mask(...)` cho prediction.

`_apply_context_mask(...)`:

- binarize activations;
- `pred_episodes = context_detector.predict_episodes_batch(...)`;
- với mỗi sample, lấy classes của episode dự đoán;
- boost logits:

```python
masked[row_idx, allowed] = masked[row_idx, allowed] + 99999.0
```

File: `fed_learning/training/local_task_loop.py`

Local IL NICE dùng cùng logic trong `_apply_local_nice_context_mask(...)`.

### Code gốc upstream

File upstream: `Source/train_eval.py`

`test(...)`:

```python
output, activations = network.get_activations(data, return_output=True)
class_preds, episode_preds = context_detector.predict_context(activations, episode_id)
output[index, episode_pred] = output[index, episode_pred] + 99999
```

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Forward logits + activations | `get_activations(..., return_output=True)` | `get_output_and_context_activations` | Tương đương, repo tránh forward 2 lần |
| Predict context | `predict_context` | `predict_episodes_batch` | Tương đương |
| Boost episode classes | `+ 99999` | `+ 99999.0` | Tương đương |
| Global unseen guard | Không cần trong Avalanche stream theo cách repo gốc | Có `_build_global_unseen_mask` cho fixed 34-head | Adapt cần thiết cho IDS fixed-head |

---

## 12. Federated aggregation và bảo vệ mature neurons

### Ý nghĩa trong repo hiện tại

Upstream NICE không có federated aggregation. Repo này phải thêm logic để sau
FedAvg, tham số mature neurons không bị client average ghi đè.

### Code hiện tại

File: `fed_learning/strategies/fed_incremental/nice.py`

`NICEAggregator.aggregate(...)`:

1. Weighted average client params.
2. Restore fully frozen parameter keys từ `global_params`.
3. Restore row-level mature neurons theo `_freeze_masks`.

File: `fed_learning/servers/nice_server.py`

`NICEServer.train_round(...)`:

- gửi global params, neuron ages, masks, freeze masks xuống workers;
- aggregate results;
- merge client neuron ages bằng `np.maximum.reduce(...)`.

File: `fed_learning/training/nice_worker.py`

Worker transfer:

- `neuron_ages`;
- `masks`;
- `freeze_masks`;
- `phase_offset`;
- `max_phases_override`.

### Code gốc upstream

Không có counterpart trực tiếp. Upstream `Learner` train một network duy nhất,
nên không cần aggregation hay merge ages.

### Đối chiếu

| Bước | Upstream NICE | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Local training | Một model duy nhất | Mỗi client train local copy | Federated adaptation |
| Aggregation | Không có | `NICEAggregator` weighted average + restore frozen | Thêm mới, hợp lý |
| Age state | Một `network.unit_ranks` duy nhất | Server state + client states + max merge | Thêm mới để xử lý non-IID client selection |
| Mask broadcast | Không có | Worker config truyền masks xuống client | Thêm mới |

---

## 13. Local IL path

### Code hiện tại

File: `fed_learning/training/local_task_loop.py`

Local NICE dùng cùng primitives:

- `_run_local_nice(...)` set `fc2` class mới thành learner;
- gọi `client.train(...)` theo phase;
- update context memory bằng train samples qua `_update_local_nice_context_memory(...)`;
- sau task gọi `increase_unit_ranks(model)` và `update_freeze_masks(model)`;
- evaluate dùng context boost trong `_apply_local_nice_context_mask(...)`.

### Code gốc upstream

Gần với `Source/learner.py` hơn federated path vì đều là một model local. Điểm
khác là upstream dùng Avalanche experience object, repo hiện tại gộp data từ
tất cả clients thành một pseudo-client local.

### Đối chiếu

| Bước | Upstream NICE | Repo local IL | Nhận xét |
|---|---|---|---|
| Một model | Có | Có | Tương đương |
| Data object | Avalanche `TCLExperience` | TensorDataset từ federated split | Adapt |
| Context memory | `get_n_samples_per_class(train_episode, n)` | sample theo class từ `X_train/y_train` | Tương đương |
| Eval boost | `test(...)` boost class context | `_apply_local_nice_context_mask` boost class context | Tương đương |

---

## 14. Bảng tổng hợp fidelity

| Cơ chế NICE | File repo hiện tại | File upstream | Mức map |
|---|---|---|---|
| Unit ages | `models/nice_model.py` | `Source/architecture.py` | 1:1 về semantic |
| Output class learner | `servers/nice_server.py:set_task` | `Source/nice_operations.py:select_learner_units` | 1:1, khác vị trí |
| Phase loop | `clients/nice_client.py`, `training/task_loop.py` | `Source/learner.py` | 1:1 semantic, federated expose phase thành round |
| Activation selection | `strategies/incremental/nice.py` | `Source/nice_operations.py` | 1:1 core, adapt CNN-GRU |
| Drop young -> non-young | `strategies/incremental/nice.py` | `Source/nice_operations.py` | 1:1 core, thêm GRU |
| Grow young incoming | `strategies/incremental/nice.py` | `Source/nice_operations.py` | 1:1 core, thêm GRU |
| MaskedOutYoung | `models/nice_model.py` | `Source/architecture.py` | 1:1 |
| LetLearner | `models/nice_model.py` | `Source/architecture.py` | 1:1 |
| Mature freeze | `models/nice_model.py`, `strategies/incremental/nice.py` | `Source/architecture.py`, `Source/nice_operations.py` | 1:1 cho conv/linear, GRU adapt |
| Context memory | `servers/nice_server.py` | `Source/context_detector.py` | 1:1 core, 0-based index |
| Logistic chain | `servers/nice_server.py` | `Source/context_detector.py` | 1:1 core |
| Test boost | `servers/nice_server.py`, `local_task_loop.py` | `Source/train_eval.py` | 1:1 |
| Federated aggregation | `strategies/fed_incremental/nice.py` | Không có | Repo-specific extension |

---

## 15. Các sai khác/điểm cần chú ý

1. `ContextDetector` trong `fed_learning/servers/nice_server.py` có method
   `predict_episode` và `predict_episodes_batch` bị định nghĩa lặp. Bản sau
   override bản trước, nên runtime vẫn dùng bản chain-probability. Nên cleanup
   để tránh đọc code nhầm.

2. Upstream episode index là `1..E`, repo là `0..E-1`. Khi so sánh công thức
   context detector phải trừ/đổi index.

3. Repo hiện tại dùng fixed output `34` từ task đầu. Nếu task đầu có 6 class,
   28 output còn lại tồn tại vật lý nhưng:
   - không vào CE của NICE batch-sliced loss;
   - bị global unseen mask trong eval;
   - chỉ được set learner khi class của nó xuất hiện.

4. GRU không có trong upstream NICE image backbone. Repo hiện tại mask GRU output
   và context activation GRU, nhưng không freeze row-level internal GRU weights
   trong `reset_frozen_gradients()`. Đây là adaptation cẩn thận nhưng không 1:1.

5. Upstream sparse modules có `SparseConv2d`, `SparseLinear`, `SparseOutput`.
   Repo hiện tại dùng standard PyTorch layers cộng manual `weight_masks`/`bias_masks`.
   Semantic mask giống nhau, implementation khác.

6. Repo hiện tại update context memory cả sau mỗi NICE train round/phase và trước
   age transition ở `end_task()`. Upstream push context trong mỗi phase và end
   episode. Đây là gần tương đương, phù hợp federated tracking.

---

## 16. Kết luận ngắn

Implementation NICE hiện tại map rất sát với upstream ở các cơ chế cốt lõi:

- age system;
- learner selection bằng activation;
- drop/grow connection masks;
- `MaskedOutYoung` và `LetLearner`;
- freeze mature gradients;
- context detector binary memory + logistic chain;
- inference bằng context-class boost `+99999`.

Các phần không 1:1 chủ yếu đến từ việc repo này adapt NICE sang IDS + federated:

- backbone CNN-GRU thay vì CNN image/sparse layers;
- GRU cần masking riêng;
- fixed 34-class head cần unseen-class guard;
- server/client/worker/aggregator phải đồng bộ age/mask và bảo vệ mature params
  sau FedAvg.
