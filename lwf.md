# LWF.md - Map 1:1 giữa LwF gốc và code hiện tại

Tài liệu này map từng bước của LwF với implementation hiện tại trong repo
`FL_IL_IDS`, đồng thời đối chiếu với repo gốc:

- Upstream: https://github.com/ngailapdi/LWF
- File upstream chính:
  - `model.py`
  - `main.py`
- Code hiện tại:
  - `fed_learning/strategies/incremental/lwf.py`
  - `fed_learning/strategies/fed_incremental/fedlwf.py`
  - `fed_learning/clients/fedlwf_client.py`
  - `fed_learning/servers/fedlwf_server.py`
  - `fed_learning/models/cnn_gru.py`
  - `fed_learning/core/trainer.py`

Ghi chú quan trọng:

- Upstream LwF dùng classifier **mở rộng động** theo class mới.
- Code hiện tại dùng fixed-head IDS: `CNN_GRU_Model.fc2 = nn.Linear(256, num_classes)`,
  với config thường là `num_classes = total_classes = 34`.
- Vì vậy, upstream không có node output cho class chưa tới lượt; node đó chưa tồn tại.
- Code hiện tại có đủ 34 output node từ đầu; class chưa tới lượt bị loại khỏi CE bằng
  seen-class cross entropy, nhưng vẫn tồn tại vật lý trong output layer.

---

## 1. Output layer

### Ý nghĩa trong LwF

LwF gốc học class-incremental bằng cách thêm output neuron khi class mới xuất hiện.
Classifier cũ được copy weight sang phần đầu của classifier mới. Class mới có weight
mới khởi tạo.

### Code gốc upstream

File upstream: `model.py`

Khởi tạo ban đầu:

```python
num_features = self.model.fc.in_features
self.model.fc = nn.Linear(num_features, classes, bias=False)
self.fc = self.model.fc
```

Trong `main.py`, model được tạo bằng:

```python
model = Model(1, class_map, args)
```

`classes = 1` ở đây là classifier ban đầu dạng mồi. Khi task đầu chạy,
`increment_classes()` thay lại `fc` theo số class thực tế của task.

Hàm expand output:

```python
def increment_classes(self, new_classes):
    n = len(new_classes)
    in_features = self.fc.in_features
    out_features = self.fc.out_features
    weight = self.fc.weight.data

    if self.n_known == 0:
        new_out_features = n
    else:
        new_out_features = out_features + n

    self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
    self.fc = self.model.fc

    kaiming_normal_init(self.fc.weight)
    self.fc.weight.data[:out_features] = weight
    self.n_classes += n
```

### Code hiện tại

File: `fed_learning/models/cnn_gru.py`

```python
self.fc2 = nn.Linear(256, num_classes)
```

File: `fed_learning/servers/fedlwf_server.py`

```python
self.global_model = CNN_GRU_Model(
    config["input_shape"], config["num_classes"]
)
```

### Đối chiếu

| Điểm | LwF upstream | Repo hiện tại |
|---|---|---|
| Output layer | Dynamic expand | Fixed-head |
| Task 1 | `fc` được thay bằng số class task 1 | `fc2` đã có đủ `num_classes` |
| Class chưa tới lượt | Chưa có neuron | Có neuron nhưng không dùng trong CE |
| Giữ class cũ | Copy weight cũ vào rows đầu | Giữ cùng shape 34, update theo optimizer |
| Thêm class mới | Tạo `nn.Linear` mới lớn hơn | Không thêm node mới |

Kết luận: LwF gốc **không set sẵn full output layer** như NICE/EWC fixed-head.
Nó expand động theo class mới.

---

## 2. Class mới được phát hiện như nào

### Code gốc upstream

File upstream: `model.py`

```python
classes = list(set(dataset.train_labels))

if self.n_classes == 1 and self.n_known == 0:
    new_classes = [classes[i] for i in range(1,len(classes))]
else:
    new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]

if len(new_classes) > 0:
    self.increment_classes(new_classes)
```

File upstream: `main.py`

```python
model.update(train_set, class_map, args)
model.n_known = model.n_classes
```

### Ý nghĩa

- `n_classes`: số output hiện tại sau khi thêm class mới.
- `n_known`: số class đã biết sau khi train xong iteration trước.
- Trước khi train task mới, upstream tìm class có `class_map[cl] >= n_known`.
- Sau khi train xong, `main.py` set `n_known = n_classes`.

### Code hiện tại

File: `fed_learning/strategies/incremental/lwf.py`

```python
self.old_classes = list(self.seen_classes)
self.new_classes = new_classes
self.current_task = task_id
self.seen_classes.update(new_classes)
```

### Đối chiếu

| Vai trò | LwF upstream | Repo hiện tại |
|---|---|---|
| Class cũ | `n_known` | `old_classes` / `seen_classes` trước update |
| Class mới | `new_classes` từ `class_map` | `new_classes` từ task loop/server |
| Sau task | `n_known = n_classes` | `seen_classes.update(new_classes)` |

---

## 3. Class chưa tới lượt làm gì

### LwF upstream

Class chưa tới lượt **không có output neuron**.

Vì classifier được expand động, nếu task 1 có 10 class thì output chỉ có 10 node.
24 class còn lại không bị mask, không bị prune, không bị freeze, vì chúng chưa tồn tại
trong layer `fc`.

Khi task sau xuất hiện thêm class, upstream tạo `nn.Linear` mới với output lớn hơn:

```python
self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
```

Sau đó copy weight cũ:

```python
self.fc.weight.data[:out_features] = weight
```

### Repo hiện tại

Class chưa tới lượt **có output neuron sẵn** nếu `num_classes = 34`.

File: `fed_learning/core/trainer.py`

```python
seen_logits = output.index_select(dim=1, index=class_tensor)
return F.cross_entropy(seen_logits, remapped, reduction=reduction)
```

Ý nghĩa:

- CE chỉ tính trên `seen_classes`.
- Unseen logits không nằm trong softmax CE.
- Đây là masking ở loss CE, không phải prune node khỏi model.

Lưu ý với KD:

File: `fed_learning/strategies/incremental/lwf.py`

```python
if self.distill_old_classes_only:
    kd_loss = self.compute_distillation_loss(
        old_logits,
        output,
        old_class_indices=self.old_classes if self.old_classes else None,
    )
else:
    kd_loss = self.compute_distillation_loss(old_logits, output)
```

Nếu `distill_old_classes_only=False` thì KD chạy trên toàn bộ output hiện tại.
Với fixed-head 34, tức là KD có thể nhìn cả 34 logit. Nếu bật
`distill_old_classes_only=True`, KD chỉ chạy trên class cũ.

---

## 4. Distillation

### Ý nghĩa trong LwF

LwF giữ tri thức cũ bằng cách copy model trước khi học task mới. Model cũ sinh soft
target. Model mới học task mới bằng CE và học lại output cũ bằng distillation.

### Code gốc upstream

File upstream: `model.py`

```python
prev_model = copy.deepcopy(self)
prev_model.cuda()
```

Trong train loop:

```python
logits = self.forward(images)
cls_loss = nn.CrossEntropyLoss()(logits, labels)

if self.n_classes//len(new_classes) > 1:
    dist_target = prev_model.forward(images)
    logits_dist = logits[:,:-(self.n_classes-self.n_known)]
    dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
    loss = dist_loss+cls_loss
else:
    loss = cls_loss
```

Ý nghĩa của slice:

```python
logits_dist = logits[:,:-(self.n_classes-self.n_known)]
```

- `self.n_classes - self.n_known` là số class mới vừa thêm.
- Slice này lấy phần logit class cũ.
- Distillation chỉ ép model mới giống model cũ trên output cũ.
- Class mới học bằng CE.

### Code hiện tại

File: `fed_learning/clients/fedlwf_client.py`

```python
if self.current_task > 0 and self.old_model_state is not None:
    self._load_old_model(self.model, self.device)
```

```python
loss = trainer.compute_loss(
    self.model, out, y_batch,
    global_params=global_params,
    inputs=X_batch,
    old_model=self.old_model,
    **kwargs
)
```

File: `fed_learning/strategies/incremental/lwf.py`

```python
with torch.no_grad():
    old_logits = old_model(inputs)
```

```python
return ce_loss + self.lwf_alpha * kd_loss
```

### Đối chiếu

| Bước | LwF upstream | Repo hiện tại |
|---|---|---|
| Teacher | `copy.deepcopy(self)` trước update | `old_model_state` snapshot |
| Task 1 | Chỉ CE | Chỉ CE nếu `current_task == 0` hoặc chưa có teacher |
| Task sau | CE + distillation | CE + `lwf_alpha * KD` |
| KD logits | Chỉ old logits qua slice | Toàn bộ logits hoặc old classes tùy config |
| Temperature | `2` hard-coded | `temperature`, default `2.0` |

---

## 5. Server trong code hiện tại

LwF upstream là single-process continual learning, không có server.

Repo hiện tại có `FedLwFServer` để chạy federated version:

File: `fed_learning/servers/fedlwf_server.py`

- tạo global model fixed-head:

```python
self.global_model = CNN_GRU_Model(
    config["input_shape"], config["num_classes"]
)
```

- tạo trainer và aggregator:

```python
self.trainer = FedLwFTrainer(...)
self.aggregator = FedLwFAggregator()
```

- set task và `seen_classes`:

```python
self.current_task = task_id
self.task_classes[task_id] = task_classes
...
self.seen_classes = list(seen_classes)
```

- train client song song rồi aggregate FedAvg:

```python
new_params = self.aggregator.aggregate(results, global_params)
self.set_global_params(new_params)
```

- lưu global snapshot làm teacher cho task sau:

```python
self.trainer.save_model_snapshot(self.global_model)
...
client.old_model_state = OrderedDict(
    (k, v.clone()) for k, v in global_state.items()
)
```

Kết luận: server không phải phần của LwF gốc. Đây là adaptation cho federated learning.

---

## 6. Kết luận nhanh

| Câu hỏi | Trả lời |
|---|---|
| LwF gốc output layer set sẵn không? | Không. Nó expand động bằng `increment_classes()`. |
| Nếu task 1 có 10 class thì 24 node còn lại bị mask không? | Không. Upstream chưa tạo 24 node đó. |
| Khi class mới tới thì làm gì? | Tạo `nn.Linear` mới lớn hơn, copy weight cũ, init weight mới. |
| Distillation áp vào class nào? | Upstream áp vào logit class cũ qua `logits_dist`. |
| Code mình giống upstream không? | Giống ý tưởng CE + KD, khác output head: repo mình fixed-head 34. |
| Code mình xử lý unseen class sao? | CE mask bằng `seen_classes`; KD tùy `distill_old_classes_only`. |

