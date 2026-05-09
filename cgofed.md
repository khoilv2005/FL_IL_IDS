# CGOFed.md - Map 1:1 giữa CGoFed gốc và code hiện tại

Tài liệu này map từng bước của CGoFed với implementation hiện tại trong repo
`FL_IL_IDS`, đồng thời đối chiếu với repo gốc:

- Upstream: https://github.com/fengjiyuan/cgofed
- Code hiện tại:
  - `fed_learning/strategies/fed_incremental/cgofed.py`
  - `fed_learning/clients/cgofed_client.py`
  - `fed_learning/servers/cgofed_server.py`
  - `fed_learning/training/cgofed_worker.py`
  - `fed_learning/models/cnn_gru.py`
  - `fed_learning/training/task_loop.py`
  - `fed_learning/training/post_task.py`

Ghi chú quan trọng:

- Upstream CGoFed là code research cho federated class-incremental learning trên
  CIFAR/Tiny/Image/graph, dùng `AlexNet` multi-head và script loop thủ công.
- Repo hiện tại adapt CGoFed sang IDS với backbone `CNN_GRU_Model`, fixed output
  head `num_classes = total_classes = 34`, server/worker đa GPU, checkpoint,
  resume, và history artifact trên disk.
- Upstream chủ yếu lưu state trực tiếp trong object `Client`; repo hiện tại tách
  state rõ hơn thành `CGoFedClient`, `CGoFedServer`, `CGoFedTrainer`,
  `CGoFedAggregator`.
- Cơ chế lõi vẫn là: lấy representation của task, SVD để tạo basis, gán
  importance, project gradient khi học task mới, dùng similarity/history để
  regularize và personalized aggregation.

---

## 1. Backbone và output head

### Ý nghĩa trong CGoFed

CGoFed cần hai loại representation:

- representation phục vụ projection space của từng layer;
- representation/prototype phục vụ đo similarity giữa client/task.

Trong upstream, model là `AlexNet` với nhiều head theo task. Khi train/test task
nào thì lấy đúng head của task đó.

### Code hiện tại

File: `fed_learning/models/cnn_gru.py`

- `CNN_GRU_Model.__init__()` tạo backbone CNN + GRU và fixed output head:

```python
self.fc1 = nn.Linear(concat_size, 256)
self.fc2 = nn.Linear(256, num_classes)
```

- `get_fused_representation(x)` trả representation trước MLP head:

```python
return torch.cat([cnn_output, gru_output], dim=1)
```

File: `fed_learning/training/task_loop.py`

```python
config["num_classes"] = config["total_classes"]
```

Nghĩa là với config Kaggle hiện tại, output layer là 34 class ngay từ task đầu.

### Code gốc upstream

File upstream: `model.py`

`AlexNet` lưu activation từng layer và dùng `ModuleList` multi-head:

```python
self.fc3 = torch.nn.ModuleList()
for t, n in self.taskcla:
    self.fc3.append(torch.nn.Linear(2048, n, bias=False))
```

Trong `forward()` upstream trả:

```python
return y, feature_x
```

Trong `client.py`/`server.py`, task hiện tại dùng head riêng:

```python
head_idx = model.tid2head[int(task_id)]
logits = output[head_idx]
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Backbone | `AlexNet` image/graph variants | `CNN_GRU_Model` cho IDS sequence | Domain adaptation |
| Head | Multi-head theo task | Fixed 34-class head | Khác lớn về output layer |
| Representation | `feature_x`, `net.act` | `get_fused_representation`, hooks layer input | Cùng mục tiêu, implementation khác |
| Unseen class | Không cần mask do multi-head | Dùng seen-class CE trên fixed head | Adapt cần thiết cho IDS |

---

## 2. Luồng task/round tổng thể

### Ý nghĩa trong CGoFed

Mỗi task gồm nhiều global round. Ở mỗi round:

1. Client train local.
2. Client cập nhật prototype/representation.
3. Server hoặc script tính similarity giữa client/task.
4. Server tạo personalized model hoặc regularization target cho round/task sau.
5. Cuối task lưu history để chống quên ở task tiếp theo.

### Code hiện tại

File: `fed_learning/training/task_loop.py`

- Server được tạo một lần rồi reuse xuyên task để giữ state CGoFed:

```python
# Create server ONCE and reuse for all tasks
server = None
```

- Với mỗi task:

```python
server.set_task(task_id, new_classes, seen_classes)
trainer.set_task(task_id, new_classes)
aggregator.set_task(task_id)
```

File: `fed_learning/servers/cgofed_server.py`

`CGoFedServer.train_round()`:

- lấy global params;
- chuẩn bị optional pre-round Eq.12/Eq.14 state;
- chia client theo GPU;
- gọi `train_cgofed_clients_on_gpu`;
- aggregate bằng `CGoFedAggregator`;
- chuẩn bị personalized model và regularization info cho round kế tiếp nếu chưa
  phải round cuối task.

### Code gốc upstream

File upstream: `main_cifar.py`

Loop gốc:

```python
for epoch in range(args.g_epochs):
    for c_id in range(args.clients_num):
        if task_id == 0:
            clients[c_id].train_first_task(...)
        else:
            old_model_list = select_old_model(...)
            clients[c_id].train_new_task(...)
```

Sau đó upstream tính khoảng cách prototype và personalized aggregation:

```python
compute_distance_curr_feature(...)
compute_distance_with_history_AvgProto(...)
clients[c_id].personalized_global_model = PFL(...)
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Task loop | Script `main_cifar.py`/`main_cifar_reg.py` | `task_loop.py` generic | Repo framework hóa |
| Client train | Gọi trực tiếp từng client | Worker đa GPU/thread | Adapt production |
| State xuyên task | Object `Client` giữ list history | Aggregator/server/client có resume state | Repo robust hơn |
| Personalized aggregation | `PFL(...)` trong script | `CGoFedServer._compute_personalized_models()` | Cùng ý tưởng, khác vị trí |

---

## 3. Task representation `R^t`

### Ý nghĩa trong CGoFed

CGoFed lấy representation từ forward pass để biểu diễn task. Representation này
được dùng cho:

- xây projection space bằng SVD;
- đo similarity giữa current task và historical tasks/clients.

### Code hiện tại

File: `fed_learning/clients/cgofed_client.py`

`compute_activation_representation()` ưu tiên dùng fused representation của
`CNN_GRU_Model`:

```python
rep = model.get_fused_representation(X_batch)
```

Kết quả là matrix `[num_samples, hidden_dim]`, không phải mean vector đơn lẻ.

File: `fed_learning/strategies/fed_incremental/cgofed.py`

`build_representation_artifact(rep)` lưu:

- `matrix`;
- `signature`;
- `shape`;
- `mean_vector`;
- `mean_norm`.

### Code gốc upstream

File upstream: `main_cifar_reg.py`

`get_representation_matrix(net, device, x, y=None)`:

- forward sample qua model;
- đọc `net.act`;
- với conv layer thì tự extract patch;
- với FC layer thì transpose activation;
- trả `mat_list`.

Đoạn chính:

```python
example_out = net(example_data)
act_key = list(net.act.keys())
...
mat_list.append(mat)
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Forward để lấy representation | `get_representation_matrix` | `compute_activation_representation` và hooks | Cùng ý tưởng |
| Matrix thay vì vector | Có, per-layer matrix | Có, task/client representation matrix | Tương đương về semantic |
| Conv patch extraction | Manual nested loops | `F.unfold` trong hook | Repo dùng API PyTorch ổn định hơn |
| IDS fused rep | Không có | CNN+GRU fused representation | Domain adaptation |

---

## 4. Projection space bằng SVD

### Ý nghĩa trong CGoFed

Sau mỗi task, CGoFed xây không gian biểu diễn cũ bằng SVD. Basis này đại diện cho
các hướng quan trọng của task cũ. Khi học task mới, gradient bị trừ bớt thành
phần nằm trong không gian cũ để giảm quên.

### Code hiện tại

File: `fed_learning/clients/cgofed_client.py`

`build_projection_space()` chạy client-local Eq.3-5:

- register forward hook cho Conv/Linear/GRU;
- lấy input activation của từng layer;
- cộng Gram matrix `X^T X`;
- `torch.linalg.eigh(gram)`;
- chọn rank theo `energy_threshold`;
- lưu `basis` và `importance` xuống disk theo client/task.

Đoạn cốt lõi:

```python
eigvals, eigvecs = torch.linalg.eigh(gram)
singular_values = torch.sqrt(eigvals.float())
rank = int((ratio < energy_threshold).sum().item()) + 1
basis = eigvecs[:, :rank].float()
importance = torch.sigmoid(beta * singular_values[:rank])
```

File: `fed_learning/training/post_task.py`

Post-task chỉ xác nhận projection spaces đã sẵn sàng; việc build thật diễn ra
trong client ở round cuối task:

```python
CGoFed: client-local Eq.3-5 projection spaces ready
```

### Code gốc upstream

File upstream: `main_cifar_reg.py`

`update_CGoFed(...)`:

```python
U, S, Vh = np.linalg.svd(activation, full_matrices=False)
sval_ratio = (S**2) / sval_total
r = np.sum(np.cumsum(sval_ratio) < threshold[i])
feature_list.append(U[:, 0:r])
```

Với task sau, upstream project activation lên basis cũ, lấy residual rồi append
basis mới:

```python
act_proj = np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
act_hat = activation - act_proj
U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
Ui = np.hstack((feature_list[i], U[:, 0:r] * (args.beta ** task_id)))
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| SVD/eigen basis | `np.linalg.svd(activation)` | `torch.linalg.eigh(X^T X)` | Tương đương về PCA/SVD subspace |
| Energy threshold | `np.cumsum(sval_ratio) < threshold[i]` | `ratio < energy_threshold` | Tương đương |
| Lưu basis | `client.feature_list` trong RAM | `projection_layer_bases` + `.pt` files | Repo hỗ trợ resume/memory |
| Client-local | Có, trong `Client` | Có, trong `CGoFedClient` | 1:1 semantic |

---

## 5. Importance weights cho basis

### Ý nghĩa trong CGoFed

Không phải mọi basis direction đều quan trọng như nhau. CGoFed gán weight cho
basis dựa trên singular values để hướng quan trọng bị project mạnh hơn.

### Code hiện tại

File: `fed_learning/clients/cgofed_client.py`

```python
importance = torch.sigmoid(beta * singular_values[:rank])
```

File: `fed_learning/strategies/fed_incremental/cgofed.py`

Trainer legacy path cũng dùng cùng ý tưởng:

```python
importance = torch.sigmoid(self.beta * S[:k])
```

Khi build projector:

```python
weighted_basis = basis * importance
```

### Code gốc upstream

Có hai biến thể trong upstream:

- `client.py`: `update_grad_basis_calculate_important_sigmoid(...)` dùng sigmoid:

```python
importance = sigmoid(args.beta * S[0:r])
```

- `main_cifar_reg.py`: `intra_task2_basis_weight(...)` scale singular value theo
  `param_alpha` rồi normalize về `[0, 1]`.

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Importance theo singular value | Có | Có | 1:1 semantic |
| Công thức | sigmoid hoặc normalized power | sigmoid | Repo chọn biến thể sigmoid |
| Apply vào basis | `diag(importance)` | `basis * importance` | Tương đương |

---

## 6. Projection matrix và gradient projection

### Ý nghĩa trong CGoFed

Khi học task mới, gradient `g` được chỉnh:

```text
g' = g - μ_t * g @ P
P  = M @ M^T
```

Trong đó `M` là basis cũ đã weight bằng importance. Mục tiêu là giảm update theo
các hướng đã quan trọng với task cũ.

### Code hiện tại

File: `fed_learning/clients/cgofed_client.py`

`_cache_projection_matrices()`:

- gom basis cũ của các task `< current_task`;
- apply importance;
- concat basis;
- re-orthogonalize bằng SVD;
- tạo projector:

```python
cached[layer_name] = torch.mm(U_orth * (S_normalized**2), U_orth.T)
```

`_apply_relax_constrained_gradient_update()` chạy sau `loss.backward()` và trước
`optimizer.step()`:

```python
projected = torch.mm(grad_2d, projector)
mu_t = trainer.mu_projection * trainer.mu_coefficient
grad_new = grad_2d - mu_t * projected
param.grad.copy_(grad_new.view_as(param.grad))
```

File: `fed_learning/clients/cgofed_client.py`

`train()` gọi projection đúng vị trí:

```python
loss.backward()
clip_grad_norm_(...)
self._apply_relax_constrained_gradient_update(self.model, trainer)
optimizer.step()
```

### Code gốc upstream

File upstream: `client.py`

`train_projected(...)`:

```python
params.grad.data = params.grad.data - mu * torch.mm(
    params.grad.data.view(sz, -1), feature_mat[kk]
).view(params.size())
```

File upstream: `client.py`

`train_new_task(...)` tạo projection matrix:

```python
Uf = torch.Tensor(np.dot(
    self.grad_basis[i],
    np.dot(np.diag(self.importance_list[i]), self.grad_basis[i].transpose())
)).to(device)
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Project gradient sau backward | `train_projected` | `_apply_relax_constrained_gradient_update` | 1:1 |
| Projector từ basis cũ | `feature_mat`/`Uf` | cached per-layer projector | 1:1 semantic |
| Importance-weighted basis | `diag(importance_list)` | `basis * importance`, SVD union | Cùng mục tiêu, repo ổn định số hơn |
| Bias gradient | Upstream fill zero cho bias một số param | Repo skip 1D grad | Khác implementation |
| GRU | Upstream không có | Project `gru.weight_ih_l0` nếu shape khớp | Domain adaptation |

---

## 7. Relaxation coefficient và Average Forgetting

### Ý nghĩa trong CGoFed

CGoFed không project gradient với hệ số cố định hoàn toàn. Hệ số relaxation giảm
theo task và có thể reset nếu average forgetting vượt ngưỡng.

### Code hiện tại

File: `fed_learning/strategies/fed_incremental/cgofed.py`

`CGoFedTrainer.set_task()`:

```python
self.mu_coefficient = self.lambda_decay ** (task_id - self.t_reset)
```

`update_forgetting()`:

```python
if self.last_af > self.theta_threshold:
    self.t_reset = self.current_task
    self.mu_coefficient = 1.0
```

Projection dùng:

```python
mu_t = self.mu_projection * self.mu_coefficient
```

### Code gốc upstream

File upstream: `client.py`

`train_projected(...)`:

```python
initial_mu = 1.0
decay_rate = args.alpha
if avg_forgetting[-1] < args.tau:
    mu = initial_mu * (decay_rate ** task_id)
else:
    scale_value = task_id - max_scale_value + 1
    mu = initial_mu * (decay_rate ** scale_value)
```

File upstream: `main_cifar.py`

Cuối task tính average forgetting:

```python
avg_forgetting.append(calculate_average_forgetting(acc_matrix, task_id))
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Power decay | `args.alpha ** task_id` | `lambda_decay ** (task_id - t_reset)` | Tương đương |
| AF threshold | `args.tau` | `theta_threshold` | Tương đương |
| Reset khi quên cao | Có logic reset scale | `t_reset = current_task` | Tương đương rõ hơn |
| Nguồn accuracy | `acc_matrix` script | `update_forgetting(task_accuracies)` | Framework hóa |

---

## 8. Local objective và cross-task regularization

### Ý nghĩa trong CGoFed

Khi học task mới, local loss không chỉ có CE mà còn thêm regularization để model
hiện tại không đi quá xa các historical model được chọn theo similarity.

### Code hiện tại

File: `fed_learning/strategies/fed_incremental/cgofed.py`

`CGoFedTrainer.compute_loss()`:

```python
ce_loss = self._seen_class_cross_entropy(output, target)
...
task_reg += torch.sum((param - hist_param) ** 2)
reg_term += weight * task_reg
total_loss = ce_loss + (lambda_reg / 2) * reg_term
```

`_seen_class_cross_entropy()` nằm ở `fed_learning/core/trainer.py`, giúp fixed
34-head không đẩy gradient vào class chưa xuất hiện.

### Code gốc upstream

File upstream: `server.py` hoặc `main_cifar_reg.py`

`Regularization.forward()`:

```python
reg_loss = self.regularization_loss(self.weight_list, self.old_model_list, p=self.p)
```

`regularization_loss(...)`:

```python
reg_loss = reg_loss + torch.norm(w_cur - w_old, p=p)
```

File upstream: `main_cifar_reg.py`

Trong `train_projected(...)`:

```python
local_loss_value = criterion(output[task_id], target)
reg_loss_value = reg_loss(model)
loss_value = local_loss_value + reg_loss_value / (...) * 2
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| CE + regularization | Có | Có | 1:1 semantic |
| Regularization target | `old_model_list` đã chọn | `historical_models` per client | Tương đương |
| Weight theo similarity | Upstream aggregate old model theo distance | Repo softmax similarity weights | Cùng mục tiêu, repo rõ ràng hơn |
| Loss scale | Heuristic digit scaling | `lambda_cross_task / 2` | Repo ổn định và có hyperparam |
| CE scope | Multi-head task CE | Seen-class CE trên fixed 34-head | Adapt quan trọng |

---

## 9. Similarity giữa task/client

### Ý nghĩa trong CGoFed

CGoFed đo độ gần giữa representation/prototype hiện tại và lịch sử để chọn model
cũ hữu ích nhất.

### Code hiện tại

File: `fed_learning/strategies/fed_incremental/cgofed.py`

`compute_representation_similarity()`:

```python
return -torch.norm(mat_a[:n] - mat_b[:n], p="fro").item()
```

Nếu matrix không khớp shape, repo fallback sang `representation_signature()`
gồm mean vector, leading singular values và row norm stats.

File: `fed_learning/servers/cgofed_server.py`

`_select_peer_clients_from_history()`:

- so sánh current representation của client với historical representations của
  các client khác;
- chọn top-k peer clients có similarity cao nhất.

### Code gốc upstream

File upstream: `server.py`

`compute_distance_curr_feature(...)`:

```python
d = np.linalg.norm(curr_feature - clients[i].curr_AvgProto, ord=2)
```

`compute_distance_with_history_AvgProto(...)`:

```python
d = np.linalg.norm(
    clients[curr_client].curr_AvgProto - clients[i].history_AvgProto[r],
    ord=2,
)
```

`select_old_model(...)` dùng `heapq.nsmallest(...)` để chọn client/history gần
nhất.

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Similarity metric | L2 distance giữa prototype | Negative Frobenius/L2 trên matrix/signature | Cùng nguyên lý khoảng cách representation |
| Chọn top-k | `heapq.nsmallest` | sort similarity giảm dần | Tương đương |
| Current vs history | Current prototype vs history prototype | Current rep vs historical client/task rep | Repo dùng matrix giàu thông tin hơn |
| Self client | Upstream loại current client | Repo cũng skip `other_id == client_id` | Tương đương |

---

## 10. Personalized aggregation Eq.12

### Ý nghĩa trong CGoFed

Mỗi client không nhất thiết nhận cùng một global model. Client có thể nhận model
đã pha giữa model của chính nó và model của các peer gần nhất.

### Code hiện tại

File: `fed_learning/servers/cgofed_server.py`

`_compute_personalized_models(results)`:

- lấy params và representation của từng client trong round hiện tại;
- tính similarity giữa client hiện tại và client khác;
- softmax similarity thành weights;
- weighted average model của peer;
- blend với own model:

```python
personalized_models[client_id] = self._blend_models(
    own_params,
    others_agg,
    self_weight=self.eq12_self_weight,
)
```

`_blend_models()`:

```python
mixed = self_weight * own_tensor.float() + (1.0 - self_weight) * other_tensor
```

File: `fed_learning/training/cgofed_worker.py`

`get_init_params()` gửi personalized init xuống đúng client nếu có:

```python
if client.client_id in self.client_init_models:
    return client_init
return self.global_params
```

### Code gốc upstream

File upstream: `server.py`

`PFL(...)`:

```python
w_g_personalized = copy.deepcopy(clients[curr_i].model.state_dict())
w_g_personalized.update((key, value * 0.9) for key, value in w_g_personalized.items())
...
w_g_personalized[k] += 0.1 * clients[curr_i].dis_with_other[i] * clients[i].model.state_dict()[k]
```

File upstream: `main_cifar.py`

```python
clients[c_id].personalized_global_model = PFL(c_id, clients, args.clients_num)
model_g[c_id].load_state_dict(clients[c_id].personalized_global_model)
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Personalized model per client | `PFL` | `_compute_personalized_models` | 1:1 semantic |
| Own model weight | hardcoded `0.9` trong upstream code | `eq12_self_weight`, default clamp `[0,1]` | Repo configurable |
| Peer weights | inverse normalized distance | softmax similarity | Cùng mục tiêu |
| Delivery to client | load vào `model_g[c_id]` | worker `get_init_params()` | Framework hóa |

---

## 11. Historical model selection cho Eq.14

### Ý nghĩa trong CGoFed

Local regularization cần biết nên regularize theo model cũ nào. Upstream chọn
historical model từ các client/task gần nhất. Repo hiện tại cũng chọn theo
history, nhưng đóng gói per-client state rõ hơn.

### Code hiện tại

File: `fed_learning/servers/cgofed_server.py`

`_prepare_reg_info_from_current_reps()`:

- với từng client hiện tại, chọn peer clients từ history;
- duyệt historical task reps/models của peer;
- tính similarity với current rep;
- softmax thành `similarity_weights`;
- đóng gói `historical_models` để worker gửi xuống client.

File: `fed_learning/training/cgofed_worker.py`

`get_train_kwargs()`:

```python
kwargs["historical_models"] = {
    key: load_model_state(model_ref)
    for key, model_ref in historical_models.items()
}
kwargs["similarity_weights"] = reg_info["similarity_weights"]
```

### Code gốc upstream

File upstream: `server.py`

`select_old_model(...)`:

- lấy top-k clients theo `history_dis`;
- average các `history_model`;
- chỉ trả về weight params:

```python
for layer_name, param in clients[key].history_model[r].items():
    avg_selected_model[layer_name] = ...
...
if ("weight" in name):
    old_weight_list.append((name, param))
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Chọn historical peer | `select_old_model` | `_select_peer_clients_from_history` | Tương đương |
| Historical model source | `clients[key].history_model[r]` | `client_historical_models[cid][task]` artifact | Tương đương, repo disk-backed |
| Weight model cũ | distance ratio | softmax similarity | Cùng mục tiêu |
| Truyền xuống local train | `old_model_list` argument | `historical_models` kwargs | Framework hóa |

---

## 12. Aggregator và history artifact

### Ý nghĩa trong repo hiện tại

Upstream chạy trong một process và giữ history trong RAM. Repo hiện tại cần chạy
Kaggle/resume/multi-GPU nên phải lưu history model/representation bền hơn.

### Code hiện tại

File: `fed_learning/strategies/fed_incremental/cgofed.py`

`CGoFedAggregator.aggregate()`:

1. FedAvg client params:

```python
agg_params = self._weighted_average(results)
```

2. Lưu representation của từng client:

```python
self._store_client_representations(results)
```

3. Cuối task, lưu client historical model:

```python
self.client_historical_models[client_id][self.current_task] = persist_model_artifact(...)
```

4. Cuối task, lưu global model:

```python
self.task_global_models[self.current_task] = persist_model_artifact(...)
```

5. Nếu task > 0, chọn top-k historical task:

```python
selected = self._select_top_k_similar()
```

### Code gốc upstream

File upstream: `server.py`

`FedAvg(models)`:

```python
w_avg = copy.deepcopy(models[0])
for k in w_avg.keys():
    for i in range(1, len(models)):
        w_avg[k] += models[i][k]
    w_avg[k] = torch.div(w_avg[k], len(models))
```

Historical model ở upstream được append trong client:

```python
self.history_model.append(self.model.state_dict())
self.history_AvgProto.append(self.curr_AvgProto)
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| FedAvg | `FedAvg(models)` | `_weighted_average(results)` | Tương đương |
| Client history | RAM list trong `Client` | `client_historical_models` artifact | Repo hỗ trợ resume |
| Global history | Script/model state | `task_global_models` artifact | Repo rõ hơn |
| Representation history | `history_AvgProto`, `history_mat_list` | `client_representations`, `task_representation_matrices` | Tương đương semantic |

---

## 13. Worker truyền Eq.12/Eq.14 state

### Ý nghĩa trong repo hiện tại

Do repo train nhiều client song song trên GPU/CPU worker, state CGoFed không thể
chỉ nằm trong script loop như upstream. Worker phải truyền đúng personalized init
và regularization info cho từng client.

### Code hiện tại

File: `fed_learning/training/cgofed_worker.py`

`CGoFedWorker.get_init_params()`:

- ưu tiên personalized model Eq.12;
- fallback về global params.

`CGoFedWorker.get_train_kwargs()`:

- truyền `global_params`;
- truyền `build_projection_space`;
- materialize `historical_models`;
- truyền `similarity_weights`.

### Code gốc upstream

Không có worker riêng. Script upstream trực tiếp:

- gọi `clients[c_id].train_first_task(...)` hoặc `train_new_task(...)`;
- truyền `old_model_list`;
- sau round gọi `PFL(...)` rồi load model vào `model_g[c_id]`.

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Truyền old models | function argument `old_model_list` | kwargs `historical_models` | Tương đương |
| Truyền personalized init | `model_g[c_id].load_state_dict(...)` | `get_init_params()` | Tương đương |
| Multi-GPU | Không có | `CGoFedWorker` | Repo-specific extension |

---

## 14. Fixed 34-head và masking unseen class

### Ý nghĩa trong repo hiện tại

Upstream dùng multi-head nên task chưa xuất hiện không nằm trong CE của task hiện
tại. Repo hiện tại fixed 34-head nên phải tránh cho class chưa thấy tham gia
softmax/loss.

### Code hiện tại

File: `fed_learning/core/trainer.py`

`_seen_class_cross_entropy()`:

- lấy `seen_classes`;
- slice logits theo class đã thấy;
- remap target về index cục bộ;
- fallback CE bình thường nếu thiếu thông tin.

File: `fed_learning/strategies/fed_incremental/cgofed.py`

`CGoFedTrainer.compute_loss()` gọi:

```python
ce_loss = self._seen_class_cross_entropy(output, target)
```

### Code gốc upstream

Upstream chọn task head:

```python
logits = output[head_idx]
loss = criterion(logits, target)
```

### Đối chiếu

| Bước | Upstream CGoFed | Repo hiện tại | Nhận xét |
|---|---|---|---|
| Không train unseen class | Tự nhiên do multi-head | Slice seen-class logits | Tương đương mục tiêu |
| Output layer | Head riêng mỗi task | 34 logits ngay từ đầu | Khác implementation |
| Class chưa xuất hiện | Không có trong current head | Tồn tại vật lý nhưng bị loại khỏi CE | Adapt cần thiết |

---

## 15. Bảng tổng hợp fidelity

| Cơ chế CGoFed | File repo hiện tại | File upstream | Mức map |
|---|---|---|---|
| Task representation `R^t` | `clients/cgofed_client.py`, `models/cnn_gru.py` | `model.py`, `main_cifar_reg.py` | 1:1 semantic, backbone khác |
| Per-layer activation matrix | `build_projection_space`, `_activation_to_features` | `get_representation_matrix` | 1:1 core, dùng `F.unfold` |
| SVD basis | `build_projection_space`, `_build_representation_space_from_activations` | `update_CGoFed`, `update_grad_basis...` | 1:1 core |
| Importance weights | `torch.sigmoid(beta * singular_values)` | `sigmoid(args.beta * S)` / normalized singular values | 1:1 với biến thể sigmoid |
| Gradient projection | `_apply_relax_constrained_gradient_update` | `train_projected` | 1:1 |
| Relaxation coefficient | `set_task`, `update_forgetting` | `avg_forgetting`, `args.alpha`, `args.tau` | 1:1 semantic |
| Cross-task regularization | `compute_loss` | `Regularization`, `train_projected` | 1:1 semantic, scale khác |
| Similarity selection | `compute_representation_similarity`, `_select_peer_clients_from_history` | `compute_distance_*`, `select_old_model` | 1:1 semantic |
| Personalized aggregation | `_compute_personalized_models`, `_blend_models` | `PFL` | 1:1 semantic |
| FedAvg | `CGoFedAggregator.aggregate` | `FedAvg` | 1:1 |
| History storage | artifacts + resume state | `history_model`, `history_AvgProto` lists | Repo-specific robust extension |
| Worker dispatch | `training/cgofed_worker.py` | Không có | Repo-specific extension |
| Output head | fixed 34-head + seen-class CE | multi-head `fc3` | Khác lớn, nhưng mục tiêu CE tương đương |

---

## 16. Các sai khác/điểm cần chú ý

1. Upstream CGoFed dùng multi-head theo task; repo hiện tại dùng fixed output
   `34` ngay từ task đầu. Vì vậy repo phải dùng seen-class CE để class chưa thấy
   không nhận gradient qua softmax.

2. Upstream có nhiều biến thể code (`client.py`, `main_cifar.py`,
   `main_cifar_reg.py`, graph scripts). Repo hiện tại lấy phần lõi của CGoFed
   rồi framework hóa vào trainer/client/server/aggregator.

3. Upstream giữ history trong RAM (`history_AvgProto`, `history_model`,
   `feature_list`, `importance_list`). Repo hiện tại lưu model/basis thành
   artifact `.pt` để giảm RAM và hỗ trợ resume.

4. Upstream prototype similarity chủ yếu dựa trên average prototype
   `curr_AvgProto`. Repo hiện tại dùng representation matrix và signature; đây
   là cùng mục tiêu nhưng giàu thông tin hơn.

5. Upstream build projection matrix đơn giản từ basis hiện có. Repo hiện tại
   concat weighted basis từ nhiều old tasks rồi re-orthogonalize bằng SVD để
   tránh projector có eigenvalue quá lớn, ổn định hơn khi nhiều task.

6. Upstream có mixup trong `client.py`; repo hiện tại không map phần mixup thành
   cơ chế bắt buộc của CGoFed. Đây là augmentation/training detail, không phải
   core CGoFed.

7. GRU không tồn tại trong upstream image backbone. Repo hiện tại có support
   projection cho `gru.weight_ih_l0` khi dimension khớp, nhưng đây là IDS-specific
   adaptation, không phải 1:1 với upstream.

8. `CGoFedTrainer.pre_step()` vẫn còn legacy trainer-side projection fallback.
   Path paper-faithful hiện tại chạy projection trong `CGoFedClient` sau
   `backward()` và trước `optimizer.step()`.

9. `cgofed_pre_round_state` là extension của repo để chuẩn bị Eq.12/Eq.14 trước
   round đầu của task mới. Nếu tắt option này và `rounds_per_task=1`, Eq.14 có
   thể không có state cho round đầu của task.

---

## 17. Kết luận ngắn

Implementation CGoFed hiện tại map sát upstream ở các cơ chế cốt lõi:

- representation từ forward pass;
- SVD/PCA basis theo energy threshold;
- importance weights theo singular values;
- gradient projection khi học task mới;
- relaxation coefficient dựa trên task/forgetting;
- chọn historical models theo similarity;
- cross-task regularization;
- personalized aggregation;
- FedAvg và lưu history theo task/client.

Các phần không 1:1 chủ yếu đến từ việc repo này adapt CGoFed sang IDS +
federated runtime:

- `CNN_GRU_Model` thay vì `AlexNet`;
- fixed 34-class head thay vì multi-head;
- seen-class CE để chặn unseen class;
- artifact/resume thay vì RAM-only history;
- worker đa GPU và server/aggregator abstraction;
- projection support cho Conv1d/GRU phù hợp dữ liệu IDS.
