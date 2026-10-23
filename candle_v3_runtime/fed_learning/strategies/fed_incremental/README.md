# Incremental Strategies Flow

Tài liệu này gom flow chạy của các thuật toán trong thư mục `fed_learning/strategies/fed_incremental/` vào một chỗ.

Mục tiêu của file này là trả lời 3 câu hỏi:
- thuật toán này dùng để làm gì
- những hàm nào là quan trọng nhất
- mỗi hàm chạy ở thời điểm nào trong pipeline train

## Quy ước chung

- Đầu task
  - thường được gọi từ `fed_learning/training/task_loop.py`
  - các trainer nhận `set_task(...)`
- Mỗi batch
  - thường được gọi từ `fed_learning/clients/client.py` hoặc client riêng
  - trainer tính `compute_loss(...)`
  - nếu có thì `pre_step(...)` chạy sau `backward()` và trước `optimizer.step()`
- Cuối task
  - thường được gọi từ `fed_learning/training/post_task.py`
  - các thuật toán sẽ tính Fisher, lưu snapshot, build replay/exemplar, v.v.
- Mỗi round federated
  - aggregator chạy `aggregate(...)`

---

## EWC

File chính: `fed_learning/strategies/fed_incremental/ewc.py`

### Mục tiêu

EWC không phải thuật toán federated hoàn chỉnh. Nó là regularization gắn vào `FedAvg` hoặc `FedProx` để giảm quên khi học task mới.

### Class chính

- `EWCMixin`
- `FedAvgEWCTrainer`
- `FedProxEWCTrainer`

### Flow hàm

- `__init__()`
  - chạy khi strategy được tạo
  - khởi tạo hyperparameter, cache RAM, thư mục lưu Fisher

- `set_task(task_id, new_classes)`
  - chạy ở đầu task
  - cập nhật task hiện tại, class đã thấy, reset cache

- `compute_loss(model, output, target, global_params, **kwargs)`
  - chạy mỗi batch
  - lấy loss nền từ FedAvg hoặc FedProx
  - nếu đã có Fisher thì cộng thêm penalty EWC

- `_ensure_cache_loaded(device)`
  - được `compute_loss()` gọi khi batch đầu cần EWC
  - nạp Fisher và optimal params lên đúng device

- `consolidate(model, data_loader, device)`
  - chạy cuối task, gọi từ `post_task.py`
  - tính Fisher mới, lưu optimal params, cộng dồn Fisher

- `compute_fisher_information(model, data_loader, device)`
  - được `consolidate()` gọi
  - duyệt từng sample để ước lượng diagonal Fisher

- `_get_prev_fisher_acc()`
  - helper lấy Fisher task trước từ RAM hoặc disk

- `update_forgetting(task_accuracies)`
  - chạy sau evaluate
  - cập nhật Average Forgetting

- `get_current_af()`
  - trả về AF mới nhất

### File liên quan

- `fed_learning/strategies/__init__.py`
- `fed_learning/training/task_loop.py`
- `fed_learning/clients/client.py`
- `fed_learning/training/post_task.py`

---

## FedLwF

File chính: `fed_learning/strategies/fed_incremental/fedlwf.py`

### Mục tiêu

FedLwF lưu model cũ làm teacher. Ở task mới, model hiện tại học cả nhãn mới lẫn bắt chước soft logits của teacher để giảm quên.

### Class chính

- `FedLwFTrainer`
- `FedLwFAggregator`
- `FedLwFWithProximalTrainer`

### Flow hàm

- `__init__()`
  - khởi tạo trọng số distillation, temperature, nơi lưu teacher snapshot

- `set_task(task_id, new_classes)`
  - chạy đầu task
  - cập nhật class cũ/class mới, reset cache teacher model

- `compute_loss(model, output, target, global_params, inputs, old_model, **kwargs)`
  - chạy mỗi batch
  - task đầu: CE
  - task sau: CE + KD loss

- `load_old_model(model_template, device)`
  - được gọi khi cần teacher model
  - load snapshot task trước từ RAM hoặc disk

- `compute_distillation_loss(old_logits, new_logits, old_class_indices)`
  - helper tính KL distillation loss

- `save_model_snapshot(model)`
  - chạy cuối task
  - lưu model hiện tại làm teacher cho task sau

- `update_forgetting(task_accuracies)`
  - cập nhật forgetting sau evaluate

- `cleanup()`
  - dọn cache/thư mục tạm
  - không phải flow bắt buộc của main loop

- `FedLwFAggregator.aggregate(...)`
  - chạy mỗi round
  - weighted-average như FedAvg

### File liên quan

- `fed_learning/clients/fedlwf_client.py`
- `fed_learning/training/post_task.py`
- `fed_learning/strategies/__init__.py`

---

## FedCBDR

File chính: `fed_learning/strategies/fed_incremental/fedcbdr.py`

### Mục tiêu

FedCBDR dùng replay buffer cân bằng theo lớp và TTS loss để giữ cân bằng giữa tri thức cũ và mới.

### Thành phần chính

- `FedCBDRTrainer`
- `FedCBDRAggregator`
- `ReplayBuffer`
- `LeverageScoreCalculator`

### Flow trainer

- `__init__()`
  - lưu `tau_old`, `tau_new`, `omega_old`, `omega_new`

- `set_task(task_id, new_classes)`
  - chạy đầu task
  - cập nhật class cũ/class mới

- `compute_loss(model, output, target, global_params, **kwargs)`
  - chạy mỗi batch
  - task đầu dùng CE
  - task sau gọi `_compute_tts_loss()`

- `_compute_tts_loss(logits, target)`
  - chia logit old/new
  - scale theo `tau_old`, `tau_new`
  - trộn loss old/new bằng `omega_old`, `omega_new`

- `update_forgetting(task_accuracies)`
  - cập nhật forgetting sau evaluate

### Flow replay buffer

- `ReplayBuffer.add_samples(...)`
  - thêm dữ liệu replay mới
- `ReplayBuffer._rebalance()`
  - cân bằng buffer theo lớp
- `ReplayBuffer._update_count()`
  - cập nhật tổng số mẫu
- `ReplayBuffer.get_all_samples()`
  - lấy toàn bộ replay data
- `ReplayBuffer.get_balanced_batch(batch_size)`
  - lấy batch replay cân bằng theo lớp
- `ReplayBuffer.update_importance(class_id, new_importance)`
  - cập nhật importance score
- `ReplayBuffer.get_class_distribution()`
  - xem phân bố lớp
- `ReplayBuffer.clear()`
  - xóa buffer

### Flow leverage score

- `LeverageScoreCalculator.compute_scores(features, normalize)`
  - tính leverage score từ đặc trưng
- `LeverageScoreCalculator.compute_scores_encrypted(features, P, Q)`
  - tính bản mã hóa bảo toàn riêng tư
- `LeverageScoreCalculator._random_orthogonal(n, device)`
  - helper sinh ma trận trực chuẩn

### Aggregation

- `FedCBDRAggregator.aggregate(...)`
  - weighted-average như FedAvg

### File liên quan

- `fed_learning/clients/fedcbdr_client.py`
- `fed_learning/servers/fedcbdr_server.py`
- `fed_learning/training/post_task.py`

---

## DER

File chính: `fed_learning/strategies/fed_incremental/der.py`

### Mục tiêu

DER học theo hai giai đoạn để mở rộng biểu diễn động và sau đó ổn định classifier.

### Class chính

- `DERTrainer`
- `DERAggregator`

### Flow trainer

- `__init__()`
  - khởi tạo stage, annealing, hyperparameter loss

- `set_task(task_id, new_classes)`
  - chạy đầu task
  - cập nhật class cũ/class mới, reset batch counter

- `set_stage(stage)`
  - chạy khi chuyển stage
  - `stage=1`: representation learning
  - `stage=2`: classifier finetuning

- `compute_loss(model, output, target, global_params, inputs, s, **kwargs)`
  - dispatcher theo stage

- `compute_annealing_s()`
  - tính annealing `s` cho mask

- `_stage1_loss(model, output, target, inputs, s)`
  - CE chính + auxiliary loss + sparsity loss

- `_stage2_loss(output, target)`
  - CE với temperature scaling

- `_remap_aux_targets(target, device)`
  - gộp lớp cũ vào `other`, remap lớp mới cho auxiliary classifier

- `update_forgetting(task_accuracies)`
  - cập nhật forgetting sau evaluate

### Flow aggregator

- `set_trainable_keys(keys)`
  - server báo param nào được phép thay đổi ở task hiện tại

- `aggregate(results, global_params, **kwargs)`
  - average toàn bộ param
  - sau đó khôi phục frozen params từ global model

- `set_task(task_id)`
  - hàm tương thích với pipeline chung

### File liên quan

- `fed_learning/training/task_loop.py`
- `fed_learning/clients/der_client.py`
- `fed_learning/servers/der_server.py`

---

## NICE

File chính: `fed_learning/strategies/fed_incremental/nice.py`

### Mục tiêu

NICE là thuật toán replay-free dựa trên quản lý tuổi neuron, phase training, gradient masking và freeze theo từng neuron.

### Thành phần chính

- `pick_top_neurons()`
- `select_learner_units()`
- `drop_young_to_learner()`
- `grow_all_to_young()`
- `increase_unit_ranks()`
- `update_freeze_masks()`
- `NICETrainer`
- `NICEAggregator`

### Flow các hàm neuron/mask

- `pick_top_neurons(scores, tau)`
  - helper toán học cho chọn neuron

- `select_learner_units(model, tau, data)`
  - chạy đầu mỗi phase ở client
  - chọn learner neuron mới theo activation

- `drop_young_to_learner(model)`
  - chạy sau khi chọn learner
  - cắt các kết nối không mong muốn từ young neuron

- `grow_all_to_young(model)`
  - mở lại kết nối vào neuron young

- `increase_unit_ranks(model)`
  - chạy cuối task
  - learner trở thành mature

- `update_freeze_masks(model)`
  - chạy cuối task
  - tạo freeze mask cho neuron mature

### Flow trainer

- `__init__()`
  - lưu `tau`, `max_phases`, `phase_epochs`, `memo_per_class`

- `set_task(task_id, new_classes)`
  - chạy đầu task

- `compute_loss(model, output, target, global_params, **kwargs)`
  - chạy mỗi batch
  - dùng CE chuẩn

- `pre_step(model, global_params, **kwargs)`
  - chạy sau `backward()` và trước `optimizer.step()`
  - khóa gradient của neuron mature

- `update_forgetting(task_accuracies)`
  - cập nhật forgetting sau evaluate

### Flow aggregator

- `set_frozen_keys(keys)`
  - đánh dấu param freeze toàn phần

- `set_freeze_masks(freeze_masks)`
  - đánh dấu freeze theo từng neuron/layer

- `aggregate(results, global_params, **kwargs)`
  - average như FedAvg
  - sau đó khôi phục tham số của neuron mature

### File liên quan

- `fed_learning/clients/nice_client.py`
- `fed_learning/servers/nice_server.py`
- `fed_learning/training/task_loop.py`
- `fed_learning/training/post_task.py`

---

## GLFC

File chính: `fed_learning/strategies/fed_incremental/glfc.py`

### Mục tiêu

GLFC bù quên cục bộ bằng distillation + gradient compensation, đồng thời bù quên toàn cục bằng entropy signal và proxy server.

### Thành phần chính

- `get_one_hot()`
- `compute_entropy()`
- `GLFCTrainer`
- `GLFCAggregator`

### Flow helper

- `get_one_hot(target, num_class, device)`
  - helper cho `compute_loss()`

- `compute_entropy(probs)`
  - helper cho `compute_entropy_signal()`

### Flow trainer

- `__init__()`
  - khởi tạo exemplar memory, proxy model state, entropy tracking

- `set_task(task_id, new_classes)`
  - chạy đầu task
  - cập nhật class cũ/mới, reset old model cache

- `compute_entropy_signal(model, data_loader, device)`
  - chạy trước hoặc trong train
  - đo entropy để phát hiện knowledge shift

- `load_old_model(model_template, device, signal)`
  - load teacher model theo entropy signal

- `efficient_old_class_weight(output, label, num_class, device)`
  - tính trọng số class-aware cho từng mẫu

- `compute_loss(model, output, target, global_params, inputs, old_model, signal, **kwargs)`
  - nếu chưa có old model: weighted BCE hiện tại
  - nếu có old model: trộn `loss_cur` và `loss_old`

- `save_model_snapshot(model)`
  - chạy cuối task

- `update_proxy_server_models(model, perf)`
  - cập nhật 2 best model của proxy server

- `get_old_models()`
  - trả về cặp model cũ dùng cho distillation

- `update_forgetting(task_accuracies)`
  - cập nhật forgetting sau evaluate

- `cleanup()`
  - dọn cache và thư mục tạm

### Aggregation

- `GLFCAggregator.aggregate(...)`
  - aggregate như FedAvg

### File liên quan

- `fed_learning/clients/glfc_client.py`
- `fed_learning/servers/glfc_server.py`
- `fed_learning/training/task_loop.py`
- `fed_learning/training/post_task.py`

---

## Re-Fed

File chính: `fed_learning/strategies/fed_incremental/refed.py`

### Mục tiêu

Re-Fed dùng PIM và importance scoring để chọn replay hiệu quả. File strategy này chỉ giữ phần trainer đơn giản; phần quan trọng nằm ở client/server.

### Class chính

- `ReFedTrainer`
- `ReFedAggregator`

### Flow trainer

- `__init__()`
  - lưu `memory_size`, `lambda_pim`, `pim_iterations`

- `set_task(task_id, task_classes)`
  - chạy đầu task
  - cập nhật class cũ/class mới

- `compute_loss(model, output, target, global_params, **kwargs)`
  - chạy mỗi batch
  - dùng CE chuẩn

### Aggregation

- `ReFedAggregator.aggregate(results, global_params, **kwargs)`
  - weighted-average như FedAvg

### File liên quan

- `fed_learning/clients/refed_client.py`
- `fed_learning/servers/refed_server.py`
- `fed_learning/training/task_loop.py`

---

## CGoFed

File chính: `fed_learning/strategies/fed_incremental/cgofed.py`

### Mục tiêu

CGoFed giảm quên bằng cách xây không gian biểu diễn của task cũ bằng activation + SVD, rồi chiếu bớt gradient theo không gian đó khi học task mới. Nó còn dùng similarity giữa task hiện tại và task cũ để regularize.

### Class chính

- `CGoFedTrainer`
- `CGoFedAggregator`

### Flow trainer

- `__init__()`
  - khởi tạo hyperparameter projection, SVD storage, cache, history, forgetting stats

- `set_task(task_id, new_classes)`
  - chạy đầu task
  - reset cache projection matrix
  - cập nhật `mu_coefficient` theo relaxation schedule

- `update_forgetting(task_accuracies)`
  - chạy sau evaluate
  - tính AF, có thể reset relaxation nếu quên vượt ngưỡng

- `get_current_af()`
  - trả về AF hiện tại

- `_get_weight_modules(model)`
  - chọn các layer để thu activation

- `_get_projection_target_modules(model)`
  - chọn các layer được phép áp gradient projection

- `_collect_activations(model, data, device)`
  - thu activation qua forward hooks

- `build_space_from_client_data(model, client_data, config, device)`
  - chạy cuối task từ `post_task.py`
  - gom dữ liệu client và gọi `build_representation_space()`

- `build_representation_space(model, X, device, task_id, save_prefix)`
  - tính activation matrix
  - chạy SVD
  - chọn basis và importance weights
  - lưu basis của task cũ

- `compute_loss(model, output, target, global_params, historical_models, similarity_weights, **kwargs)`
  - chạy mỗi batch
  - tính loss cơ sở
  - có thể cộng thêm regularization theo historical models tương đồng

- `pre_step(model, global_params, **kwargs)`
  - chạy sau `backward()`, trước `optimizer.step()`
  - chiếu bớt gradient theo không gian biểu diễn cũ

- `_cache_projection_matrices(device)`
  - được `pre_step()` gọi
  - nạp basis cũ và dựng projection matrices

- `get_optimizer_class()`
  - trả optimizer dùng cho CGoFed

- `get_projection_stats(reset)`
  - lấy thống kê projection

- `log_projection_stats(reset)`
  - in thống kê projection ra console

### Flow aggregator

- `__init__()`
  - tạo kho lưu representation, historical models, similarity info

- `set_task(task_id)`
  - chạy đầu task
  - reset round counter và similarity info hiện tại

- `aggregate(results, global_params, **kwargs)`
  - chạy mỗi round
  - average model như FedAvg
  - lưu representation từ client
  - cập nhật lịch sử model và similarity info

- `_store_client_representations(results)`
  - lưu representation client trả về

- `_compute_similarity(R1, R2)`
  - tính similarity giữa hai task

- `_select_top_k_similar()`
  - chọn các task lịch sử gần nhất

- `get_local_regularization_info()`
  - server gọi sau aggregate để cấp historical models và similarity weights cho round sau

### File liên quan

- `fed_learning/clients/cgofed_client.py`
- `fed_learning/training/cgofed_worker.py`
- `fed_learning/servers/cgofed_server.py`
- `fed_learning/training/post_task.py`
