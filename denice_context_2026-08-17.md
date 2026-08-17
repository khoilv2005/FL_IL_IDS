# DeNICE / CANDLE handoff context — 2026-08-17

## 1. Mục tiêu hiện tại

Repository này nghiên cứu intrusion detection theo **federated incremental
learning (FL-IL)** trên CIC-IoT-2023, dùng biến thể decentralized DeNICE/CANDLE.
Mục tiêu trước mắt không còn là smoke/debug ngắn: đã chốt một cấu hình DeNICE
đã qua audit, chạy full seed 42 hợp lệ và cần bảo toàn protocol để các kết quả
sau có thể so sánh được.

Dataset người dùng đã đặt tại:

```text
E:\School\UIT\FL_IL_IDS\Dataset\2023\federated_splits\100-clients
```

Trên Kaggle, notebook dùng:

```text
/kaggle/input/datasets/khoilv2005/100-clients/100-clients
```

Split có 100 clients, 34 classes và 6 incremental tasks. P6 đánh giá
class-balanced 100 samples/class, có replacement, tức 3,400 samples.

## 2. Git và workspace

- Remote: `https://github.com/khoilv2005/FL_IL_IDS.git`, branch `main`.
- Commit HEAD lúc lập tài liệu: `cafa6db` — `docs(experiments): record valid full D2 seed 42`.
- Commit code full runner: `afc9e87` — `feat(experiments): freeze D2 full-run configuration`.
- Những file/directory local không thuộc commit và phải giữ nguyên:
  - `Dataset/`
  - `d1_local/`
  - `denice_full_context_2026-08-13.md`
- Checkpoint D4 được chọn đã version bằng Git LFS ở commit `8d285cc`.
- Artifact ZIP Kaggle chỉ nằm ở `C:\Users\khoak\Downloads\`; không commit
  toàn bộ ZIP/checkpoint output vào Git trừ checkpoint đã chọn qua LFS.

## 3. Algorithm hiện tại

### 3.1 DeNICE decentralized IL

Mỗi client học incremental task cục bộ. Sau local update, clients được
cluster/gắn peer graph theo context similarity; peer aggregation dùng weighted
mean với alpha dựa trên similarity/effective graph score. Các cơ chế chính vẫn
được giữ:

- CANC/age ranks cho plasticity/capacity;
- capsule/adapters và router multiclass;
- consensus age merge để tránh capacity collapse từ union/max propagation;
- context-edge collaboration, label-overlap và collaboration guards;
- NICE objective (không tự ý đổi sang learner-only CE; đó là ablation riêng);
- full checkpoint/resume state, không dùng raw checkpoint làm continuation.

Các sửa correctness/audit quan trọng trước D2 đã được đưa vào code và test,
gồm strict evaluation coverage, nhất quán effective similarity khi cluster và
aggregate, clone/restore novelty state cho late client, resume state đầy đủ,
và tách semantic oracle evaluation. Chi tiết/trace lịch sử nằm trong
`plans/denice_consolidated_audit_action_plan_2026-08-14.md`.

### 3.2 D2 — phương pháp đã được chọn

Vấn đề causal được tìm thấy: peer aggregation có negative transfer khi peer
không hề học một class nhưng vẫn thay `fc2[c]` của receiver.

Flag đã chốt:

```python
denice_aggregation_mode = "peer"
denice_selective_fc2_peer_rows = True
```

Ý nghĩa chính xác:

1. Chỉ áp dụng với **plastic** `fc2.weight[c]` và `fc2.bias[c]`.
2. Một peer chỉ được góp row `c` nếu local capsule training-label support của
   peer có class `c`; không suy support từ test set.
3. Alpha của `{self + peer hợp lệ}` được renormalize riêng theo từng row.
4. Nếu không còn peer hợp lệ cho row đó, receiver giữ self update.
5. Hidden/shared layers, adapters, routing và age aggregation vẫn peer
   aggregate như trước. Đây không phải self-only training.
6. Mỗi cluster history ghi `selective_fc2_row_protection`, gồm số/rate blocked
   rows, peer alpha bị block và audit theo client/row.

Implementation nằm chủ yếu ở:

```text
fed_learning/training/decentralized_denice_il.py
run_denice_d2_kaggle.py
tools/analyze_denice_d2.py
```

`tools/analyze_denice_d2.py` đã được sửa ở commit `a50b194`: khi mọi evaluated
class từng bị blocked ở ít nhất một client/round thì tập supported-only rỗng.
Khi đó safeguard supported-only là N/A/vacuously pass, không phải false reject.

## 4. Protocol đánh giá cố định

Không được đổi protocol hoặc denominator giữa variants/seeds:

- P6 protocol: `coverage_aware_local`;
- class-balanced global test sample, 100/class, replacement;
- strict coverage bắt buộc: `assigned == requested`, `unsupported == 0`;
- lưu fixed source-index SHA-256 và prediction trace;
- báo cáo E0/E1/E2/E3/E3b/E4/E5/E6 với semantics đã tách;
- **E3** = oracle adapter + oracle true-episode class mask, tức routed-system
  ceiling, không phải deploy metric;
- **E4** = predicted hard routing, là metric routed deploy phù hợp hơn;
- validator: `tools/validate_denice_run.py`.

Không diễn giải chênh lệch E3/E4 như lỗi classifier: chênh lệch đó chủ yếu là
headroom/loss ở router, adapter availability hoặc episode routing.

## 5. Kết quả lịch sử và quyết định

### 5.1 D3/D4/D5

- D3 (class-imbalance sampler) không được chọn làm thay đổi mặc định.
- D4 xác định schedule capacity reserve dùng cho run chốt: 10% ở tasks 0–4,
  sau đó 0% tại task 5 terminal.
- D5 là router/reference-memory ablation. Router vẫn là bottleneck; không có
  thay đổi D5 nào được đưa vào full config hiện tại.

### 5.2 D6 — bằng chứng causal mở D2

D6 giữ nguyên D4 schedule và chỉ thay aggregation mode (`peer` so với
`self_only`). Kết quả ở strict fixed support:

| Seed | Peer default E3 | Self-only E3 | Chênh E3 | Peer default E4 | Self-only E4 | Chênh E4 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 31.932% | 39.609% | +7.677 pp | 24.500% | 31.037% | +6.537 pp |
| 43 | 31.336% | 38.417% | +7.082 pp | 24.089% | 29.672% | +5.583 pp |

Kết luận: bottleneck là peer parameter/age/adapter aggregation, không chỉ là
router. Không chọn self-only làm method cuối; thay vào đó D2 cô lập đúng
classifier-row negative transfer.

### 5.3 D2 — xác nhận hai seeds, quyết định KEEP_D2

D2 dùng schedule D4 ở 20 clients × 5 rounds/task, so sánh duy nhất
`peer_default` và `peer_supported_fc2` trên cùng fixed support.

| Seed | Delta E3 | Delta E4 | Delta blocked-class recall | Delta old-class recall | Quyết định per-seed |
|---|---:|---:|---:|---:|---|
| 42 | +3.538 pp | +2.199 pp | +1.853 pp | +2.267 pp | candidate |
| 43 | +2.644 pp | +2.006 pp | +2.412 pp | +2.867 pp | candidate |

Tất cả gate D2 pass ở cả hai seed. Đây là bằng chứng direct/counterfactual hợp
lệ để **KEEP_D2**. Artifact liên quan:

- `results (6).zip`: D2 seed 42, strict 3,400/3,400, source hash
  `f0b269…4ca60`.
- `results (7).zip`: D2 seed 43, strict 3,400/3,400, source hash
  `72cf27…f260d`.

Không chạy thêm D2/D6 diagnostic trừ khi thay algorithm/protocol.

### 5.4 Full D2 seed 42 — đã hoàn tất hợp lệ

Artifact: `C:\Users\khoak\Downloads\results (8).zip` (khoảng 2.77 GB).

Runner: `run_denice_full_d2_kaggle.py`.

Frozen configuration:

```text
100 clients
6 tasks (0–5)
20 rounds/task = 120 round records
peer aggregation
selective_fc2_peer_rows = true
tasks 0–4: min_free_capacity_ratio = 0.10
task 5: min_free_capacity_ratio = 0.0
full checkpoints
terminal P6 strict: 100 samples/class = 3,400 samples
```

Validation:

- Base tasks 0–4: validator `valid=true`, 100 round records. Warning
  `no P6 evaluation summary found` là **expected** vì P6 chỉ chạy terminal.
- Terminal task 5: validator `valid=true`, 120 accumulated round records,
  final checkpoint `checkpoint_task_5_round_19.pt`.
- P6: strict `3,400 / 3,400`, `unsupported=0`, source hash
  `f0b269c0f9d588b751764661308a095cfd80f2185ab0404411bc03d37814ca60`.

P6 `coverage_aware_local` metrics:

| Policy | Accuracy | Macro-F1 | Route accuracy |
|---|---:|---:|---:|
| E0 backbone/no-mask | 8.324% | 6.995% | 0.000% |
| E1 predicted adapter/no-mask | 8.618% | 7.279% | 51.765% |
| E2 oracle adapter/no-mask | 8.618% | 7.280% | 51.765% |
| E3 oracle routed-system ceiling | 39.676% | **35.676%** | 51.765% |
| E3b oracle hard/no adapter | 39.118% | 35.080% | 51.765% |
| E4 predicted hard routing | 28.235% | **26.093%** | 51.765% |
| E5 top-k 2 | 19.412% | 16.939% | 51.765% |
| E5 top-k 3 | 15.382% | 13.294% | 51.765% |
| E6 adaptive | 28.118% | 26.189% | 51.765% |

E3 macro-F1 − E4 macro-F1 = **9.583 pp**. Đây là router/route-selection gap
còn lại, không phải một validator failure và không phủ định evidence D2.

Lưu ý so sánh: full seed 42 dùng budget khác (100 clients × 20 rounds), nên
không được quy chênh so với D2 20 × 5 hoàn toàn cho D2. Hiện **không có full
peer-default matched control**; theo trao đổi gần nhất, không cần chạy thêm
control này chỉ để so sánh. Causal evidence cho D2 đã nằm ở D2 paired runs.

## 6. Cách chạy hiện tại trên Kaggle

File `train.ipynb` đã clone `main` từ GitHub, kiểm tra GPU và load Git LFS.
Tại thời điểm handoff nó mặc định:

```python
RUN_MODE = 'FULL_D2'
SEED = 43
EVAL_DEVICE = 'cuda'
```

`FULL_D2` gọi runner riêng, không gọi `train_incremental_kaggle.py` trực tiếp,
để bảo toàn D2 flag và schedule terminal reserve. Runner idempotent theo
artifact: có thể resume base/terminal nếu continuation/checkpoint đã tồn tại,
nhưng không được lấy raw `checkpoint_task_4.pt` làm continuation. Output dự
kiến:

```text
/kaggle/working/denice_full_d2_seed_43/
  base_tasks_0_to_4/
  terminal_task_5/
    p6_evaluation/p6_evaluation_summary.json
  full_d2_manifest.json
```

Sau Kaggle, download **Results ZIP** rồi inspect:

1. `base_tasks_0_to_4/audit_validation.json` must be `valid=true`.
2. `terminal_task_5/audit_validation.json` must be `valid=true`.
3. `full_d2_manifest.json` must show correct budget/schedule/flag.
4. P6 must have strict 3,400/3,400 coverage and no unsupported sample.
5. Compare seed 43 only with seed 42 under exactly the same config/protocol.

## 7. Việc tiếp theo

1. Nếu cần reproducibility/mean ± std: chạy **FULL_D2 seed 43** bằng notebook
   hiện tại, không sửa config. Có thể chạy seed 44 sau khi seed 43 hợp lệ.
2. Không chạy lại D1–D6 hoặc full peer-default trừ khi có một hypothesis mới
   được preregistered; user đã nói không cần full peer-default control lúc này.
3. Sau tối thiểu seed 42/43, tổng hợp mean ± std E3/E4, route accuracy,
   coverage, per-class recall và artifacts/checkpoint hashes.
4. Router gap 9.583 pp là research issue còn lại. Nếu cải thiện router, phải
   mở một ablation/factor mới riêng (không trộn vào confirmation seed của
   current full config).

## 8. Kiểm tra code

- Sau `afc9e87`, `run_denice_full_d2_kaggle.py` qua `py_compile` và
  `train.ipynb` JSON valid.
- Suite đã từng cho `342 passed, 1 skipped, 1 failed`.
- Failure duy nhất: `tests/test_plexus.py::test_decentralized_plexus_il_writes_fed_il_output_contract`.
  Test assertion cũ không chấp nhận hai metric output có chủ ý
  `precision_weighted` và `recall_weighted`. Nó thuộc Plexus, không liên quan
  DeNICE/D2/full runner; không sửa trong nhánh D2 để tránh scope creep.

## 9. Files quan trọng

```text
train.ipynb
run_denice_full_d2_kaggle.py
run_denice_d2_kaggle.py
run_denice_p6_eval.py
train_incremental_kaggle.py
fed_learning/training/decentralized_denice_il.py
tools/validate_denice_run.py
tools/analyze_denice_d2.py
plans/denice_consolidated_audit_action_plan_2026-08-14.md
denice_algorithm_current_2026-08-13.md
```

Use tài liệu này cùng `denice_full_context_2026-08-13.md` và audit plan để
tiếp tục. Không suy diễn metrics legacy trước các correctness fixes là baseline
trực tiếp cho full D2.
