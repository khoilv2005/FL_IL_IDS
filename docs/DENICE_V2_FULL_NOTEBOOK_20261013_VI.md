# DeNICE V2 — Full training bằng notebook Kaggle

## File chạy

`train_denice_v2_full_kaggle.ipynb` ở thư mục gốc repo.
Upload lên Kaggle, bật GPU + Internet và gắn dataset gốc chứa
`metadata.json`, `client_*_train.npz`, `global_test_data.npz`.
Notebook clone GitHub tại commit đã pin; không nhúng source/checkpoint.

Điền `DATA_DIR` nếu mount nhiều dataset. `ROLES_DIR` trỏ tới `denice_clean_roles`
chứa role manifest/lock và indices. Để trống: notebook tìm lock phù hợp hoặc
tạo split content-disjoint trước fresh training. Không dùng guarded-head roles
thay cho clean roles toàn federation.

## Main đã khóa

| Thành phần | Cấu hình |
|---|---|
| Training | Fresh seed 42, T0–T5, 20 rounds/task |
| BASE | Toàn bộ BASE hợp lệ, không cap samples/class |
| Local budget | Batch 2048, Adam LR .001, một phase/một epoch mỗi round |
| Numerics | FP32, không AMP, một GPU `cuda:0` |
| Clustering | Auto K, complete-linkage, ξ=.8; membership cố định trong task |
| Similarity | T0 Jaccard current BASE support; sau đó current BASE Conv2 means |
| Context budget | Tối đa 1024 current BASE rows/client, không giới hạn training BASE |
| Handoff | Split/dormant/rejoin giữ state; reject merge không tương thích |
| Coordinator | Số class thực sự train local, tie-break ID, không ưu tiên aggregation weights |
| Model/router | Một shared model + binary-cosine router mỗi cluster |
| Protection | `cnn_coordinate_router_v1`, Conv2 |
| Aggregation | P1 class-wise masked common-reference deltas + canonical BN |
| Evaluation | CHỈ Task 5 / Round 19; mỗi cluster chạy cùng FULL TEST 34 class |
| Headline | Mean Cluster Accuracy, Mean Cluster Macro-F1, trọng số đều |

Quality/capacity receipt là report-only. NaN, sai state/version/authority,
protected mutation và checkpoint checksum sai vẫn dừng. Không có learner
capacity thì ghi zero-step receipt; không bỏ protection. Binary memory toàn
zero được ghi nhận, dùng binary-cosine tie-break sẵn có, không dùng true task.
Không gọi APPLIANCE/CGoFed/CME; DeNICE V1 và các variants vẫn giữ trong repo.

## Checkpoint và resume

- Lưu mỗi global round hoàn tất; resume tại round boundary, không mid-epoch.
- Tensor chia khối 4 KiB, deduplicate bằng SHA-256, không quantize.
- Cuối task seal ZIP self-contained; verify trước khi bỏ các round ZIP parts.
- Index 0–19 là training snapshots; index 20 là continuation sau consolidation
  (và final evaluation ở Task 5), không phải round training bổ sung.
- Budget 18 GiB gồm cache và peak temporary trong output. Compression không
  bảo đảm dung lượng cố định với mọi K/state. Current BASE cache bỏ sau task;
  TEST chỉ được nạp khi hoàn tất Task 5 / Round 19, cache bỏ sau final evaluation.
- Persist model/router/masks/BN/protection, RNG, ledger, partition, dormant
  states, protocol/data lock và evaluation history. Không cần historical raw CAL.

Mount output cũ, giữ clean role lock/index, các ZIP và `execution_provenance.json`:

```python
RESUME_ROOT = '/kaggle/input/.../previous_output'
RESUME_TASK = 2
RESUME_ROUND = None
```

`None` chọn snapshot mới nhất, gồm continuation. Resume yêu cầu đúng source,
Torch runtime, device/config và data lock. Task đang dang dở import/verify prefix
checkpoints vào output mới; archive task cũ có thể giữ ở dataset mount.
Auto K lớn làm tăng số predictor chạy final TEST; có thể cần tiếp tục qua nhiều session Kaggle.

## Kết quả xuất

`metrics_task_5.json`: headline + từng cluster. Không đo average forgetting vì
không chạy intermediate evaluation; trường forgetting ghi N/A với lý do rõ ràng.
`evaluation_task_t_<cluster>.json`: class/router confusion, per-class/per-task,
cursor/checksum. `partition_task_t.json`: K, members, coordinators, dormant,
rejected merges. `round_task_t_*.json`: optimizer/aggregation/BN/router receipts.
`communication.json`: tổng modeled P2P application-wire bytes, gồm native model,
BN, router, metadata/handoff; chưa gồm TCP/TLS của distributed deployment.
`checkpoint_task_t_all_rounds.zip`, `execution_provenance.json`,
`completion.json` hoặc `failure.json`: checkpoint/provenance/trạng thái run.
Forgetting ghi N/A khi lineage nhập nhằng hoặc thiếu task evidence.

## Engineering receipts đã có trước yêu cầu dừng local

- 205/205 tests PASS, không skipped.
- Fixture 4 clients / 2 clusters / 2 tasks × 3 rounds: resume khớp state, RNG,
  ledger, partition, evaluation và communication.
- Fixture T0–T5, 1 round/task: archive/restore xuyên 6 task PASS.
- RTX 4060: clients 9/30, toàn bộ BASE T0=20.393 rows, T1=96.273 rows.
  Evaluation lượt nhỏ dùng VALIDATION, không đọc final TEST.

Receipts `artifacts/denice_v2_p3_*.json` ghi source hashes local đã thực thi;
Kaggle ghi provenance riêng của checkout pinned. Đây không phải full benchmark.
Theo yêu cầu mới, không chạy thêm thí nghiệm local. Full 6×20 trên Kaggle chưa
được khởi chạy tự động; dùng notebook nói trên.
