# DeNICE V2 — Full training bằng notebook Kaggle

## File chạy

`train_denice_v2_full_kaggle.ipynb` ở thư mục gốc repo.
Upload lên Kaggle, bật **T4 ×2 + Internet** và gắn dataset gốc chứa
`metadata.json`, `client_*_train.npz`, `global_test_data.npz`.
Notebook clone GitHub tại commit đã pin; không nhúng source/checkpoint.

`DATA_DIR` mặc định `/kaggle/input/datasets/khoilv2005/100-clients`; notebook tìm
đúng thư mục dữ liệu nếu Kaggle giải nén thêm một tầng.
Main dùng **100% TRAIN gốc của mỗi client**, chỉ lấy những rows thuộc current
task. Không chia CAL/FIT/validation, không dùng role indices, không deduplicate
hoặc subsample TRAIN. `ROLES_DIR` đã được bỏ khỏi notebook/CLI.
TEST có file riêng và chỉ được đánh giá ở Task 5 / Round 19.

## Main đã khóa

| Thành phần | Cấu hình |
|---|---|
| Training | Fresh seed 42, T0–T5, 20 rounds/task |
| TRAIN | 100% `client_<id>_train.npz`, lọc current task, không cap/holdout |
| Local budget | Batch **8192/client**, Adam LR .001, một phase/một epoch mỗi round |
| Local execution | Hai persistent spawn processes: `cuda:0`, `cuda:1` |
| Numerics | FP32, không AMP; aggregation/BN/router/final TEST chạy ở parent trên GPU 0 |
| Clustering | Auto K, complete-linkage, ξ=.8; membership cố định trong task |
| Similarity | T0 Jaccard current TRAIN support; sau đó current TRAIN Conv2 means |
| Context budget | Tối đa 1024 current TRAIN rows/client, không giới hạn training rows |
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

## Hai GPU

Scheduler phân công local jobs của tất cả cluster bằng longest-job-first dựa
trên số current TRAIN rows; mỗi GPU xử lý một client tại một thời điểm.
Singleton clusters cũng có thể chạy song song với những cluster khác.
Các workers trong cùng cluster cùng bắt đầu từ reference/masks/version,
giữ optimizer và RNG riêng. Chỉ gom contributions sau barrier local training,
sắp thứ tự client ID và dùng lại P1 aggregation/BN/protection/broadcast checks.
Đây là client-job parallelism, không phải DDP của một global model.

Workers đọc current TRAIN cache bằng mmap, nhận reference CPU có checksum;
không truyền toàn bộ TRAIN hoặc CUDA tensors qua multiprocessing IPC.
Log in `GPU=cuda:0` / `GPU=cuda:1`, step count và loss của mỗi client.
`timing_task_t_round_rrr.json` ghi prepare/local/checkpoint timings để kiểm tra
overhead còn lại; không mặc định hứa tốc độ toàn run tăng gấp đôi.

Theo yêu cầu, lượt mới chạy fresh. Output mặc định:
`/kaggle/working/results_denice_v2_conv2_xi_0p8_seed_42_2gpu_bs8192`.
Notebook sparse-clone code, bỏ các JSON kết quả cũ như `.tmp_denice2_json`.

## Checkpoint và resume

- Lưu mỗi global round hoàn tất; resume tại round boundary, không mid-epoch.
- Tensor chia khối 4 KiB, deduplicate bằng SHA-256, không quantize.
- Cuối task seal ZIP self-contained; verify trước khi bỏ các round ZIP parts.
- Index 0–19 là training snapshots; index 20 là continuation sau consolidation
  (và final evaluation ở Task 5), không phải round training bổ sung.
- Budget 18 GiB gồm cache và peak temporary trong output. Compression không
  bảo đảm dung lượng cố định với mọi K/state. Current TRAIN cache bỏ sau task;
  TEST chỉ được nạp khi hoàn tất Task 5 / Round 19, cache bỏ sau final evaluation.
- Persist model/router/masks/BN/protection, RNG, ledger, partition, dormant
  states, protocol/data lock và evaluation history. Không cần historical raw CAL.

Mount output cũ của cùng full-TRAIN protocol, giữ các ZIP và `execution_provenance.json`:

```python
RESUME_ROOT = '/kaggle/input/.../previous_output'
RESUME_TASK = 2
RESUME_ROUND = None
```

`None` chọn snapshot mới nhất, gồm continuation. Resume yêu cầu đúng source,
Torch runtime, device/config và original TRAIN checksums. Task đang dang dở import/verify prefix
checkpoints vào output mới; archive task cũ có thể giữ ở dataset mount.
Run BASE80 trước đây thuộc protocol khác; không resume nó thành run full-TRAIN.
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
`training_data_lock.json`: source checksums và protocol dùng 100% original TRAIN.
Log `Round complete ... checkpoint <absolute path>` xuất hiện sau khi checkpoint
được ghi và readback thành công. Round đang train chưa có checkpoint mới.
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
Các receipts BASE80 ở trên thuộc protocol trước; bản sửa full-TRAIN hiện tại
chưa chạy lại local theo yêu cầu, cần chạy notebook Kaggle.
Phần hai-GPU/batch8192 cũng chưa chạy thực nghiệm local; notebook có GPU
preflight và giữ các kiểm tra correctness khi chạy trên Kaggle.
Theo yêu cầu mới, không chạy thêm thí nghiệm local. Full 6×20 trên Kaggle chưa
được khởi chạy tự động; dùng notebook nói trên.
