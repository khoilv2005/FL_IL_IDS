# DeNICE V2 — Native P1 một round

## Trạng thái

**Gate native P1 trong phạm vi smoke: PASS.** Đã chạy local trên CPU với dữ
liệu BASE thật của Task 0, hai client thuộc cùng cluster baseline. Regression
P0/P1 và 15 tests native mới: **89/89 PASS**, 0 failure/error/skipped.

- [Receipt smoke dữ liệu thật](../artifacts/denice_v2_native_p1_smoke_20261011.json).
- [Receipt tests, environment và source hashes](../artifacts/denice_v2_native_p1_test_receipt_20261011.json).

Đây là một round bounded, không phải full runner 6 task. Chưa fit cluster
router, chưa đánh giá TEST, chưa có accuracy benchmark V2. Không triển khai
P2 hoặc thay clustering trong mốc này.

## 1. Đã nối những gì?

| Thành phần | Implementation |
|---|---|
| Canonical allocation | `native.py:allocate_cluster()` gọi allocation của DeNICE một lần theo union current BASE classes; mở FC2 learner rows tại cluster |
| Native worker | Dùng `DeNICEClient/DeNICETrainer`, `forward_output`, CE, clipping và optimizer callbacks thật |
| Shared layout | `cluster_layout_locked=True` bỏ local selection/pruning/regrowth; V1 mặc định vẫn chạy các bước này |
| Protection | Gradient permissions và projection sau optimizer; chặn drift của blocked coordinates kể cả weight decay/momentum |
| Deltas | Export state worker, kiểm tra topology, successful class counts, rồi dùng P1 engine |
| BN collector | Hook input BN trên post-aggregate canonical model ở eval mode; moments FP64, current BASE, binding plan/model hashes |
| Broadcast | Restore model từ từng state clone; kiểm tra hash và logits giống hệt trên cùng current BASE witness |

Native integration hiện hỗ trợ **plain legacy DeNICE**; từ chối adapters,
continual head hoặc các auxiliary registries chưa có contract thay vì bỏ state
âm thầm. Optimizer reset ở phase mới như native loop. Smoke dùng một phase,
một epoch, natural shuffle, FP32, Adam `lr=0.001`, không regularization bổ sung.
SGD có weight decay/momentum chỉ dùng trong controlled protection tests.

Workers không tự đổi ages/masks/topology; có kiểm tra các attributes của model
thật sau training, không chỉ giữ một bản metadata cũ trong packet. Không đủ
trainable capacity hoặc optimizer không tạo contribution thực → reject.

## 2. Smoke bằng BASE thật

Nguồn roles: `results (11).zip`, locked trước backbone training. Nguồn train:
`archive (1).zip`. Kiểm tra metadata, source shard và role index checksums.
Chỉ giải nén metadata, hai train shards và role indices; **không giải nén TEST**.
Các train shard nguồn chứa nhiều task; sau khi đọc chỉ chọn BASE rows thuộc
Task 0 để training/BN/witness. Không dùng raw CAL hay dữ liệu task khác.

Chọn cluster 2 từ dry-run complete-linkage/Jaccard ξ=0,8 đã khóa: members
**9, 30**. Không ép partition mới, không chọn cặp theo accuracy. Model khởi tạo
mới seed 42, không dùng checkpoint V1 divergent hoặc weights T5.

| Client | Current BASE support | Rows smoke | Optimizer steps |
|---|---|---:|---:|
| 9 | 1, 2, 3, 4, 5 | 160 | 10 |
| 30 | 1, 2, 4, 5 | 128 | 8 |

Cap 32 BASE rows/class/client, theo thứ tự indices đã khóa. Đây là subset cho
engineering smoke, không phải full Non-IID training budget. P1 weighting dùng
đúng counts của subset được cấp quyền; không giả số mẫu full shard đã train.

Shared allocation: conv1=10, conv2=19, conv3=38, GRU=15, FC1=38, FC2=5 learner
neurons. Class 0 vắng ở cả cluster, không được mở learner row. Class 3 chỉ có
ở client 9.

| Gate | Kết quả |
|---|---|
| Common reference và topology | Cả hai worker khớp cùng hashes |
| Optimizer thực sự thay đổi weights | Cả hai thay đổi CNN, GRU, FC1 và FC2 permitted coordinates |
| FC2 class có một contributor | Row class 3 thay đổi; bằng chính xác reference + delta của client 9; không bị pha loãng bởi client 30 |
| Declared routing dependency | `gru.weight_hh_l1` giữ byte exact |
| BN collection | bn1 → bn2 → bn3, đúng input/output hash chain |
| Reference transaction | State ban đầu không bị mutate |
| Broadcast | State hash và raw classifier logits giống hệt giữa members |

Tổng 9/9 smoke checks PASS. Thời gian tool sau imports: khoảng **12,84 giây**
trên CPU, hai threads. Không suy thời gian full campaign từ con số này.

## 3. Phân biệt protection và router consistency

Fresh Task 0 không có mature historical neurons hoặc activation memory. Vì vậy
smoke này dùng một declared GRU dependency làm protection witness; **không coi
nó là dependency closure đầy đủ của router**. Router memory, stable-feature
mask và registrations còn ở trạng thái bootstrap rỗng.

Mature protection được kiểm tra riêng trên native controlled fixtures: mature
Conv/FC1/FC2 rows và protected BN channels không thay đổi, kể cả SGD với weight
decay/momentum. Các tests không chứa dữ liệu task cũ.

Logits identical chứng minh model broadcast đồng nhất; chưa chứng minh native
routed inference đúng, retained historical sketches không drift, hoặc khả năng
continual learning. Các điều đó thuộc P2/task transition.

## 4. Test coverage

44 tests P0 + 30 pure P1 + 15 native P1 = **89 tests**. Bộ native bổ sung:

- Allocation union, cùng initial hashes, optimizer steps và changed coordinates.
- Single-contributor FC2 và negative-only contributor exclusion.
- Mature/routing protection với decay/momentum; reference không bị mutate.
- BN protected channels, sequential hash chain; đối chiếu hook moments với
  phép Conv1D trực tiếp trên cùng inputs.
- Broadcast hashes/logits, singleton minibatch tail không làm rơi sample.
- Historical/wrong-role BASE, content mutation, shape mismatch bị reject.
- No-capacity bị reject.
- Locked loop không gọi local selection/prune/grow; V1 default vẫn gọi đủ.

```powershell
.\.venv-audit\Scripts\python.exe -m unittest discover -s tests -p 'test_denice_v2_p*.py' -v
```

Chạy lại smoke trên local:

```powershell
.\.venv-audit\Scripts\python.exe tools/run_denice_v2_native_smoke.py `
  --results-zip 'C:\Users\khoak\Downloads\results (11).zip' `
  --dataset-zip 'C:\Users\khoak\Downloads\archive (1).zip' `
  --cache audit_denice/denice_v2_native_p1 `
  --out audit_denice/denice_v2_native_p1/new_smoke_receipt.json
```

Output receipt phải là path mới. Không overwrite kết quả cũ để đổi cấu hình
âm thầm. Các cache input được kiểm tra lại bằng locked checksums.

## 5. Coverage và bước tiếp theo

Đã tính lại từ dry-run: sum class unions=76, mean=76/23=3,3043/6. Nếu mỗi
cluster chỉ được predict class trong union đã học, macro-F1 trên đủ 6 class
có upper bound trung bình **76/(23×6)=55,0725%**. Đây là coverage ceiling có
điều kiện, không phải measured F1, accuracy hay một oracle inference result.

Baseline clustering ξ=0,8 giữ nguyên. Mốc này chưa giải quyết 16 singleton
và chỉ 11/90 gaps có peer support. Chính sách complementarity phải là variant
development riêng, dùng BASE metadata và state compatibility, trước benchmark.

Gate native P1 đã đủ để bắt đầu **P2 shared cluster router**: dependency closure,
current memory theo shared encoder, activation drift, không historical raw replay.
Full multi-task runner, consolidation, live registry commit, training resume,
communication ledger, quota ZIP và full-test evaluator vẫn cần integration theo
kế hoạch. Chưa tạo notebook full hoặc chạy full training.
