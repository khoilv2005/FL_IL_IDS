# CANDLE V4 — Native DRAFT/shadow integration

Đã triển khai và chạy **20 native rounds**, Task 2, receiver 30 ← donor 16,
class 13. Engineering PASS: shadow bật/tắt giữ nguyên native state; dừng R8 và
resume đến R19 khớp hoàn toàn với chạy liền. Chưa mở CAL, TEST, ACTIVE hoặc full training.

Kết quả tái hiện lợi ích giảm false activation, **chưa chứng minh tăng accuracy
so với donor-only**. Đối chứng học cuối task cho cùng expert weights và Adam states.

Receipt: [artifacts/candle_v4_native_shadow_20261023.json](../artifacts/candle_v4_native_shadow_20261023.json).

## Phạm vi đã chạy

| Hạng mục | Thiết lập |
| --- | --- |
| Native source | CANDLE V1 terminal T1, giữ hai model hiện có của client 16 và 30 |
| Graph | Tính lại trong từng round, `paper`, ξ=0,8; không replay graph T2 cũ |
| Native learning | Optimizer thật, aggregation thật, BN/router refresh thật; 20 rounds T2 |
| Native data | Current T2 BASE, dedup/cap 4.096/class trước native local-validation split 10% |
| Expert data | Chỉ `client.X_train/y_train` sau split đó; không dùng FIT để train |
| Expert | CPU FP32 MLP 39→64→1; Adam 0,001, clip 5; seed 42 |
| Scheduling | Quota 44/round; max 88 khi có cạnh; carry khi thiếu peer |
| Primary | Joint class-balanced; donor-only và joint unbalanced là controls |
| Work mỗi nhánh | 880 Adam steps, 225.280 lượt mẫu: 112.640 positive + 112.640 negative |
| Boundary cuối | Sau hook R19, **trước task finalization**, chưa kiểm tra task transition |
| Native compute | Một RTX 4060 Laptop GPU; giữ AMP của V1 |

Đây là nhánh engineering/development hai client từ checkpoint, không phải fresh
100-client run. Membership của fixture được khóa trước khi chạy; donor đã được
chọn trong development trước đó, không phải kết quả blind online discovery.
Trong graph mới, cạnh 30←16 có alpha dương ở cả 20 rounds; không ép cạnh để đủ 880 steps.

BASE thực sự sau native local-validation split:

| Owner | Class → số mẫu |
| --- | --- |
| 16 | 13: 3.687; 14: 3.687; 15: 414; 16: 375; 17: 1.872 |
| 30 | 12: 3.687; 14: 3.687; 15: 760; 16: 10; 17: 23 |

Do sử dụng post-split BASE, endpoint không phải bản sao số học của standalone
pilot cũ. Các controls trong run mới dùng cùng initialization và ngân sách.
Joint balanced/unbalanced dùng cùng sampling; weighting negative loss là khác biệt.

Preparation giải nén shard NPZ gốc để lọc current BASE/FIT. Runtime chỉ đọc staged
T2; không có historical raw replay hoặc CAL/TEST example access. Các API đọc raw
roles không giới hạn và native evaluation loader bị chặn trong harness.
Input authority dựa trên dataset metadata/role lock; chưa có fingerprint riêng
của scaler artifact để đưa ra claim deployment rộng hơn.

## Implementation

- `candle_v4_runtime/fed_learning/training/candle_v4_shadow.py`: receiver-owned
  draft bank, shared-reference gradients, sum một lần, global clipping, Adam,
  quota, reference IDs, wire ledger và FIT shadow diagnostics.
- Hook trong private native runner sau aggregation/router refresh, trước metrics
  và round checkpoint. Default tắt; không gắn expert vào native `nn.Module`.
- Native weights/masks/ages/BN/router/availability không bị expert thay đổi.
- Init dùng fork CPU RNG; sampling dùng generators riêng theo step. Không reseed
  native CPU/CUDA RNG. Giữ nguyên training flags của từng native submodule khi đánh giá.
- Atomic continuation lưu cả native và draft/Adam/quota/last-round/bytes. Không
  lưu stale donor gradients để dùng khi peer vắng hoặc task đã đổi.
- Current FIT chỉ tạo diagnostic; threshold và FIT metrics không điều khiển training,
  scheduler, early stopping hoặc activation.

Original clone manifest vẫn mô tả baseline V1. Sau extension, 365/367 file gốc
còn khớp SHA; hai file private được sửa là runner và clean-config validator;
thêm một module shadow. **Root V1 và V3 không bị sửa.**

## Engineering receipts

| Đối chiếu | Kết quả |
| --- | --- |
| Native tensors + algorithm/router state + RNG: shadow on/off | Exact |
| Native state: uninterrupted vs stop R8/resume R9–R19 | Exact |
| Graph: on/off và resume | Exact |
| Expert weights, Adam states, quota, history, steps, bytes sau resume | Exact |
| Native parameters hữu hạn ở checkpoints cuối các nhánh | PASS |
| Không có peer: 0 steps/0 bytes, quota carry | PASS, controlled probe |
| Peer đến muộn: 88 fresh updates, không dùng gradient cũ | PASS, controlled probe |
| Qua task khác: expire pending, từ chối train lại generation cũ | PASS, controlled probe |

Các probe no-peer/late-peer/expiry dùng fixture riêng; không được mô tả thành
cạnh thực sự vắng mặt hoặc task transition tự nhiên của native run này.
Đây không phải receipt unit tests tổng quát.

AMP có skipped steps ở đầu run do overflow; native optimizer vẫn thực hiện bước
cập nhật. Không có nonfinite native parameter ở endpoint, và các nhánh đối chiếu
khớp tuyệt đối. Checkpoint Kaggle có RNG nhiều GPU được ánh xạ về stream CUDA0
trong fixture local; không claim tái lập số học run Kaggle gốc.

Hai lỗi harness đã được sửa trước receipt cuối: restore RNG nhiều GPU trên máy
một GPU; thiếu history key khi đi vào task finalization. Pilot cuối khóa dừng ở
round boundary R19 theo phạm vi thí nghiệm. Không tune learning rate/threshold
hoặc chọn endpoint bằng những lần chạy đó.

## Endpoint FIT, round 19 đã khóa

Donor FIT: 5.242 positives + 750 negatives. Receiver FIT: 4.237 negatives,
trong đó class 12 có 516 mẫu. Base là **receiver model đang thay đổi thật** qua native rounds.

| Nhánh / threshold | Donor target recall | Donor FP | Receiver FP | Class 12 FP | Receiver break |
| --- | ---: | ---: | ---: | ---: | ---: |
| Donor-only / 0,5 | 5.231/5.242 = 99,79% | 1 | 508 | 506 | 1 |
| Donor-only / FIT-recall95 | 4.981/5.242 = 95,02% | 0 | 2 | 2 | 1 |
| **Joint balanced / 0,5** | **5.228/5.242 = 99,73%** | **1** | **0** | **0** | **0** |
| Joint balanced / FIT-recall95 | 4.980/5.242 = 95,00% | 0 | 0 | 0 | 0 |
| Joint unbalanced / 0,5 | 5.229/5.242 = 99,75% | 1 | 1 | 0 | 0 |
| Joint unbalanced / FIT-recall95 | 4.980/5.242 = 95,00% | 0 | 0 | 0 | 0 |

**Fixed threshold 0,5 chưa đạt đầy đủ per-owner/class risk gate:** joint balanced
còn 1 FP trên donor class 15 (1/45 = 2,22%; donor negatives 1/750 = 0,133%).
Không được lấy receiver 0 FP để gọi expert đã an toàn. FIT-recall95 là calibration
control trên FIT đã quan sát, không phải CAL acceptance độc lập.

Pooled current FIT bỏ 43 cross-owner duplicates, còn 10.186 unique rows:

| Predictor | Đúng / unique rows | Accuracy | Rescue / break |
| --- | ---: | ---: | ---: |
| Native receiver baseline | 3.801/10.186 | 37,32% | — |
| Donor-only / 0,5 | 9.031/10.186 | 88,66% | 5.231 / 1 |
| Joint balanced / 0,5 | 9.029/10.186 | 88,64% | 5.228 / 0 |
| Joint unbalanced / 0,5 | 9.030/10.186 | 88,65% | 5.229 / 0 |
| Ba nhánh / FIT-recall95 | 8.781/10.186 mỗi nhánh | 86,21% | donor 4.981/1; joint 4.980/0 |

Như vậy joint giảm FP/break, nhưng **không tăng endpoint accuracy so donor-only**.
Phần lớn FP donor-only thay một dự đoán vốn đã sai bằng một dự đoán sai khác.
Các accuracy trên chỉ thuộc panel current T2 FIT; không phải full-system TEST.
Current evidence không chứng minh retention T0/T1 hoặc safety khi tích lũy nhiều experts.

## Đối chứng học cuối task

Chạy lại cùng current BASE, initialization, sample order và 880 Adam steps,
dồn toàn bộ expert training vào cuối task. Cả ba nhánh có **weights và Adam state
khớp tuyệt đối** với round-wise endpoint.

Không có bằng chứng round-wise schedule tự nó làm tăng final accuracy trong ca
này. Learning curve sớm vẫn là DRAFT diagnostics; chưa có ACTIVE deployment để
claim online utility. Đối chứng cuối task dùng cùng expert work và payload training;
nó không cần lặp current-base FIT verification 20 lần, nên không bị ép tiêu traffic
thừa cho bằng tổng bytes của round-wise diagnostics.

## Communication

Đo độ dài packet serialization trong mô phỏng cùng process, không phải network
benchmark. Native ledger mô hình hóa all-to-all Context Capsules và dense neighbor
weights + masks + ages + adapter metadata, không có transport headers/compression.

| Thành phần, 20 rounds | Bytes |
| --- | ---: |
| Native capsules | 83.112.000 |
| Native parameter/state messages | 164.583.400 |
| **Native modeled total** | **247.695.400** |
| Current base verification reference, 20 versions | 134.638.220 |
| Expert reference + gradient, hai joint trials | 43.085.104 |
| FIT expert references, ba trials | 714.300 |
| FIT metric receipts | 44.346 |
| **Shadow diagnostics, ba trials cùng chạy** | **178.481.970** |
| **Combined modeled total** | **426.177.370 ≈ 406,43 MiB** |

Current-base verification được chia sẻ giữa ba đối chứng. Đây không phải cost
của một production primary riêng lẻ, và không dùng để khẳng định tiết kiệm bandwidth.
Nó giải thích vì sao 23,44 MiB của pilot frozen-base cũ không dự báo đúng native cost.
Không coi model đã có trong RAM là truyền miễn phí.

## Reproduce

Local small development pilot, không phải notebook full training:

```powershell
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode prepare
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode shadow
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode baseline
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode split
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode resume
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode timing-control
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode no-peer
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode endpoint
python tools/run_candle_v4_native_shadow.py --out audit_denice/new_v4_native --mode summarize
python tools/summarize_candle_v4_native_shadow.py --out audit_denice/new_v4_native
```

Các paths dataset/roles/source archive trong harness trỏ đến các artifact local
đã audit. Không upload raw samples, model hoặc optimizer checkpoints lên Git.

Mốc này hoàn thành bounded native shadow integration. ACTIVE acceptance, historical
retention, multi-expert bank và full Kaggle training vẫn chưa được xác nhận.
