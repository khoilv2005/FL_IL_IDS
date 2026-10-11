# DeNICE V2 — P1 Pure Class-Wise Masked Aggregation

## Trạng thái

**P1 pure engine: PASS — 30/30 tests.** Chạy cùng 44 regression tests P0:
**74/74 PASS**, 0 failure/error/skipped. Receipt và source hashes:
[denice_v2_p1_test_receipt_20261011.json](../artifacts/denice_v2_p1_test_receipt_20261011.json).

Đây là kiểm tra aggregation engine trên fixtures và native tensor state, chưa
phải local optimizer training hoặc native smoke. P2, notebook full và full
training chưa triển khai trong commit này. DeNICE V1 và code P0 giữ nguyên.

## 1. Các module mới

| File | Chức năng |
|---|---|
| `denice_v2/aggregation.py` | Round authority, worker clones, local deltas, class-wise contributor weighting, protected permissions, shadow result, acceptance/rollback và broadcast |
| `denice_v2/normalization.py` | BN canonical từ current-BASE moments, protected channels, sequential layer hashes |
| `tools/dry_run_denice_v2_partition.py` | Metadata-only Task0 dry-run từ locked role manifest/ZIP |
| `tests/test_denice_v2_p1.py` | 30 tests P1 và dry-run regression |

## 2. Common-reference contract

`prepare_round()` nhận canonical state, current class IDs, positive current BASE
counts của đúng members và BASE role hash. `start_worker()` trả về state clone
độc lập cùng reference/topology hashes. Target round = reference round + 1.

`finish_worker()` yêu cầu successful local-learning receipt đúng client/task/round,
current BASE, completed=true. Không suy class contribution từ availability hoặc
weights broadcast. Incomplete/stale/future/unsupported receipts bị từ chối.

Workers không được tự đổi architecture, ranks, topology, masks, adapter metadata
hoặc router. Mature, frozen và declared routing dependencies bị chặn. P1 dùng
**learner-only** permissions (rank=1); chưa tự allocate hoặc consolidate neurons.
GRU áp mask theo ba gate ở cả hai layer, hỗ trợ native recurrent mask keys.
Parameter families chưa có permission contract được giữ frozen.

Reference hash dùng cho aggregation loại coordinator identity, ownership/lineage
và RNG metadata ra khỏi weighting binding. Model, versions, algorithm topology,
router và protection vẫn được bind. Đổi leader không yêu cầu đổi contributions
và không có leader multiplier. Full ClusterState hash vẫn lưu coordinator.

## 3. Công thức đã triển khai

\[
\Delta_j=\theta_j-\theta_{ref},\qquad
\bar\Delta_p=\frac{\sum_j w_{j,p}M_{j,p}\Delta_{j,p}}{\sum_jw_{j,p}M_{j,p}},
\qquad \theta'_p=\theta_{ref,p}+\eta\bar\Delta_p.
\]

Denominator=0: giữ chính xác reference coordinate. Masks cũng giữ canonical;
không OR worker masks, không tăng tuổi một lần cho mỗi member.

- FC2 weight **và bias** row c: chỉ client thực sự train c có contribution;
  trọng số là số BASE samples của c đã được cấp quyền ở current task.
- Backbone: trọng số tổng BASE counts của các class có successful contribution
  tại client, chuẩn hóa riêng từng tọa độ có mask.
- Epoch repetitions trong receipt không nhân thêm BASE weighting.
- FC2 negative-only changes cho class không có positive training contribution
  không được upload vào row đó. Illegal mature/routing changes bị reject.
- Packet order được sort theo client ID để deterministic accumulation bằng FP64;
  kết quả được cast về dtype của reference. Eta mặc định 1, phạm vi [0,1].

Kết quả là shadow state, reference không bị mutate. Acceptance callback có thể
reject; reject/invalid packet/thiếu BN evidence giữ nguyên reference. Callback
không được tự sửa candidate. Sau PASS, broadcast clones có cùng state hash và
không chia sẻ tensor buffers. Live registry installation thuộc runner integration.

## 4. BN canonical

Không average local running means/variances và không chọn BN của leader.
Mature/protected/routing channels giữ reference. Mutable channels dùng current
BASE packets gồm counts, sums, squared sums và sample authority.

Provider nhận clone của **post-aggregation model**. Các layer xử lý theo thứ tự
bn1 → bn2 → bn3; packet bn2 phải bind model hash sau recalibration bn1, tương tự
bn3. Role/task/round/plan/model hash, unique contributor, dimensions, observation
counts và realizability của moments đều được kiểm tra.

Mutable running mean = tổng sums / tổng counts; variance dùng unbiased global
variance. Canonical `num_batches_tracked` giữ reference, không average local
counters hoặc giả lập số training batches từ moment packets.
Thiếu moments cho mutable BN → reject. BN hoàn toàn protected → giữ reference,
không gọi provider. Raw inputs không thuộc packet.

**Giới hạn:** native hook/collector để tạo moments trên dữ liệu BASE thực tế
chưa nối vào runner. Tests dùng moment fixtures. P1 bảo vệ dependency names
được khai báo, chưa chứng minh closure đầy đủ hoặc historical activation drift;
đó vẫn là gate P2.

## 5. Dry-run Task0 trên metadata thực tế

Nguồn: locked clean-role manifest trong `results (11).zip`, SHA-256
`b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.
Chỉ dùng BASE support của classes 0–5 và eligible Task0 population. Không đọc
raw train/CAL, TEST, không inference/training hoặc dùng counts của future classes.
Assume fresh shared initialization khi kiểm tra bootstrap partition; không dùng
model T5 cũ để giả lập một fresh V2 model.

| Metric | Kết quả |
|---|---:|
| Active / dormant clients | 50 / 50 |
| ξ | 0,8 |
| K tự động | 23 |
| Singleton clusters | 16 |
| Cluster có union đủ 6 class hiện tại | 1/23 |
| Data-support gap pairs | 90 |
| Gap có peer sở hữu class trong cùng cluster | 11/90 = 12,22% |
| Gap chưa có peer support trong cluster | 79/90 |

Receipt đầy đủ có members/class union từng cluster:
[Task0 partition dry-run](../artifacts/denice_v2_task0_partition_dry_run_20261011.json).

Kết quả xác nhận bootstrap Jaccard có giới hạn complementarity. **Data support
không chứng minh functional competence**, và 11 gap không phải 11 transfer thành
công. Không thay ξ/P0 hoặc chọn cấu hình bằng TEST. Đây là rủi ro cần giữ trong
báo cáo trước benchmark, không phải lỗi implementation của Non-IID allocation.

## 6. Tái lập và bước tiếp theo

```powershell
python -m unittest discover -s tests -p 'test_denice_v2_p[01].py' -v
python tools/dry_run_denice_v2_partition.py --source 'results (11).zip' --classes 0 1 2 3 4 5 --out task0_partition.json
```

Dry-run output immutable: dùng path mới khi chạy lại. Test môi trường CPU:
Python 3.14.7, NumPy 2.5.2, PyTorch 2.14.0+cpu.

P1 đã chứng minh math/permissions/BN packet protocol/rollback/broadcast bằng
unit tests; regression đổi leader giữ nguyên kết quả model. Có test aggregate
native `DeNICEModel` tensor state rồi `load_state_dict(strict=True)` thành công.
Chưa chứng minh gain accuracy hoặc forgetting qua training.

Bước kế tiếp: nối P1 với native workers, shared topology/allocation và BASE BN
moment collector; kiểm tra một round có optimizer thật. Chỉ chuyển P2 khi native
P1 tạo được shared model/state nhất quán. Chưa khởi động full training.
