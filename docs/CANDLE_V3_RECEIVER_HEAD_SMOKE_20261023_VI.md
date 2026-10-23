# CANDLE V3 — smoke học head trên encoder receiver

**Full training HOLD:** xem [audit sâu](CANDLE_V3_RECEIVER_HEAD_DEBUG_20261023_VI.md).
Người dùng yêu cầu ổn định chất lượng trước khi chạy full; smoke kỹ thuật dưới đây
không còn được dùng như điều kiện đủ để đề nghị full training.

## Kết luận

Engineering smoke PASS; hiệu quả transfer chưa được chứng minh. Không chạy full training local.
V1 và notebook baseline được giữ nguyên. Variant C không fallback sang Direct FC2.

| Kiểm tra | Kết quả |
| --- | --- |
| Structural/control tests | 25/25 PASS, không failure/error/skipped |
| Native integration T0 → T1 | 83/83 PASS |
| Native optimizer steps | 6, dữ liệu BASE thật |
| Head session thực sự chạy | 1 session, 8 vòng aggregate primary + 8 vòng donor-only control |
| Resume sai lệch tham số lớn nhất | 0 |
| Cài có kiểm soát | 1, synthetic fixture |
| Cài tự nhiên | 0 |
| TEST/validation để acceptance | Không truy cập |

## Phạm vi smoke

`tools/smoke_candle_v3_receiver_head.py` kiểm tra gradient tổng hợp tương đương objective
pooled class-balanced; mini-batch partition không đổi kết quả ngoài dung sai số học;
không gradient/BN drift ở frozen encoder; không đổi mature head; từ chối reference stale,
duplicate contributors và mixed head rounds. Kiểm tra commit, rollback khi lỗi, persistence,
prediction restore, survival qua optimizer/native aggregation và role access hiện tại.

Ca commit có kiểm soát dùng encoder synthetic được dựng từ BASE để tách hai nhóm input.
FIT/CAL là các panel riêng. Inference CNN-GRU, 8 vòng head optimizer, router,
acceptance và transaction đều chạy thật. Không mock chúng. Kết quả 64 rescue / 0 break
trên 64 dương + 64 âm chỉ chứng minh đường code, không phải benchmark knowledge transfer.

`tools/smoke_candle_v3_native.py --variant receiver_head_learning` chạy client 9 và 30
trên locked real BASE, T0/T1, 1 round/task, tối đa 64 BASE/client/task. Chỉ giới hạn
participant list thành induced subgraph. Không mock optimizer, model, graph aggregation,
router hoặc acceptance. TEST readers bị chặn để phát hiện truy cập ngoài phạm vi.
Ba lần chạy: uninterrupted T0/T1; T0 riêng; restore ZIP T0 rồi T1. Model weights,
algorithm state, registry và prediction/route trước/sau restore trùng nhau.

## Candidate dữ liệu thật: 30 ← 9 / class 3

Availability-only FIT đúng 17/34. Đã encode dữ liệu donor bằng frozen receiver,
học head và chạy đủ 8 vòng primary cùng donor-only control. Trạng thái cuối là UNKNOWN:

- Current CAL chỉ có 3 mẫu dương, dưới gate 32; không được cấp phép commit.
- Shadow candidate đúng 1/3 dương; 2 false activation trên 233 âm, negative break=2.
- Recall, FAR và gain gates cũng không đạt. Không hạ gate hoặc đổi learning rate để cứu ca này.
- Đó là kết quả shadow, không phải prediction sau một update đã được cài.
- Hai receipt còn lại SKIP. Không có natural install.

Native teaching traffic Task 0: 6.702.087 bytes; gồm receiver reference 6.314.442,
head references 38.816, head deltas 27.344, router sketches 32.915, verification request
279.735, receipts 7.981 và announcements 854 bytes. Task 1 không có teaching traffic.
Native DeNICE traffic chưa đo trong ledger này; không tuyên bố tiết kiệm toàn hệ thống.

## Runtime và regression

Chỉ mở rộng bounded smoke validator để nhận cả hai transfer modes đã khai báo.
Giới hạn hai client/task/rounds, current roles, FP32 và unopened TEST được giữ nguyên.
Config full không bật smoke flag. Không thay optimizer, gate hay protocol production.
Direct FC2 được chạy lại cả structural và native smoke trên source mới trước khi lock.
Các receipt cũ được giữ như lịch sử, không sửa kết quả hoặc source hash của chúng.

## Notebook full

`train_candle_v3_receiver_head_kaggle.ipynb` bật `--peer-teaching`, dùng private runtime
được pin GitHub commit. Không nhúng source/checkpoint/dataset. One GPU + AMP, batch 2048,
seed 42, 6 task × 20 rounds, current BASE split V1. Chỉ TEST sau T5/R19 và terminal
consolidation. Có round checkpoints, ZIP mỗi task và output guard 12 GiB.
Đây là notebook fresh-run, không có CLI arbitrary-round resume.

### Giới hạn so sánh với Direct FC2

Cùng backbone, native aggregation/router, seed, local training, role split và TEST protocol.
Variant C gốc có max 2 donor/3 target và teaching graph announced-current-peers; Direct FC2
có 1 donor/1 target, chỉ trong native neighborhood. Vì vậy đây là so sánh hai cấu hình,
chưa phải ablation chỉ thay cơ chế transfer. Head C còn chạy donor-only control để báo cáo.
Phải ghi cả khác biệt này và toàn bộ traffic/compute khi báo cáo kết quả.

### Artifacts

- `artifacts/candle_v3_receiver_head_structural_smoke_20261023.json`
- `artifacts/candle_v3_receiver_head_native_smoke_20261023.json`
- `artifacts/candle_v3_direct_fc2_structural_after_encoder_20261023.json`
- `artifacts/candle_v3_direct_fc2_native_after_encoder_20261023.json`

Manifest bind SHA-256 của source và từng receipt; engineering PASS không chứng nhận
accuracy, historical safety hoặc bandwidth saving. Chưa chạy full training.
