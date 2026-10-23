# CANDLE V3 Direct FC2 — Smoke receipt

Ngày: 2026-10-23. Engineering smoke: **PASS**.
Hiệu quả transfer trên cặp thật: **chưa được chứng minh**.

| Phần | Kết quả |
| --- | --- |
| Structural/control tests | 27/27 PASS; 0 failure/error/skipped |
| Native runner checks | 80/80 PASS |
| Native training | BASE thật, clients 9/30, T0 → T1, 1 round/task, cap 64 BASE/client/task |
| Optimizer | 6 successful steps trong run liên tục, FP32 |
| Save/resume | Trùng toàn bộ tensors và algorithm state; max parameter difference = 0 |
| Controlled installation | 1 commit, forward CNN-GRU thật trên fixture tổng hợp |
| Natural installation | 0; 2 SKIP và 1 INCOMPATIBLE |
| TEST | Không truy cập; evaluator bị chặn trong smoke |

## Đã chạy những gì

### Structural và controlled service

`tools/smoke_candle_v3_direct_fc2.py` import runtime V3 riêng và kiểm tra:

- Encoder eval features khớp masked inference; không cho dùng training-mode BN/dropout.
- Packet codec/checksum, 1.028 byte FP32, 32 byte packed weight mask.
- Reject donor chưa train local hoặc row chưa mature.
- Copy chính xác row/bias/masks, không trộn row receiver random.
- Reject unprotected FC1 dependencies và mature receiver row.
- Multi-donor chỉ tổng hợp cùng feature frame, normalize đúng, giữ coordinate không contributor.
- Shadow apply không mutate live model hoặc tensors/rows cũ.
- FAIL receipt và stale reference không được commit; lỗi sau khi ghi model phải rollback.
- Save/restore model, router, masks, ranks, registry; prediction giữ nguyên.
- Imported row giữ nguyên qua optimizer step và native age-masked aggregation.
- Current task/role guard; ledger budget/restore; acceptance, FAR, break và accounting.
- Chạy toàn service với actual CNN-GRU forward, acceptance rồi commit; chạy lại task
  không cài trùng hoặc cộng lại bytes.

Controlled fixture tạo một trường hợp phân biệt được bằng row tuyến tính từ BASE
tổng hợp. Dữ liệu FIT/CAL tổng hợp riêng. Không mock inference, acceptance hoặc commit;
model weights/router và ownership receipts được thiết lập để buộc exercise commit.
Đây là bằng chứng integration correctness, **không phải kết quả accuracy nghiên cứu**.

### Native runner trên BASE thật

`tools/smoke_candle_v3_native.py` dùng original locked roles:

`role_manifest_sha256 = b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.

Thực hiện ba lần chạy:

1. Liên tục T0 → T1.
2. T0 riêng, lưu sealed task ZIP.
3. Resume ZIP đó vào T1, đối chiếu terminal state với lần chạy liên tục.

T0 có cả clients 9/30; client 9 không có Task 1 BASE nên chỉ client 30 train T1.
Đây là hành vi Non-IID hợp lệ, không ép client 9 train lại task cũ.

Chỉ participant list được giới hạn thành induced subgraph {9,30}. Không mock model,
gradients, native aggregation, router, discovery, head transfer hoặc acceptance.
Các wrapper dữ liệu chỉ audit task/class access và chặn evaluator/TEST.

Native chạy trên RTX 4060 Laptop GPU, Torch 2.10.0+cu128; structural chạy CPU.
Lần cuối structural khoảng 5,60 giây, ba native phases tổng khoảng 17,98 giây,
không tính startup/import và tổng thời gian viết/debug smoke harness.

## Candidate thật đã bị reject vì sao?

`receiver 30 ← donor 9 / class 3`, Task 0:

- Có cạnh native neighborhood hợp lệ và donor có current local training evidence.
- Availability-only FIT: 17/34 mẫu đúng.
- Row donor có 40 active nonzero FC1 connections.
- **8/40 connections trỏ tới FC1 coordinates receiver chưa mature.**
- Kết quả: `INCOMPATIBLE`, không clip row, không đổi masks để cưỡng ép commit,
  không fallback sang variant C và không hạ gate.

CAL acceptance của candidate này chưa chạy vì dependency precheck đã reject trước.
Kết quả không chứng minh row sẽ thất bại về recall nếu bỏ protection; cũng không
cho phép kết luận row an toàn khi bỏ protection. Đây là một giới hạn tương thích
quan sát được trên cặp cụ thể trong smoke, không phải kết luận cho toàn graph.

## Communication thực đo

| Hạng mục | Bytes |
| --- | ---: |
| Row FP32 raw | 1.028 |
| Packet row đã serialize, gồm metadata/masks | 1.689 |
| Receiver reference trong candidate native | 6.314.442 |

Reference phục vụ FIT/CAL và receiver-space sketches, không phục vụ donor train
head ở variant A. Candidate này bị reject trước CAL nên chưa trả toàn bộ chi phí
một phiên install thành công. Native DeNICE wire chưa được ledger đo; chưa thể
tuyên bố tổng bandwidth tiết kiệm hoặc toàn phiên chỉ khoảng 1 KB.

## Thay đổi để chạy smoke

Thêm flag `candle_v3_integration_smoke` vào validator **private V3**. Flag chỉ cho
bounded direct-FC2 smoke với ≤2 clients/tasks/rounds, ≤256 BASE/client/task, FP32,
validation role và tắt evaluator. Main production profile vẫn giữ budget, AMP và
data-role constraints V1; không đổi gates recall/FAR/break.

Manifest bind SHA-256 của cả hai execution receipts với hash source đã chạy:

`283d8e9a5725435bc27eee28d893c1a6d0a951cc4bc599f36a5070652554b870`.

Status là `ENGINEERING_SMOKE_PASS_QUALITY_UNPROVEN`, chỉ áp dụng variant A.
Variant C chưa được smoke. Chưa chạy full training, chưa cập nhật notebook full
đang pin baseline clone, chưa có benchmark Accuracy/Macro-F1 mới.

## Reproduce

Tại repo root, dùng Python có Torch, NumPy và dependencies V1:

```powershell
python tools/smoke_candle_v3_direct_fc2.py --out audit_denice/new_direct_smoke
python tools/smoke_candle_v3_native.py --data-dir <original-data-dir> --roles <locked-role-dir> --out audit_denice/new_native_smoke --clients 9 30
```

Output directory phải mới. Native cần original shards của clients đã chọn,
metadata và locked role indices/checksums tương ứng. Không cần global TEST file.

## Receipts

- Summary: `artifacts/candle_v3_direct_fc2_smoke_20261023.json`.
- Tests và controlled service: `artifacts/candle_v3_direct_fc2_structural_smoke_20261023.json`.
- Native checks/resume: `artifacts/candle_v3_direct_fc2_native_smoke_20261023.json`.

Smoke đã xác nhận đường commit trong controlled case và đường reject trên cặp
thật. Còn cần một ca install tự nhiên trên dữ liệu thật, với đủ current evidence
và dependency tương thích, để chứng minh functional transfer thực tế.
