# Audit nút thắt DeNICE gốc

## Phạm vi đã khóa

- Checkpoint `results (11)`, Task 5 / Round 19, seed 42, ξ = 0,8.
- DeNICE legacy; không CGoFed, CME, APPLIANCE hoặc donor inference.
- Giữ frozen `BinarySelf` và `MulticlassSelf` đã xuất cùng checkpoint. Không fit lại.
- Toàn bộ 13.505.771 test rows, 34 class, đúng 98 receiver terminal.
- Dùng nguyên global row ID, receiver assignment và thứ tự từ predictions cũ.
- Không đọc raw train hoặc CAL của task cũ; role manifest chỉ dùng làm metadata provenance.
- Đây là nguồn test đã dùng cho development, không gọi là untouched confirmation.

## Các phép đo

| Phép đo | Nhãn thật tham gia prediction? | Ý nghĩa |
|---|---|---|
| BinarySelf / MulticlassSelf | Không | Accuracy thật của một model local |
| OracleMatched | Có, diagnostic | Thay predicted task bằng true task nếu task đó hợp lệ; giữ mask và fallback |
| BestAllowedRoute | Có, diagnostic | Mẫu có thể được đoán đúng bởi ít nhất một route hợp lệ hay không |
| AllClassesDiagnostic | Không | Bỏ task mask, giữ weights; khác prediction policy gốc |
| MaskOnlyDiagnostic | Không | Giữ predicted task, thay local mask bằng global task mask; diagnostic, chưa là production repair |
| OracleGlobalTaskMaskDiagnostic | Có, diagnostic | True task + global task mask; ngoài local policy, không phải router ceiling |

Route hợp lệ là task có activation memory không rỗng. Mask lấy từ đúng helper
native. Task có class entry rỗng giữ fallback native về toàn bộ seen classes.
Không tự tạo task profile, mở class mask hoặc dùng donor weights trong audit.

## Phân rã lỗi không chồng lấn

Theo từng receiver và từng policy:

1. `correct`: normal prediction đúng.
2. `coverage_unreachable`: normal sai và true class không có trong union các legal route masks.
3. `classifier_within_coverage`: class reachable, nhưng mọi legal route vẫn đoán sai.
4. `routing_recoverable`: normal sai nhưng ít nhất một legal route đoán đúng.

Hai identity bắt buộc:

```
N = correct + coverage_unreachable + classifier_within_coverage + routing_recoverable
BestAllowedCorrect = correct + routing_recoverable
```

Các flag như missing true-task profile, missing true-task class, router sai
được ghi riêng vì có thể chồng lấn. Không cộng các flag này thành tổng lỗi.

## Tính toàn vẹn và numerical replay

- Xác minh SHA checkpoint, role manifest, frozen routers, metadata, test arrays và original predictions.
- Xác minh global row unique/exhaustive, label alignment và receiver population.
- Fingerprint model + algorithm state + router trước/sau mỗi receiver.
- Pure checkpoint không có adapters: một forward cho logits và context activations,
  rồi tái dùng cùng logits để xét các legal task masks. Nếu có adapters, script reject.
- Đối chiếu fast path với native evaluator ở batch đầu của từng receiver, hai policies.
- Đối chiếu normal prediction với toàn bộ predictions cũ; ghi mọi mismatch.
- Ngân sách numerical disagreement khóa trước: 0,01% cho mỗi phép đối chiếu.
  Nếu vượt, không kết luận ưu tiên sửa từ audit đó trước khi giải thích mismatch.
- Tắt TF32, không AMP; batch 512 như original evaluation.
- Lưu kết quả từng receiver để resume; không đổi protocol khi resume.

## Quy tắc chọn thành phần cần sửa

Chỉ chọn sau khi kiểm tra integrity và xem ba nhóm lỗi. Coverage phải được
tách tiếp thành class đã học local nhưng mất registration và class không có
BASE local. Nhóm thứ hai không tự động là bug; không tự mở 34 class.

Nếu dominant coverage không có lỗi persistence, bước sửa phải có đối chứng
availability-only trên development hợp lệ. Không chọn thresholds, donors hay
weight updates bằng nhãn của full-test audit. Nếu model không có competence
cho class ngoài mask, mở mask đơn thuần không được gọi là knowledge transfer.

Giữ baseline DeNICE gốc; mọi sửa đổi prediction policy được đặt tên riêng.
Không đổi graph, ξ, backbone hoặc optimizer cùng lúc với routing/availability.
