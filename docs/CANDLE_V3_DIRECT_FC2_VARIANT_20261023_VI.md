# CANDLE V3 — Variant A: Direct FC2 Transfer

Trạng thái: đã viết code variant A, chưa chạy tests/smoke/training/evaluation.
V1 giữ nguyên. Variant C `receiver_head_learning` vẫn giữ riêng, không tự fallback
từ A sang C sau khi nhìn CAL.

## Quyết định triển khai

Ưu tiên thử một donor và một class trước:

1. Native training và age-masked aggregation V1 chạy như cũ.
2. Cuối task, xét donor thuộc native neighborhood thực tế của receiver ở round cuối.
   Không dùng K=1 sentinel làm cluster. Neighborhood là điều kiện sàng lọc;
   không suy ra rằng feature spaces đã được chứng minh tương thích.
3. Donor phải có current local training evidence, successful optimizer steps của
   target class và FC2 row mature. Không gửi row random/chỉ được router đăng ký.
4. Copy nguyên row FP32, bias, weight/bias masks sang shadow receiver.
5. Stage availability và receiver-space current-task router sketches.
6. Đánh giá head, controls và negative risk trên current CAL; đạt mới commit.

Không có optimizer step của head trong A. Không trộn 50% receiver row với donor.
Không mở khóa mature receiver row. Chỉ class thiếu trong current BASE và receiver
FC2 row chưa allocate mới được xét ở phiên bản đầu.

## Packet

Module: `candle_v3_runtime/fed_learning/strategies/decentralized/candle_v3/direct_fc2.py`.

- Raw FP32: 256 weights + 1 bias = **1.028 byte**.
- Weight mask bit-packed: 32 byte; bias mask: một trường riêng.
- Class/task/donor, architecture, preprocessing, maturity/local training receipt,
  feature fingerprint và packet checksum.
- Ledger đo serialized message bytes gồm overhead/header; không báo 1.028 byte
  là kích thước toàn gói hoặc toàn phiên.

Với một donor, copy toàn bộ raw row, bias và masks, không thực hiện phép chia
q*w/q gây sai khác làm tròn. Source không bị sửa. Head được từ chối nếu active
nonzero connections sẽ phụ thuộc vào FC1 coordinates receiver chưa mature.
Không clip/bỏ những connections này rồi gọi đó là kết quả direct copy.

Với nhiều donor, cần feature fingerprint giống nhau giữa các donor trước khi
average. Đây là precheck bảo thủ của variant đầu, không phải chứng minh donor
phù hợp với receiver. Dùng current FIT quality cho trọng số, chuẩn hóa theo
contributors từng coordinate. Coordinate không có contribution giữ nguyên trạng
thái receiver. Default một donor tránh đưa multi-donor alignment vào phép thử đầu.

Native donor recall không là final eligibility gate của single-donor transfer;
nó được report. Acceptance phải đo actual receiver inference, giữ bài học từ các
audit trước rằng native router có thể loại nhầm donor có head dùng được.

## Controls

Tất cả chạy trên cùng current CAL, inference dự đoán task từ X:

| Điều kiện | FC2 | Router |
| --- | --- | --- |
| Availability-only | Receiver row cũ, masks cũ | Chỉ mở class |
| Router-only | Receiver row cũ, masks cũ | Receiver-space sketches mới |
| Mask-only | Receiver row cũ, masks/ranks như candidate | Router như candidate |
| Direct + native router | Copy donor row | Chỉ mở class, giữ memory native |
| Direct + enriched router | Copy donor row | Receiver-space sketches mới |

Candidate chính đã chọn trước là Direct + enriched router. Direct + native router
là diagnostic để tách nút thắt routing. Không chọn condition thắng trên CAL rồi
đổi nó thành candidate khác. Gates recall/FAR/break/minimum evidence giữ theo
protocol trước. Candidate phải vượt controls availability/router/mask-only để
chứng minh phần weights có đóng góp.

## Chi phí reference vẫn còn

**A không gửi encoder để donor train head.** Tuy nhiên, production acceptance vẫn
cần kiểm tra positive của target và router sketches trong hệ quy chiếu receiver.
Receiver không có dữ liệu target; donor còn current FIT/BASE/CAL. Implementation
hiện gửi receiver reference để thực hiện các bước đó và ghi đủ vào ledger.

Vì vậy, A giảm payload chuyển weights và loại bỏ head-training rounds, nhưng chưa
có bằng chứng tổng communication chỉ còn khoảng 1 KB/class. Nếu muốn cấm cả
reference cho verification thì khi không có cached reference đúng version hoặc
receiver evidence phù hợp, quyết định phải là UNKNOWN, không tự cho PASS.

## Cấu hình và trạng thái chạy

- Profile: `configs/candle_v3_direct_fc2.json`.
- `transfer_mode=direct_fc2`, `max_donors=1`, `max_targets=1`.
- Launcher experimental: `tools/train_candle_v3.py --direct-fc2` cùng các tham số
  `--data-dir`, `--output`, `--roles` như trước.
- `--direct-fc2` và `--peer-teaching` loại trừ nhau.
- Method identity: `candle_v3_direct_fc2`.
- Head learning variant C không bị đổi ngầm hoặc bật khi A bị reject.
- Notebook hiện tại vẫn pin baseline clone; chưa cập nhật full-run notebook khi
  chưa có native smoke/restore receipt.

Source manifest đã được cập nhật, nhưng chưa có execution receipt. Chưa kết luận
copy head thành công, cải thiện accuracy hoặc tiết kiệm communication toàn hệ thống.
