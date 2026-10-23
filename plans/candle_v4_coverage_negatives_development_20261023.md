# CANDLE V4 — đăng ký coverage-aware negatives development

Kết quả sau execution: [báo cáo development](../docs/CANDLE_V4_COVERAGE_NEGATIVES_DEVELOPMENT_20261023_VI.md).
Rule/work budget bên dưới giữ nguyên bản đăng ký; không có CAL/ACTIVE/full.

Đây là protocol development mới, không sửa hồi tố kết quả bank REJECT tại `644d116`.
Giữ fixture receiver6, positive donors1/2, classes12/13 và owners `{1,2,3,6}`.
Không chạy thêm native smoke để tìm bank PASS; không mở CAL/TEST/ACTIVE/full.

## Giả thuyết và rule tổng quát

Một expert có thể thiếu negative classes vì chỉ học từ primary donor + receiver.
Tại từng current graph, lấy hợp negative classes hiện diện ở hai participant đó.
Với class ngoài target có **0 BASE samples** trong hợp này, tìm current eligible
neighbors có ≥32 current BASE samples của class đó. Chọn owner ID nhỏ nhất cho
mỗi class thiếu; chỉ yêu cầu negative gradients của đúng class được chọn.

Selector chỉ dùng graph và BASE count metadata, không FIT/CAL outcome hoặc tên
class ngoại lệ. Trên fixture đã khóa, rule sẽ tự chọn contributor2/class13 cho
expert12; expert13 vốn đã có mọi negative class current nên không thêm contributor.

## Ba nhánh và work matching

| Arm | Cơ chế |
| --- | --- |
| current | Gọi đúng original V4 balanced joint trainer |
| coverage-aware | Generic multi-peer trainer, thêm missing-class negatives |
| no-extra ablation | Cùng generic trainer, tắt contributor bổ sung |

- Initialization seed42, MLP39→64→1 FP32 CPU, Adam0,001, weight_decay0, clip5.
- 8 logical rounds, quota44/round, max88 catch-up; 352 steps/expert.
- 128 positive + 128 negative samples/update cho mọi arm. Extra negatives thay
  một phần 128 negatives hiện có; không tăng tổng lượt mẫu hoặc số bước.
- Positive loss mass0,5; negative mass0,5 với class balancing theo BASE counts.
  Khi thêm class mới, objective negative uniform-class cũng đổi; không tách claim
  coverage khỏi hiệu ứng redistribution bằng một thí nghiệm này.
- Mỗi participant tính gradients trên cùng reference, receiver sum một lần rồi
  global clip + Adam. Không average optimizer states, không chuyển raw samples.
- No-extra phải khớp weights original control; original control phải tái hiện
  hashes của hai experts ở native smoke trước. Không chỉnh implementation dựa
  trên FIT để làm khớp outcome mong muốn.

Đây là offline development replay của recorded current graph T2 rounds0–7, không
train lại native V1 hoặc claim thêm bằng chứng native integration/early utility.

## Data và evaluation đã định trước

Chỉ dùng staged current T2 BASE/FIT. Train từ cùng BASE sau original native
local-validation split10%, cap/dedup như native smoke. Không đọc original CAL
partition hoặc ba mẫu đã gây reject; không lựa hard samples bằng CAL scores.

Chia FIT theo content hash, owner-independent: first8 bytes SHA256 của FP32 input,
big-endian, modulo2. Nhóm0 dùng chọn threshold; nhóm1 chỉ dùng assessment.
Không dùng label/owner để thay đổi partition, không chọn lại split seed sau outcome.

- Primary: threshold donor FIT-select recall95 order statistic, rồi đánh giá
  trên FIT-assess của cả4 owners.
- Secondary cố định: threshold0,5.
- Giữ arbitration normalized excess như bank cũ, native prediction làm fallback.
- Đánh giá target recall, pairwise false activation, từng owner/class, candidate
  rescue/break, accuracy/Macro-F1 và thay đổi correct predictions so với controls.
- Báo cả owner pools và pooled unique `(feature hash,label)` để không cộng duplicates
  thành unique test population. Endpoint round7 cố định, không chọn winning round.
- Chỉ mở FIT sau khi mọi arm đã train xong. FIT đã là development từ trước;
  partition mới không biến nó thành independent confirmation holdout.

## Communication và quyết định

Đếm serialized expert references, mọi remote gradient, metadata discovery,
FIT references/threshold receipts/prediction receipts. Native model reference và
assessment identity cache bất biến dùng chung được đo riêng; báo cả cost từng
isolated arm và combined research experiment. Budget128MiB/arm.

Không claim tiết kiệm communication bằng cách bỏ additional peer gradients hoặc
verification. Không có real network benchmark hoặc DP claim.

Kết quả có thể giảm FP, không đổi hoặc làm recall/accuracy xấu đi. Báo từng tiêu chí,
không dùng FIT để quyết định thêm steps/đổi threshold/tìm donor khác. Thí nghiệm này
không cấp phép ACTIVE hoặc full seed42; gate observed FAR0,1% của protocol cũ giữ
nguyên, và confirmation cuối cùng vẫn cần evidence độc lập phù hợp.
