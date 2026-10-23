# CANDLE V3 encoder: phân rã lỗi và sửa preflight

## Trạng thái

Full training tiếp tục **HOLD**. Đã sửa việc phát hiện/điều phối trial; chưa tuyên bố
giải quyết chất lượng end-to-end. V1 và Direct FC2 không thay đổi thuật toán.
Không đổi recall/FAR/break gates, không đổi 8 head rounds/lr, không train backbone.

## 1. Audit trên encoder đã trưởng thành

Dùng checkpoint V1 Task 2 sau đủ 20 rounds, đúng role manifest
`b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.
Chỉ dùng current Task 2 BASE/FIT/CAL của hai client đã có local 9, 30. Không mở raw
task cũ, validation hoặc TEST. Hai ca được khóa bằng metadata trước khi đọc checkpoint:
class nhỏ nhất đủ FIT/CAL trong mỗi chiều, tối đa hai ca. Không thêm donor để cứu kết quả.

| Ca | Availability-only FIT | Full head FIT, ép đúng task (diagnostic) | Native routed full FIT | Kết luận |
| --- | --- | --- | --- | --- |
| 9 ← 30 / class 12 | 235/256 | 255/256 | 253/256 | Router-only đã 253/256, không cần học head |
| 30 ← 9 / class 13 | 0/256 | 189/256 | 3/256 | Có head gain thật, nhưng chưa 95% và router chặn |

Với class 13, staged native router chỉ đưa **11/256** target FIT vào episode chứa
class này (4,30%). Chỉ thay FC2 không thay được Conv/GRU routing activations. Trên
chính FIT đó, dù head hoàn hảo thì recall cũng bị giới hạn bởi khả năng mở class của
router. Đây là bound thực nghiệm của panel FIT, không phải chứng nhận cho CAL/test.

CAL baseline trial class 12: 122/122 target, 0 FP, 0 negative break, nhưng router-only
cũng 122/122; FAIL vì không có locked target gain vượt controls.
CAL class 13: 16/378 target, 0 FP, 0 negative break; FAIL recall. Các số này là shadow
trial. Không có update nào được cài vào native run hoặc checkpoint gốc.
Đây là development audit trên checkpoint có sẵn, không phải confirmation benchmark.

## 2. Ba thay đổi trong luồng encoder

### Evidence preflight trước snapshot

`CurrentTaskRoles.count_upper_bounds()` chỉ lấy count của current classes từ locked
manifest đã được BASE loader xác minh. Không đọc CAL scores hoặc raw CAL để chọn donor.
Mỗi donor gửi receipt count nhỏ, có tính serialized bytes. Nếu ngay cả upper bound
không đủ gate, trả UNKNOWN trước khi gửi receiver snapshot, encode hoặc học head.
Upper bound đủ không có nghĩa PASS: dedup và CAL acceptance cuối cùng vẫn bắt buộc.

### Functional gap phải vượt đúng router/mask controls

Sau Availability-only, tạo router sketches từ current BASE, rồi đo router-only và
mask-only trên current FIT. Nếu một control đã đạt recall gate, trả REGISTRATION_GAP,
không học head. Không tự cài router hoặc mở availability: risk/authorization chưa được
chứng nhận bằng việc target FIT tốt.

### Routing feasibility trước head optimization

Nếu không có registration gap nhưng tỷ lệ FIT được route tới một episode cho phép
target thấp hơn recall gate, trả DEFERRED / routing_preflight. Không đổi router hoặc
coi thiếu routing capability là lỗi optimizer. Đây là eligibility rule bảo thủ trên FIT,
không bảo đảm candidate sẽ FAIL trên mọi CAL/phân bố khác; có thể làm giảm trial coverage.

## 3. Version và regression

Config encoder mới khai báo `discovery_policy=control_fit_preflight_v1`. Default
`availability_only_v1` giữ semantics cũ cho replay. TeachingConfig khác policy thì
native resume từ chối; không âm thầm tiếp tục một experiment với discovery mới.
Direct FC2 giữ policy cũ và không chạy các preflight dành cho encoder.

| Kiểm tra sau sửa | Kết quả |
| --- | --- |
| Encoder structural/control tests | 28/28 PASS |
| Encoder native T0 → T1 | 84/84 PASS, task-boundary resume max difference 0 |
| Encoder native head sessions | 0, candidate dừng sớm do CAL upper bound=3 |
| Controlled synthetic install | 1, optimizer 8 rounds thật |
| Direct FC2 structural regression | 27/27 PASS |
| Direct FC2 native regression | 80/80 PASS |

Teaching traffic trong native smoke thiếu CAL giảm từ **6.702.087** xuống **1.007 bytes**.
Đây là chi phí của cùng một trial/task bị từ chối, không phải tổng communication toàn
hệ thống; native traffic chưa đo. Không gọi đây là cải thiện accuracy.

Mature paired replay dùng cùng checkpoint/cases, chỉ khác discovery policy:
class 12 chuyển từ học thừa rồi FAIL thành REGISTRATION_GAP; class 13 chuyển từ học rồi
FAIL thành DEFERRED tại FIT routing preflight. Cả hai không học head, không đọc raw
CAL để acceptance và không commit trong policy mới. Chưa có natural install PASS.

## 4. Blocker còn lại

Nút thắt đã rõ hơn: binary_cosine trong `ContextDetector.predict_episodes_with_scores()`
so sánh input với **một mean prototype mỗi episode**. Gộp thêm per-sample sketches vẫn
bị rút về mean này ở inference. Không có căn cứ rằng thêm sketches hoặc train head
lâu hơn sẽ tự sửa episode discrimination của class 13.

Head cũng mới đạt 73,83% khi ép đúng task trên FIT, chưa đạt gate. Vì vậy cần nghiên cứu
routing discrimination và head convergence riêng bằng development hợp lệ; không thể
chỉ sửa router rồi tuyên bố đủ 95%. Không hardcode class 12/13 trong production.
Không hạ gate và chưa bật full notebook. Một thay đổi router/optimizer tiếp theo cần
có protocol và đánh giá giữ kiến thức trước khi đưa vào cấu hình chính.

## Artifacts

- `artifacts/candle_v3_receiver_head_mature_before_preflight_20261023.json`
- `artifacts/candle_v3_receiver_head_mature_after_preflight_20261023.json`
- `artifacts/candle_v3_receiver_head_structural_preflight_20261023.json`
- `artifacts/candle_v3_receiver_head_native_preflight_20261023.json`
- `artifacts/candle_v3_direct_fc2_structural_preflight_20261023.json`
- `artifacts/candle_v3_direct_fc2_native_preflight_20261023.json`

Các receipt trước sửa vẫn giữ nguyên. Manifest runtime bind các receipt mới; notebook
encoder tiếp tục hiển thị HOLD, không được xem engineering PASS là quality PASS.
