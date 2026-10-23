# CANDLE V3 receiver-encoder — HOLD full training

## Quyết định

Theo yêu cầu người dùng, chưa chạy full biến thể encoder. Notebook được giữ để dùng
sau khi debug chất lượng; không xem engineering smoke là điều kiện đủ để chạy full.
Không đổi gates, không tuning bằng CAL/TEST, không sửa production runtime hoặc V1.

## Audit đã chạy

`tools/debug_candle_v3_receiver_head.py` nạp đúng checkpoint T0 của smoke, tái dựng
BASE qua cả smoke cap và local validation split, restore model/router rồi replay session
30 ← 9 / class 3. Frozen reference hash, mọi control metric và cả primary/donor-only
head histories khớp receipt gốc. Tự tính lại chuỗi 8 bước SGD và kiểm tra fitted head
khớp tuyệt đối với production `learn_head`. Rejected trial không đổi predictor live.

Audit chỉ quan sát checkpoint tại đúng task 0; không dùng dữ liệu Task 1 để fit,
không TEST, không historical replay trong training, không thay hyperparameters.
Old-only/task-mask scores trong trace chỉ là diagnostic cho objective head,
không phải headline inference. Terminal acceptance vẫn dùng native predicted routing.

## Kết quả

| Thành phần | Quan sát |
| --- | --- |
| Encoder đã train | Chỉ 1 native round, tối đa 64 BASE/client trước local split |
| BASE thực sự vào teaching | Receiver 60; donor 59 mẫu, class đích 12 mẫu |
| FIT class đích | 34 mẫu |
| CAL class đích | 3 mẫu, dưới gate 32 |
| FC1 coordinates cho phép học head | 32/256 |
| Head SGD rounds | 8, mỗi round một bước gradient local đầy đủ rồi aggregate |
| Weighted BASE loss | 1,60123193 → 1,60109997 |
| Head change L2 | 0,00324988 |
| Target recall trên BASE, cùng task-mask objective | 0/12 → 0/12 |
| Target recall trên FIT, cùng task-mask objective | 0/34 → 0/34 |
| Native routing đúng task trên CAL | 236/236, Task 0 mới có một task memory |
| Terminal CAL full head | 1/3 target; 2 FP/233 negatives; 2 negative break |
| Terminal full so với mask-only | Tất cả metric theo owner/class giống nhau |

Loss giảm khoảng 0,0082% tương đối. Optimizer có cập nhật thật, nhưng trong ca này
8 bước chưa thay đổi quyết định theo hướng có ích. Đây không phải bằng chứng rằng chỉ
cần tăng learning rate/rounds sẽ sửa được: chưa xác định feature representation có đủ
thông tin phân biệt dưới maturity permissions và evidence hợp lệ hay không.

## Ba kết luận cần giữ riêng

1. **Evidence không đủ là blocker chắc chắn của cặp này.** Manifest đã ghi CAL class 3
   của donor 9 chỉ 3 mẫu. Train native lâu hơn không tự tạo thêm CAL cho cùng cặp và
   cùng role allocation. Không nên dùng cặp này làm quality acceptance pilot.
2. **Head chưa tạo gain là blocker độc lập.** Không chỉ thiếu CAL: ngay trên BASE/FIT,
   target vẫn 0%. Primary CAL kém hơn availability-only và không vượt mask-only.
   Router Task 0 chọn đúng nên không đổ thất bại hiện tại cho router.
3. **Không kết luận kiến trúc đã thất bại.** Smoke encoder mới học 1 round và chỉ dùng
   32 mature FC1 coordinates. Nó kiểm tra integration, chưa đại diện encoder đủ trưởng
   thành sau training chính thức, cũng chưa kiểm tra routing nhiều task về chất lượng.

## Phân biệt implementation và hạn chế protocol

- Trong phạm vi audit, chưa thấy sai reference, sai phương trình aggregate hoặc update
  không được apply. Tham số thay đổi, loss giảm, head tái tính khớp code production.
- 8 head rounds hiện là **8 full-gradient SGD updates**, không phải 8 epoch mini-batch SGD.
  Cần giữ cách đếm này trong log và mô tả phương pháp; chưa tự đổi ý nghĩa config.
- Native V1 còn `_split_local_validation(..., denice_validation_fraction=0.1)` sau
  BASE-role selection và sau smoke sample cap. Vì vậy teaching nhận 60/59 thay vì 64/64.
  Đây là hành vi V1 kế thừa, không phải mất mẫu ngẫu nhiên do codec hoặc head trainer.
- Nếu chỉ nhìn mask/hash và test receipt PASS, sẽ bỏ sót việc target learning chưa có gain.
- Eligibility hiện trả UNKNOWN sau khi đã học, dù CAL count không đủ có thể biết từ
  locked manifest trước session. Đây là điểm nên sửa về scheduling/cost, không phải
  phương án làm tăng recall hoặc đủ điều kiện commit.

## Bước tiếp theo được giới hạn

1. Preflight counts trên metadata đúng current task để chọn ca có thể đủ FIT/CAL;
   tuyệt đối không lấy CAL scores để chọn donor/tune. Chọn ca trước khi đánh giá kết quả.
2. Dùng encoder checkpoint đã học đủ, ở đúng current task, cho một bounded trial.
   Đo masked-feature information, old/new margins, BASE/FIT learning và native routed
   inference riêng. Không mở full 6 task để chẩn đoán một candidate.
3. Nếu có lỗi correctness, sửa đúng lỗi rồi chạy lại structural/native regressions.
   Nếu chỉ là thiếu convergence/feature discrimination, đăng ký thử nghiệm development
   riêng trước khi đổi optimizer budget hoặc permissions; không cứu kết quả bằng CAL.
4. Chỉ đề nghị full khi có gain dữ liệu thật vượt controls, đủ current acceptance
   evidence và kiểm tra retention/restore. Nếu không đạt thì báo kết quả âm; không
   tự kéo dài việc tìm donor hoặc tune vô hạn.

Artifact: `artifacts/candle_v3_receiver_head_debug_20261023.json`.
Full training vẫn HOLD; chưa có natural install thành công được xác nhận.
