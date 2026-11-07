# Kiểm chứng bổ sung routing features từ donor

Đã chạy một ca chọn bằng metadata, không chọn theo accuracy: receiver 0,
donor 60, class32, Task5/R19. Trong 70 candidate hợp lệ, lấy tuple
receiver/class/donor nhỏ nhất. Donor có 3.624 positive TRAIN và 291 negatives,
được phép đóng góp expert ở round19 theo receipt trong checkpoint.

## Cơ chế

Donor nhận bản serialize của receiver function và feature indices. Dùng seed42
chọn 20 mẫu class32 từ Original TRAIN Task5, chạy qua encoder receiver, trả
20 × 85 continuous routing features. Không chuyển raw samples. Receiver fit
một bản router thử nghiệm từ memory TRAIN sẵn có cộng features này. Backbone,
experts, native detector, quá trình training và notebook mặc định không đổi.

Kiểm tra feature encoder sau truyền khớp tuyệt đối encoder receiver. All fitting
hoàn tất trước khi mở TEST. Threshold expert vẫn0,5; mỗi mẫu chỉ chạy scope
adapter/mask/expert của một task được chọn.

## Kết quả development

Panel toàn cục cân bằng class: 128 mẫu/class ×34 =4.352 mẫu, seed20261107.
Đây là một receiver trên panel diagnostic, **không phải accuracy toàn hệ thống**
và không so trực tiếp với full pooled accuracy48,08%.

| Phương án | Accuracy | Macro-F1 | Recall class32 | Route đúng trên class32 | FP class32 | Rescue / Break |
|---|---:|---:|---:|---:|---:|---:|
| Router gốc |19,5772%|9,2076%|12,50%|17,1875%|190|0 /0|
| Chỉ refit memory cũ |19,5542%|9,1978%|12,50%|17,1875%|191|0 /1|
| Thêm features, chọn trong Top2 gốc |19,6461%|9,2638%|14,0625%|20,3125%|205|11 /8|
| Thêm features, chọn lại mọi task |19,7610%|9,3819%|23,4375%|30,4688%|235|25 /17|

Top2 cứu ròng3 mẫu so với baseline, +0,0689 điểm accuracy. Chọn lại mọi task
cứu ròng8 mẫu, +0,1838 điểm. Recall class32 tăng nhưng false positives cũng tăng;
chưa đủ để triển khai hoặc dự báo full accuracy >50%.

Task đúng chỉ nằm trong Top2 gốc ở63/128 mẫu class32 (49,21875%). Bộ phân xử
bị giới hạn trong Top2 không thể chọn task đúng cho65 mẫu còn lại, bất kể
bộ chọn tốt đến đâu. Đây là hạn chế candidate coverage của ca này, không phải
trần toàn hệ thống. Không dùng true task để quyết định prediction.

Đối chứng refit chênh1 prediction so baseline: môi trường local sklearn1.9.0
khác bản lưu1.6.1 và fit lại không đảm bảo bitwise prediction giống hệt. Vì vậy
đã báo cả refit control. Không cài/đổi dependency local.

## Chi phí và kiểm tra

- Receiver capsule serialize:6.184.342 bytes.
- Feature packet:7.390 bytes.
- Tổng trao đổi mô phỏng application-wire:6.191.732 bytes, khoảng5,905 MiB.
- Capsule ở pilot chứa receiver model hoàn chỉnh; chưa tối ưu chỉ routing encoder
  hay reuse reference. Không được chỉ báo7,39KB rồi bỏ receiver capsule.
- Native model hash và service state hash (expert/Adam/router gốc) giữ nguyên.
- 8 protected V1/native file hashes khớp manifest.
- Chỉ đọc donor current-task TRAIN để lấy features; lịch sử chỉ dùng saved
  activation memory. Không train backbone, không replay raw task cũ.

## Phạm vi kết luận

Feature transfer đúng tọa độ thực hiện được và có tín hiệu cải thiện nhỏ ở ca
này; chưa chứng minh gain tổng quát hoặc giải quyết router. Đây là cùng-process
wire simulation tại checkpoint cuối, chưa tích hợp round-wise/native network.
Reference là encoder sau consolidation; chưa kiểm tra drift ở round/task sau.
Graph authorization dựa round19 đã lưu, không dựa graph tương lai.

Chưa thay inference chính thức, chưa tạo notebook full mới, chưa train full lại.
Nếu mở rộng, cần một cohort chọn trước bằng metadata, đo cả candidate coverage,
negative transfer và reference cost; không thay fixture để tìm PASS. TEST ở đây
được khai báo development, không phải xác nhận độc lập.

Script: `tools/pilot_candle_v4_receiver_routing_features.py`.
Receipts: `audit_denice/candle_v4_receiver_routing_features_20261107/fixture.json`
và `audit.json`. Public receipt: `artifacts/candle_v4_receiver_routing_features_20261107.json`.
