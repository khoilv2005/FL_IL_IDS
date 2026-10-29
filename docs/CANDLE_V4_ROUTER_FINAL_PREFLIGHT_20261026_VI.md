# Router audit cuối trước full CANDLE V4-only

## Phạm vi đã khóa

Một bounded read-only audit trên results18: toàn bộ counts đã có của100 clients, thêm frozen feature panel6 clients (19/38 coverage thấp,1/40 giảm sau repair,35/43 positive controls). Mỗi panel tối đa128 samples/class, RNG20261026+client; tổng số samples dùng để chẩn đoán, không thay final TEST metrics.

Không fit, không optimizer, không thay threshold/routing rule, không sửa V1/training/evaluation/notebook. Đọc stored LR coefficients bằng phép tính scaler/linear/softmax cố định, không refit bằng sklearn local khác version. Không oracle task prediction được dùng trong inference.

## Correctness

- **0/100 router support mismatch** sau repair: continuous tasks khớp nonempty native memory tasks.
- Group true/predicted classes theo task từ full V4 confusion matrix khớp đúng runtime task-router confusion matrix; không phát hiện sai class→task mapping hoặc ghi metric nhầm.
- Six panels đều finite, model weights trước/sau không đổi; protected V1/shared-helper file SHA256 không đổi.
- Không có bằng chứng numerical failure phổ biến từ scaler: median standardized absolute features khoảng0,43–0,97; p99 khoảng2,09–2,75. Có outliers max19–67, chưa đủ cơ sở gọi scaler bug hoặc đặt clipping mới.

Không phát hiện correctness bug mới bắt buộc sửa trước full training. Đây không phải chứng minh mọi implementation path đều hoàn hảo.

## Phân rã full TEST errors

N=13.505.771; V4 Accuracy48,0823%, route accuracy51,0915%.

| Nguồn lỗi | Rows | % toàn TEST | % lỗi dự đoán |
|---|---:|---:|---:|
|True task không có trong router memory của client|3.041.412|22,5194%|43,3751%|
|Task có memory nhưng chọn nhầm task|3.564.061|26,3892%|50,8289%|
|Chọn đúng task nhưng sai class|406.411|3,0092%|5,7960%|

Các nhóm mutually exclusive, cộng đúng7.011.884 lỗi. Missing-task coverage là hệ quả local task participation/protocol hiện tại, **không phải lỗi bootstrap còn sót**. Không tự tạo task profiles bằng TEST hoặc ép mọi client có đủ34 classes.

Khi true task có memory, route accuracy **65,9410%**. Khi chọn đúng task, class accuracy **94,1102%**. Trần do task support hiện tại77,4806% là trần lạc quan, không phải accuracy có thể bảo đảm hoặc mục tiêu được phép tune bằng TEST.

## Nút thắt quan trọng: class chưa có trong router memory

| Nhóm mẫu | TEST rows | Chọn đúng task |
|---|---:|---:|
|True class có saved feature memory|7.692.013|80,6254%|
|True class chưa có feature memory, nhưng task đã có|2.772.346|25,1981%|
|True task hoàn toàn chưa có|3.041.412|0%|

Router LR hiện fit **class identity**, sau đó map class→task. Nó phải generalize các class không có feature memory sang đúng task thông qua những class đại diện khác. Khoảng cách80,63% so25,20% cho thấy hạn chế này lớn; chưa chứng minh đơn lẻ nguyên nhân là loss, first-task feature subspace hay domain shift. Không nên quy tất cả thành forgetting.

Weighted accuracy trên chính saved feature memory85,3932%; đây là training-memory diagnostic, không phải independent validation.

Client19 chỉ cóT0/T4: trần task coverage full-shard6,26%, V4 thực tế1,81%. Client38 chỉ cóT0/T4/T5: trần7,07%, V4 thực tế3,45%. Không kỳ vọng sửa thứ hạng router đơn thuần đưa hai client này lên50% trên toàn TEST.

## Fixed panel rank diagnostic

| Client | Rows | Top1 route | True task trong Top2 |
|---|---:|---:|---:|
|19|3.337|20,44%|30,78%|
|38|3.338|24,27%|30,80%|
|1|3.335|61,05%|80,06%|
|40|3.338|49,55%|61,59%|
|35|3.338|68,54%|86,25%|
|43|3.338|63,63%|83,43%|

Panel gần class-balanced nên không so trực tiếp với full-shard route accuracy sample-weighted. Top2 chỉ đo khả năng hiện diện của true task, **không phải deployable route accuracy**; chưa có rule input-only phân xử để đạt con số đó. Không bật Top2 hoặc đổi ngưỡng dựa vào bảng.

## Quyết định trước run

**GO về engineering** cho `train_eval_candle_v4_only_kaggle.ipynb` với cấu hình đã khóa. Không thêm vòng redesign/tune TEST trước run. Full fresh nhằm xác nhận bootstrap-fixed lifecycle/checkpoint chạy từ đầu, không tự sửa các task/classes chưa có memory và không có bằng chứng bảo đảm vượt50% Accuracy.

Nếu mục tiêu nghiên cứu tiếp theo là tăng Accuracy, ưu tiên routing cho classes chưa có memory và coverage theo đúng quyền dữ liệu; triển khai trong protocol development riêng sau khi có baseline fresh. Không sửa V1, không mở toàn bộ class/task hoặc dùng true task khi inference.

Artifact: `artifacts/candle_v4_router_final_preflight_20261026.json` (global counts,100 client summaries,6 panel results). Script: `tools/audit_candle_v4_router_final_preflight.py`.
