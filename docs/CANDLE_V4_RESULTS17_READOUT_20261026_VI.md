# CANDLE V4 results17 — terminal readout

Nguồn: `C:\Users\khoak\Downloads\results (17).zip` (2.314.262.296 bytes). Source `37a277c8f6ae85e74c9553d4d30e5e27f052ca93`, continuous_class_v1, seed42. Resume từ task-boundary Task4; receipt khớp continuation SHA đã kiểm tra ở results16.

## Hoàn tất

Completion true; phase summary completed_task5, 6 task records, 120 round records. Final TEST task5/round19, after_consolidation true. Đánh giá đủ 13.505.771 rows/34 classes, phân bổ class-stratified disjoint shards trên 100 receivers, seed523687; mỗi row tính một lần. Không phải mỗi client chạy full TEST riêng. Giữ nguyên protocol V1 của lần chạy này.

Đã cộng lại confusion matrices từng client, kiểm tra khớp pooled matrices và tự tính Accuracy/Macro-F1 từ counts. Không train hoặc chạy lại inference/TEST trong audit này.

| Readout | Accuracy | Pooled Macro-F1 |
|---|---:|---:|
| Native V1 multiclass control, cùng backbone/rows | 38,4413% | 19,8637% |
| Continuous router + native classifier, chưa fusion expert | 36,7871% | 20,8237% |
| V4 continuous + coverage expert fusion | 41,4923% | 23,3672% |

Router task accuracy của phương pháp continuous: **44,5513%**. Không gán con số này cho native multiclass control; report chỉ ghi một router confusion matrix cho phương pháp chính.

## Phân rã

- V4 so native control: +3,0511 điểm Accuracy, +3,5034 điểm F1; rescue1.762.886, break1.350.814, net+412.072 dự đoán đúng. Đây là hiệu ứng tổng router+experts.
- Continuous router so native control: -1,6541 điểm Accuracy, +0,9600 điểm F1. Có 28/100 clients giảm accuracy. Chưa có căn cứ coi continuous là cải thiện chung.
- Expert fusion so continuous-native: +4,7052 điểm Accuracy, +2,5434 điểm F1; net+635.472 đúng. 16 clients giảm accuracy khi thêm expert; tổng gain không có nghĩa mọi client đều được lợi.
- 667 ACTIVE experts, 227 donor handoffs. ACTIVE là training-only unverified, không phải quality acceptance.
- Expert serialized application-wire simulation19.503.031.378 bytes (~18,16 GiB), discovery32.189.766 bytes. Tổng estimate405.220.563.410 bytes gồm native dense upper bound; không coi đó là network traffic đo thực tế.
- Forgetting chưa đo được theo trajectory vì không có intermediate TEST/validation. Development report ghi TEST_opened false vì được xuất trước terminal TEST; completion/final_test xác nhận TEST đã mở và hoàn tất.

## Các client cần debug trước

| Client | Native MC | Continuous-native | V4 fusion |
|---|---:|---:|---:|
|35|65,52%|0,79%|0,79%|
|43|63,84%|0,58%|0,89%|
|59|62,40%|8,12%|18,47%|
|5|47,41%|12,12%|19,75%|

Các ca trên chỉ ra suy giảm ở nhánh đổi router trước khi expert fusion, chưa xác định nguyên nhân sâu là representation/memory/fit hay implementation. Không được gọi chắc chắn là forgetting hoặc sửa V1 dựa trên bảng này. Tổng4 clients có V4 accuracy<10%:35,43,19,38.

Kết quả scoped+multiclass cũ từng audit là42,9073%/23,0388%; endpoint mới thấp hơn1,4149 điểm Accuracy, cao hơn0,3283 điểm F1. Đây là so sánh với readout checkpoint cũ, không phải matched ablation riêng continuous router hay donor handoff. Cần giữ phân biệt này khi quy đóng góp.

Bước hợp lý tiếp theo: audit read-only router state/predictions của35,43 trước, cùng native MC control và cùng input rows. Giữ V1, training/evaluation protocol nguyên trạng; chưa chạy lại full hoặc chọn threshold/router theo TEST.

Chi tiết100 clients: `artifacts/candle_v4_results17_summary_20261026.json`.
