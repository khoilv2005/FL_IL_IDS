# Results18: full readout sau phục dựng memory V4

Archive `C:\Users\khoak\Downloads\results (18).zip` (13.842.027 bytes) là output notebook historical repair, **không phải fresh V4-only training**. Protocol `candle_v4_historical_memory_counterfactual_v1`, source training37a277c, readout842a59a; sklearn1.6.1 như Kaggle gốc. Backbone/expert training false; raw TRAIN và TEST-for-fitting false.

## Kết quả cuối

Completion true, phục dựng50 clients, đánh giá đủ13.505.771 original TEST rows/34 classes trên100 receivers. Cùng disjoint partition seed523687 và pooled metrics; mỗi row tính một lần.

| Metric V4 | Results17 | Results18 sau phục dựng | Chênh lệch |
|---|---:|---:|---:|
|Accuracy|41,4923%|48,0823%|+6,5900 điểm|
|Pooled Macro-F1|23,3672%|27,6110%|+4,2438 điểm|
|Task routing accuracy|44,5513%|51,0915%|+6,5402 điểm|

V4 đúng6.493.887 rows, tăng **890.027** so results17. Đây là net count từ full confusion matrices, không phải số rescue đơn thuần; paired rescue/break so results17 không suy ra được chỉ từ aggregate matrices.

Trong50 clients được sửa:46 tăng Accuracy,4 giảm.50 clients không cần sửa giữ nguyên toàn bộ V4 confusion matrices. Native control cũng khớp đúng pooled và tất cả100 per-client confusion matrices của results17, chứng minh phần đọc baseline/native không bị thay đổi.

## Tách các readout

| Predictor | Accuracy | Pooled Macro-F1 |
|---|---:|---:|
|Native multiclass control|38,4413%|19,8637%|
|Continuous router + native classifier sau phục dựng|43,7696%|25,1247%|
|V4 sau expert fusion|48,0823%|27,6110%|

Expert fusion tăng4,3128 điểm Accuracy so repaired continuous-native. Đây là readout cùng checkpoint, không chứng minh riêng hiệu quả training so randomized control.

Client35:0,7945%→70,2335%. Client43:0,8915%→67,8694%. Hai kết quả gần local counterfactual trước đó nhưng môi trường lần này dùng đúng sklearn1.6.1.

Hai client V4 còn Accuracy<10%:19 (1,81%),38 (3,45%). Các client thấp tiếp theo:70 (11,74%),14 (20,31%),0 (20,67%),47 (20,70%),99 (20,78%),33 (21,62%).

## Kiểm tra và giới hạn

Tự cộng lại pooled confusion matrices từ100 per-client matrices; tính lại Accuracy/Macro-F1 từ counts, đều khớp. Kiểm tra đầy đủ TEST rows và các branches cùng sample counts. Tổng evaluator wall time1.926,27 giây (~32,10 phút).

Chưa vượt50% Accuracy. Routing vẫn là giới hạn lớn: trong hard task-scoped inference, dự đoán đúng class yêu cầu task phù hợp; route accuracy hiện51,09% chỉ cho một trần lạc quan, không bảo đảm đạt được trần do class availability và within-task errors.

Không dùng readout này như kết quả fresh bootstrap-fixed training hoặc independent benchmark. Đây là development counterfactual trên checkpoint đã audit TEST. Không đổi threshold hoặc fit từ TEST để tạo kết quả. Không chạy lại inference/training trong bước đọc archive18 này.

Chi tiết100 client: `artifacts/candle_v4_results18_summary_20261026.json`. Full JSON đã trích local ở `audit_denice/candle_v4_results18_20261026/`.
