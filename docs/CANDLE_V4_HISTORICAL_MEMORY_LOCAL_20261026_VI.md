# CANDLE V4 — local counterfactual sau phục dựng historical memory

## Kết quả

Đã test trên **hai client khóa trước35/43**, mỗi client đủ135.058 TEST rows đúng shard cũ; tổng270.116 rows. Không train backbone/expert, không mở historical raw TRAIN, không dùng TEST fit/chọn threshold. Giữ threshold0,5, native routing/evaluation và V4 expert weights cũ. Chỉ fit lại class LR theo công thức cũ trên **TRAIN feature memory đã lưu** sau bổ sung đúng inherited memory.

| Client | Native MC control | V4 cũ | Phục dựng: native + continuous | Phục dựng: V4 fusion | Macro-F1 V4 mới |
|---|---:|---:|---:|---:|---:|
|35|65,5178%|0,7945%|70,2402%|70,2402%|41,2256%|
|43|63,8444%|0,8915%|67,7901%|67,8612%|37,2950%|

| Client | Route accuracy cũ | Route accuracy phục dựng | Rescue so V4 cũ | Break so V4 cũ |
|---|---:|---:|---:|---:|
|35|0,8167%|73,3655%|94.394|602|
|43|3,0069%|72,8176%|91.091|643|

Net đúng thêm184.240 trên hai shard. Có break nên không mô tả đây là sửa không rủi ro. Nhánh native MC và V4 cũ local **khớp chính xác toàn bộ confusion matrices34×34** của results17 tại cả hai client, không chỉ khớp Accuracy. Model state_dict kiểm tra trước/sau không đổi. V1 source/config/notebook không sửa.

Đây là local counterfactual có chọn client theo lỗi đã phát hiện, **không phải Accuracy mới của toàn bộ100 clients** hoặc kết quả fresh V8 training. Không ngoại suy trực tiếp hai client này thành48–50% toàn hệ thống.

## Phục dựng theo provenance

Không lấy arbitrary memory của source cuối run. Native bootstrap có chuỗi cùng task: ví dụ85 clone20 ở đầuT3, nhưng20 cũng vừa clone74 ngay đầuT3; checkpointT2 chưa có20. Do đó phải truy về client thực sự thu feature của từng task:

| Task bị thiếu | Client tạo memory | Snapshot dùng |
|---|---:|---|
|T0|15|CuốiT0|
|T1|22|CuốiT1|
|T2|74|CuốiT2|
|T3|85|CuốiT3|
|T4 (Client35)|85|CuốiT4|

Client43 giữ nguyên local T4/T5; Client35 giữ nguyên local T5. Các snapshotT0–T4 từ results16, full continuation cuốiT5 và old metrics từ results17.

Mỗi task import phải khớp native binary sketches, thresholds, stable/routing mask và episode class list giữa origin và target; selected continuous indices cũng khớp. Đây là feature memory, không phải raw inputs.

Thêm geometry probe cố định tối đa64 samples/class trên shard: **1.780 input/client**. Selected routing activations giữa encoder origin lịch sử và target hiện tại có **max absolute drift0 và relative L1 drift0**, với cả5 origin snapshots của35 và4 origin snapshots của43. Đây là empirical identity trên panel, không phải formal proof ngoài panel.

## Giới hạn môi trường và phạm vi

- Kaggle cũ dùng sklearn1.6.1; local hiện có1.9.0. Không cài/downgrade dependency. LR phục dựng fit bằng bản local, nên chưa là exact environment reproduction của fresh V8 run. Native/old V4 readout dùng estimator đã lưu vẫn tái tạo đúng matrices.
- Raw TEST chỉ dùng đo và kiểm tra drift, không fit router/calibrate/chọn ngưỡng. Đây là development diagnostic sau khi TEST đã có lịch sử audit.
- Không sửa results17, không ghi đè checkpoint gốc, không xuất checkpoint có thể resume giả là V8. Chưa có migration production cho cả100 clients.
- Một lỗi shape trong script diagnostic đã được sửa: expert input phải flatten39D giống terminal evaluator; runtime training/evaluation chính không đổi. Client35 không có expert kích hoạt nên readout đầu đã hoàn tất; Client43 được chạy lại với đúng input layout.

## Hành động tiếp theo

Chưa cần bắt đầu fresh full training. Ưu tiên mở rộng cùng quy trình provenance cho50 clients bị thiếu inherited memory, kiểm tra tương thích từng lineage và đánh giá terminal readout toàn bộ. Khi đó mới có Accuracy/Macro-F1 mới để quyết định có cần full retraining không. Không đổi V1/loss/optimizer/data/evaluation nhằm cứu điểm.

Script: `tools/debug_candle_v4_historical_memory_repair.py` (local paths, fixed35/43). Artifact tóm tắt: `artifacts/candle_v4_historical_memory_local_20261026.json`. Full local confusion matrices ở `audit_denice/candle_v4_bootstrap_reconstruction_20261026/readout.json`.
