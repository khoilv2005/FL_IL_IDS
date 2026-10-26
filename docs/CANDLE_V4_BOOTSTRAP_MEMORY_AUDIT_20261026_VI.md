# Audit results17: thiếu continuous memory khi bootstrap client

## Kết luận

Đã xác nhận **lỗi tích hợp state V4**, không phải lỗi thiết kế Non-IID: native bootstrap clone model và native context detector, nhưng không clone continuous memory được đặt ngoài native state. Client tham gia muộn thừa hưởng khả năng/model/router history ở native, trong khi nhánh continuous chỉ giữ classes đã thu thập từ TRAIN local kể từ khi client tham gia.

Audit read-only full continuation cuối Task5 của results17 và terminal confusion matrices đã có. Không mở raw TRAIN/TEST, không forward model, không fit router, không sửa runtime hay V1. Việc load checkpoint local có sklearn version warning; kết luận này chỉ dựa trên metadata/support và counts có sẵn, không dùng estimator prediction/fit khác version.

## Bằng chứng

1. `decentralized_denice_il.py:2831` gọi `restore_context_detector()` khi `representative_clone` có parameter distance0, nhưng không đồng bộ `v4_shadow.continuous_routers`.
2. `candle_v4_continuous_runtime.py:71` tạo `ContinuousClassMemory` rỗng theo receiver khi update đầu tiên; historical native memories không được chuyển sang continuous.
3. `ContinuousClassMemory.predict()` chỉ có thể dự đoán các task tương ứng với classes đã fit. `predictions()` chỉ fallback khi cả continuous memory chưa tồn tại, không khi continuous support thiếu một phần native support. Scoped expert cũng chỉ được cạnh tranh trong predicted task.
4. **50/100 client** thiếu ít nhất một inherited task trong continuous memory. Tập này trùng chính xác50 client có event `representative_clone` trong debug history.
5. Trên cả100 terminal per-client matrices, predicted task support của continuous-native và V4 fusion luôn là subset của continuous task support; zero true positives tại các inherited task bị thiếu.

| Client | Clone từ / thời điểm | Native task memory | Continuous task memory | Native Acc | V4 Acc |
|---|---|---|---|---:|---:|
|35|85 / T5|T0–T5|T5|65,52%|0,79%|
|43|85 / T4|T0–T5|T4–T5|63,84%|0,89%|
|59|20 / T3|T0–T5|T3–T5|62,40%|18,47%|
|5|22 / T2|T0,T1,T2,T3,T5|T2,T3,T5|47,41%|19,75%|

Client35 có133.955/135.058 TEST rows nằm ngoài các inherited task mà continuous còn giữ. Ngay cả hoàn hảo trên T5, hard task routing hiện tại chỉ có trần Accuracy **0,8167%** trên shard này. Thực tế0,7945% đã gần trần. Đây không phải bằng chứng backbone của client35 bị mất kiến thức; native control cùng checkpoint vẫn65,52%.

## Phạm vi ảnh hưởng

Tổng **1.253.856 rows**,9,2839% full TEST, thuộc inherited tasks có ở native nhưng vắng ở continuous. Native control đã nhận đúng **913.811** trong số đó; continuous-native và scoped V4 đều đúng0 trên nhóm này.913.811 là số dự đoán native đang bị làm unreachable, **không phải gain có thể bảo đảm sau sửa**.

| Nhóm | Clients | Native MC Acc | Continuous-native Acc | V4 fusion Acc |
|---|---:|---:|---:|---:|
|Thiếu inherited continuous memory|50|48,7154%|37,7370%|44,3075%|
|Không có mismatch này|50|28,1671%|35,8373%|38,6772%|

Đây là phân rã mô tả từ run hiện tại, không phải randomized matched ablation. Nó cho thấy không thể kết luận mọi continuous routing đều kém chỉ từ endpoint chung. Cũng chưa chứng minh bootstrap là nguyên nhân duy nhất của mọi lỗi routing.

## Sửa đúng phạm vi tiếp theo

- Chỉ ở đường **V4 exact representative clone**, deep-copy continuous memory/scaler/classifier/indices tương ứng từ source **tại thời điểm clone**, cùng model/native detector. Ghi source/version/provenance và số bytes state đi kèm, tránh alias giữa clients.
- Nếu source continuous memory đã thiếu từ bootstrap trước đó, copy cuối run không tự phục hồi phần thiếu. Phải theo đúng lịch sử/bootstrap chain; không gắn tùy tiện memory của source cuối Task5 vào receiver khác encoder.
- Không áp dụng copy này cho `rejoining_plastic_catch_up`: đó chỉ là partial parameter sync, không bảo đảm representation identity. Giữ nguyên native behavior.
- Test riêng late join, clone chain, checkpoint/resume, no-alias và native isolation. Không đổi loss, optimizer, số bước, graph, threshold hay evaluation data.
- Có thể audit/counterfactual trên snapshot nếu có đủ state đúng thời điểm; không ghi đè kết quả results17 hoặc tuyên bố đã sửa quality. Chưa cần full rerun trước khi fix state contract.

Không tự dùng historical raw data để bù memory. Mục tiêu là chuyển đúng feature-memory state vốn phải đi cùng exact model clone, không ép mỗi client có đủ tất cả tasks/classes.

Reproduce: `tools/audit_candle_v4_bootstrap_coverage.py --checkpoint <results17 continuation_state_latest.pt> --report <candle_v4_final_test.json> --output <audit.json>`.

Artifact: `artifacts/candle_v4_results17_bootstrap_coverage_20261026.json` chứa support, bootstrap provenance, counts và accuracy của100 clients. Đây là **audit**, chưa áp dụng fix training/runtime.
