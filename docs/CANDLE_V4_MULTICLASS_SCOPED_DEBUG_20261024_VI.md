# CANDLE V4 — debug scoped fusion + multiclass trên checkpoint

## Kết quả đầy đủ

Đã xử lý 100 clients, đủ 13.505.771 TEST rows, cùng partition seed523687. Không train lại backbone/expert; không chọn threshold trên TEST; threshold giữ 0,5. Multiclass balanced router chỉ fit từ activation memory đã lưu. Đây là readout development sau khi đã quan sát TEST, không phải confirmation độc lập.

| Cấu hình | Pooled Accuracy | Pooled Macro-F1 |
|---|---:|---:|
| V1 + binary | 34,61% | 16,96% |
| V1 + multiclass | 38,44% | 19,87% |
| V4 scoped + multiclass | **42,91%** | **23,04%** |

So với V1 multiclass cùng weights/router/class masks: +4,4667 pp Accuracy, +3,1726 pp Macro-F1. Rescue 705.677, break 102.420, ròng +603.257 đúng. Tổng đúng V4 scoped là 5.794.957. Có 86 clients tăng, 13 giảm, 1 không đổi. 667 experts được xét; kích hoạt trên 2.895.679 lượt, trong đó 366.150 có nhiều expert cùng vượt threshold trong predicted task.

Không quy toàn bộ chênh lệch so với V4 cũ 20,46% cho scoping: phép so đó đồng thời đổi router binary sang multiclass. Đối chứng chính là V1 multiclass 38,44%.

## Cách tính và kiểm tra

Dùng source package từ run gốc `2ec3d0e43b4cccd4ebc56a9ad78ddacdd9f41a20`, checkpoint V4 cuối Task5, original TEST. SHA256 của checkpoint/TEST/reference, threshold, router, batch và seed được khóa trong `comparison.json`.

Tận dụng confusion matrix native multiclass đã kiểm chứng ở lượt V1: tính lại task predictions trên mọi TEST row, yêu cầu task confusion matrix từng client khớp chính xác reference. Chỉ forward classifier lại trên những mẫu expert thực sự override; trừ native confusion cells và cộng expert cells. Kiểm tra không trừ vượt matrix gốc, không đổi weights, không mất mẫu và identity gain = rescue − break.

Hàm executor có tên `oracle_hard` được dùng để thực thi adapters/class masks theo **task đã dự đoán từ input** trên subset. Không truyền true task hoặc labels vào executor. Nhãn chỉ dùng sau dự đoán để tính metrics. Có script cross-check direct full forward trên Client9/71, không tái sử dụng confusion cells, và receipt riêng.

Đây là post-hoc inference change. Notebook scoped-v5 đã publish vẫn dùng binary router; audit này không tự đổi training config, không resume protocol mới từ checkpoint cũ và không chạy full training.

## Nút thắt còn lại: routing và coverage

Đã kiểm tra 519 populated router profiles: mọi profile có local classes không rỗng và đúng task-group. Vì native hard mask lẫn scoped experts đều chỉ dự đoán trong task được chọn, accuracy bị chặn bởi routing accuracy **45,5192%** trên checkpoint này.

| Phân rã | Số mẫu |
|---|---:|
| TEST thuộc task client có memory | 10.464.359 (77,48%) |
| TEST thuộc task client không có memory | 3.041.412 (22,52%) |
| Router chọn đúng task | 6.147.717 (45,52% toàn TEST) |
| Router chọn sai dù task có memory | 4.316.642 |
| V4 scoped: sai class dù chọn đúng task | 352.760 |

Router accuracy trên phần task có memory là 58,75%. Không gọi missing-task coverage do Non-IID/participation là bug implementation. Metadata class coverage native là 56,95%; native + expert là 76,59%, không phải functional accuracy.

Sau scoped fusion, 7.358.054/7.710.814 lỗi còn lại (95,43%) liên quan đến chọn sai task. Ngay cả nếu sửa hết 352.760 lỗi class trong correct-task region, cấu hình vẫn không đạt 50% accuracy nếu giữ nguyên task routing.

Chỉ xét receiver có memory task tương ứng, các confusion task lớn nhất: T2→T3 943.462, T3→T2 584.079, T2→T1 452.393, T1→T3 401.280. Đây là nhóm nên ưu tiên audit tiếp. Metadata cả 100 detector ghi fresh=True; cờ này không chứng minh sketches lịch sử còn khớp representation cuối.

## Lỗi class/experts còn lại

Các break pairs lớn nhất:

| True class | Expert class | Break |
|---|---|---:|
| 22 | 23 | 33.218 |
| 23 | 22 | 23.429 |
| 13 | 12 | 18.853 |
| 32 | 30 | 4.450 |

Client9 giảm nhiều nhất: 28,06%→23,51%, rescue1.044 / break7.197. Một expert riêng có thể gây hại dù toàn client vẫn gain; cần giữ cả attribution theo expert và theo client.

Receiver71/class12: 544 rescue / 12.071 break. Receiver9/class12: 973 rescue / 6.780 break. Cả hai chỉ nhận 44 updates. Trong khi đó một số expert23 đã học đủ 880 updates vẫn nhầm class22; không quy mọi lỗi cho thiếu số bước.

Đã đối chiếu local-class metadata của receiver/donor cho năm expert gây nhiều break: confuser class đều có trong parent metadata. Ví dụ receiver9 có class13, donor10 có class12; không thể giải thích toàn bộ 13→12 bằng việc không tồn tại negative class13.

## Xác nhận vấn đề donor liveness

Cả hai expert `(9,12)` và `(71,12)` được discover ở Task2/Round0 với donor10. Đã join đủ 20 actual graph snapshots với training history bằng graph SHA256, không dùng graph tương lai để điều khiển training.

- Donor10 chỉ là positive-alpha neighbor ở round0.
- Round1–19, training receipt ghi eligible=False, updates=0, steps=44; receiver vẫn có peer khác sở hữu class12.
- Client48 thuộc positive-alpha graph của **cả hai** receiver trong mọi round1–19.
- Metadata audit chỉ đọc `y_train` của Client48, không load `X_train` hay chạy optimizer: có **3.075 mẫu class12**, vượt min donor32. Đây là thống kê để giải thích run đã có, không dùng historical samples để tạo transfer.

Code `_discover` bỏ qua key đã tồn tại; `_after_round` giữ `entry['donor']` cố định. Khi donor này rời graph, expert không thể dùng donor hợp lệ khác. Đây là thiếu cơ chế donor handoff trong lịch học, không phải graph không có nguồn kiến thức. Chưa có bằng chứng rằng sửa handoff chắc chắn làm expert12 pass về chất lượng.

Không áp một ngưỡng min-updates mới chỉ để loại expert xấu: toàn nhóm25 expert chỉ học44 updates vẫn net +8.972 đúng. Nhóm522 expert880 updates net +555.329, nhóm120 partial khác net +38.956. Loại blanket partial experts có thể làm mất ích lợi.

## Thứ tự sửa tiếp theo

1. Giữ scoped fusion làm hướng đã có gain; bỏ unconditional global override khỏi cấu hình thử nghiệm chính. Muốn dùng multiclass trong notebook cần contract/config riêng, không coi notebook binary đã tự được cập nhật.
2. Sửa donor handoff theo graph **hiện tại**: khi primary donor không hợp lệ, chọn donor khác có current-task TRAIN count≥32 theo deterministic rule. Tiếp tục cùng expert reference, weights, Adam state và quota; ghi provenance từng round/handoff. Không đọc lại task cũ, không dùng future graph, không hạ threshold/gate.
3. Regression handoff cần kiểm tra old donor rời graph, alternate donor xuất hiện, không có alternate, resume ở boundary đổi donor, training proof qua nhiều donor, native isolation. Ghi một summary count theo round vào file, console vẫn gọn; không log từng gradient/mẫu.
4. Nếu mục tiêu vượt50%, ưu tiên audit task memory coverage và routing T2↔T3. Không chỉ tăng expert capacity/epochs vì ceiling45,52% vẫn giữ nguyên.
5. Thêm availability/global-task-mask-only control trước khi khẳng định gain là kiến thức mới do expert learning: CANDLE trước đây đã có các ca missing metadata nhưng weights có sẵn năng lực. Chưa chạy control này trong audit hiện tại.

Không tự thêm lại validation/acceptance gate và không loại expert theo kết quả TEST. Chưa thay hyperparameters, chưa triển khai handoff hoặc chạy lại full seed.

## Artifacts

Summary công khai: `artifacts/candle_v4_multiclass_scoped_readout_20261024.json`. File local trong `audit_denice/candle_v4_multiclass_scoped_20261024/`:

- `comparison.json`, `summary.json`, `per_client.csv`, `per_class.csv`, `per_expert.csv`.
- `client_<id>_overrides.npz`: chỉ các rows expert can thiệp, gồm truth/predicted route/native/expert/score.
- `routing_ceiling.json`, `coverage_decomposition.json`, `available_task_confusions.json`, `router_freshness_metadata.json`.
- `training_budget_breakdown.json`, `remaining_break_parent_support.json`.
- `donor_graph_liveness.json`, `alternate_donor_count_proof.json`.
- `direct_forward_verification.json`: đối chiếu hai client bằng full forward độc lập với cách cộng/trừ matrix.
