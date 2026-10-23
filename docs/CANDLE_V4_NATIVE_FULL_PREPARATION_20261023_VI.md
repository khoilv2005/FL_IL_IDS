# CANDLE V4 — tích hợp native coverage-aware và chuẩn bị full development

## Quyết định

GO để chạy full development seed42 trên Kaggle. Protocol mới không yêu cầu
hai expert ACTIVE trước khi chạy; số ACTIVE và tỷ lệ rejection là kết quả.
Giữ recall≥95%, observed FAR từng owner–class≤0,1%, negative break=0.
Không mở thêm thí nghiệm tìm LR, threshold, architecture hoặc donor thắng.

## Đã triển khai

- `FullCoverageService`: discovery nhiều receiver/class sau native aggregation,
  xét positive-alpha neighbors của round hiện tại. CoverageService cung cấp
  gradients cho expert theo đúng cùng reference.
- Receiver/donor negatives và auxiliary negatives theo missing-class coverage,
  minimum owner ID, ít nhất32 BASE samples; không hardcode class12/13.
- Ba nhánh cùng initialization và lịch cập nhật: coverage-aware, no-aux và
  donor-only class-balanced. V1 native là đối chứng dùng chung, không bị expert
  thay đổi weights, masks, optimizer hay router.
- Quota44 updates/round, catch-up≤88;128positive+128negative/update.
  Nếu không có negatives hợp lệ cho donor-only, các nhánh cùng defer cơ hội đó.
- Gate atomic trên toàn bank pending cũ và draft mới, arbitration cố định bằng
  normalized threshold excess; không chọn subset theo CAL để cứu acceptance.
- Chuyển task đóng quyền đọc historical BASE/FIT/CAL. Expert cũ giữ weights,
  threshold; PENDING_NEW_TASK cần kiểm tra negatives hiện tại trước ACTIVE.
  Rủi ro historical chưa quan sát vẫn UNKNOWN, không có chứng nhận an toàn rộng.
- Continuation atomic mỗi round lưu cả expert, Adam, quota, ledger, native RNG.
  ZIP từng task có continuation, run lock và threshold locks đúng thư mục.
- Report development: Accuracy, Macro-F1, recall từng class, forgetting, từng
  client, status/steps/hash expert, FP/FAR, rescue/break, collisions, bytes,
  runtime và thời gian inference tăng thêm.

## Regression kỹ thuật

11/11 tests PASS bằng dữ liệu tổng hợp. Có gradient thật và Adam thật;
checkpoint save/load rồi chạy tiếp cho state khớp tuyệt đối. Kiểm tra fingerprint
model/router NICE thật không đổi khi gọi service, nhiều receiver và class,
graph biến đổi, quota catch-up, budget, transition, simulated ACTIVE/collision,
threshold tamper và archive restore.
Gate journal cho phép khôi phục quyết định đã commit mà không đọc CAL lại;
nếu chỉ có threshold lock do bị ngắt giữa gate thì defer, không thử CAL lại.

ACTIVE trong regression là fixture score có kiểm soát, không phải quality PASS
trên CAL thật. Các test này không chứng minh full100-client runner đã hoàn tất
hay mức cải thiện accuracy tự nhiên. Không chạy lại smoke nghiên cứu chất lượng.
Receipt: `artifacts/candle_v4_full_integration_regression_20261023.json`.

## Notebook và protocol

`train_candle_v4_kaggle.ipynb` clone private runtime từ GitHub ở SHA được pin.
Bật GPU và Internet; dataset mặc định:
`/kaggle/input/datasets/khoilv2005/100-clients`.

Giữ profile V1: một GPU, native batch2048, AMP, split seed20261006 với
BASE80%/FIT8%/CAL2%/validation10%, không phải100% raw TRAIN. Expert dùng CPU.
Sáu task ×20 rounds, seed42. Mỗi task chỉ học dữ liệu current task.

Đánh giá cumulative VALIDATION sau mỗi task trên shard của từng receiver,
pooled metrics cho V1 và ba nhánh. Không tự động mở TEST. Cấu hình này là
development; CAL đã có lịch sử development. Đánh giá TEST cuối cần protocol
riêng được khóa; không dùng kết quả TEST để chọn thiết kế.

Application-wire cap128MiB/expert/arm,64GiB/arm; giữ10% để discovery/gate.
Mức trần10200 expert cho100receiver/34class/3arm tương đương107,1MB FP32 weights
và khoảng214,2MB Adam moments, chưa gồm objects/native models/cache.
Native traffic được ghi dưới dạng dense payload upper-bound; tổng với expert
wire là estimate, không phải physical network measurement.12GiB output guard
giữ nguyên. Không bỏ chi phí verification khỏi ledger.

Output chính: `results_candle_v4_seed42/candle_v4_development_report.json`,
`candle_v4_completion.json`, `continuation_state_latest.pt` và các
`checkpoint_task_*_all_rounds.zip`. Dùng Kaggle Save Version để giữ output.

## Phạm vi hoàn thành

Runtime, protocol và notebook đã được chuẩn bị cho full. Full seed42 chưa được
khởi chạy từ máy local. Môi trường hiện tại không có Kaggle CLI/credentials để
gửi job; notebook là điểm chạy trên Kaggle, không có kết quả full mới để báo cáo.
