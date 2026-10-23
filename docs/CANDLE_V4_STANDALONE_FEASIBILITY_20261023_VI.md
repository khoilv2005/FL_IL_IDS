# CANDLE V4 — standalone expert feasibility và ReachableCoverage

Cập nhật: đã có [positive–negative aggregation pilot](CANDLE_V4_POSITIVE_NEGATIVE_PILOT_20261023_VI.md)
và coverage qua đủ 20 graph snapshots; kết quả standalone dưới đây giữ nguyên.

## Kết luận

**Có bằng chứng capacity cho MLP 39→64→1 trên ca được thử. Chưa có deployment PASS.**
Expert học tốt class 13 trên FIT của donor. Trên receiver, threshold cố định 0,5
gây false activation lớn với class donor chưa có. Chưa chạy gradient exchange,
native integration, activation gate, CAL acceptance hoặc full training.

## Protocol khóa trước

- Task 2, class 13, receiver 30; graph V1 thật ở Round 19, ξ=0,8.
- Client 9 không phải native neighbor của 30 tại snapshot này nên không dùng cặp 30←9.
- Chọn donor 16, ID nhỏ nhất trong actual neighbors có current positives và đủ count
  FIT/CAL. Collaboration alpha 30←16 = 0,06684574216. Không chọn bằng FIT/TEST scores.
- Receiver/class được chọn vì đây là difficult case đã thấy trong development trước;
  không gọi đây là một benchmark confirmation hoặc ca được chọn mù hoàn toàn.
- Model Linear(39,64)→ReLU→Linear(64,1), 2.625 parameters, FP32 10.500 bytes.
- Train chỉ donor BASE Task 2, tối đa 4.096 unique samples/class; positive/negative
  groups có tổng loss weight bằng nhau. Không dùng receiver data để fit.
- Adam lr=0,001, batch=256, 20 epochs, seed=42, clip gradient norm=5, không weight decay,
  không đổi learning rate, không early stopping hoặc chọn epoch tốt nhất theo FIT.
- Epoch 20 là model cuối đã khóa. Quan sát FIT tại 0/1/5/10/20, không cập nhật dựa vào FIT.
- Threshold 0,5 dùng để báo cáo cố định. FAR@Recall95 dùng threshold từ donor FIT theo
  một rule đã khóa và chỉ là diagnostic; không trở thành deployment threshold.
- **20 epochs standalone, 880 optimizer steps; không phải 20 communication rounds.**
  Khoảng 11 giây trong process sau imports trên RTX 4060 Laptop; không dự báo thời gian full.

Role manifest: `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.
Graph payload SHA-256: `6b6c595caf31be6a8a33d1be307cc5b2c86504b7b4ae643d5a7e6fb1fd4eecc7`.

## Dữ liệu và integrity

| Panel | Class 13 | Negatives | Tổng unique |
| --- | --- | --- | --- |
| Donor BASE train | 4.096 | 7.051 | 11.147 |
| Donor FIT | 5.242 | 750 | 5.992 |
| Receiver FIT | 0 | 4.237 | 4.237 |

Receiver FIT negatives: class 12=516, class 14=3.644, class 15=72,
class 16=3, class 17=2. Donor không có class 12 trong current training panel.
Không có content overlap giữa train và hai FIT panels. Hai FIT pools báo riêng,
không gộp sample counts có thể trùng giữa các owner thành evidence độc lập.

Original compressed shards và role indices được checksum theo manifest trong bước
preparation. Preparation chỉ trích current Task 2 BASE/FIT; trainer mở các file
current-task đã stage, không đọc lại original shards. NPZ decoding lúc preparation
có thể materialize original arrays; không gọi đó là physical I/O chỉ đọc current rows.
Không mở CAL examples, validation hoặc TEST. CAL counts chỉ dùng metadata feasibility.
Không train từ historical task examples.

Input giữ nguyên preprocessing của dataset đã xác minh; không fit scaler mới hoặc dùng
TEST statistics. Chưa thu hồi được fingerprint riêng của scaler/feature-order artifact;
matching shard/metadata authority đủ cho local donor pilot này, chưa hoàn tất input
contract cho distributed deployment. Recall donor FIT không chứng minh target recall
trong miền receiver vì receiver không có target positives.

## Kết quả ở threshold cố định 0,5

| Epoch | Donor recall | Donor FP / 750 | Donor AP | Receiver FP / 4.237 |
| --- | --- | --- | --- | --- |
| 0 | 19,19% | 367 | 0,819346 | 2.337 |
| 1 | 98,95% | 4 | 0,999115 | 497 |
| 5 | 99,64% | 0 | 0,999980 | 507 |
| 10 | 99,71% | 1 | 0,999999 | 508 |
| **20, khóa trước** | **99,77% (5.230/5.242)** | **1** | **0,999999563** | **506** |

AP dùng `sklearn.metrics.average_precision_score`, không phải trapezoidal integration
của PR curve. Donor FIT positive prevalence=87,48%; cần đọc AP cùng prevalence và FAR,
không chỉ nhìn AP gần 1. ROC-AUC cuối=0,999996948.

Donor FAR cuối 1/750 = **0,1333%**, vẫn trên 0,1% nếu áp dụng gate đó trực tiếp.
Một FP nằm ở class 15 (1/45). Không chọn epoch 5 thay epoch 20 để báo kết quả đẹp hơn.

Receiver FAR cuối 506/4.237 = **11,9424%**. Trong đó **505 FP thuộc class 12**,
1 thuộc class 14. Đây là false activations của expert standalone; chưa đo baseline
CANDLE predictions nên không gọi 506 FP là 506 negative breaks.

Learning curve cho thấy donor recall cao xuất hiện sớm nhưng receiver FP không tự
giảm khi tiếp tục chỉ train donor. Điều này hỗ trợ việc kiểm tra receiver-negative
gradients ở bước tiếp theo; chưa chứng minh thêm chúng chắc chắn sửa được vấn đề.

## Diagnostic tại donor recall ít nhất 95%

Threshold 0,9987466335 được lấy từ donor FIT theo order statistic đã khóa:

- Donor: 4.981/5.242 target đúng = **95,021%**, 0/750 FP.
- Receiver: **2/4.237 FP = 0,0472% pooled FAR**.
- Cả 2 FP ở class 12: **2/516 = 0,3876% per-class FAR**, trên gate 0,1% nếu dùng per-class.

Vì threshold được suy từ FIT đã quan sát và chưa có CAL acceptance, không gọi đây là
certificate hoặc deployment PASS. Pooled FAR thấp không được che per-class failures.
Không thay threshold 0,5 trong kết quả primary của standalone bằng diagnostic này.

## ReachableCoverage từ graph thật

| Chỉ số Task 2 / Round 19 | Kết quả |
| --- | --- |
| Active receivers trong graph | 70 |
| Current classes | 12–17 |
| Metadata ownership gaps | 126 |
| Gaps có donor trong positive-alpha native neighborhood | 121 |
| ReachableCoverage theo metadata | **96,03%** |
| Gaps có ít nhất một donor đủ FIT/CAL counts + receiver đủ negative CAL count | 85 |
| Evidence-count eligible / mọi metadata gap | **67,46%** |
| Receivers không có positive-alpha peer | 2 |

Đây là một graph snapshot, không phải trung bình qua 20 rounds hoặc cả 6 tasks.
Denominator là class thiếu dữ liệu current local, chưa xác nhận functional gap.
Count eligibility chưa chứng minh unique acceptance đủ hoặc expert có năng lực.
Vì vậy 96,03% không phải tỷ lệ transfer thành công; 67,46% cũng không phải coverage
được chứng nhận. Snapshot này chưa tạo lý do để mở ngay cross-neighborhood discovery.

## Diễn giải và bước tiếp theo

1. MLP nhỏ không bị bác bỏ về capacity trên ca này; expert có thể học class 13 bằng
   direct input features. Không suy rộng sang mọi class hoặc so causal với V3 dùng donor khác.
2. Donor-only expert chưa đủ để dùng ở receiver; class 12 là negative chưa được donor
   thấy và chiếm phần lớn false activation.
3. Bước tiếp theo: cùng expert/case, kiểm tra positive–negative gradient aggregation
   có contribution thực của receiver BASE, với reference/version và wire accounting rõ.
4. Khóa lại optimizer work cho so sánh đó. Không so 880 minibatch steps ở đây với
   20 single-gradient rounds rồi quy khác biệt cho decentralized training.
5. Chưa bật gate, chưa mở CAL/TEST để cứu kết quả, chưa tích hợp core/native hoặc full.

Core `candle_v4_runtime/` vẫn nguyên bản V1. Prototype standalone nằm ngoài core tại
`tools/pilot_candle_v4_standalone.py`. Peer communication thực nghiệm bằng 0 vì bước
này là local donor training; 420/630 KB là ước lượng cho giao thức tương lai, chưa đo ở đây.

Artifacts:

- `artifacts/candle_v4_standalone_20261023.json`
- `artifacts/candle_v4_reachable_coverage_20261023.json`
- `audit_denice/candle_v4_standalone_20261023/standalone_expert.pt` — local checkpoint, không push Git.

Protocol lock, preparation access và staged data giữ local để tái lập; chỉ metrics,
provenance và source code được đưa vào repository. Chưa có historical safety claim.
