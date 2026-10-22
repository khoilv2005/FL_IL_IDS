# CANDLE V1 — mở khóa head bằng FC1 prototype trên checkpoint

Ngày: 22/10/2026. Nguồn: DeNICE V1 legacy `results (11)`, Task 4 / Round 19,
seed 42, binary cosine router, ξ=0,8. Đây là development audit trên checkpoint,
không phải kết quả một run CANDLE vừa train lại hoặc full TEST.

## 1. Kết luận chính

**Client 7 có thể dự đoán class 24 bằng input X mà không cần oracle task hoặc
memory Conv/GRU của donor.** Direct head của donor 11 kết hợp cổng FC1 prototype
đạt 97,27% recall, 249 rescue, 0 false activation và 0 break trên panel development
ở cấu hình phụ q=1% đã khóa trước khi mở panel trong lượt này.

Nhưng cần giữ ba kết luận riêng biệt:

1. Cấu hình chính q=5% chỉ đạt 94,53% recall, dưới gate 95%. Không đổi tên q=1%
   thành cấu hình chính sau khi thấy kết quả. Cả hai primary đều FAIL gate recall.
2. Prototype-only cho prediction giống hệt head + gate trên panel này. Vì vậy
   chưa chứng minh head mang lại lợi ích thêm ngoài một prototype classifier.
3. 0 break trên panel hữu hạn không phải bảo đảm an toàn toàn bộ class/phân bố.
   Chưa install vào training runner, chưa chạy survival hoặc full campaign.

Không optimizer, không epoch, không train backbone/head, không merge binary
memory, không đọc final TEST. Source model/router giữ nguyên complete-state hash.

## 2. Dữ liệu và thứ tự thực hiện

- Hai case giữ nguyên từ audit trước: `7 ← 11 / class 24` trong neighborhood
  (alpha 0,184805); `65 ← 11 / class 24` ngoài neighborhood (alpha 0).
  Case 65 là counterfactual cho thiết kế trao đổi mới, không phải edge native hợp lệ.
- Load full FP32 terminal và graph đúng Round 19 từ archive; role manifest SHA:
  `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.
- Fitting/selection chỉ dùng **current CAL Task 4**, split theo content hash;
  không đọc raw historical CAL. CAL-FIT trong report này là phần con của CAL,
  khác với original META-FIT role.
- Exploratory sweep: 6.112 candidates/case; chọn trên CAL selection rồi khóa
  trước khi đọc validation. Historical validation chỉ dùng để đánh giá lỗi.
- Sau khi thấy validation failure, thiết kế một amendment: threshold là phân vị
  positive CAL-FIT q∈{0; 0,01; 0,05; 0,1}; q=0,05 là primary. Head/gate primary
  theo receiver và 40 ablations/case được ghi trong lock trước khi mở META-FIT.
- Amendment dùng original META-FIT **chỉ để chấm điểm**, không fit head,
  prototype hoặc threshold bằng role này. Mỗi class cap 256, dedup theo content;
  kiểm tra không overlap với CAL. Positives từ donor 11; negatives từ những
  class receiver thực sự có local training support đến Task 4.
- Đây là development đã qua nhiều vòng audit. Role khác nội dung không đồng
  nghĩa pristine confirmation; không tuyên bố đánh giá khoa học độc lập.
- Đọc historical validation/META-FIT là đánh giá offline có kiểm soát, không
  phải dữ liệu được phép dùng lại cho production adaptation.

CAL-FIT: receiver 7 có 64 target positives và 33 negatives; receiver 65 có
64 positives và 71 negatives. Selection CAL tách riêng có 64 positives mỗi
case và 44/60 negatives. Cỡ mẫu nhỏ không đủ chứng nhận historical safety.

## 3. Thử nhiều cơ chế và kết quả exploratory

Các head: availability control, direct mixing eta=0,25/0,5/1; norm matching;
norm+bias matching; diagonal moment transport. Các policy: native router;
Top-2/3 với raw, weight-normalized hoặc cosine scores; targeted gates bằng
donor/receiver FC1 prototype, weight cosine, donor/receiver FC1 binary sketch.
Threshold/margin lấy từ FIT, chọn trên CAL selection, không chọn từ validation.

| Winner từ CAL selection | Client 7 | Client 65 |
|---|---:|---:|
| Recall trên validation | 99,61% | 98,44% |
| Rescue | 255 | 252 |
| Break | **644** | **92** |
| False activation | 1.050/2.832 | 140/1.794 |
| Observed FAR | 37,08% | 7,80% |

Đây là failure cần giữ lại trong báo cáo: current-task negatives không đại diện
đủ các class cũ. Điểm prototype phân biệt tốt vẫn có thể đi kèm threshold quá lỏng.

### Task 4 thực sự nằm thứ mấy?

Trên 256 target validation samples mỗi case, xếp hạng raw cosine native router:

| Task-4 rank | Client 7 | Client 65 |
|---|---:|---:|
| Rank 1 | 1 | 0 |
| Rank 2 | 1 | 0 |
| Rank 3 | 2 | 0 |
| Rank 4 | 0 | 256 |
| Rank 5 | 252 | 0 |

Vì vậy Top-2/3 không mở được phần lớn class 24 ở hai case này. Không thể mặc
định episode đúng nằm ngay sau Top-1.

### FC1 cosine và norm

Validation AUROC target-vs-negative: donor prototype 0,998674 ở Client 7 nhưng
chỉ 0,715358 ở Client 65; receiver-space prototype lần lượt 0,998270 và 0,992671.
Đây là diagnostic, không dùng các nhãn validation để đặt threshold amendment.

Donor effective row norm = 0,884013, bias = 0,067567. Client 7 current rows 25–28
có norm 0,501092–0,657698; Client 65 rows 25–27 có norm 0,344769–0,393700.
Chênh lệch norm có thật nhưng không chứng minh normalize sẽ giải quyết routing.

Oracle recall cao **không tự suy ra** cosine prototype cao, feature alignment
toàn cục hay negative safety. Những đại lượng này phải đo riêng như trên.

## 4. Amendment: cổng phân vị positive FIT

Rule cho mọi input, không dùng nhãn thật:

1. Giữ native router để có fallback prediction.
2. Tính FC1 bằng context 4 cố định trong metadata capability, cho mọi input.
3. Thay target score bằng row ứng viên; yêu cầu class 24 thắng competition
   với các class native đã đăng ký trong context 4.
4. Chỉ trả class 24 nếu cosine FC1 với prototype ≥ tau; còn lại trả fallback.

Context 4 là thông tin class-to-task của capability, không phải oracle task
được cấp theo sample. Policy này thay đổi inference và có thể gây false activation,
dù không mutate native router memory.

Primary Client 7 dùng direct donor row và donor FC1 prototype. Client 65 dùng
receiver prototype và diagonal moment transport fit trên paired positive CAL-FIT:

```
a = clip(std(h_D) / max(std(h_R), 1e-3), 0.25, 4)
o = mean(h_D) - a * mean(h_R)
w_R = w_D * a
b_R = b_D + dot(w_D, o)
```

Đây là biến đổi row theo moments, không phải bằng chứng neuron alignment hoàn
chỉnh; không tối ưu gradient. Receiver reference cần có ở donor để tính các
receiver features. Ngay cả gate donor-prototype của Client 7 cũng cần receiver
features trên positive FIT để hiệu chuẩn tau.

### Kết quả trên META-FIT evaluation

Mỗi case có 256 target positives. Client 7: 2.740 negatives, gồm 2.292 historical
rows thuộc class <24. Client 65: 1.711 negatives, gồm 1.274 historical rows.

| Receiver / fixed configuration | Recall | Rescue | FP | Break | Panel accuracy |
|---|---:|---:|---:|---:|---:|
| 7 — native baseline | 0% | 0 | 0 | 0 | 52,50% |
| 7 — **primary q=5%** | **94,53%** | 242 | 0 | 0 | 60,58% |
| 7 — secondary q=1% | 97,27% | 249 | 0 | 0 | 60,81% |
| 65 — native baseline | 0% | 0 | 0 | 0 | 44,28% |
| 65 — **primary q=5%** | **92,58%** | 237 | 0 | 0 | 56,33% |
| 65 — secondary q=1% | 96,09% | 246 | 0 | 0 | 56,79% |

Baseline rescue/break bằng 0 theo định nghĩa so với chính baseline. Panel
accuracy không phải accuracy full TEST hoặc metric toàn hệ thống.

Giữ nguyên gate recall≥95%, observed FAR≤0,1%, negative break=0: primary 0/2
PASS; các secondary đạt empirical gate là 8/40 ở Client 7 và 1/40 ở Client 65.
Không chọn lại primary hoặc gọi mọi ablation đạt gate là một phương pháp đã xác nhận.

## 5. Đối chứng cơ chế bắt buộc

Giữ nguyên prototype và threshold đã khóa, dùng lại cùng META-FIT panel. Đây
là phân tích hậu nghiệm về cơ chế, không phải một evaluation độc lập mới.

| q=1%, cùng gate | Client 7 recall | Client 65 recall |
|---|---:|---:|
| Original masked receiver row | 0% | 0% |
| Receiver row + donor connection mask | 0% | 0% |
| Receiver row, unmask toàn bộ connections | 0% | 0% |
| Direct donor row | 97,27% | 0% |
| Frozen capability: direct / moment-transport | 97,27% | 96,09% |
| **Prototype gate alone, bỏ head competition** | **97,27%** | **96,09%** |

Tất cả dòng trên có 0 FP và 0 break trên panel này. Prototype-only và frozen
capability có predictions giống hệt ở cả q=1% và q=5%.

Do đó có bằng chứng direct donor head hoạt động trong local feature space của
Client 7, và moment-adjusted row hoạt động ở Client 65. Nhưng **chưa có bằng chứng
head + gate vượt prototype-only**. Không quy toàn bộ gain cho weight transfer.
Đây là đối chứng quan trọng trước khi tuyên bố novelty hay tích hợp production.

## 6. Runtime và communication

`tools/candle_head_unlock_runtime.py::predict_packet()` chỉ nhận X, capability,
model/router và declared seen classes; không nhận y hoặc task thật của input.
Replay 4 packets q=5%/1% trên hai receiver khớp từng prediction với kết quả đã khóa.
Không donor model inference, không thay original parameters/router/memory.

Đây là sidecar scoring của experimental head, chưa phải transactional installer
ghi row/masks vào checkpoint và quản lý lifecycle. Một local model nhưng có thể
cần thêm context-specific forward; không tuyên bố miễn phí hoặc chỉ một forward.

| Serialized inventory | Client 7 | Client 65 |
|---|---:|---:|
| FP32 row+bias+one prototype, raw | 2.052 B | 2.052 B |
| Capability NPZ + metadata | 2.595 B | 2.640 B |
| Receiver function reference | 7.374.584 B | 6.868.664 B |

Setup loại raw reference-input memory, nhưng vẫn gồm model/router algorithm
state. Chỉ được tái dùng reference nếu exact version hợp lệ đã có ở donor.
Đây là số bytes serialize trong harness, **không phải full network traffic**.
Chưa tính đầy đủ native exchanges, discovery, responses và wire headers.

## 7. Giới hạn và bước tiếp theo

- Positives là donor-origin, chưa chứng minh receiver-domain generalization.
- Negatives không phủ mọi class; có class rất ít mẫu. Không có population FAR
  guarantee hoặc strict cumulative safety certificate từ 0 lỗi quan sát được.
- Shared FC1 prototype có tín hiệu tốt ở cặp intra nhưng không mặc nhiên portable
  sang mọi receiver. Binarize donor prototype cũng không tự chuyển hệ quy chiếu.
- Không tiếp tục chỉnh tau từ các panel vừa xem. Nếu tiếp tục, khóa candidate
  cùng prototype-only control rồi đánh giá evidence độc lập có coverage phù hợp.
- Chỉ sau đó mới xét install/survival. Chưa sửa runner hoặc train lại T0–T5.

## 8. Tái lập và artifacts

```
.venv-denice-gpu\Scripts\python.exe tools/audit_candle_v1_head_unlock.py --out audit_denice/head_unlock_new
.venv-denice-gpu\Scripts\python.exe tools/audit_candle_v1_positive_quantile_gate.py --out audit_denice/positive_quantile_new
.venv-denice-gpu\Scripts\python.exe tools/replay_candle_head_unlock_packets.py --experiment audit_denice/positive_quantile_new --out audit_denice/replay_new
.venv-denice-gpu\Scripts\python.exe tools/audit_candle_v1_head_unlock_controls.py --experiment audit_denice/positive_quantile_new --out audit_denice/controls_new
```

Commands dùng local inputs mặc định trong repo; checkpoint/data không được
commit. Scripts không ghi đè output directory đã tồn tại.

Local receipts: `audit_denice/candle_v1_head_unlock_20261022_v2`,
`audit_denice/candle_v1_positive_quantile_20261022_v1`,
`audit_denice/candle_v1_head_unlock_replay_20261022_v1`,
`audit_denice/candle_v1_head_unlock_controls_20261022_v1`.
Public compact receipt: `artifacts/candle_v1_head_unlock_development_20261022.json`.
