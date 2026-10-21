# CANDLE V1 — Target-Free Anchor Calibration, Test A/B

Ngày: 22/10/2026. Tiếp nối `f7bdeda`, cùng checkpoint legacy Task 4 / Round 19.
Không optimizer, không train lại, không final TEST, không merge binary memory.

## 1. Kết luận

**Công thức chính chưa đạt cả Test A lẫn Test B.** Có một tín hiệu phụ đáng giữ:
Client 7 dùng mean-shift prototype và ngưỡng chỉ từ background đạt 98,44% recall
trên validation tail mới, nhưng FAR=0,521%, 1 break. Không điều chỉnh threshold
để biến kết quả gần đạt thành PASS.

| Primary, validation tail mới | Recall target | FP | Break |
|---|---|---:|---:|
| Client 7 / class 24, Test A | 0% | 0 | 0 |
| Client 65 / class 24, Test A | 0% | 0 | 0 |
| Client 23 / classes 24,25,26, Test B | 0% / 3,125% / 1,5625% | 21 | 7 |

Không có variant nào trong ma trận đã khóa đạt recall từng target≥95%, pooled
FAR≤0,1%, break=0 trên panel mới. Cũng không variant nào đồng thời đạt
recall từng target≥95%, FAR≤0,5%, break=0 ở panel này.

Kết quả bác bỏ các cấu hình anchor calibration đã thử, **không chứng minh**
zero-shot transfer ngoài neighborhood bất khả thi hoặc mọi giải pháp đều phải
căn chỉnh trong training. Client 23 và donor ở Test B đều là intra-neighborhood.

## 2. Điều cần phân biệt với các giả thuyết ban đầu

- Trong `f7bdeda`, primary bằng donor-only vì tau_D lớn hơn tau_N ở mọi case.
  Recall0 lúc đó không thể quy cho một max lấy từ 8.125 negatives: tau_N chỉ
  được fit trên 77/21/131 current CAL negatives; 8.125 là evaluation negatives.
- Cosine qua hai encoder không bắt buộc giảm hoặc giảm cùng một lượng cho mọi
  class. Ở audit này, attenuation sau mean-shift còn có thể âm. Không clamp nó
  về số dương để phù hợp giả thuyết.
- 206 multi-gate matches trước đó gồm 44 target samples và 162 negatives.
  Chúng chưa chứng minh representation của ba target thực sự collapse. Nhiều
  gate quá lỏng và background không đại diện cũng tạo ra hiện tượng này.
- Centering không bảo đảm phục hồi thông tin phân biệt hoặc loại hết giao thoa;
  z-score của một score không phải full multivariate Mahalanobis distance.
- Receiver-space prototype trong lượt cũ được compile bằng receiver function
  trên donor current positives và đã được khai báo. Đó là protocol khác,
  không phải âm thầm dùng positive data local của receiver. Lượt này áp ràng
  buộc mạnh hơn: **không target nào chạy qua receiver trước evaluation**.

## 3. Case selection và nguồn neo

Chỉ sử dụng common classes thực sự có current local BASE ở cả hai phía.
Không mở raw historical BASE/CAL. Các views `CurrentBaseData` và
`CurrentCalibrationData` bind owner, Task 4 và role-manifest hash.

| Receiver | Donor | Targets | Current anchors |
|---|---|---|---|
| 7 | 11 | 24 | 25,26,27 |
| 65 | 11 | 24 | 25,26,27 |
| 23 | 13 | 24,25,26 | 27,29 |

Donor 13 là common mature intra donor của cả ba target, có hai current anchors
đủ BASE support. Trong các common donors có hai anchors≥32 BASE rows, nó có
minimum target BASE support lớn nhất (480). Chọn bằng metadata trước scoring,
không lấy kết quả của donor cũ rồi đổi donor theo accuracy.

Thay đổi donor ở Test B được báo rõ: không gọi chênh lệch với `f7bdeda` là
ablation chỉ đổi centering. Các scorer trong run mới dùng cùng donor, cùng
prototype, cùng data và cùng gates để so sánh trực tiếp với nhau.

BASE chia bằng content-hash parity, cap256/class/half. Anchor cần ≥32 BASE
rows tổng và ít nhất8 unique samples mỗi half. FIT tạo prototype; half còn
lại đo anchor attenuation, không fit trên evaluation. Đây vẫn là BASE, không
được đổi nhãn thành CAL acceptance evidence.

Receiver 7/65 có 3 anchors; donor11/class27 chỉ có 19 FIT +14 hold samples.
Receiver23/donor13/class27 có 50+44 / 31+39, class29 mỗi phía cap256+256.
Hai/ba class means không xác định được phép căn chỉnh tổng quát của 256 chiều.
Lượt này không fit Procrustes hoặc một mapper 256×256.

Donor target prototype từ current BASE FIT. Target mean/std từ donor current
CAL (min8 unique); donor13/class26 chỉ có10 CAL samples nên moments có giới hạn
độ tin cậy. Receiver background chỉ từ current CAL negatives: 160, 252 và55
rows tương ứng Client7,65,23. Không dùng validation để chọn threshold.

## 4. Công thức đã khóa

Với P_R/P_D là các anchor means:

```
center_R = mean(P_R); center_D = mean(P_D)
Delta = center_R - center_D
raw:        score = cos(h_R, p_D)
mean_shift: score = cos(h_R, p_D + Delta)
centered:   score = cos(h_R - center_R, p_D - center_D)
```

Tại mỗi geometry, đo donor self-score và receiver cross-score trên các anchor
hold halves; delta_cross là trung bình chênh lệch của các class means đó.

```
tau_positive = mu_D_target - delta_cross - 2 * sigma_D_target
tau_background = quantile(receiver_background_score, 0.999, method='linear')
tau_primary = max(tau_positive, tau_background)
```

Hai controls khóa trước: background-only và donor-unadjusted `mu_D−2σ`.
Với ít background samples, q99,9% vẫn gần max; không thể coi đây là bảo đảm
robustness trước outlier hoặc chứng nhận tail FAR0,1%.

Scorers dùng cùng gates: prototype cosine; background-normalized prototype z;
affine head z; column-centered sub-head z. Z-score dùng background std floor1e-6.
Không giả định score distribution Gaussian hoặc so z-score như probability.

**Transform của head:** nếu h_R≈h_D+Delta thì w_R=w_D,
b_R=b_D−dot(w_D,Delta). Không cộng Delta vào classifier weight như với prototype.
Trong centered coordinates, bias được đổi tương ứng. Column-centering sub-head
trừ cùng mean weight vector khỏi mọi class; riêng phép này giữ raw pairwise
logit margins. Z-normalization theo class mới có thể đổi thứ tự ranking.

Primary A: mean_shift + primary threshold + prototype cosine.
Primary B: centered + primary threshold + prototype z; so với sub-head z dùng
chung gate. Mỗi geometry/policy/scorer được ghi trong lock trước evaluation:
3×3×4=36 variants/case, không chọn lại primary sau khi xem dữ liệu.

Nếu không gate nào mở, giữ native prediction. Nếu có gate, chọn scorer argmax
trong imported candidates đã qua gate. Không dùng true task/class ở inference.

## 5. Evaluation và mức độ độc lập

1. Paired panel: khôi phục chính xác 8.381 rows /34 class của `f7bdeda` để so
   trước/sau và xác nhận lại 206 collisions. Đây là panel đã xem, không confirmation.
2. Fresh tail: 3.775 rows /31 class, loại toàn bộ hashes của paired panel và các
   validation prefixes đã dùng trước đó. Mỗi target có128 samples. Class0,28,31
   không còn content mới; class33 chỉ còn3 samples. Không gọi đây là full34-class
   independent validation hoặc evidence đủ rộng cho mọi class.

Cả hai là validation role đã tách global content khỏi BASE/CAL. Historical
validation chỉ dùng đánh giá offline, không tham gia adaptation. Class30–33 là
future-class stress ở checkpoint T4 và luôn có metrics tách riêng.
Development source đã qua các audit khác; fresh tail trong chuỗi này không bảo
đảm hoàn toàn chưa được xem ở mọi nghiên cứu trước đó.

## 6. Test A: attenuation và mean-shift

| Receiver | Raw delta_cross | Mean-shift delta_cross | Mean-shift tau primary |
|---|---:|---:|---:|
| 7 | 0,1317 | −0,0004 | 0,9709 |
| 65 | 0,4652 | −0,0034 | 0,9738 |

Mean-shift làm anchor means phù hợp hơn, nhưng mức phù hợp đó không generalize
đủ sang target24. Primary recall vẫn0 ở cả paired lẫn fresh tail.

Secondary **mean-shift + background-only**, Client7:

| Panel | Recall24 | Rescue | FP / negatives | FAR | Break |
|---|---:|---:|---:|---:|---:|
| Paired | 99,61% | 255 | 29/8.125 | 0,357% | 0 |
| Fresh tail | **98,44%** | **126** | **19/3.647** | **0,521%** | **1** |

Đây là tín hiệu transfer target-free có thật trong một secondary đã khóa,
nhưng không qua strict gate hoặc đồng thời qua FAR0,5% và break0 trên fresh tail.
Break mới thuộc class4; FP trải trên nhiều class. Không đặt lại q để cứu run.

Client65: mean-shift background-only recall0 trên fresh tail; centered
background-only đạt96,88% recall nhưng gây491 FP và167 break. Vì vậy recall cao
khi đổi geometry vẫn chưa tương đương safe transfer ở cặp này.

## 7. Test B: centering + background z-score

Primary Client23:

| Panel | Recall24 /25 /26 | FP | Break | Multi-gate matches |
|---|---|---:|---:|---:|
| Paired | 0% /3,125% /4,297% | 53 | 12 | 0 |
| Fresh tail | 0% /3,125% /1,563% | 21 | 7 | 0 |

206→0 multi-gate matches không phải quality PASS: target recall cũng gần bằng0.
Prototype/sub-head cho cùng prediction ở primary vì mỗi input còn tối đa một
gate mở. Accuracy trên chính 206 mẫu cũ chỉ3/206, không cho thấy disambiguation
được giải quyết.

Ở secondary centered + donor-unadjusted (nhiều gate còn mở), fresh tail:

| Scorer | Recall24 /25 /26 | FP | Break |
|---|---|---:|---:|
| Prototype z | 0% /91,41% /69,53% | 3.248 | 934 |
| Centered sub-head z | 0% /96,09% /49,22% | 3.248 | 934 |

Có734 collisions. Head tăng recall25 nhưng giảm recall26; không thắng đồng đều
và cả hai gây negative transfer rất lớn. Không chọn “head tốt hơn” chỉ từ một
class, cũng không chốt prototype thay thế head từ một primary đóng gần hết cổng.

## 8. Quyết định

- Các tham số đã khóa của Test A/B chưa đạt; không sửa production runner.
- Giữ secondary Client7 làm evidence rằng current-anchor mean-shift có thể giúp
  mở target mà không chạy target qua receiver trong lúc fitting. Chưa đủ để triển khai.
- Thất bại ở Client65 giới hạn ở donor/checkpoint và họ phép biến đổi đã thử;
  chưa là định lý rằng ngoài neighborhood bắt buộc phải training-time alignment.
- Hai/ba anchors không bảo đảm target extrapolation. Cả lỗi gate, score calibration,
  donor compatibility và feature discrimination vẫn cần được phân biệt; không gọi
  chúng là hai “giới hạn cứng” đã chứng minh chỉ từ collisions/recall0.
- Không tiếp tục chọn threshold trên các evaluation panels vừa xem. Một thiết kế
  tiếp theo cần protocol và evidence riêng trước khi đánh giá.

## 9. Tái lập

```
.venv-denice-gpu\Scripts\python.exe tools/audit_candle_v1_anchor_calibration.py --out audit_denice/anchor_calibration_new
```

Default inputs là legacy checkpoint, current BASE/CAL stores và roles đã có local.
Output gốc: `audit_denice/candle_v1_anchor_calibration_20261022_v1`.
Runtime khoảng46 giây trên RTX4060 Laptop. Source complete-state hashes không đổi;
packet locks giữ nguyên; source snapshots và panel content receipts lưu local.
Single-process harness không phải network protocol; chưa đo full wire traffic.
Prototype/anchor/stats messages vẫn có bytes, không tự coi chúng đã tồn tại miễn
phí trong capsule đúng version.

Compact receipt: `artifacts/candle_v1_anchor_calibration_20261022.json`.
