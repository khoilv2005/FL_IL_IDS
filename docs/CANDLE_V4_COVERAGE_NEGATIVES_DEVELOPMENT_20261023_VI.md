# CANDLE V4 — coverage-aware negative learning: development có đối chứng

**Có tín hiệu tích cực trên FIT-assess; chưa mở CAL, ACTIVE hoặc full training.**
Theo endpoint primary đã đăng ký, bổ sung negative class còn thiếu giảm class13→12
false activation từ **6/2.146 xuống 0/2.146**, tăng recall class12 từ **97,38% lên
99,44%**, và tăng **101 dự đoán đúng / 11.586 unique assessment rows**, không tăng break.

Đây là một development experiment mới, không thay kết quả bank REJECT tại `644d116`.
Không mở lại ba mẫu CAL đã gây reject và không claim chúng đã được sửa.

- [Protocol đăng ký](../plans/candle_v4_coverage_negatives_development_20261023.md)
- [Receipt đầy đủ](../artifacts/candle_v4_coverage_development_20261023.json)

## 1. Rule tổng quát, không hardcode class12/class13

Tại mỗi recorded current graph:

1. Primary donor và receiver vẫn cung cấp positives/negatives như V4 hiện tại.
2. Lấy các negative classes có **0 current BASE samples** trong hợp của hai bên.
3. Trong current eligible neighbors, chọn owner ID nhỏ nhất có ≥32 BASE samples
   cho từng class thiếu; chỉ request gradients của đúng negative class đó.
4. Mọi peer dùng cùng receiver-owned expert reference/version. Receiver sum gradients
   một lần, global clip5 rồi Adam; không truyền raw samples hoặc average Adam states.

Fixture giữ nguyên: receiver6, primary donor1/class12 và donor2/class13; owners
`{1,2,3,6}`. Rule tự chọn **peer2/class13 cho expert12**. Expert13 đã có mọi current
negative class nên không thêm contributor; weights của nó giữ nguyên so với control.
Owner3 được dùng để đánh giá, không được chọn đặc biệt để fit ba lỗi CAL.

Contributor selection chỉ dùng current graph và BASE count metadata, không FIT/CAL
scores. Đây là offline replay graph T2 rounds0–7, không phải native training mới.

## 2. Work matching và implementation checks

| Arm | Training |
| --- | --- |
| Current V4 | Gọi nguyên original balanced joint trainer |
| Coverage-aware | Generic multi-peer implementation, thêm missing-class negatives |
| No-extra ablation | Cùng generic implementation, tắt phần bổ sung |

Mỗi arm có hai expert; mỗi expert khởi tạo seed42 giống nhau, MLP39→64→1 FP32 CPU,
Adam0,001/weight_decay0/clip5, **352 steps**, **128 positive + 128 negative/step**.
Tổng **90.112 lượt mẫu/expert**, gồm 45.056 positive và 45.056 negative. Không tăng
updates, tổng samples hoặc thay backbone/native optimizer.

Trong expert12 coverage-aware, 11.141 negative sample presentations đến từ peer2/class13,
thay một phần 45.056 negatives hiện có. Extra negative gradients có norm>0 ở cả352 steps.

Checks:

- Original control tái hiện **đúng hashes** của hai expert native smoke trước.
- No-extra ablation khớp **weights và assessment metrics tuyệt đối** với current V4.
- Expert13 không thêm nguồn: weights/hash khớp giữa cả ba arms.
- Mọi generic update giữ đúng128 positive/128 negative, finite gradients/parameters.
- Remote native function reference phục hồi cho cùng predictions chính xác.

Việc thêm class negative cũng phân bổ lại class-balanced objective: từ bốn negative
classes thành năm cho expert12. Thí nghiệm này không tách riêng hiệu ứng nguồn dữ
liệu mới khỏi hiệu ứng redistribution của fixed negative budget.

## 3. Data và partition

Chỉ đọc staged current T2 BASE/FIT của fixture. BASE sau original native local-validation
split10%, dedup/cap như trước. Không đọc CAL partition, không sử dụng raw rejected
CAL samples để train, threshold tuning hoặc confirmation.

Sau khi **mọi arm đã train xong**, chia FIT bằng SHA256 của input FP32, owner-independent:
first8 bytes big-endian modulo2; nhóm0 chọn threshold, nhóm1 assessment. Những input
trùng nhau giữa owners luôn vào cùng fold.

| Owner | FIT-select rows | FIT-assess rows |
| --- | ---: | ---: |
| 1 | 798 | 805 |
| 2 | 7.776 | 7.761 |
| 3 | 2.865 | 2.868 |
| 6 | 254 | 269 |

Pooled assessment bỏ117 cross-owner duplicate `(feature hash,label)` records, còn
**11.586 unique rows**. Owner pools được báo riêng; các paired owner events không
được gọi là unique samples khi chưa dedup.

FIT đã có lịch sử development. Partition này giúp tách threshold selection khỏi
assessment trong experiment, **không biến FIT thành confirmation holdout độc lập**.

## 4. Endpoint primary: cùng rule FIT-select recall95

Mỗi expert dùng donor FIT-select để khóa threshold theo cùng order-statistic rule;
assessment không tham gia chọn threshold, round hoặc variant.

| Metric, pooled unique FIT-assess | Current / no-extra | Coverage-aware |
| --- | ---: | ---: |
| Accuracy | 7.084/11.586 = **61,14%** | 7.185/11.586 = **62,01%** |
| Macro-F1, current labels12–17 | **50,67%** | **50,88%** |
| Class12 recall | 4.727/4.854 = **97,38%** | 4.827/4.854 = **99,44%** |
| Class13 recall | 2.075/2.146 = **96,69%** | 2.076/2.146 = **96,74%** |
| Expert12 activation sai trên class13 | **6/2.146 = 0,280%** | **0/2.146** |
| Expert13 false activation | **0** | **0** |
| Rescue so native fixed predictor | **6.802** | **6.903** |
| Break so native fixed predictor | **0** | **0** |
| Multi-gate rows | **1** | **0** |

So current control: **+101 correct unique predictions**, +0,872 điểm phần trăm
accuracy. Paired owner-event analysis cũng ghi101 gained /0 lost trong primary.

Required donor assessment recall:

| Expert / owner | Current | Coverage-aware |
| --- | ---: | ---: |
| Class12 / donor1 | 71/72 = 98,61% | 72/72 = 100% |
| Class13 / donor2 | 895/928 = 96,44% | 895/928 = 96,44% |

Class13→12 FP theo owner:

- Owner2: **2/928 → 0/928**.
- Owner3: **4/1.233 → 0/1.233**.

Đây là các FIT-assess rows khác bộ CAL cũ; không phải ba lỗi CAL đã được kiểm tra lại.

Threshold class12 thay đổi tự nhiên theo score distribution: current `0.9797241688`,
coverage-aware `0.9028619528`. Không điều chỉnh hai con số này bằng assessment/CAL;
cùng một rule recall95 trên FIT-select tạo ra chúng. Class13 threshold và weights
không đổi: `0.9856781363`.

## 5. Secondary đã khóa: threshold0,5

| Metric | Current / no-extra | Coverage-aware |
| --- | ---: | ---: |
| Correct /11.586 | 7.256 | 7.262 |
| Accuracy | 62,63% | 62,68% |
| Class12 recall | 4.838/4.854 = 99,67% | 4.838/4.854 = 99,67% |
| Class12 FP trên class13 | 6/2.146 | 0/2.146 |
| Class13 recall | 2.136/2.146 = 99,53% | 2.142/2.146 = 99,81% |
| Class13 false activation | 8 | 8 |
| Multi-gate rows | 2.128 | 3 |

Net +6 correct; paired events có **9 gained và3 lost**. Không giấu ba lost events
hoặc gọi secondary đạt mọi safety gate. Class13 còn8 FP tại threshold0,5. Kết quả
secondary cho thấy giảm class13→12 không chỉ đến từ đổi numeric threshold primary.
Class13 model không train khác; gain của class13 là hiệu ứng arbitration khi expert12
ít tranh nhãn sai hơn.

## 6. Communication tăng thêm

Đo serialized application messages, không real transport. Không gửi raw BASE/FIT
samples; có model/gradient packets, count metadata, content IDs và prediction/label
receipts phục vụ evaluation. Không có DP claim.

| Thành phần | Bytes |
| --- | ---: |
| Common native FIT reference + assessment identity cache | 21.247.436 |
| Current V4 arm | 17.841.133 |
| No-extra generic arm | 18.000.237 |
| Coverage-aware arm | 26.696.574 |
| **Extra coverage vs matched no-extra** | **8.696.337 ≈ 8,29 MiB** |

Isolated arm total khi tính cả common setup: current39.088.569 bytes; no-extra39.247.673;
coverage-aware47.944.010. Combined three-arm experiment83.785.380 bytes, common setup
được tính một lần. Header của generic protocol khiến no-extra cao hơn original
159.104 bytes; chi phí extra contributor nên so với matched generic ablation.

Giữ cùng optimization/sample budget **không có nghĩa cùng communication**. Phần tăng
thêm chủ yếu do reference + gradient của peer bổ sung. Native không train lại ở
experiment này; không cộng recorded native traffic cũ thành traffic mới hoặc suy
ra full-system saving từ con số trên.

## 7. Kết luận giới hạn

Giả thuyết negative-class coverage có bằng chứng development tích cực trên fixture
này: FP giảm, required/pooled recall không giảm, accuracy tăng trên primary,
ablation xác nhận hiệu ứng không đến từ thay initialization hoặc thêm optimizer work.

Chưa có independent confirmation, native integration của variant mới, ACTIVE gate,
historical retention/safety hoặc full TEST. Không gọi kết quả này là PASS deployment;
không thay gate0,1% hoặc status bank REJECT của protocol cũ. Cần protocol/evidence
confirmation phù hợp trước bước acceptance tiếp theo, không gọi lại CAL cũ để tìm PASS.

## Reproduce

```powershell
python tools/pilot_candle_v4_coverage.py --out audit_denice/new_coverage_dev --mode prepare
python tools/pilot_candle_v4_coverage.py --out audit_denice/new_coverage_dev --mode run
python tools/summarize_candle_v4_coverage.py --out audit_denice/new_coverage_dev
```

Source/BASE/FIT/graph/native-reference hashes được khóa trước training. Hashes,
metadata-selected contributors, round curves, complete owner/class metrics và wire
breakdown nằm trong receipt. Private checkpoints và raw samples không upload Git.
