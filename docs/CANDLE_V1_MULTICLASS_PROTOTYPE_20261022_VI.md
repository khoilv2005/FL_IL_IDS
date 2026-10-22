# CANDLE V1 — nhiều class import, threshold không dùng positive receiver, panel 34 class

Ngày: 22/10/2026. Tiếp nối commit `18d21b2`. Offline checkpoint audit,
không optimizer, không train lại, không merge donor routing memory, không final TEST.

## 1. Kết luận

1. **Cổng đã khóa trước đó tiếp tục hoạt động trên panel validation mới, rộng hơn.**
   Client 7 q=1% đạt 98,05% recall class 24, 251 rescue, 0 break, 1 false activation
   trên 8.125 negatives thuộc đủ 33 class khác. Client 65 q=1% đạt 97,66%,
   250 rescue, 0 break, 0 false activation trên cùng 8.125 negatives.
2. **Quy tắc threshold tự hiệu chuẩn đã thử chưa dùng được.** Donor self-cosine
   `mean−2σ`, kết hợp current receiver negative maximum, cho recall 0% ở mọi
   target của ba case. Negative-only mở lại recall một số class nhưng tạo nhiều FP.
3. **Đã tạo được tình huống nhiều cổng cùng mở.** Ở Client 23 có 206 collisions;
   head và prototype thực sự đưa ra các prediction khác nhau. Tuy nhiên toàn bộ
   cấu hình này FAIL risk/recall gates. Chưa có cơ sở chốt bỏ head hoặc chọn head
   làm thành phần tốt hơn trong một cơ chế multi-class an toàn.

Không thay primary q=5% của lượt trước thành q=1%. Không hạ gate recall≥95%,
observed FAR≤0,1%, negative break=0 theo scope đã báo cáo.

## 2. Các diễn giải cần sửa

- “Row gốc + unmask = 0%” ở audit trước là **receiver row**, không phải donor
  row đang cạnh tranh với logits của task mà native router chọn nhầm. Vì vậy
  không thể dùng con số này để kết luận donor logit nhỏ hơn mọi native winner.
- Cặp Client 7 cho thấy router chọn sai gần hết target class 24. Không đủ để
  kết luận binary router mù trước mọi class chưa xuất hiện cục bộ.
- Chỉ có một import class không bắt buộc head+gate bằng prototype-only:
  head competition vẫn có thể từ chối mẫu đã qua gate. Audit trước đo thấy
  competition không thay prediction trên panel cụ thể đó.
- Client 65 ở lượt trước dùng **receiver-space prototype**, được tạo bằng
  receiver function trên donor current positives, cùng moment-adjusted head.
  Không phải chỉ L2-normalize donor prototype là tự đạt 96,09%. Panel âm cũ
  của Client 65 là 1.711 mẫu, không phải 2.740 mẫu của Client 7.
- Không thể gọi communication tăng bằng 0 nếu prototype, calibration statistics
  hoặc đúng receiver reference/version chưa thực sự có sẵn ở phía cần dùng.

## 3. Protocol được khóa trước evaluation

Checkpoint: legacy results(11), Task 4 / Round 19, cùng role manifest và source
hashes như audit trước. Rules ghi vào `protocol_before_data.json`; mỗi receiver
có packet/threshold lock hoàn tất trước khi mở validation panel.

| Receiver | Actual missing classes | Donors | Ý nghĩa |
|---|---|---|---|
| 7 | 24, 29 | 11, 3 | Hai gap thật, trong neighborhood |
| 23 | 24, 25, 26 | 21, 13, 96 | Ba gap thật, trong neighborhood |
| 65 | 24 | 11 | Extra-neighborhood counterfactual control |

Client 7 giữ continuity của case cũ. Client 23 được chọn bằng metadata:
trong các receiver có ít nhất ba gap đủ support, chọn minimum donor CAL count
lớn nhất. Donor từng class ưu tiên current CAL support, tie-break client ID.
Không chọn bằng inference quality, không ẩn class nhân tạo.

Construction chỉ đọc current Task-4 CAL. Prototype là donor FC1 mean trên
construction positives, không đọc positive receiver để tạo threshold.
Số donor positives theo thứ tự các class: Client 7 là 128/43; Client 23 là
128/120/83; Client 65 là 128. Receiver current negative counts là 77/21/131.

Threshold chính cho class c:

```
tau_D = mean(cos(h_D(x_c), p_D(c))) - 2 * std(cos(h_D(x_c), p_D(c)))
tau_N = nextafter(max cos(h_R(x_negative), p_D(c)), +inf)
tau_primary = max(tau_D, tau_N)
```

Hai secondary đã khóa: donor-only tau_D và receiver-negative-only tau_N.
Không quét z, không fit threshold từ validation. “Target-free” ở đây chỉ có
nghĩa **threshold không cần positive receiver features**; donor vẫn có current
positive CAL hợp lệ. Secondary adjusted head vẫn cần receiver function tại
donor và paired current positives để tính diagonal moments như audit trước.

Cùng một bộ cổng cho bốn scorers: prototype cosine, direct-head cosine,
moment-adjusted-head cosine, moment-adjusted-head raw logits. Nếu nhiều import
candidates qua gate, chọn argmax score trong các candidates đó; nếu không có
candidate, giữ exact native fallback. Không cấp true task/class cho inference.
Không thêm native-logit competition vào phép tách scorer này.

## 4. Panel validation rộng hơn

- **8.381 unique rows, đủ 34 class**: cap 256/class; class 0 có 210, class 28
  có 151, class 31 có 84; các class còn lại có 256.
- Loại 4.879 content hashes của các validation panels đã dùng trong vòng audit
  head-unlock trước. Không overlap current CAL; original roles content-disjoint.
- Mỗi class lấy từ owner có validation support lớn nhất, rồi owner khác nếu
  thiếu unique rows. Target class 24 vẫn có 256 mẫu donor 11, ngoài prefix đã dùng.
- Class 30–33 có 852 rows, là **future-class stress diagnostic** ở checkpoint T4.
  Cumulative T0–T4 có 7.529 rows. Hai scope được ghi riêng trong metrics.
- Đây là development source đã từng có nhiều audit; content mới so với panels
  vừa rồi không bảo đảm chưa được xem ở mọi nghiên cứu trước đó.
- Historical validation chỉ để đánh giá offline; không dùng historical raw CAL
  hoặc đưa các nhãn evaluation vào adaptation/threshold fitting.

## 5. Đối chứng threshold không dùng positive receiver

| Receiver / targets | tau_D theo target | Primary recall từng target | FP / break |
|---|---|---|---|
| 7 / 24,29 | 0,956087 / 0,923329 | 0% / 0% | 0 / 0 |
| 23 / 24,25,26 | 0,974696 / 0,866771 / 0,921494 | 0% / 0% / 0% | 0 / 0 |
| 65 / 24 | 0,956087 | 0% | 0 / 0 |

Trong các case này tau_D luôn lớn hơn tau_N nên primary bằng donor-only. Các
cổng không mở; kết quả 0 FP không phải một transfer thành công. Donor 11
self-cosine class 24 có mean 0,995114, std 0,019513; threshold đó không chuyển
được sang receiver similarity score distribution đã thấy ở audit trước.

Negative-only, prototype scorer:

| Receiver | Recall từng target | FP | Break | Pooled FAR |
|---|---|---:|---:|---:|
| 7 | 24:100%; 29:2,34% | 2.210 | 377 | 28,08% |
| 23 | 24:100%; 25:8,98%; 26:1,95% | 3.546 | 1.230 | 46,58% |
| 65 | 24:0,39% | 233 | 18 | 2,87% |

Current local negatives không đủ để suy ra threshold cho mọi protected class.
Đây là failure của các quy tắc đã khóa, không chứng minh mọi cách target-free
calibration đều bất khả thi. Không chỉnh z/threshold để cứu kết quả sau evaluation.

## 6. Đối chứng nhiều class import

Client 7 không có collision ở cả ba threshold policies; vì vậy sự bằng nhau
giữa head/prototype ở case này chưa phân xử được scorer nào tốt hơn.

Client 23 / negative-only có 206 samples qua ít nhất hai cổng; 44 thuộc target
classes. Các scorers thực sự khác prediction:

| Scorer | Khác prototype-only | Accuracy trên 206 collisions |
|---|---:|---:|
| Prototype cosine | 0 | 2,91% |
| Direct head cosine | 80 | 0,97% |
| Adjusted head cosine | 28 | 1,46% |
| Adjusted head raw | 153 | 0,49% |

Các con số rất thấp vì collisions còn chứa nhiều negatives bị mở nhầm. Tất cả
configurations FAIL; không lấy scorer thắng tương đối trong một gate thất bại
làm bằng chứng đủ để loại head khỏi kiến trúc hoặc triển khai prototype-only.

## 7. Quét packets cũ, không hiệu chuẩn lại

Giữ đúng prototype, head và tau của packets từ `18d21b2`; đánh giá trên panel
mới. Không thay weights, không chọn q bằng kết quả panel này.

| Receiver / packet | Recall 24 | Rescue | FP / 8.125 negatives | Break |
|---|---:|---:|---:|---:|
| 7 / primary q=5% | 94,53% | 242 | 1 | 0 |
| 7 / secondary q=1% | **98,05%** | **251** | **1** | **0** |
| 65 / primary q=5% | 94,14% | 241 | 0 | 0 |
| 65 / secondary q=1% | **97,66%** | **250** | **0** | **0** |

Prototype-only vẫn cho prediction giống hệt head+gate ở cả bốn packets.
Primary q=5% vẫn FAIL recall95%; secondary q=1% đạt pooled empirical gate
trên panel này, chưa được đổi thành primary hoặc production acceptance.

False activation duy nhất của Client 7 thuộc class 22: pooled FAR = 1/8.125
= 0,0123%, nhưng **per-class FAR22 = 1/256 = 0,3906%**. Break bằng 0 vì baseline
đã sai mẫu đó. Nếu áp gate FAR≤0,1% cho từng class, Client 7 không đạt gate này.
Không được dùng zero break để thay cho đánh giá false activation.

Trên cumulative T0–T4: Client 7 q=1% FAR = 1/7.273 = 0,01375%, 0 break.
Client 65 q=1% FAR = 0/7.273, 0 break. Future class30–33 có 0 false activation
cho cả hai, nhưng native baseline cũng không nhận diện được các future classes.

## 8. Quyết định sau audit

- Giữ bằng chứng feasibility của **cổng được hiệu chuẩn bằng receiver function
  trên donor positives**: đã lặp trên panel rộng hơn với lợi ích thực tế.
- Chưa đạt mục tiêu portable self-calibration chỉ bằng donor statistics và
  current receiver negatives. Không tuyên bố chỉ dùng capsule sẵn có là đủ.
- Không có evidence head cải thiện trên prototype-only ở một multi-class
  configuration đạt acceptance. Chưa chốt bỏ head hoặc đưa mechanism vào main.
- Nếu nghiên cứu tiếp, phải chốt protocol calibration/alignment mới trước
  evaluation; không tiếp tục tune trên panel 34 class vừa mở.

## 9. Code, receipts và tái lập

Script: `tools/audit_candle_v1_multiclass_prototype.py`.

```
.venv-denice-gpu\Scripts\python.exe tools/audit_candle_v1_multiclass_prototype.py --out audit_denice/multiclass_prototype_new
```

Default paths là legacy checkpoint/roles/current CAL store đã có local.
Runtime đo được khoảng 37 giây trên RTX4060 Laptop, không phải dự báo full training.

Local output: `audit_denice/candle_v1_multiclass_prototype_20261022_v1` gồm
protocol, packet locks, panel content receipt, per-class metrics, completion và
source snapshots. Source model/router complete hashes giữ nguyên. Raw data và
checkpoints không commit; public compact receipt ở
`artifacts/candle_v1_multiclass_prototype_20261022.json`.
