# CANDLE V2 so với V1 — audit matched router trên checkpoint

Ngày 22/10/2026. Đã chạy inference development cho 98 client ở mỗi checkpoint T5/R19.
**Không train backbone, không chạy thêm TEST, không đổi production router.**

## 1. Kết luận

V2 hiện thua **V1 Multiclass**, nhưng vượt V1 Binary trên full TEST đã công bố.
Matched development cho thấy lựa chọn Binary thay vì Multiclass giải thích phần lớn chênh lệch
trên hai panel đã khóa. Đồng thời tồn tại điểm yếu classifier riêng ở class 6; không thể
quy mọi lỗi cho router hoặc gọi prototype transfer đã thành công.

## 2. So sánh cùng dữ liệu và cùng kiểu router

Panel reference: 2.176 unique validation inputs, 64/class. Tail: 2.132 inputs,
64/class trừ class 31 còn 20. Hai panel content-disjoint nhưng đều thuộc development đã
được quan sát. Mỗi model chạy toàn bộ panel; đây là **mean client metrics**, không phải
các số full TEST 35,70%/37,15%. Không dùng nhãn panel để fit router.

| Checkpoint / router | Reference Accuracy | Tail Accuracy | Tail mean Macro-F1 | Tail route accuracy |
|---|---:|---:|---:|---:|
| V1 Binary | 18.7003% | 18.9134% | 11.9194% | 34.4177% |
| V1 Multiclass đã lưu | 24.3365% | 24.3788% | 17.1759% | 39.8333% |
| V1 Multiclass refit | 24.3350% | 24.3754% | 17.1702% | 39.8366% |
| V2 Binary repair | 21.2405% | 21.7540% | 13.8626% | 34.9198% |
| V2 Multiclass diagnostic | 28.2394% | 28.3158% | 20.4350% | 43.5425% |

Refit dùng đúng cấu hình V1: balanced Logistic Regression, C=1, max_iter=1000,
random_state=0, **chỉ fit saved binary activation memory của chính client đó**. Không trộn
memory giữa hai encoders; không đọc historical raw CAL để fit. Saved router V1 tạo bằng
sklearn 1.6.1, môi trường local là 1.9.0: giữ frozen V1 làm control và refit cả V1/V2
trong cùng 1.9.0. Tail V1 frozen/refit chỉ lệch 0,0034 điểm phần trăm accuracy.

Phân rã chênh lệch trên tail, theo đường counterfactual đã khóa:

- V1 refit MC → V2 refit MC: **+3.9404 pp**.
- V2 MC → V2 Binary: **-6.5618 pp**.
- Đây là tác động thay router trên cùng checkpoint và chênh lệch giữa hai checkpoint.
  Không phải ablation cô lập alignment/virtual CE trong quá trình training.

## 3. Classifier có yếu đi không?

Oracle dưới đây biết task thật **chỉ để chẩn đoán**. Không phải inference chính thức.

| Tail diagnostic | V1 | V2 |
|---|---:|---:|
| Đúng task, availability riêng | 42.7629% | 48.3024% |
| Đúng task, mở toàn bộ class của task | 51.4143% | 52.9746% |
| Đúng task, cùng intersection availability | 40.2798% | 40.1654% |

Availability coverage trên tail: V1 64,02%, V2 84,82%. Khi dùng cùng class mask,
V2 không còn gain tổng thể rõ ràng; một phần lợi ích đang đến từ coverage. Đây chưa
phải bằng chứng weight transfer cải thiện các class chung.

| Recall class 6 trên tail | V1 | V2 |
|---|---:|---:|
| Binary thực tế | 39.4930% | 18.1760% |
| Multiclass thực tế | 48.6288% | 49.7449% |
| Đúng task, cùng 6 class T1 | 95.2168% | 81.0746% |

Class 6 có cả hai vấn đề: Binary định tuyến kém, và khả năng phân biệt trong T1
của checkpoint V2 cũng thấp hơn V1. Đổi raw affine trên chính weights V2 không cứu được:
ở reference, oracle recall class 6 raw chỉ 55.0223%
so với cosine 81.7761%.
Do đó chưa có căn cứ sửa bằng một phép bỏ normalization ở evaluator.

V1 oracle có/không adapter cho cùng accuracy trên cả hai panel; chưa thấy adapter giải thích
chênh lệch trong diagnostic này. Không suy ra adapter luôn vô ích ở mọi checkpoint.

## 4. Phần thua trên full TEST đã lưu nằm ở đâu?

Chỉ tính lại confusion matrices đã công bố; không mở TEST rows hay chạy predictor mới.

| True task | V1 Multiclass | V2 Binary repair | Delta số đúng |
|---|---:|---:|---:|
| T0 | 40.0953% | 34.2673% | -23,877 |
| T1 | 43.5497% | 37.2943% | -348,817 |
| T2 | 31.3388% | 33.1595% | +75,551 |
| T3 | 34.1882% | 40.3073% | +172,845 |
| T4 | 28.4768% | 8.9260% | -85,087 |
| T5 | 30.9110% | 44.0264% | +14,460 |

Net: **−194.925** prediction đúng. Riêng class 6: **−574.115**; class 24:
**−77.869**. Các class tăng bù lại một phần. Vì TEST mất cân bằng, class 6 có hơn
2 triệu mẫu nên ảnh hưởng headline accuracy rất lớn. Không dùng những test labels này
để chọn threshold hay hardcode class guard.

## 5. Training log nói gì?

Thống kê mô tả theo client-round, không weighted theo số mẫu; không dùng để quy kết nhân quả.

| Task | Mean real CE | Mean virtual CE | AMP skipped / attempted | Zero-update client-rounds |
|---|---:|---:|---:|---:|
| T0 | 0.9251 | 0.5529 | 4.3681% | 81 |
| T1 | 0.2127 | 0.0244 | 0.2470% | 0 |
| T2 | 0.4847 | 0.0446 | 0.0192% | 1 |
| T3 | 0.6438 | 0.0531 | 0.0104% | 0 |
| T4 | 1.7715 | 0.2521 | 0.0000% | 0 |
| T5 | 2.9124 | 0.2581 | 0.0000% | 0 |

T4/T5 real CE tăng mạnh, trong khi hai task này không có AMP skipped steps. Vì vậy
không có căn cứ đổ failure cuối run cho AMP. Virtual prototype CE thấp hơn real CE
không chứng minh class đã học tốt trên input thật của receiver; tương thích prototype
với receiver encoder vẫn cần ablation riêng.

Code preservation KL chỉ chuẩn hóa trong các old classes, và teacher được chụp trước
mỗi local round. Nó không trực tiếp bảo vệ cạnh tranh old-vs-new hoặc bù logit calibration.
Đây là giới hạn objective, chưa phải bằng chứng cần thêm một regularizer mới.

## 6. Việc tiếp theo

1. Giữ nguyên kết quả Binary repair 35,7030% / 18,4767%.
2. Khóa một candidate V2 Multiclass dùng đúng saved-memory protocol của V1; nếu đánh giá
   full TEST thì dùng notebook Kaggle riêng, không cần train lại. Kết quả development
   28,3158% không được thay thế hoặc dự đoán thành số full TEST.
3. Chỉ sau khi kiểm soát router, làm ablation training có giới hạn cho classifier T1
   và late-task learning: V2 full so với bỏ alignment / bỏ virtual CE, cùng reference,
   data budget, masks và seed; dùng development. Không tune riêng class 6 theo TEST.
4. Chưa retrain full hoặc thay kiến trúc thêm trong audit này.

## 7. Receipts và tái lập

- Local output: `audit_denice/candle_v1_v2_matched_20261022_v1`.
- Tool: `tools/debug_candle_v1_v2_matched.py`.
- Summary: `tools/summarize_candle_v1_v2_matched.py`.
- 196 model hash receipts trước/sau khớp; canonical V1 prediction control không mismatch.
- V2 Binary confusion matrices khớp lượt repair trước ở 98 models × 2 panels.
- Error partition khớp tổng errors, unexplained = 0; rescue/break khớp delta correct.
- Một seed, development đã quan sát, không có untouched confirmation.

```powershell
.venv-denice-gpu\Scripts\python.exe -X utf8 tools/debug_candle_v1_v2_matched.py --out <NEW_OUTPUT>
```
