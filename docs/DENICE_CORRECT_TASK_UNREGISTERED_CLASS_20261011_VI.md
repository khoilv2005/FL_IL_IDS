# DeNICE: chọn đúng task nhưng true class chưa đăng ký

## Phép đo

Trên đúng full test của `results (11)`, Task 5 / Round 19, xét event:

```
predicted_task == true_task
AND true_class không thuộc class mask đã đăng ký của receiver cho task đó
```

Đây là phân tích sau inference. Nhãn thật chỉ dùng xác định event và tính
metrics; không đưa vào router hoặc classifier. Không forward/fit/train thêm.

## Kết quả pooled trên 98 receiver

| Chỉ số | BinarySelf gốc | MulticlassSelf variant |
|---|---:|---:|
| Toàn test | 13.505.771 | 13.505.771 |
| Tổng mẫu router chọn đúng task | 5.070.494 | 5.782.706 |
| Đúng task nhưng class chưa đăng ký | **663.015** | **557.790** |
| Tỷ lệ trên toàn test | **4,9091%** | **4,1300%** |
| Tỷ lệ trên các mẫu chọn đúng task | **13,0759%** | **9,6458%** |
| Tỷ lệ trên 2.807.415 mẫu thiếu class mask nhưng có task profile | 23,6166% | 19,8685% |
| Mẫu đúng task, class đã đăng ký nhưng classifier vẫn sai | 187.657 | 208.022 |

Cả 98 receiver đều có ít nhất một mẫu thuộc event đang xét. Không gộp nhóm
thiếu task profile (2.902.000 mẫu) vào event này; nhóm đó không thể chọn đúng
true task trong legal route bank của fixture.

**Trong event đang xét, normal prediction sai 100%:** class thật bị loại khỏi
argmax bởi mask. Điều này xác nhận tình huống 2 có xảy ra, không chỉ là ví dụ.
Nó không chứng minh model đã học class đó hoặc registration bị mất do bug.
Audit provenance trước không có class có BASE local nhưng bị bỏ khỏi registry.

## Mở class mask có cứu được toàn bộ nhóm này không?

Không. Dùng lại diagnostic đã khóa: giữ **cùng predicted task và weights**,
chỉ thay local task mask bằng global class set của task đó.

| Trong nhóm đúng task nhưng thiếu class | BinarySelf | MulticlassSelf |
|---|---:|---:|
| Được cứu bởi global-task-mask diagnostic | **348.911** | **263.429** |
| Tỷ lệ được cứu trong nhóm | **52,62%** | **47,23%** |
| Vẫn sai sau mở task mask | 314.104 | 294.361 |

Vì vậy 4,91% / 4,13% toàn test bị block ở bước class mask **không phải** mức
accuracy gain bảo đảm khi mở mask. Classifier còn sai trên gần nửa nhóm.
Ngoài nhóm này, mở mask còn tạo competing logits và gây break:

| Toàn test, cùng diagnostic mở task mask | BinarySelf | MulticlassSelf |
|---|---:|---:|
| Rescue | 348.911 | 263.429 |
| Break | 43.029 | 44.134 |
| Accuracy normal → diagnostic | 31,2446% → 33,5094% | 37,1463% → 38,7700% |

Diagnostic không được coi là registration repair đã được chứng nhận; chưa
thay production mask hoặc chọn acceptance threshold bằng test.

## Những class đóng góp nhiều blocked rows nhất

| Class | BinarySelf blocked | MulticlassSelf blocked |
|---|---:|---:|
| 6 | 71.681 | 94.943 |
| 9 | 78.033 | 64.867 |
| 13 | 58.067 | 44.309 |
| 14 | 63.394 | 73.636 |
| 20 | 61.575 | 32.274 |
| 21 | 96.790 | 62.470 |
| 22 | 43.066 | 39.162 |

Đây là counts chẩn đoán, chịu ảnh hưởng class frequency và receiver support.
Không dùng bảng này để hardcode class, tune trên test hoặc suy ra priority
theo per-class rate mà chưa tính mẫu số tương ứng.

## Integrity và artifacts

**1.912/1.912 checks PASS.** Route confusion suy từ prediction class khớp
actual route receipts cho cả normal và mask diagnostic; class/task mapping
disjoint, không có all-class fallback ở fixture. Đối chiếu counts với
valid-scope audit; tất cả target bị block có normal TP=0 và rescue không vượt
số mẫu đúng task. Role/dataset/completion hashes được kiểm tra trước khi đọc.

- Script: `tools/audit_denice_correct_task_unregistered_class.py`.
- Summary: `artifacts/denice_correct_task_unregistered_class_20261011.json`.
- ZIP local: `artifacts/denice_correct_task_unregistered_class_20261011.zip`.
- ZIP chứa `receiver_class_counts.csv` theo policy/client/task/class và báo cáo;
  không chứa model/dataset hoặc raw historical train/CAL.

Các tỷ lệ trong báo cáo đều pooled theo mẫu, không phải mean client accuracy.
Full test đã là development source; không gọi đây là untouched confirmation.
