# DeNICE + APPLIANCE V2: audit full test Task 5

## Nguồn và protocol

Artifact: `C:\Users\khoak\Downloads\denice_appliance_full_eval_task_5_20261010_031923_632938.zip`.

Training từ log `fl-il-lk-na-786cce.log`: commit `92fe732`, DeNICE legacy + APPLIANCE, seed42, ξ=0.8, 6 task × 20 round. Eval chạy trên CUDA, một local backbone; không CGoFed, CME hoặc peer-model inference. Router Multiclass được fit từ binary activation memory checkpoint trước test, không refit raw historical data.

Checkpoint SHA256: `3003593d64e3d63a81200b90d1a6e7487fff6c0875e3518cd58c2ccad22fe391`.

Role manifest SHA256: `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.

`completion=true`; 13.505.771 samples, 98 receivers, 34/34 classes. Partition class-stratified disjoint, seed523687. Test source đã được quan sát trong các vòng development trước; không gọi đây là untouched confirmation.

## Kiểm tra artifact

Đọc toàn bộ prediction CSV theo chunk và recompute confusion matrices, accuracy, Macro-F1, per-client accuracy, transition accounting, row uniqueness và changes ở receiver chưa được authorize. **11 kiểm tra đều khớp**; không chạy lại inference hoặc training.

Audit local: `audit_denice/appliance_eval_v2/summary.json`, `receiver_transitions.csv` và script `audit_output.py`.

## Kết quả

| Decision policy trên cùng checkpoint | Accuracy | Macro-F1, 34 class | Số đúng |
|---|---:|---:|---:|
| BinarySelf | 31,2087% | 15,3676% | 4.214.981 |
| MulticlassSelf | 37,1530% | 19,6432% | 5.017.795 |
| APPLIANCE | **37,8620%** | **20,0777%** | **5.113.558** |

APPLIANCE so với MulticlassSelf: **+0,7091 điểm accuracy**, **+0,4345 điểm Macro-F1**. Đây là inference ablation trên cùng checkpoint được train với APPLIANCE; Self với backup head không thay thế một fresh DeNICE control training độc lập.

Eval elapsed field: 3843,49 giây (~64,1 phút), không bao gồm toàn bộ setup/router fitting.

## Transition accounting

- Route authorized: **7**, tất cả `CARRY_FORWARD`, certificate current.
- Route suspended: **1**, receiver9/class12, `new_protection_conflict`; activation=0.
- Imported activation: **137.793**, bằng đúng số prediction thay đổi.
- Self sai → APPLIANCE đúng: **95.790 rescue**.
- Self đúng → APPLIANCE sai: **27 break**.
- Self sai → APPLIANCE vẫn sai, nhưng đổi class: **41.976**.
- Changed correct → correct: 0.

Identity kiểm tra:

```text
5.113.558 - 5.017.795 = 95.790 - 27 = 95.763
95.763 / 13.505.771 × 100 = 0,709052 điểm %
```

Không có prediction thay đổi ở receiver không có authorized route.

## Theo capability đã cài

Recall/precision dưới đây tính trên full test shard của receiver, không phải CAL acceptance.

| Receiver / class | Recall | Precision | Rescue | Break | Gain accuracy receiver |
|---|---:|---:|---:|---:|---:|
| 0 / 21 | 96,83% | **37,58%** | 9.418 | 2 | +6,83 điểm |
| 13 / 6 | 94,36% | 100% | 19.912 | 0 | +14,45 điểm |
| 55 / 20 | 97,41% | **36,70%** | 7.628 | 16 | +5,52 điểm |
| 64 / 9 | 96,57% | 100% | 11.449 | 0 | +8,31 điểm |
| 65 / 20 | 97,14% | **36,56%** | 7.607 | 9 | +5,51 điểm |
| 76 / 6 | 94,37% | 100% | 19.914 | 0 | +14,45 điểm |
| 97 / 6 | 94,12% | 100% | 19.862 | 0 | +14,41 điểm |

Class21 có 15.643 false positive; hai patch class20 có 13.158 và 13.202 false positive. Tổng **42.003 false activation**, trong đó 41.976 mẫu Self đã sai sẵn. Vì thế break=27 không chứng minh routing sạch hoặc population FAR thấp. Đây là quan sát đúng với limitation của empirical acceptance ngoài finite CAL scope.

## Coverage và giới hạn gain hiện tại

Chỉ 7/98 receivers có route đang hoạt động; mỗi receiver một imported capability, bốn class import khác nhau (6,9,20,21). Tổng target-class test rows trên bảy shard đó là **100.550**, tương đương **0,7445%** toàn test. Self không đúng target class trên các pair này; APPLIANCE phục hồi 95.790/100.550 (~95,27%).

Với đúng bảy capability và chỉ sửa target classes, ceiling gain là khoảng **0,7445 điểm accuracy**. Gain thực tế +0,7091 đã gần bound này. Đây không phải bound cho toàn bộ APPLIANCE nếu bổ sung capabilities; cũng không phải global classifier/router oracle.

Class28 vẫn 648 test rows, **0 true positive**, 16.661 prediction thành class28; không có imported patch class28. Class được giữ trong evaluation, không chặn run.

## So với results (12)

| Metric | Run cũ | Run mới |
|---|---:|---:|
| MulticlassSelf accuracy | 37,1847% | 37,1530% |
| APPLIANCE accuracy | 37,1847% | 37,8620% |
| APPLIANCE activation | 0 | 137.793 |

Self gần như giữ nguyên; thay đổi đã làm imported capabilities thực sự tham gia inference. Chưa có evidence rằng backbone training đã tốt hơn. Đây là một seed và có nhiều thay đổi discovery/deployment, không tách causal effect từng component giữa hai run.

## Bước tiếp theo

1. Giữ artifact hiện tại: install → protect → carry-forward → normal single-model inference đã hoạt động và có gain quan sát được trên full test.
2. Audit coverage bằng metadata/live graph/FIT/CAL: vì sao chỉ tám install, bảy route active; phân loại receiver thiếu BASE protection, thiếu CAL, donor chưa mature/quality thấp, dependency không phù hợp hoặc acceptance reject. Không dùng test labels để chọn donor hoặc chỉnh guard.
3. Với class20/21, phân tích false activation trên CAL/development hợp lệ; threshold hoặc routing variant mới phải được chọn và khóa trước evaluation mới. Không tune trên full-test CSV vừa đọc.
4. Trước full rerun tiếp theo, chứng minh automatic discovery tăng số capability hợp lệ bằng bounded integration. Không kỳ vọng gain 10 điểm chỉ từ việc bật lại bảy patch hiện tại.
