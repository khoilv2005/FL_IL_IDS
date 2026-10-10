# DeNICE gốc: audit nút thắt và variant class availability

## 1. Kết luận

Audit toàn bộ **13.505.771 test rows, 34 class, 98 receiver** của `results (11)`
đã hoàn tất. Checkpoint là **DeNICE legacy, ξ = 0,8, seed 42, Task 5 / Round 19**.
Không CGoFed, CME, APPLIANCE, donor inference hoặc retraining.

**Coverage là thành phần chiếm phần lớn lỗi:** 5.709.415 mẫu không có true class
trong bất kỳ legal route mask nào của receiver. Đây là **42,2739% toàn test** và
**67,2576% tổng lỗi MulticlassSelf**.

Không phát hiện class từng có BASE local nhưng bị mất registration. Vì vậy
không được diễn giải toàn bộ coverage loss là lỗi checkpoint/persistence.

Đã triển khai variant chỉ sửa **class-mask policy**, giữ router, weights, ranks,
task profiles, graph và optimizer. Variant tăng accuracy **1,6237 điểm %** và
macro-F1 **0,9105 điểm %**. Đây là một cải thiện nhỏ có thể kiểm chứng, nhưng
**chưa giải quyết phần lớn coverage loss**. Default legacy vẫn giữ `local`.

## 2. Kết quả trên cùng checkpoint và toàn bộ test

### Prediction thực tế, không dùng true class/task để quyết định

| Policy | Accuracy | Macro-F1, 34 class | Route accuracy |
|---|---:|---:|---:|
| BinarySelf, local mask | 31,2446% | 15,4274% | 37,5432% |
| MulticlassSelf, local mask | 37,1463% | 19,7131% | 42,8166% |
| BinarySelf + global-task mask, diagnostic control | 33,5094% | 16,4426% | Giữ nguyên |
| **MulticlassSelf + global-task mask** | **38,7700%** | **20,6236%** | **Giữ nguyên** |
| AllClassesDiagnostic, không task mask | 37,1011% | 17,2604% | Không dùng để gate class |

`global-task mask` nghĩa là router vẫn tự chọn task. Model được chọn trong
global class set đã khai báo cho **task dự đoán**, không nhận true task.
Không tự tạo profile của task mà receiver chưa có.

AllClassesDiagnostic không giúp hơn MulticlassSelf và làm macro-F1 giảm.
Vì vậy không có căn cứ bỏ toàn bộ task masking.

### Oracle: chỉ là phép đo chẩn đoán

| Diagnostic | Accuracy | Phạm vi |
|---|---:|---|
| OracleMatched | **54,8568%** | True task nếu thuộc legal bank; native local mask/fallback giữ nguyên |
| BestAllowedRoute | **54,8568%** | Bất kỳ legal route nào dự đoán đúng; cùng local mask policy |
| BestAllowed sau global-task mask | **65,1958%** | Giữ legal bank cũ, chỉ mở class mask của các task trong bank |
| OracleGlobalTaskMaskDiagnostic | **80,6801%** | True task + global task mask, kể cả task ngoài bank; ngoài policy gốc |

BestAllowed của mask variant được tính lại từ từng client/class receipt:
chỉ cộng global-task oracle TP của class có true task trong bank của receiver.
Task/class mapping là disjoint, không có local empty-entry fallback ở fixture này.

**80,6801% không phải accuracy deployable và không phải router ceiling của
baseline.** Nó dùng true task và bỏ giới hạn local availability/profile.

### Đính chính mục tiêu 50%

Trên **checkpoint DeNICE gốc này**, router-only có upper bound **54,8568%**.
Do đó chưa thể nói router-only không đạt được 50%.

Từ MulticlassSelf tới 50% cần +12,8537 điểm, trong tổng routing headroom
17,7105 điểm. Đây là khả năng lý thuyết, chưa phải một router đã đạt được.
Ceiling 45,37% của checkpoint/protocol cũ không được dùng thay cho số này.

## 3. Phân rã lỗi không chồng lấn

| MulticlassSelf | Số mẫu | % toàn test | % trong tổng lỗi |
|---|---:|---:|---:|
| Đúng | 5.016.894 | 37,1463% | — |
| **Class không reachable trong legal bank** | **5.709.415** | **42,2739%** | **67,2576%** |
| Router sai nhưng legal route khác cứu được | 2.391.944 | 17,7105% | 28,1774% |
| Class reachable, mọi legal route vẫn sai | 387.518 | 2,8693% | 4,5650% |

Đã kiểm tra exact accounting:

```
13.505.771 = 5.016.894 + 5.709.415 + 2.391.944 + 387.518
7.408.838 BestAllowed correct = 5.016.894 + 2.391.944
```

Classifier error ở đây là lỗi **trong coverage hiện tại**. Không được suy ra
classifier đã biết chắc mọi class ngoài coverage.

## 4. Coverage thiếu ở đâu?

- **74/588 receiver–task pairs** không có nonempty activation memory.
- **2.902.000 mẫu, 21,4871% test** thiếu true-task profile.
- **2.807.415 mẫu, 20,7868% test** có true-task profile nhưng thiếu true class
  trong local mask. Hai nhóm này cộng thành coverage loss ở fixture này.
- **0 pair / 0 row** có BASE local của class nhưng class bị mất khỏi legal mask.
- 430 receiver–class pairs reachable dù không có BASE local. Số này chỉ cho
  biết metadata/availability, không tự chứng nhận competence hoặc nguồn transfer.

| Task | Coverage-unreachable rows | % toàn test |
|---|---:|---:|
| T0 | 59.386 | 0,4397% |
| **T1** | **2.434.871** | **18,0284%** |
| **T2** | **1.863.105** | **13,7949%** |
| T3 | 1.164.822 | 8,6246% |
| T4 | 154.447 | 1,1436% |
| T5 | 32.784 | 0,2427% |

Phần lớn mất coverage nằm ở T1/T2. Chỉ sửa/cài khi tới T5 không thể xử lý
phần lớn nhóm này dưới protocol không đọc raw historical CAL.

Class 28 có 648 test rows: 0 TP ở hai self policies, chỉ 4 TP BestAllowed.
Nó cần được ghi nhận, nhưng không phải nguồn chiếm phần lớn accuracy loss.
Không có policy dừng training/evaluation do recall class 28 trong audit này.

## 5. Variant chỉ sửa class availability

Module mới: `fed_learning/strategies/incremental/denice_class_availability.py`.

Hai policy rõ ràng:

- `local`: giữ nguyên behavior legacy.
- `global_task`: shadow detector dùng global classes của predicted task,
  intersect với seen classes; không nhận y_true hoặc oracle task.

Variant không thay class ownership, activation memory, binarizer, LR coefficients,
fc2 rows, masks/ranks của model hoặc adapter policy. Không có donor models.
Không import patch, tạo certificate hoặc tuyên bố tất cả class đã có competence.

`run_legacy_self(..., include_global_task_mask=True)` ghi thêm policy
`MulticlassGlobalTaskMask`, còn BinarySelf/MulticlassSelf giữ nguyên. Flag mặc
định `False`; không tự chọn variant làm main method. Không chạy cùng APPLIANCE
certificates để tránh gộp hai prediction scopes khác nhau.

### Rescue/break trên cùng toàn bộ test

| Transition, so với MulticlassSelf | Số mẫu |
|---|---:|
| Đúng → đúng | 4.972.760 |
| **Đúng → sai, break** | **44.134** |
| **Sai → đúng, rescue** | **263.429** |
| Sai → sai | 8.225.448 |

```
Net correct gain = 263.429 − 44.134 = 219.295
Accuracy gain = 219.295 / 13.505.771 = 1,6237 điểm %
```

77 receiver cải thiện, 10 không đổi, 11 giảm. Descriptive receiver-bootstrap
CI cho mean client gain: [+0,9058; +2,4469] điểm. Receivers có liên hệ qua graph;
đây không phải multi-seed significance hoặc untouched-test confirmation.

Ngay cả khi true task đã được cung cấp trong diagnostic, mở global mask vẫn
phá 59.649 correct decisions cũ và thêm 1.456.009 correct decisions ở class mới
reachable. Net +1.396.360 correct tạo ceiling 65,1958%. Đây là conditional
oracle accounting, không thay cho rescue/break của inference thật ở bảng trên.

## 6. Tính toàn vẹn

- SHA checkpoint archive, role manifest, frozen routers, dataset metadata,
  test arrays và saved receiver records khớp.
- Test rows unique/exhaustive; receiver assignment và label/global-row alignment
  giữ nguyên từ predictions gốc.
- **0/27.011.542 self prediction mismatch** so với file cũ.
- First-batch native replay: 50.176 rows mỗi policy, **0 mismatch**.
- Recompute 16 full-audit checks: **16/16 PASS**.
- Native variant/oracle replay trên 98 receiver, 50.176 rows:
  **882/882 checks PASS**, gồm mask control, route preservation, state
  preservation, legal forced actions và matched oracle.
- Toàn bộ model/router fingerprint trước/sau không đổi.
- Không optimizer steps, không fit router trong audit, không đọc raw train/CAL.
- GPU local RTX 4060, Torch 2.10.0+cu128, TF32/AMP tắt, batch 512.
- Full forward audit khoảng **24 phút 8 giây**, chưa tính chuẩn bị môi trường.

Frozen LR được serialize bởi sklearn 1.6.1, môi trường audit là 1.9.0.
Warning được giữ trong log; exact full replay và native checks đã xác nhận
predictions của fixture này. Không suy rộng compatibility sang artifact khác.

PowerShell redirect stderr ghi warning thành NativeCommandError và shell
reported exit 1 dù Python đã hoàn tất; completion và các verifier đều PASS.
Launcher `tools/run_denice_bottleneck_local.py` tee stdout/stderr qua subprocess,
giữ warning và actual child exit code để lần chạy sau không bị nhầm trạng thái.

## 7. Quyết định tiếp theo

1. **Ưu tiên class/task coverage**, không tiếp tục donor-update/Head-Agg hoặc
   đổi backbone, graph, ξ, optimizer từ audit này.
2. Global-task mask đã có implementation và control, nhưng chỉ cứu 4,61% nhóm
   originally-unreachable trong inference thật. Chưa gọi là complete repair.
3. Tách tiếp legitimate missing profile khỏi class registration; không tự tạo
   profile từ nhãn test hoặc mở mọi class để giả định competence.
4. Coverage/profile admission trong training phải dùng evidence hợp lệ lúc
   class còn current, giữ native router và không replay raw CAL lịch sử.
5. Router vẫn có headroom lớn, nhưng không đổi router cùng lượt với coverage
   để tránh mất khả năng attribution. Nếu ưu tiên riêng mục tiêu 50%, ceiling
   mới cho phép nghiên cứu router-only như một experiment tách biệt.

**Chưa train full lại, chưa đổi default của baseline, chưa giải quyết xong
coverage.** Test source này đã được xem nhiều lần; kết quả là development
evidence. Cấu hình final phải được khóa trước independent evaluation/seeds.

## Artifacts

- `artifacts/denice_bottleneck_full_20261010.json`: metrics, bounds, decomposition,
  protocol hashes, verifier và native replay summary.
- `artifacts/denice_bottleneck_status_20261010.json`: trạng thái và phần chưa giải quyết.
- `audit_denice/denice_bottleneck_20261010/full_cuda/`: immutable protocol và receiver receipts.
- `artifacts/denice_bottleneck_diagnostics_20261010.zip`: receipts/report, không chứa raw dataset.
