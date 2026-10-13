# APPLIANCE V2 — ablation 2×2 trên update class 6 đã khóa

## Kết luận quyết định

**Cặp `2 ← 62 / class 6` chưa chứng minh donor update tạo thêm lợi ích.** Chỉ đăng ký class 6 vào class availability của router hiện có đã đạt **97,2656% recall**, cao hơn Full V2 **97,1680%**. Không học lại update, không refit router, không sửa threshold.

Do đó cần sửa cách hiểu kết quả trước: tăng từ 0 lên khoảng 98% ở native inference chưa đủ để kết luận receiver học được knowledge mới từ donor. Trong case này, classifier row gốc đã hoạt động tốt khi được phép cạnh tranh trong mask của Task 1. Ablation không xác định được nguồn của khả năng này, và không suy ra receiver có khả năng phân loại tương đương khi bỏ task mask.

**Chưa native survival, chưa tích hợp runner và chưa full training.** Việc tiếp theo là functional-gap discovery: phân biệt class thực sự thiếu khả năng nhận diện với class chỉ chưa được đăng ký availability. Không tiếp tục survival của update class 6 chỉ vì Full V2 vượt baseline đóng mask.

## Protocol mới và authority

Protocol native routed được ghi vào `protocol_before_validation.json` trước khi mở VALIDATION và tính prediction. Đây là một evaluation mới; báo cáo All-seen primary cũ giữ nguyên, không đổi nhãn secondary cũ thành primary.

| Thành phần | Giá trị |
|---|---|
| Checkpoint | DeNICE legacy Task 1 / round 19, lineage results (13) |
| Terminal SHA256 | `b057f27a763b63bd5ffff936926f5e7d4bb0696c07b35ada8c2d4de209799a87` |
| Graph SHA256 | `0c4c7acd41b31627ef32591c31aa72d561adfc2721028dead91e459d697d3a22` |
| Clean roles SHA256 | `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15` |
| Frozen trained-row file SHA256 | `5b6f0bc79cf95a6b7152a810373b103e212f183c8e84933f4888bb2fb3e5c664` |
| Full V2 function fingerprint | `fca68ce9a4cabdb4bbdf7c50e1484261693ff5a7` — khớp lock cũ |
| Inference primary mới | `pred_hard`, router `binary_cosine`, seen classes 0–11 |
| Fitting | Không; dùng đúng row đã seal ở run01 |
| Availability intervention | Chỉ thêm class 6 vào `episode_classes[1]` |
| Update intervention | Row/bias đã học cùng mask=1 và maturity=2 của update cũ |

Availability-only giữ nguyên weights, masks, ranks và mọi state của model. Update-only giữ nguyên availability. Hai nhánh update giữ nguyên tất cả prefix/buffer và các head row không phải class 6. Tất cả nhánh giữ nguyên routing memories, learned router state và task decision trên từng sample; chỉ mask availability khác nhau. Không CGoFed, CME, imported route, donor query hoặc ground-truth task ở inference.

Update intervention gồm cả đăng ký mask/maturity của update đã khóa; không được mô tả là một thay đổi thuần weight không có metadata. Dù vậy, Availability-only không cần các thay đổi đó vẫn đạt kết quả tương đương và tốt hơn một mẫu.

## Panel và mức độc lập

Đọc original VALIDATION role, tách role/content với BASE/CAL/FIT của lượt train-time trước. Không đọc BASE, FIT hoặc raw CAL trong lần này; không mở final test.

- Positive: class 6 của donor 62, cap 1.024 unique inputs.
- Current negatives: classes 6–11 của receiver 2, cap 1.024/class, thực tế không có class 6/8.
- Old witnesses: peers **15, 78, 81, 4**, có positive-alpha edge trong graph Task 1 của receiver 2. Chọn greedy bằng role counts để đạt 32 mẫu/class cũ, tie theo ID; không chọn bằng prediction quality.
- Dedup toàn panel bằng SHA256 của feature input float32; một input chỉ tính một lần. Không có conflicting labels.
- Tổng **9.428 unique rows**: 1.024 positive, 8.404 negative; trong đó 5.496 old-class rows.

| Class | Unique rows | Class | Unique rows |
|---:|---:|---:|---:|
| 0 | 32 | 6 | 1.024 |
| 1 | 3.072 | 7 | 819 |
| 2 | 47 | 8 | **0** |
| 3 | 107 | 9 | 1.024 |
| 4 | 2.162 | 10 | 82 |
| 5 | 76 | 11 | 983 |

**Không gọi đây là untouched independent confirmation.** VALIDATION từng được dùng trong CME và các audit APPLIANCE trước. Chưa có ledger đầy đủ để loại toàn bộ những row/content đã ảnh hưởng tới development. Role-disjoint với lượt update vừa qua không chứng minh độc lập với toàn bộ lịch sử nghiên cứu. Content dedup cũng không chứng minh iid sampling hoặc population representativeness.

Việc bổ sung witnesses ở đây là offline mechanism qualification, không phải quyền đọc raw historical evidence trong production. Witness data không đi vào optimizer hay inference của receiver. Cặp vẫn được chọn hồi cứu từ development Task 5, không phải prospective automatic discovery.

## Kết quả 2×2 — cùng panel, cùng router

| Classifier / availability | Recall class 6 | Accuracy panel | Rescue / break so baseline | FP class 6 trên negatives |
|---|---:|---:|---:|---:|
| Original row / chưa đăng ký — Baseline | 0/1.024 = **0%** | 61,4977% | 0 / 0 | 0/8.404 |
| Original row / đăng ký — Availability-only | 996/1.024 = **97,2656%** | **72,0619%** | **996 / 0** | **0/8.404** |
| Updated row / chưa đăng ký — Update-only | 0/1.024 = **0%** | 61,4977% | 0 / 0 | 0/8.404 |
| Updated row / đăng ký — Full V2 | 995/1.024 = **97,1680%** | 72,0513% | **995 / 0** | **1/8.404** |

Accuracy là trên panel stratified gồm dữ liệu nhiều owner cho một receiver cố định; **không phải accuracy toàn hệ thống, không phải full test**.

So trực tiếp **Full V2 với Availability-only**:

- Rescue = **0**, break = **1**, correct delta = −1.
- Recall class 6 giảm **0,097656 điểm %**.
- Accuracy panel giảm **0,010607 điểm %**.
- Thêm 1 target false positive ở một sample vốn baseline đã sai, nên không tăng old break.

Toàn bộ gain quan sát được so baseline đóng mask được tái tạo bằng Availability-only. Không kết luận một chênh lệch một mẫu là regression thống kê; kết luận cần giữ là **không quan sát được lợi ích bổ sung của learned update trong case này**.

## Old risk, FAR và acceptance

Native old accuracy cả bốn nhánh giữ **3.498/5.496 = 63,6463%**; old rescue/break của Full V2 so baseline đều 0.

Full V2 có 1 false positive:

- Pooled negatives: **1/8.404 = 0,011899%**.
- Old negatives: **1/5.496 = 0,018195%**.
- Class 4 pooled: **1/2.162 = 0,046253%**; các class âm được quan sát khác đều 0.
- Riêng owner **81 / class 4**: **1/567 = 0,176367%**, vượt ngân sách 0,1% nếu xét riêng pool này. Pooled FAR thấp không bảo đảm mọi owner có FAR thấp.

Theo đúng gate mới đã khai báo ở mức pooled và từng class pooled, Full V2 đạt các điều kiện thực nghiệm trên panel: recall≥95%, FAR≤0,1%, negative break=0, old drop≤1pp, rescue>0. Đây chỉ là `empirical_observed_gate_pass`, **không phải acceptance của production CAL**.

Không cấp PASS/native smoke vì:

1. Chưa xác lập independent untouched evidence.
2. Không có negative class 8; scope seen 0–11 chưa được kiểm tra đủ.
3. Pool owner81/class4 có FAR cao hơn pooled result.
4. Không có benefit vượt Availability-only — cơ chế donor learning chưa được chứng minh.

Không có claim population FAR≤0,1%. Riêng class 0 chỉ có 32 unique rows; 0 observed FP trên số mẫu ít không chứng nhận tail risk. Không hạ gate hoặc biến VALIDATION/provenance thành CAL acceptance.

## Integrity và communication

- Full frozen function khớp lock cũ; live receiver và bốn function không đổi sau inference.
- Task route trên từng sample giống nhau ở bốn nhánh.
- Predictor chỉ nhận `x`, model và router; `y` dùng cho chọn scope đã khai báo và tính metric sau prediction.
- Tính lại độc lập từ prediction records: **9.428 rows × 4 variants**, đúng recall/FAR/counts/accuracy, không mismatch; `Δcorrect = rescue − break` khớp tuyệt đối.
- Không học lại, không đổi LR/regularization, không tune policy theo panel.

Không có transfer mới. Chi phí tạo update cũ giữ nguyên: packet **2.921 bytes**, receiver capsule **6.685.240 bytes**, tổng **6.688.161 bytes ≈6,38 MiB**. Không được gọi tổng transfer là 2,85 KiB. Qualification hiện chạy emulator local, chưa có measured decentralized witness communication.

## Quyết định tiếp theo

**Không đẩy cặp class 6 sang native survival của donor update.** Trước khi nghiên cứu transfer mới, discovery phải chạy reference Availability-only để nhận ra knowledge có sẵn nhưng bị local class mask chặn. Chỉ một cặp có functional gap thực sự và update vượt reference đó mới là ứng viên survival.

Availability-only là ứng viên sửa registration, không tự được triển khai rộng: cần prospective provenance hợp lệ và risk qualification. Không mở toàn bộ class mask hoặc tự coi mọi inherited class là đã được chứng nhận. Không kết luận toàn bộ train-time transfer thất bại từ một cặp; kết luận đúng là cặp này không làm bằng chứng bổ sung cho donor learning.

## Files

- Script: `tools/audit_appliance_train_time_2x2.py`.
- Public aggregate artifact: `artifacts/appliance_train_time_native_2x2_20261010.json`.
- Local protocol, per-row predictions và coordinates: `audit_denice/appliance_train_time_transfer_v1/native_2x2_01/`.

Chạy lại bằng cùng authority và prepared update:

```powershell
.venv-audit/Scripts/python.exe -m tools.audit_appliance_train_time_2x2 `
  --checkpoint audit_denice/appliance_train_time_transfer_v1/checkpoint_task_1_all_rounds.zip `
  --prepared audit_denice/appliance_train_time_transfer_v1/run01 `
  --roles audit_denice/appliance_historical_calibration_local/_runtime_5f5bbe06cee041de/roles `
  --data audit_denice/appliance_transfer_results9/inputs/dataset `
  --out audit_denice/appliance_train_time_transfer_v1/native_2x2_repeat `
  --publish artifacts/appliance_train_time_native_2x2_repeat.json
```

Các checkpoint/data/row private không nhúng vào source hoặc notebook. Script từ chối overwrite output directory để bảo toàn lock của mỗi run.
