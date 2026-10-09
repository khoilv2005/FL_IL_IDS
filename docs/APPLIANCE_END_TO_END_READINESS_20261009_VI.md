# APPLIANCE: báo cáo tới mốc training end-to-end

Ngày: 09/10/2026. Mục tiêu `Hoàn thành bước 1 2 3` đang **paused** theo yêu cầu người dùng.
Báo cáo này đọc artifact và code hiện tại; không khởi chạy hoặc sửa thuật toán trong lúc pause.

## 1. Kết luận hiện tại

**Chưa đủ bằng chứng để chốt full training DeNICE + APPLIANCE.**

Parameter patch có thể hoạt động trên receiver, được bảo vệ và sống qua các round thật.
Nhưng phiên bản survival đã chứng minh còn dùng raw CAL lịch sử để tái chứng nhận.
Phiên bản mới không đọc raw CAL lịch sử đã cài được hai patch; chưa có native survival
hoặc lifecycle hoàn chỉnh của phiên bản này trong runner chính.

Không có accuracy cuối cùng của full APPLIANCE campaign. Các recall 95–100% dưới đây
là recall của class được import trên các pool được ghi rõ, không phải accuracy toàn dataset.

## 2. Cấu hình và phạm vi cần giữ

- Nền: DeNICE legacy; seed 42; xi = 0.8; paper threshold neighborhood.
- Không CGoFed, không CME, không ensemble tại inference.
- Receiver inference bằng model của mình + imported route; không chạy donor model.
- Aggregation và transfer là hai đường riêng.
- Không thêm backbone đóng băng hoặc đóng băng toàn model.
- Không chọn donor, pair hoặc threshold bằng final test; không thay pair sau HOLDOUT reject.
- Class 28 thiếu coverage được báo cáo; không chặn cả campaign.
- BASE là dữ liệu training/support; CAL dùng cho calibration/acceptance. Không đổi nhãn vai trò.

## 3. Bước 1 — Một patch sống qua ba native rounds

Artifact: `audit_denice/appliance_active_native_survival_local/run_20261009_014900`.

| Nội dung | Bằng chứng |
|---|---|
| Pair | donor 27 → receiver 54, class 24 |
| Native training | Task 5, rounds 0/1/2; vẫn giữ lịch gốc 20 rounds/task |
| Federation | 98 active clients, 205.015 BASE rows |
| Update thật | 6 optimizer steps có parameter change, 0 skipped |
| Endpoint recall | 47/48 = 97,9167% |
| Incremental rescue/break | 47 rescue, 0 break; FAR = 0 trên pool đo |
| Protection/restore | Valid qua các stage quan sát; save/restore khớp |
| Independent audit | 104 comparisons, 0 mismatch |

Đạt bằng chứng survival **trong phạm vi ba rounds đã đo**. Không hoàn tất cả task,
không chứng minh chuyển task trọn vẹn hoặc replay-free calibration.
Completion giữ `passed_feasibility=false`, `replay_free_calibration_verified=false`:
observer còn giữ CAL-HOLDOUT lịch sử và dùng nó để tái chứng nhận.

## 4. Bước 2 — Nhiều pair/classes

Artifact: `audit_denice/appliance_multiclass_native_local/task3_to4_recovery_3rounds_v2`.
Các pair khóa trước; không đổi donor theo kết quả HOLDOUT.

| Donor → receiver / class | Endpoint recall | Parameter-changing steps |
|---|---:|---:|
| 37 → 92 / 19 | 100% | 3 |
| 1 → 44 / 20 | 96,0784% | 48 |
| 3 → 20 / 21 | 100% | 24 |
| 27 → 36 / 22 | 97,8261% | 6 |

Task 4 rounds 0/1/2: 89 active clients, 796.386 BASE rows; 81 changing steps tổng,
0 skips. Bốn pair đều valid ở endpoint, incremental patch break/FAR = 0 trên pool đo,
restore khớp. Independent audit: 432 checks, 0 mismatch.

Đạt survival giới hạn cho bốn pair. Vẫn dùng CAL lịch sử, chưa chứng minh phiên bản mới
hoặc toàn task transition. Không suy ra native training không gây forgetting.

## 5. Bước 3 — Calibration hiện tại, bảo vệ class cũ, drift

### Đã có code và bằng chứng

- Discovery dựa trên current local FIT và graph thật với alpha dương; không dùng inherited
  binary memory làm bằng chứng đã train class. Campaign đã khóa năm pair classes 19–23.
- CurrentCalibrationData và CurrentBaseData kiểm tra owner/role/task/provenance trước I/O;
  runtime không có fallback về unscoped historical raw data.
- Client-local calibration trao aggregate counts; receiver có staging/commit/rollback.
- Shield lưu hình học/counts của own BASE support, không giữ raw/per-example rows trong state.
  Có checkpoint restore và loại bỏ state của receiver khi representative bootstrap.
- Registry bảo vệ imported FC2 row; phát hiện head/reference/prefix thay đổi để fail closed.

Shield audit: `appliance_BASE_shield_local/task3_v2`, 66 checks, 0 mismatch.
Chứng minh tính đúng của component và bảo vệ finite recorded BASE support;
không phải population FAR certificate hoặc main-method pass.

### Hai patch theo activation policy mới đã cài thật

Artifact: `audit_denice/appliance_stable_current_acceptance_local/task3_v2`.
50 checks, 0 mismatch, hai transactional installs và Torch restore khớp.

| Pair / class | Positive CAL-HOLD | Target hits | Recall | Receiver negative CAL-HOLD | Receiver FAR / break |
|---|---:|---:|---:|---:|---|
| 60 → 0 / 21 | 336 | 321 | 95,5357% | 353 | 0 / 0 |
| 72 → 4 / 23 | 91 | 90 | 98,9011% | 119 | 0 / 0 |

Đây là current-task fixture trên checkpoint Task 3; shield được dựng hồi cứu.
HOLDOUT đã được dùng trong development, không gọi là independent final confirmation.
Chỉ hai pair được staged; ba pair còn lại vẫn được ghi nhận, không coi là đã pass.
Các artifact vẫn giữ `main_install_authorized=false` và
`actual_native_current_only_lifecycle_verified=false`.

**Thay đổi thiết kế phải ghi rõ:** activation mới dùng signature + fixed-context margin
+ BASE veto. Không áp dụng packet beta/self-confidence; confidence router chỉ được log.
Vì vậy không được lấy survival của guard cũ làm proof cho guard mới.

### Nút đang vướng ở task kế tiếp

Log: `audit_denice/appliance_current_negative_preflight.log`.

| Receiver / imported class | Task 4 negative CAL-HOLD | FAR / break | Monitor pass |
|---|---:|---|---|
| 0 / 21 | 10 | 0 / 0 | Không: dưới 32 mẫu |
| 4 / 23 | 9 | 0 / 0 | Không: dưới 32 mẫu |

Function certificate chưa đổi tại preflight. Nhưng chưa đủ mẫu CAL để pass monitor hiện tại.
Code hiện tại vô hiệu route nếu monitor fail; chưa có policy duy trì/refresh qua nhiều task
được chứng minh đúng trong native training. Không hạ gate 32 hoặc lấy BASE thay CAL.

BASE Task 4 preflight:
`audit_denice/appliance_current_BASE_negative_local/task4_preflight_v1/completion.json`.

| Receiver | Own current BASE rows | Imported false activations |
|---|---:|---:|
| 0 | 1.877 | 0 |
| 4 | 1.478 | 0 |

Kết quả hữu ích cho finite BASE protection. Chưa chạy native training trong preflight,
không giải quyết tự động CAL monitor và không chứng nhận dữ liệu chưa thấy.

### Những phần còn thiếu thật sự

1. Chốt lifecycle certificate/refresh khi task mới ít CAL, chức năng thay đổi hoặc support thiếu;
   không raw lịch sử, không tự nới acceptance.
2. Native survival ba rounds cho guard mới: update thật, current-only access, protection,
   invalidation/rollback đúng, prediction/save-restore và old-damage measurement độc lập.
3. Discovery + calibration + install chạy bên trong training, thay vì fixture trước training.
4. Shield được thu thập khi BASE còn là current data trong training từ đầu; không dựa vào
   retrospective seed để tuyên bố main replay-free.

## 6. Tình trạng nối runner

Runner có hooks cho diagnostic observer cũ và checkpoint có shield state.
Chưa thấy đường production gọi `install_stable_head` hoặc `StableHeadRegistry` từ
`fed_learning/training/decentralized_denice_il.py`.
`select_current_pairs` còn ghi round = 19; chưa có scheduler tùy round hoàn chỉnh.
Không thể coi bật notebook training hiện tại là đã có full APPLIANCE tự động.

Partial diagnostic endpoint lưu một số receiver; **không phải full federation resume state**.
Main campaign cần checkpoint đầy đủ model/router/registry/version/support/certificate,
cùng trạng thái training cần thiết và RNG; kiểm tra resume thật.

## 7. Ba mốc trước full campaign

| Mốc | Công việc | Điều kiện đóng |
|---|---|---|
| A. Đóng bước 3 | Current-only calibration, certificate lifecycle, old protection và native survival guard mới | Không historical raw recert; acceptance không nới; update/protect/invalidation/restore đúng |
| B. Nối runner chính | Detect → discover → compile → current CAL → staging → commit/rollback → protect/register | Chạy thực trong training, đúng graph/task/round; full checkpoint/resume; một receiver model inference |
| C. Smoke hai task | Hai task liên tiếp với training/aggregation/finalization/transition | Patch sống hoặc disable/refresh đúng; không future/test leakage; resume khớp |

Sau đó mới chạy từ đầu sáu task × 20 rounds/task, xi = 0.8, checkpoint mỗi round
và archive theo task. Final evaluation dùng toàn test set theo protocol đã khóa;
missing-class coverage được báo rõ, không dừng vì class 28.

Số lượt debug còn lại không xác định. Không có cơ sở đưa phần trăm readiness, thời gian
training tăng hoặc accuracy full APPLIANCE dự kiến từ các pairwise recall hiện tại.
Setup encoder/capsule traffic đã từng đo khoảng 6,6–7,3 MB; patch vài KiB không phải
tổng traffic. Chưa có benchmark overhead production end-to-end.

## 8. Trạng thái pause

Các worker stable install, CAL preflight và BASE preflight đã exit 0.
Chưa khởi chạy native ba rounds cho guard mới sau các preflight trên.
Trong báo cáo này không chạy thêm experiment, không commit/push hoặc sửa thuật toán.
Mục tiêu giữ paused, chưa complete.

## 9. Cập nhật sau khi user yêu cầu tiếp tục

Phần 8 là snapshot lúc pause. Sau đó user đã yêu cầu tiếp tục và khóa lifecycle
V1: thiếu CAL mới không tự phá certificate cũ; carry chỉ trong scope cũ,
function/version không đổi; domain/conflict mới chưa chứng nhận phải suspend.

Đã tích hợp observer current-only vào native runner. Lifecycle audit sau sửa
pass 60/60, current Task4 CAL monitor pass 18/18. Native round0 đã giúp tìm ra
false drift do `+0.0/-0.0` trong query fingerprint; effective tensors không
thay đổi giá trị. Numeric-equivalence repair pass 36/36, giữ immutable seal,
acceptance, thresholds và scope; vẫn reject một ULP thay đổi thật.

Native v3 đang chạy lại 3 rounds trên toàn 89 clients, legacy / xi=0.8. Không
tính run v2 bị ngắt là native pass. Chưa chốt automatic discovery/install,
hai-task smoke hoặc full end-to-end. Xem specification/evidence mới tại
`APPLIANCE_LIFECYCLE_V1_20261009_VI.md` và log
`audit_denice/appliance_current_native_scope_v3.log`.

### Kết quả sau khi native v3 hoàn tất

3actual rounds đã chạy xong,89clients/round. Hai imported functions được giữ
nguyên qua localtraining, aggregation và routerrefresh;6trackedparameter-changing
optimizersteps,0skip. Endpointchecker pass56/56, oldscope carry và newdomain
suspend đúng, không historicalrawCAL. Đây là pass của **scope-lifecycle và
certified-function retention probe**, chưa phải full end-to-end pass.

Livecurrentinstallation service cũng đã tạo2patch trực tiếp từ donor/currentCAL
(12checks0mismatch), live-discovery API reproduce đúng5originalpairs trên đủ80
activeclients ởTask3 (6checks0mismatch). Preparation đủ100ownedBASE/CALstores.
Đã có services để tiếp tục automaticintegration; callback native install và
two-taskautomatic smoke vẫn chưa được xác nhận.

## 10. Production integration và smoke đã hoàn tất

Trạng thái mới thay phần “callback/smoke chưa xác nhận” ở trên. Automatic service
đã được nối vào runner, có 2 scoped automatic commits, 3 rollback đúng. Native
Task4 chạy 2 rounds trên 89 clients; uninterrupted vs round-resume pass20/20,
0mismatch. Không historicalCAL, head/certificates còn nguyên, communication
bao gồm setup. Có launcher fresh6×20/seed42/xi.8 và external runtime nhỏ.

Full training chưa khởi chạy. Scoped certificates vẫn suspend khi application
domain chứa class ngoài scope đã chứng nhận; đây là giới hạn thực tế quan sát
được, không phải authorization cho cumulative inference. Báo cáo hiện tại:
`APPLIANCE_PRODUCTION_INTEGRATION_20261009_VI.md`.
