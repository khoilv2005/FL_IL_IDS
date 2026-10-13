# APPLIANCE — tích hợp runner và smoke Task 3 → Task 4

## Kết quả kỹ thuật

Automatic discovery/install đã được nối vào `decentralized_denice_il.py`, sau router refresh mỗi round. Có thêm callback sau task finalization vì head của task hiện tại chỉ đủ mature tại bước này. Không promote head trẻ để làm donor hợp lệ.

Service dùng đúng graph/positive alpha của round vừa chạy, CAL của đúng client/task và guard đã khóa. Không nhận file patch chuẩn bị sẵn, không gọi historical CAL, không dùng CGoFed/CME.

Hai lần cài tự động ở terminal Task 3:

| Donor → receiver | Class | CAL recall | Rescue | Break |
|---|---:|---:|---:|---:|
| 60 → 0 | 21 | 95,5357% | 321 | 0 |
| 72 → 4 | 23 | 98,9011% | 90 | 0 |

Ba yêu cầu khác bị từ chối vì thiếu owned BASE support cho knowledge cũ; physical weights không thay đổi. Controlled second-capability request cũng rollback đúng. V1 vẫn giới hạn một capability/receiver và giữ selector một receiver/class đã kiểm chứng; chưa phải scheduler phục vụ mọi missing class đồng thời.

## Smoke và resume

Task 3 dùng checkpoint terminal thật, toàn bộ 80 endpoint của graph thật, compile patch mới qua chính production callback. Sau đó chạy native Task 4, 89 active clients, hai round, tối đa 256 BASE rows/client, CPU FP32.

Owned BASE sketch summaries được seed hồi cứu để bắt đầu giữa chừng; không seed patch thủ công và không đọc CAL lịch sử. Đây là integration smoke từ checkpoint, chưa chứng minh fresh Task 0 → Task 5.

So sánh nhánh dừng sau round 0 rồi resume với nhánh chạy liền hai round:

- **20/20 checks, 0 mismatch**.
- Toàn bộ model weights, algorithm state, novelty, neuron ages, reference bank và RNG khớp exact.
- Registry/certificate khớp; hai head và certified functions còn nguyên sau training, aggregation và finalization.
- Prediction/activation trên probe không truyền nhãn khớp.
- Communication ledger giữ nguyên prefix qua resume; callback không bị chạy lại.
- Task 4 chỉ có 10/9 CAL âm HOLDOUT ở hai receiver; training vẫn hoàn tất.

Evidence:

- `audit_denice/appliance_production_smoke_local/task3_to4_v2/controlled/controlled_automatic_endpoint.pt`
- `audit_denice/appliance_production_smoke_local/task3_to4_v3/controlled_integration_independent_audit.json`
- `audit_denice/appliance_production_smoke_local/task3_to4_v3/native/continuation_state_task_4.pt`
- `audit_denice/appliance_production_smoke_local/resume_verification_v1/completion.json`
- `artifacts/appliance_production_integration_smoke.json`

Checker controlled ban đầu so cả metadata runtime scope được callback chủ động cập nhật, nên báo mismatch ở invariant “source unchanged”. Independent checker đối chiếu weights với checkpoint gốc xác nhận cả ba rollback giữ nguyên weights. Bản smoke v3 được ngắt sau khi đã lưu terminal nhánh resume để thay nhánh kiểm tra bị trùng bằng phép so **uninterrupted vs resumed** thực sự. Không tính completion chưa hoàn tất của các driver cũ là PASS.

## Checkpoint

Mỗi round lưu atomic `continuation_state_latest.pt`, gồm toàn federation, model/router/registry/certificate, owned summaries, authority state, communication ledger, CANC preparation, reference data, previous valid cluster, guard streak, RNG và GradScaler nếu có.

Resume giữa task bỏ qua task preparation đã thực hiện. Optimizer Adam vẫn được khởi tạo lại mỗi NICE phase như thuật toán nền; không tạo state optimizer giả giữa round. Giữ một latest full continuation để tránh nhân dung lượng full snapshot theo số round. Checkpoint delta vẫn mỗi round; archive từng task như trước. Task-end continuation thay latest sau finalization.

## Communication

Controlled install + hai round/finalization Task 4 ghi **15.136.996 application bytes**, bao gồm discovery, receiver function capsules, selection/acceptance counts và capability packets. Riêng setup capsules là **13.613.360 bytes**; hai capability packets tổng **8.557 bytes**.

Không gọi vài KiB patch là toàn bộ transfer cost. Byte accounting là application-payload simulation; không đo socket/TLS overhead. Communication nền của graph aggregation được log riêng.

## Giới hạn quan trọng: scope

**Integration PASS không đồng nghĩa imported route được phép hoạt động trên toàn cumulative test.**

Certificate class 21 chỉ bao phủ `[19,20,21,23]`; class 23 bao phủ `[18,19,20,21,23]`. Application domain ở Task 3/4 chứa toàn bộ class đã thấy. Nó rộng hơn hai certificate nên lifecycle suspend ngay khi kiểm tra domain tích lũy. Nguyên nhân là scope, không phải head hỏng hay CAL ít làm mất certificate.

Trong scope cũ: carry-forward hợp lệ và certificate giữ nguyên. Ngoài scope: giữ/protect head nhưng route suspend, inference dùng original local prediction. Không suy ra scope từ `y_true` hoặc predicted task. Không mở rộng certificate bằng BASE veto.

Vì vậy chưa có claim APPLIANCE tăng accuracy trên full cumulative test. Nếu chạy cấu hình strict hiện tại, phải báo cả fraction/number route được authorize và suspend. Không diễn giải fallback accuracy thành hiệu quả knowledge transfer.

## File full đã chuẩn bị

- `train_denice_appliance_full_kaggle.ipynb`
- `train_denice_appliance_full_kaggle.py`
- Runtime source: clone latest GitHub `main`; no runtime ZIP upload required.

Cấu hình: fresh Task 0 → Task 5, 20 round/task, seed 42, ξ=0.8, legacy DeNICE, không CGoFed/CME, guard/scope policy giữ nguyên. Full test toàn bộ 34-class source, chia shard disjoint giữa receiver, sau Task 5 finalization và khóa balanced multiclass routers. Baseline và APPLIANCE prediction được tính trên cùng rows; nhãn chỉ đi vào metrics sau prediction. Không có class-28 stop gate.

On Kaggle enable Internet and attach the 100-clients dataset. The notebook clones GitHub main and verifies the smoke source checksums. Optional DENICE_CLEAN_ROLES_DIR and APPLIANCE_RUNTIME_DATA_DIR may point to existing compatible roles/current stores.

Baseline fresh DeNICE gốc dùng cùng launcher với `DENICE_APPLIANCE_ENABLED=0`, cùng clean roles, seed và test partition. Default output directory của baseline là `results_denice_legacy_control_...`, tách khỏi `results_denice_appliance_...`. `BinarySelf`/`MulticlassSelf` trong output APPLIANCE chỉ là self-decision ablation trên weights của run đó, không thay thế run DeNICE control độc lập.

Full campaign **chưa khởi chạy**. Đây là launcher strict scoped; không tự mở route chưa được chứng nhận để đạt gain trên test.

## GitHub launcher

Notebook clones latest `main` from `https://github.com/khoilv2005/FL_IL_IDS.git`, skips LFS checkpoint downloads, and logs the actual commit. No commit pin and no runtime ZIP. Smoke checksums still verify the validated algorithm sources.
