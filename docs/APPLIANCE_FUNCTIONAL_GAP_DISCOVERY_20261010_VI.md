# APPLIANCE V2 — Functional-Gap-Aware Discovery

## Đã thay đổi gì

Discovery V2 không còn dùng việc thiếu class trong metadata làm kết luận model thiếu knowledge. Entry point prototype `tools/run_appliance_train_time_transfer.py` mặc định chạy **functional native discovery**, trước khi đọc BASE để học update.

```text
Class thiếu trong native availability
  → quét tất cả neighbor có positive alpha trong graph hiện tại
  → donor có current owned BASE + mature target row?
  → clone receiver, chỉ đăng ký class availability
  → donor và receiver kiểm tra clone bằng current CAL-FIT của từng owner
  → phân loại gap
     registration_gap: không học donor update; xét registration riêng
     functional_gap: chỉ xét trial nếu đủ FIT và qua current risk precheck
     unknown_gap: thiếu positive evidence; skip, không xem là PASS
  → khóa candidate trước CAL-HOLDOUT
  → registration so với closed-mask baseline
  → transfer so với Availability-only trên cùng HOLDOUT rows
  → benefit + acceptance + independent risk evidence
  → mới được xét native survival/installation
```

Các helper không cài vào live model. Production installer và full runner chưa chuyển sang V2; notebook full cũ vẫn không phải implementation của phương pháp này.

## Các quy tắc giữ nguyên

- Native `pred_hard`, router `binary_cosine`; không true task/class lúc inference.
- Availability-only không đổi weights, masks, ranks, router memories hoặc learned router parameters.
- Donor eligibility chỉ dựa vào graph hiện tại, current owned BASE và maturity; không loại donor bằng native router accuracy.
- CAL-FIT dùng để phân loại/chọn candidate; CAL-HOLDOUT dùng sau khi candidate khóa.
- Positive tối thiểu 32 unique inputs, receiver negatives tối thiểu 32; recall ≥95%, FAR ≤0,1%, negative break=0.
- Transfer cần net rescue dương **và target hits vượt Availability-only**; không chỉ vượt baseline đóng mask.
- FAR kiểm tra riêng từng observed owner/class, không dùng pooled FAR che tail lỗi của một owner.
- Giữ donor steps128, integration steps64, Adam lr0,02, L2/proximal cũ; không tune lại.
- Không đọc raw CAL task cũ, không BASE/FIT/validation/test trong đường prescreen. BASE chỉ được mở khi thực sự chạy trial.
- Thiếu independent old-risk evidence không cấp phép registration/transfer/native smoke.

CAL-HOLDOUT bỏ content trùng CAL-FIT/SELECTION tại cùng owner, dedup trong từng endpoint. Chưa xác lập cross-owner independence/iid; không suy ra population FAR.

## Veto của tất cả peer, không chọn pool thuận lợi

Các donor khác nhau kiểm tra **cùng một Availability-only function** của receiver cho class đó. Nếu một peer ghi nhận false activation vượt gate hoặc negative break thì evidence đó không mất hiệu lực khi đổi donor.

`select_probes()` gom mọi current CAL-FIT negative veto cho cùng class/function. Registration không được đi tiếp chỉ nhờ chọn donor có pool thuận lợi hơn. Nếu học ra function mới, các score cũ không chứng nhận được function mới: nhánh transfer gửi row đã tích hợp tới tất cả endpoint đã query, thu protected CAL-HOLDOUT receipts mới rồi kiểm tra từng pool.

Đây là current-task protection; không biến nó thành bằng chứng bảo vệ đầy đủ các task cũ. Mọi quyết định install/native smoke trong API development vẫn `false`; verified cumulative risk receipt authority chưa được triển khai.

## Lượt chạy local

Checkpoint DeNICE legacy **Task1/round19**, receiver **2**, cùng lineage results (13):

- Terminal: `b057f27a763b63bd5ffff936926f5e7d4bb0696c07b35ada8c2d4de209799a87`.
- Graph: `0c4c7acd41b31627ef32591c31aa72d561adfc2721028dead91e459d697d3a22`.
- Roles: `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.

Quét **68 candidate pairs / 41 donor hợp lệ**, không random-k, không chọn thêm donor ngoài graph. Requests là class6 và class8 vì chúng thiếu trong router availability; không dùng cặp cố định 62/64 làm toàn bộ discovery.

| Target | Registration gap | Functional gap | Unknown gap | Tổng |
|---|---:|---:|---:|---:|
| Class 6 | **34** | 0 | 2 | 36 |
| Class 8 | **29** | **2** | 1 | 32 |
| Tổng | **63** | **2** | **3** | **68** |

Đây là số pair/probe, không phải số client mới cài patch. Gap chỉ mô tả competence trên donor's current CAL-FIT distribution, không phải khả năng trên toàn bộ global class distribution.

### Class 6

34 donor có đủ positive FIT đều cho signal registration gap. Hai pool nhỏ dưới 32 unique positives giữ UNKNOWN mặc dù observed recall cao.

Không có current negative veto được quan sát cho Availability-only/class6. Rule chọn donor27 bằng positive FIT count lớn nhất rồi donor ID, không chọn theo HOLDOUT. Cùng candidate đã khóa cho kết quả development HOLDOUT:

- Target recall **1.052/1.092 = 96,3370%**.
- Donor negatives **0 FP / 107**, receiver negatives **0 FP / 369**.
- **1.052 rescue / 0 break** so với baseline đóng mask.
- Không học update, không đọc BASE.

Những số này tiếp tục chứng minh registration có ích; **không chứng minh train-time transfer**. Old-risk độc lập chưa có, các class có ít negatives vẫn cần báo coverage, nên **chưa được install/native smoke**.

### Class 8

29 pair có target recall FIT≥95%, nhưng nhiều peer-negative pools bị false activation. Ví dụ Availability-only/class8 trên donor3 CAL-FIT:

- Class6 negatives: **82/86 bị đoán thành class8**, FAR **95,3488%**.
- Receiver2 current negatives riêng của nó không có FP; điều đó không đủ để bỏ qua negative evidence của donor3 và các peer khác.

Hai functional gaps là donor51 (**32/35 = 91,4286%**) và donor82 (**50/53 = 94,3396%**), nhưng cả hai đều không qua current risk precheck. Không kích hoạt trial theo policy hiện tại.

**Class8 registration bị chặn bởi current negative veto**, thay vì chọn một donor có negative pool thuận lợi rồi mở class. Không suy ra không thể tạo function mới an toàn; chỉ kết luận candidate Availability-only hiện tại chưa đủ điều kiện.

Việc class8 ở panel 2×2 trước không có mẫu không được sửa thành một claim independent validation: lần này có class8 trong current CAL-FIT/HOLDOUT development, nhưng đó vẫn là logical Task1 replay, không phải một confirmation mới độc lập.

## Tính minh bạch của lượt replay

Scan đầu thu đủ 68 FIT summaries và chạy HOLDOUT của hai selected registration candidates dưới risk check theo pair. Khi review, phát hiện lựa chọn donor như vậy có thể bỏ qua negative evidence từ peer khác cho cùng function.

Đã sửa default discovery bằng veto tổng hợp trên **FIT**, rồi áp dụng lại trên sealed FIT summaries qua `tools/replay_appliance_functional_gap_fit.py`. Không học lại, không mở CAL thêm, không chọn threshold theo HOLDOUT:

- Class6 giữ đúng donor27 và cùng function; HOLDOUT numbers là **reuse**, không independent replication.
- Class8 bị chặn bằng FIT veto; HOLDOUT ở pool thuận lợi của scan đầu không được dùng để cấp PASS hoặc cứu registration.
- Không install hoặc native smoke ở cả scan đầu lẫn replay.

Fresh default runner hiện kiểm tra veto trước khi mở registration HOLDOUT. Nhánh training/update và fanout protected-HOLDOUT đã được nối vào code nhưng **không được thực thi trong fixture này**: trial training runs=0. Không gọi đó là survival/integration PASS.

## Communication

Scan gửi một receiver function capsule tới mỗi donor, cache/reuse giữa class6 và class8; gửi request và aggregate FIT/HOLDOUT summaries. Không gửi raw examples, raw per-example labels hoặc receiver features trong messages.

- **41 donor capsules**; toàn scan **274.190.320 bytes ≈261,49 MiB** application egress.
- Có tính setup capsules và metadata; không gọi đây là vài KiB transfer.
- Không trained-row packet nào vì không trial learning.
- Replay sealed summaries không tạo thêm application-wire traffic; 261,49 MiB là chi phí scan trước, không tính hai lần.
- Đây là endpoint emulator, không phải socket/TLS, full FL-round cost hoặc federation-wide quota. Pairwise transport budgets không phải tổng quota toàn hệ thống.

Discovery overhead hiện vẫn lớn; việc tránh học update thừa không đồng nghĩa toàn discovery đã nhẹ. Chưa tối ưu communication vì mục tiêu lần này là sửa điều kiện xác định gap.

## Quyết định

**Discovery đã phân biệt được registration/functional/unknown và chặn learned update không cần thiết.** Chưa có candidate đủ điều kiện cho native survival trong fixture này; không full training.

Bước tiếp theo hợp lý là dùng discovery mới tìm **functional gap có current risk evidence phù hợp** trên receiver/task khác. Candidate đó phải chứng minh learned update tốt hơn Availability-only và có independent cumulative risk evidence trước native smoke. Không chạy survival class6 để lấy bằng chứng donor learning, không tự động mở class8, không sửa LR/regularization để cứu một request xuất phát từ metadata.

## Files và entry points

- `appliance/functional_gap_discovery.py`: shadow, enumerator, endpoint comparison, gap classification, negative veto, HOLDOUT assessment.
- `tools/run_appliance_functional_gap_transfer.py`: development Task1 replay, all-peer discovery → conditional trial → qualification.
- `tools/run_appliance_train_time_transfer.py`: mặc định dispatch sang discovery mới. `--protocol legacy_all_seen` chỉ tái tạo historical diagnostic.
- `tools/replay_appliance_functional_gap_fit.py`: áp dụng policy lên sealed summaries, không đọc CAL hoặc học lại.
- `artifacts/appliance_functional_gap_discovery_20261010.json`: aggregate counters, decisions, evidence/function bindings và communication; không raw inputs.

Fresh local replay, ghi vào output directory mới:

```powershell
.venv-audit/Scripts/python.exe -m tools.run_appliance_functional_gap_transfer `
  --checkpoint audit_denice/appliance_train_time_transfer_v1/checkpoint_task_1_all_rounds.zip `
  --roles audit_denice/appliance_historical_calibration_local/_runtime_5f5bbe06cee041de/roles `
  --base-store audit_denice/appliance_current_runtime_data_local/full100_v1/base_store `
  --cal-store audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store `
  --out audit_denice/appliance_train_time_transfer_v1/functional_gap_fresh `
  --publish artifacts/appliance_functional_gap_fresh.json
```

Dữ liệu và checkpoint không nhúng vào source. Local full scan lưu tại `audit_denice/appliance_train_time_transfer_v1/functional_gap_01/`; replay quyết định tại `functional_gap_negative_veto_01/`. Artifact final không phải independent acceptance hoặc kết quả test toàn hệ thống.
