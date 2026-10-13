# Audit coverage, donor selection và false activation của APPLIANCE

## Nguồn và cách audit

- Training output: `C:\Users\khoak\Downloads\results (13).zip`.
- Full eval: `denice_appliance_full_eval_task_5_20261010_031923_632938.zip`.
- Native training: commit `92fe732`, legacy DeNICE, ξ=0.8, seed42.
- Terminal checkpoint checksum trong task archive: `3003593d64e3d63a81200b90d1a6e7487fff6c0875e3518cd58c2ccad22fe391`, **khớp checkpoint SHA trong evaluation lock**. SHA của ZIP chứa checkpoint khác SHA của `checkpoint_task_5.pt`; eval Kaggle dùng full terminal đã giải nén.

Đọc 126 callback records, live graph đã lưu, task-start BASE histograms, 57 transactions, guard locks, lifecycle, communication và prediction CSV. Không chạy inference/training, không fit model, không mở raw BASE/CAL lịch sử, không đổi threshold hoặc certificate.

Script: `tools/audit_appliance_discovery_output.py`. Artifacts local: `audit_denice/appliance_coverage_v2/` (`summary.json`, `request_funnel.csv`, `transactions.csv`, `lifecycle.csv`, `false_activation_by_true_class.json`, `missing_owned_provenance.json`, `selection_vs_holdout.json`).

## 1. Tổng kết: lifecycle không làm mất patch

| Task | Requests ở callback cuối | Donor offers | Thử install | Commit |
|---|---:|---:|---:|---:|
| 0 | 53 | 38 | 13 | 0 |
| 1 | 104 | 137 | 16 | 4 |
| 2 | 101 | 79 | 12 | 1 |
| 3 | 130 | 95 | 16 | 3 |
| 4 | 80 | 6 | **0** | **0** |
| 5 | 10 | 7 | **0** | **0** |

Tổng **57 attempts → 8 commit, 49 reject**:

- **30 reject** `STABLE_CURRENT_AGGREGATE_ACCEPTANCE_REQUIRED`: tất cả có target recall CAL HOLDOUT **<95%**. Đây là rejection theo gate đã khóa, không phải bằng chứng cần giảm threshold.
- **19 reject** `NONMATURE_STABLE_GUARD_DEPENDENCY`: compiler không thể chứng nhận function ổn định khi dependency còn young. Không đủ log để gán từng rejection cho một layer/neuron cụ thể.
- Tất cả rejected transactions có rollback verified.
- Tất cả head/certificate observations trong lịch sử vẫn exact/current. Không có evidence patch bị optimizer/aggregation ghi đè.
- 7 capability còn authorize. Receiver9/class12 bị suspend đúng vì Task3 CAL FAR=18,4332%, break=40; conflict được latch. Không bật lại patch này dựa trên FAR nhỏ hơn ở task sau.

**120 round callbacks đều zero offers/zero transactions**. Offers xuất hiện sau native task finalization khi output mature. Automatic discovery được gọi mỗi round, nhưng run này chỉ transfer ở cuối task.

## 2. Vì sao Task4/5 không install?

### Task4: funnel trên 80 requests

| Bước chặn đầu tiên | Request/class pairs |
|---|---:|
| Thiếu owned old BASE protection | 35 |
| Receiver đã có một capability | 10 |
| Có offer class đó toàn federation, nhưng không có trong positive neighborhood | 14 |
| Không có offer cho class cần import | 21 |
| Được chọn | **0** |

Các offer chỉ thuộc classes24/25/26; classes27/28/29 không có offer. Đây không phải callback bị skip hoặc budget hết. Không có eligible receiver–class–donor pair sau các gate.

### Task5: funnel trên 10 requests

- 3 bị thiếu owned old BASE protection.
- 2 receiver đã có capability.
- 5 request còn lại không có offer cho requested class.
- Requests là **30/31/33**, cả **7 offers đều là class32**: không có class overlap để matching. Tăng budget hoặc đổi alpha không tự tạo donor cho 30/31/33.

### Hạn chế của định nghĩa request

`local_requests` chỉ nhận **current-task class**, không có BASE/CAL local và `fc2 rank==0`, router refreshed, đủ current CAL negatives. Vì thế:

- Class đã có slot/memory nhưng classifier yếu không được request repair.
- Không phát hiện/revisit class thiếu của task cũ ở task mới.
- Receiver đã có một capability bị chặn toàn bộ request sau đó, kể cả capability đang suspended.

Đây là constraints của protocol V1, không phải lỗi runtime. Mở multi-capability, repair hoặc historical capability lookup cần thiết kế và integration riêng; không bỏ gate âm thầm.

## 3. Owned protection thiếu là provenance thật, không phải restore bug

Đối chiếu từng missing class với BASE histogram của tất cả task trước mà receiver thực sự active:

- **1.611 task–receiver–class missing observations** ở sáu terminal snapshots.
- **0** trường hợp đã có previous active local BASE rows cho class đó nhưng shield vẫn bị ghi thiếu.

Những mature class rows không có local observation có thể đến từ representative bootstrap/aggregation. Model có knowledge/mature row không tạo ra **owned BASE evidence**. Task-start debug có representative clone events; shield là own-only, nên không được tự gán dữ liệu donor thành owned receiver support.

Số 1.611 là observations có lặp qua task, không phải 1.611 clients hoặc capabilities riêng biệt. Artifact này không chứng minh mọi inherited prediction đều tốt; nó chỉ xác định đúng provenance gap của protection gate.

Nếu muốn tăng coverage mà giữ bảo vệ inherited knowledge, cần một protocol **peer-provided protection summaries** có provenance riêng. Shared sketch cho phép nghiên cứu trao đổi summary; đây chưa phải implementation đã được chứng nhận. Không được sửa thành “coi mature row là owned” hoặc đọc lại raw BASE/CAL task cũ lúc runtime.

## 4. Donor quality chưa đo portability sang receiver

Offers xếp donor bằng own FIT Wilson quality LCB. Nhưng donor dự đoán tốt trên model của nó không bảo đảm head + guard hoạt động tốt trên receiver.

Trong 38 transactions tới được guard selection/HOLDOUT:

- 30 fail HOLDOUT recall; 24 trong số đó đã có SELECTION recall dưới95%.
- 8 commit; **2 commit cũng có SELECTION recall dưới95%**.

Vì vậy không thêm cutoff SELECTION95% chỉ để loại 24 reject: cutoff đó cũng sẽ loại các patch đã pass acceptance. Chưa có controlled evidence cho cutoff mới.

Hướng phù hợp hơn là **receiver-specific FIT compatibility** trước khi chọn duy nhất donor cuối, vẫn trong live graph, khóa trước SELECTION/HOLDOUT. Maturity compatibility nên được kiểm tra trước setup capsule. Không thử donor thay thế theo kết quả HOLDOUT.

## 5. False activation có dạng rất cụ thể

| Receiver / patch | True class bị nhận nhầm nhiều nhất | Số nhận nhầm / samples class đó | Tỷ lệ |
|---|---|---:|---:|
| 0 / class21 | **class14** | 15.637 / 15.861 | **98,59%** |
| 55 / class20 | **class13** | 13.085 / 13.182 | **99,26%** |
| 65 / class20 | **class13** | 13.116 / 13.182 | **99,50%** |

Ba patch có **42.003 false activations**. Trong đó **41.995 nằm ngoài initial CAL scope** (chỉ 8 nằm trong scope). Các đối thủ class13/14 đều không có owned BASE support được guard yêu cầu trên receiver tương ứng:

- Receiver0: đã học Task0; không có local observation class14 từ Task2.
- Receiver55: đã học Task0/1; không có local Task2 class13.
- Receiver65: có Task2 BASE cho12/14/15/16, không có class13.

Patch acceptance lúc Task3 chỉ có current CAL negatives ở18–23. SELECTION/HOLDOUT FAR=0 trên scope đó không đo separation của imported class20/21 với13/14.

Các threshold đã khóa: receiver55 τ≈0,1669; receiver65 τ≈0,1399. Chúng cho thấy cosine gate khá rộng, nhưng **không đủ chứng minh** rằng tăng τ sẽ sửa được lỗi mà giữ target recall. CSV không lưu cosine/margin/veto per sample; không thể quy toàn bộ aliasing cho random projection16D hoặc chọn threshold mới từ đây.

Lifecycle monitors Task4/5 cũng không retest Task2 negatives. Receiver0 cuối Task5 có0 current negatives; receiver55 có9, receiver65 có4. FAR=0 ở các monitor này không phủ class13/14. Empirical carry-forward thực hiện đúng policy đã chốt; nó không chứng nhận population FAR ngoài CAL evidence.

Ít break chỉ vì phần lớn các mẫu13/14 đó Self đã sai sẵn. Không được diễn giải 27 break là imported route có precision gần100%.

## 6. Class28: protocol evidence không đủ

Clean roles toàn federation: BASE1208, CAL30, FIT121, VALIDATION151. CAL30 phân tán trên16 clients; client nhiều nhất chỉ **4 CAL class28**.

Donor offers cần own current FIT positive evidence và reserved HOLDOUT support (hiện min32). Với tối đa4 CAL/client, không donor nào có thể pass. Class28 có **zero terminal offers**. Không phải class bị xóa hoặc final evaluation bị chặn.

Đổi rare-class allocation/evidence policy là protocol thay đổi cần khóa trước training. Không lấy FIT/VALIDATION/test rồi đổi tên thành CAL, không giảm gate theo test để ép class28 pass. Class28 chỉ có648 test samples; riêng sửa class này không giải quyết mục tiêu global50%.

## 7. Communication: patch nhỏ nhưng setup vẫn lớn

APPLIANCE application egress tổng **428.974.131 bytes (~409,1 MiB)**, baseline aggregation tách riêng:

- Receiver function capsules: **379.703.672 bytes**, ~88,5% tổng.
- Capability payloads: **162.524 bytes** (~158,7 KiB), gồm rejected attempts.
- Transaction wire theo outcome: committed **53.732.487 bytes**; recall reject **202.839.904**; nonmature reject **123.949.992**.
- Discovery còn chi phí gửi requests/offers qua từng edge ở mọi callback.

Không gọi capability vài KiB là toàn bộ transfer cost. Maturity preflight có thể tránh setup cho structural rejects; lượng tiết kiệm thực tế cần đo sau integration, không coi mọi byte reject đều có thể xóa mà vẫn chọn được cùng patch.

## 8. Kết luận và thứ tự sửa trước full run tiếp theo

Pipeline đã cài, protect, carry-forward và dùng patch trong single-model inference. **Không cần thêm survival/pairwise audit để chứng minh lại tám patch này.** Blockers hiện tại là coverage/provenance và receiver-specific portability; full-test false activation là vấn đề separation ngoài CAL scope.

1. **Preflight dependency maturity theo receiver–donor–class** trước capsule; ghi reason theo layer và FIT transfer bottleneck. Chỉ metadata/model checks, không mở HOLDOUT.
2. **Receiver-specific FIT donor selection** trên một shortlist graph đã khóa; thử bounded integration, chọn một donor trước SELECTION/HOLDOUT. Không tăng budget vô hạn hoặc fallback theo HOLDOUT.
3. **Peer protection summaries có provenance rõ** cho inherited/absent local classes, cập nhật từ current data trong quá trình training. Tách owned và peer evidence; không giả ownership. Cần thử bounded cases cho protection, recall loss và payload trước đổi main protocol.
4. **Negative separation ngoài local support** dùng development/current streaming summaries hợp lệ. Đánh giá class13/14 chỉ như failure diagnosis; không tune từ test CSV. Chưa đủ artifact để chốt dimension/τ/γ mới.
5. Multi-capability/repair/rare-class protocol là mốc riêng sau khi các bước trên có bằng chứng. Không chạy full6-task chỉ để thay budget trước khi chứng minh số eligible/accepted capability tăng.

Giữ ξ=0.8, backbone, optimizer, guard thresholds và certificate hiện tại trong audit này. Không có production algorithm changes hoặc native test mới. Kết quả một seed, không phải causal ablation giữa mọi component.
