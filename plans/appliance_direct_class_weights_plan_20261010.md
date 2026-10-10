# Kế hoạch APPLIANCE — Direct Class-Specific Weight Transfer

Ngày: 2026-10-10. Đây là protocol trước execution tại `e738737`; **P1 đã implement, P2 đã hoàn tất 0/3 PASS → NO-GO**. Xem [báo cáo kết quả](../docs/APPLIANCE_DIRECT_HEAD_DEVELOPMENT_20261010_VI.md). Các lựa chọn/gates bên dưới được giữ làm lịch sử preregistration, không dùng để mở thêm pilot sau NO-GO.

Phương pháp chính: **APPLIANCE — Decentralized Class-Aware Masked Knowledge Transfer for Federated Continual Learning**.

Cấu hình nghiên cứu chính: **APPLIANCE-Head-Agg**. Internal protocol ID dự kiến: `appliance_direct_head_v1`.

Thiết kế này thay thế ưu tiên triển khai của proposal receiver-aligned sparse donor training tại `8d91488`. APPLIANCE V2 readout-only vẫn NO-GO; proposal sparse trước đó chưa được chạy và được giữ làm lịch sử thiết kế. Không đổi tên các kết quả cũ thành evidence của direct Head-Agg.

## 1. Quyết định kiến trúc

| Thành phần | Chốt cho kế hoạch |
|---|---|
| Nền training | DeNICE legacy, decentralized, paper graph, ξ=0,8 |
| Donor | Xuất trực tiếp head đang có trong model; **không optimizer step để tạo packet** |
| Receiver | Aggregate heads trên shadow, cùng native router; không donor inference lúc deployment |
| Backbone | Giữ nguyên CNN/GRU/FC1/BN trong direct-head transaction |
| Parameter transfer | Chỉ target FC2 row/bias và approved target masks/registration metadata |
| Dữ liệu | Current BASE/CAL đúng owner/roles; không raw historical train/CAL, không replay |
| Global model/server | Không; mỗi receiver cập nhật model riêng |
| Risk policy | Empirical acceptance đã chọn trước đây; old-risk thiếu evidence ghi UNKNOWN |
| Main pilot | Head-Agg không receiver optimizer; Head-Reg là variant riêng dùng current BASE |

Simulation hiện chạy trên một runner. Không mô tả nó là deployment P2P trên 100 máy. Phải đi qua message codec/ledger; lấy peer model Python object không được tính là giao tiếp miễn phí.

Head transfer có thể tái phân phối kiến thức donor đã học. Không cần donor train lại task cũ. Tuy nhiên một FC2 row là classifier **trong feature space của donor**, không phải mô tả đầy đủ một class độc lập với backbone.

## 2. Câu hỏi quyết định

Trước full training phải trả lời:

> Với native inference của receiver, head trích trực tiếp từ hai donor có thể được tổng hợp, vượt Availability-only, đạt current empirical gates, và sống qua native training/aggregation mà không cần imported route hoặc gửi donor model lúc inference không?

Ba yêu cầu độc lập:

1. **Transfer utility:** vượt Availability-only và control chỉ thay mask/rank.
2. **Functional compatibility/risk:** target recall và protected negatives đạt gates đã khóa.
3. **Practical cost:** tính đầy đủ setup/verification/native traffic; packet nhỏ chưa chứng minh bandwidth toàn hệ thống giảm.

Nếu average heads không hơn donor đơn tốt nhất, không thể tuyên bố multi-donor aggregation là contribution đã chứng minh. Single-donor Head vẫn có thể có feasibility nhưng là variant khác với main Head-Agg.

## 3. Data, ownership và evidence

### Metadata cần tách

- `ownership_mask`: cumulative class ownership đã quan sát từ BASE khi task đó current, kèm birth task/count/provenance; không suy ownership từ inherited binary router memory.
- `current_data_mask`: class có current BASE/CAL hợp lệ; historical ownership không cấp quyền đọc lại dữ liệu cũ.
- `availability_mask` và task-context mapping: class có được native router/classifier dùng không.
- `request_mask`: functional gap đủ evidence, không phải `1-ownership_mask`.
- Class/output maturity và feature dependency/version fingerprints.
- Quality receipts có owner, role, function/version, task, sample counts và scope; certificate cũ khác function không tự xác nhận function hiện tại.

Bit-packed mask 34 classes = 5 bytes. Cumulative ownership là evidence provenance, không phải current recall/FAR evidence hoặc quyền install.

### Temporal policy

| Target/evidence | Hành vi |
|---|---|
| Target còn current, đủ positive CAL-FIT/HOLDOUT | Được xét probe/acceptance |
| Target cũ, chỉ còn weights + ownership | Được announce/export khi graph cho phép; installation **UNKNOWN/DEFERRED** nếu không có evidence hợp lệ |
| Target class chưa xuất hiện | Không request/announce future support như đã học |
| Không có positive target | Không dùng confidence/metadata thay recall; không auto-commit |

Historical head xuất từ model hiện tại có thể đã drift; không gọi đó là snapshot y nguyên lúc donor học task cũ. Không tạo replay buffer hoặc per-task packet archive để giải quyết evidence thiếu. Current model, masks và compact provenance là state được giữ. Cache cần thiết cho verification có lifecycle/byte budget, không phải raw-data memory.

BASE chỉ dùng ở native training và optional receiver Head-Reg. CAL-FIT dùng gap/selection/quality/eta; CAL-HOLDOUT chỉ mở sau khóa candidate. Development diagnostic tách khỏi final test. Old-risk development nếu đã xem trước không được gọi independent confirmation.

## 4. Functional gap và router

Kế thừa `availability_shadow()` và label-blind `Predictor`:

1. Clone exact receiver model/router.
2. Chỉ thêm target class vào **đúng task-context mapping công khai**; không mở target vào mọi contexts để cứu routing.
3. Weights, connection masks, ages và router learned state giữ nguyên.
4. Donor endpoints chạy clone trên own current CAL-FIT, receiver chạy trên own current negatives.
5. Báo registration/functional/unknown gap cùng owner scope. Khi pools khác nhau cho kết luận khác nhau, giữ từng receipt; không chọn pool thuận lợi và bỏ negative veto đã biết.

Primary inference dùng native router restored từ checkpoint, không true task ID/class. Binary-cosine/multiclass phải ghi rõ từ config; không đổi hoặc refit router giữa DeNICE baseline, Availability-only và Head-Agg. Campaign full sau này phải chọn duy nhất router protocol bằng development trước và dùng cùng protocol cho các methods.

Route không tới task target thì head có thể không được xét. Forced-context recall chỉ dùng diagnostic phân biệt **head incompatibility** với **router unreachability**, không quyết định commit, không dùng làm main accuracy. Kết quả class 24 + imported route cũ không chứng minh native Head-Agg sẽ đạt 99,5%.

## 5. Packet trực tiếp từ donor

Export snapshot từ live/current model, không đọc historical raw shards:

```text
schema_version, sender, receiver, target_class, class_mapping_hash
task, round, graph_hash, donor_model_version, receiver_reference_hash
architecture_hash, preprocessing_hash, feature_dependency_fingerprint
head_weight[256] FP32, head_bias FP32
weight_mask[256] bit-packed, bias_mask, output_rank
ownership/maturity provenance, scoped quality binding, payload checksum
```

FC2 dimensions phải lấy từ configured model, không hardcode mọi variant đều 256×34. Với model đúng kích thước này: 257×4 = **1028 bytes** weights+bias; weight mask =32 bytes. **1060 bytes chưa bao gồm bias mask, header, hashes hoặc serialization.** Codec phải đo serialized packet thực tế.

Donor weight mask khác receiver permission mask. Donor không thể gửi một bit mask rồi tự cấp quyền sửa receiver mature row. Effective head dùng raw weights×connection mask, bias×bias mask. Một learned weight bằng 0 vẫn là contribution nếu mask=1; không loại nó chỉ vì value=0.

Để tránh quên semantics: export raw row cùng masks; aggregator tính effective contributions và trả application mask. Packet không chứa raw examples, per-example features/labels, reference inputs hoặc replay memory của router.

## 6. Compatibility và aggregation

### Hai mức compatibility

**Structural:** architecture, dimensions, class mapping, preprocessing, dtype, graph/version và mask schema hợp lệ. Shape equality chưa chứng minh feature alignment.

**Functional:** exact effective feature-boundary fingerprint giống nhau là điều kiện đủ cho common coordinates khi forward policy cũng khớp. Boundary khác không bị gọi là compatible chỉ vì cùng initialization/architecture. Có thể stage direct head trên receiver và kiểm tra current FIT/CAL bằng empirical contract; đây là evidence hữu hạn, không proof exact function equivalence.

Nếu muốn kiểm tra từng donor trước aggregate, làm trên CAL-FIT. Negative evidence của từng candidate version phải gắn đúng hash. Negative veto của Availability-only không tự chứng nhận hoặc bác bỏ function Head-Agg mới; function mới cần evidence mới. Không bỏ qua risk đã quan sát của chính candidate khi chuyển sang owner khác.

Không thêm mapper, donor retraining, dependency closure hoặc shared encoder để cứu failed direct-head pilot. Nếu cần các cơ chế đó, phải redesign/variant mới, không còn là packet Head-Agg đang xét.

### Per-coordinate normalization

Với weight hoặc bias coordinate p, đặt `a_j = alpha_ij*q_j,c` và `m_j,p` là donor contribution mask sau structural validation:

\[
Z_p=\sum_j a_jm_{j,p},\qquad
W^{agg}_p=\frac{\sum_j a_jm_{j,p}W_{j,p}}{Z_p}\quad(Z_p>0).
\]

`Z_p=0`: **không apply coordinate**, giữ nguyên value/mask của receiver. Đây là aggregation của absolute weights, không phải delta nên không được ghi zero vào model ở tọa độ không có contributor. Bias cũng normalize theo actual contributors.

- Alpha được renormalize trong selected donors tại từng coordinate, không giữ denominator của toàn neighborhood chứa peers không đóng góp.
- Quality q phải được khóa bằng FIT của candidate head trên receiver function. Native donor confidence/recall không được coi là quality transfer duy nhất. Alpha-only (`q=1`) và uniform averaging là controls.
- Primary pilot giữ alpha-only để tách cơ chế direct aggregation khỏi learned quality; functional FIT vẫn dùng để eligibility. Quality weighting là secondary preregistered variant, không chọn thêm sau HOLDOUT.
- Aggregated row có thể kém cả hai row đơn dù từng donor pass. Chỉ final combined candidate mới được quyết định acceptance.

## 7. Integration, masks và protection

Receiver output slot phải chưa mature/protected và không có conflicting installed transfer. Prototype ưu tiên rank0, không current owned target BASE. Class có metadata thiếu nhưng mature row là candidate registration review, không tự overwrite.

Với receiver permission `A_p` và có contributor:

\[
W'_p=W^{old}_p+\eta A_p(W^{agg}_p-W^{old}_p).
\]

Phải xác định `W_old` effective semantics và target connection mask sau apply. Không blend raw masked-out weights rồi bật masks để làm lộ những weights chưa được validation. Coordinates không được cấp quyền, non-target rows, CNN/GRU/FC1, BN buffers và existing router learned state giữ nguyên.

Nếu cần bật target connections hoặc promote target output rank, đó là transaction metadata change được ghi rõ; không thay FC1 ranks hoặc làm dependency graft. `eta=0` control phải exact no-op về weights/masks/ages so với reference tương ứng; mask-only changes là control riêng.

### Control bắt buộc

**Registration + Mask-only:** dùng cùng target connection/bias mask, target-rank transition và availability như final head candidate, nhưng giữ receiver classifier parameters. Điều này tách gain do unmask/promote khỏi gain do donor row. Transfer phải vượt control này và Availability-only để được ghi nhận donor-weight benefit.

### Optional Head-Reg

Receiver có thể train target row/bias trên own current BASE bằng preservation/proximal loss. Conv/GRU/FC1/BN và other classifier rows frozen. Không target positives thì không giả lập target CE bằng labels từ validation; chỉ current negative anchors/teacher preservation.

Head-Reg có optimizer/compute cost tại **receiver**, dù donor vẫn không train. Không gọi cả variant này là zero-training transfer. Không dùng `donor_proposal()` hoặc receiver-aligned donor training của V2/V3 cũ.

## 8. Acceptance, transaction và lifecycle

### Empirical acceptance

Giữ numerical gates làm starting contract, lock trước execution:

- Target recall ≥95% trên required current CAL-HOLDOUT owners đủ unique positives.
- Observed FAR ≤0,1% theo từng required owner/class; negative break=0.
- Min 32 unique target positives và min 32 receiver negatives là empirical support threshold; 0 FP với số mẫu nhỏ không chứng minh population FAR ≤0,1%.
- Target benefit và positive net rescue vượt Availability-only **và Registration+Mask-only** trên identical owner rows.
- Protected weights/buffers không đổi; snapshot/class mapping/versions khớp.
- Recheck final multi-class transaction nếu cùng round cài nhiều classes.

FIT quyết định eta/donors. HOLDOUT không dùng để chọn donor/eta/variant. Failed candidate không được thay bằng donor thứ ba hoặc thêm threshold sweep.

Trạng thái rõ ràng: `EMPIRICAL_CURRENT_PASS`, `FAIL`, `UNKNOWN_EVIDENCE`, `DEFERRED`, kèm `old_risk=UNKNOWN` nếu cumulative independent evidence thiếu. Empirical policy có thể cho prospective install sau current pass; không tuyên bố strict safety. **Thiếu target positive CAL không được empirical policy biến thành target recall PASS.**

### State và atomicity

`IDLE → PROBED → STAGED → VERIFIED → COMMITTED` hoặc `ROLLED_BACK/DEFERRED`.

Backup trong transaction gồm row/bias/masks/rank, target availability entries, freeze/protection state và receipt metadata. Rollback phải exact, không chỉ reset weights. Commit lưu source donors/versions, mix weights, eta, packet checksum, acceptance function hash/scope và target protection trong checkpoint. Không lưu một donor packet archive cho từng task; keep compact provenance trong model state, chỉ transient packet/reference caches theo budget.

### Survival không tự được bảo đảm

- Pin imported target row khỏi invalid local optimizer/aggregation writes; kiểm tra post-step equality, không chỉ zero gradients với Adam momentum/weight decay.
- Không freeze toàn backbone để ép pilot pass.
- Upstream feature/router changes có thể làm head không còn hoạt động. Ghi dependency drift, version và risk status.
- Khi target còn current, dùng newly valid current CAL theo protocol để re-evaluate; không dùng lại raw CAL task cũ.
- Khi target thành historical và không còn valid evidence, chỉ carry nếu phạm vi/versions đáp ứng policy đã khóa; drift ngoài phạm vi → suspend newly introduced availability/defer refresh. Không xóa legitimate availability đã có trước transfer hoặc tự tạo inference scope bằng true test task.
- Không có đường sustain hữu ích qua next task thì native integration NO-GO cho cấu hình này, dù packet initial recall cao.

## 9. Kế hoạch thực hiện theo milestones

| Mốc | Công việc / deliverables | Gate chuyển mốc |
|---|---|---|
| P0 — Freeze protocol | Chốt schemas, variants, target slot rules, evidence policy, candidate metadata và hard budgets | Không còn nhầm absolute weights/deltas, donor masks/receiver permissions hoặc current/historical evidence |
| P1 — Core mechanics | Codec direct heads, masked absolute aggregation, shadow apply, mask-only control, full rollback | Mask/zero-denominator/version/protection accounting đúng; chưa training backbone |
| P2 — Bounded feasibility | 3 receiver–class cases tối đa, 2 fixed donors/case; native FIT chọn candidate rồi H một lần | Có transfer vượt both controls và current gates; không rescue bằng imported route |
| P3 — Cost + lifecycle | Cold/exact-cache verification, 3 native rounds, save/resume, next-task smoke | Measured total bytes; useful accepted head tồn tại hoặc chuyển state đúng, không old raw reads |
| P4 — Runner integration | Service sau router refresh; automatic discovery, staging, ledger và resume | Có automatic accepted install + reject/rollback; không manual prebuilt patch |
| P5 — Full campaign | Fresh 6 tasks×20 rounds, seed42 trước; sau lock ≥3 independent seeds | So DeNICE cùng protocol; report forgetting, coverage, risk và cost |

Không đi thẳng P5. P2/P3 fail thì đóng cấu hình theo stopping rule. Class 28 không được hardcode loại khỏi metrics hoặc dùng làm blanket training stop; báo coverage/eligibility/unknown per class.

### P0 — Choices phải lock

- Checkpoint và live graph hashes, class mapping, preprocessing, roles/store authority.
- Pilot dùng adapter-free/auxiliary-head-free endpoints, không imported-route registry đang active. Reject/skip cấu hình residual classifier hoặc context adapter chưa có dependency contract; không mặc định `fc2[c]` là toàn bộ classifier function.
- Receiver/donors/class pairs từ metadata, không prediction/test labels.
- Fixed variant list, eta grid và FIT selection rule, alpha normalization, tie rule.
- Bit-packing, serialization precision, mask/rank apply policy.
- Sampling/dedup/source-exposure ledger, current owner/class support counts.
- Empirical acceptance gates, unknown/defer/lifecycle states.
- Total attempt/time/byte budget và final summary protocol.

### P1 — Files cần implement

| File dự kiến | Trách nhiệm |
|---|---|
| `appliance/direct_head_contract.py` | Schemas, metadata eligibility, dependency/function/version binding |
| `appliance/direct_head_codec.py` | Exact row/mask export, finite/shape checks, serialization byte accounting |
| `appliance/direct_head_aggregation.py` | Per-coordinate absolute-weight normalization, application mask |
| `appliance/direct_head_transaction.py` | Shadow apply, mask-only control, rollback, protected commit state |
| `appliance/direct_head_verification.py` | Owner-local current evidence, quality/FIT selection, combined H receipts |
| `tools/run_appliance_direct_head_feasibility.py` | Locked bounded experiment, x-only inference, variants và ledger |
| `tools/build_appliance_direct_head_notebook.py` | Kaggle notebook builder sau CLI hoạt động |
| `eval_appliance_direct_head_kaggle.ipynb` | Feasibility runner; clone GitHub, đọc dataset/checkpoint paths, không embedded artifacts |

Đây là danh sách files đã chốt trước implementation. Năm core modules và feasibility CLI hiện đã có callable implementation; notebook chưa tạo vì P2 NO-GO. Mặc định không dùng `appliance/closure.py` compiler exact (có boundary-equality rejection và dependency graft), `guarded_head.py` (yêu cầu imported shared-sketch route), hoặc `train_time_transfer.py` donor learning làm core của phương pháp mới.

P1/P2 ưu tiên chạy local CPU từ checkpoint/data stores hiện có. Notebook Kaggle chỉ tạo sau CLI, clone branch GitHub hiện tại và ghi actual commit/source hashes vào artifact; không ép user phải clone một commit cố định, không nhúng source hoặc roles vào notebook. Dataset/checkpoint/roles paths là parameters, tìm đúng archive hoặc extracted manifest với provenance thay vì đoán một ZIP bất kỳ trong mount.

### P2 — Scope bounded đề xuất

Khóa metadata lại trước execution; không chọn thêm sau thấy metrics:

| Case đề xuất | Donors | Vì sao dùng |
|---|---|---|
| Task1, receiver3, class7 | 54, 51 | Receiver có measured functional gap trong development cũ; direct native donor rows chưa được chứng minh pass |
| Task2, receiver1, class13 | 74, 34 | Control cho representation/transfer khó; không coi tiny V2 gain là direct-head success |
| Task2, receiver3, class14 | 74, 26 | Metadata/current FIT available; chưa có V2 trial update do budget cũ |

Các coordinates này là **proposed development fixtures**, không final independent confirmation. Chỉ dùng khi checkpoint/role/graph authority và native direct-donor eligibility vẫn khớp. Nếu một case không đủ metadata/CAL/capacity, giữ UNKNOWN/SKIP, không substitute case mới.

Budget đề xuất: tối đa 3 cases, 6 selected donor endpoints/reference transmissions, 3 final primary aggregate candidates, 128 MiB application egress toàn chiến dịch, 15 phút execution kiểm tra tại phase boundaries; preparation/extraction tính thời gian riêng. Không có donor training, không current BASE reads trong main feasibility.

FIT preregistered grid `eta ∈ {0,25; 0,5; 1}`; alpha-only main aggregation. FIT dùng chọn eta, cố định selected donors từ metadata; zero/FAR violation tại FIT của candidate ghi rõ và không bỏ đi qua receiver pool benign. Variant-uniform/quality weighting có thể preregister như diagnostics nhưng không tăng grid sau H.

Primary H: final Head-Agg candidate được chọn trên FIT. Baseline, Availability-only, Registration+Mask-only, single Head donor1, single Head donor2 là predefined controls. Controls không được dùng chọn lại headline sau xem H. Nếu FIT chọn single donor thì main Head-Agg claim không đạt; báo outcome thay vì rename best control thành main.

Ghi native task predictions, head/logit margins như diagnostics để phân biệt incompatibility/routing; không chuyển primary metric thành forced-context sau fail. Overall metrics trên pools này phải ghi scope, không gọi full test/federation accuracy.

### Điểm dừng P2

- Không case nào final Head-Agg vượt both controls và đạt gates: **NO-GO direct Head-Agg theo protocol này**. Không train full, không hạ 95%, không thêm mapper/route/donor.
- Single Head pass nhưng Head-Agg fail: multi-donor averaging chưa đủ điều kiện; chỉ xem xét Head variant qua một quyết định protocol riêng.
- Current empirical pass nhưng old risk unknown: báo đúng giới hạn; chỉ tiếp tục P3 dưới empirical policy đã khóa, không claim strict transfer safety.
- Communication/attempt budget hết: stop với measured outcomes và unexecuted cases, không tự kéo dài.

## 10. Communication và computation

Mỗi donor chỉ gửi row packet chính; verification gửi **receiver reference** tới selected donor endpoint khi không có exact cache. Không gửi donor full model để receiver inference. Packet class24 hay 1KB không phải whole transaction cost.

Donors dùng reference clone + candidate row/mask/availability để chạy own current CAL, trả aggregate counts và hashes. Có thể reuse một exact receiver reference cho các FIT variants và final H bằng tiny candidate row changes. Cache hit chỉ khi backbone/BN/masks/router/versions khớp; native post-aggregation receiver model không mặc định đã được peer cache.

Trong simulator, tất cả function packets/candidate masks/evidence requests phải vào transport ledger. Không dùng global access tới local CAL arrays làm một centralized validation pool; từng owner endpoint giữ raw arrays.

Ví dụ corrected raw payload (MB thập phân):

- 200×1028 =205.600 bytes weights/bias.
- 200×32 =6.400 bytes weight masks.
- 800×3×5 =12.000 bytes class masks.
- Tổng **224.000 bytes =0,224 MB**, chưa bias masks/header/hashes/evidence/setup.
- 20 receiver references×6.685.240 bytes ≈133,705 MB, nếu measured codec của đúng function có kích thước đó. Không hardcode mọi model capsule đều 6,685 MB.

Log categories: native FL, metadata, direct heads, receiver setup/refresh, FIT/H queries/receipts, rejected packets, cache retained bytes, checkpoint size, forward/optimizer time. Cold vs exact-cache và amortization phải report riêng; không chỉ báo payload 1,06KB.

\[
Comm_{total}=Comm_{native}+Comm_{metadata}+Comm_{heads}+Comm_{verification}.
\]

Overhead O(R(EC+Qd+VP)) theo assumptions; nếu native full-state exchange còn O(REP), whole system vẫn phải cộng term đó. Direct packet không thay native exchange trừ khi có ablation đổi baseline protocol riêng. Head-Reg/verification có forwards qua backbone dù chỉ target row thay; computation không phải O(d) đơn thuần.

## 11. Native integration và full evaluation sau gates

Service đặt sau native aggregation/age merge/router refresh, trước round checkpoint. Trước commit xác nhận live receiver vẫn đúng reference; lưu complete registry/protection/availability vào full and delta checkpoints. Round-resume không duplicate transfer; repeated packet idempotent, stale packet rejected.

P3/P4 dùng evolving native graphs và local updates, không cố định peers/reload donor terminal mỗi round rồi gọi đó là survival. Smoke hai tasks liên tiếp với current-only data providers; calibration/version changes phải tự đổi state, không đọc historical raw arrays.

P5 dự kiến:

- Fresh Tasks0→5, 20 rounds/task, ξ=0,8; graph không bị APPLIANCE đổi thành random expert pool.
- Transfers theo bounded policy mỗi round, không chỉ last-task evaluation.
- Checkpoint mỗi round; dùng compression/retention quota code hiện có sau kiểm tra restore + registry inclusion. Đừng hứa zip giữ mọi checkpoint luôn đủ quota.
- Lưu cumulative task-boundary metrics trên **toàn bộ test của các classes đã seen**, chia disjoint receiver shards; mỗi row tính một lần. Cuối Task5 dùng full 34-class test. Không infer true task ID lúc dự đoán.
- Test readers tách khỏi donor selection/eta/acceptance, không dùng metrics test điều chỉnh live transfer.
- DeNICE baseline và variants cùng backbone/data/router/seeds/test receiver assignment. Paper campaign ≥3 seeds chỉ sau protocol lock.
- Pilot giữ existing locked role allocation. Fresh campaign phải khóa role allocation chung cho DeNICE và APPLIANCE trước training, tính toàn bộ labeled CAL dùng cho discovery/selection/acceptance; không tái phân bổ FIT/validation thành BASE sau khi xem outcomes để tăng điểm.
- Dataset test hiện đã được dùng cho nhiều diagnostics trước đây. Lock thiết kế mới không biến panel đã xem thành untouched test; cần exposure ledger và confirmation nguồn/panel chưa dùng để phát triển khi claim confirmatory. Independent training seeds kiểm tra variance, không tự xóa test-set exposure.
- Report pooled accuracy, macro-F1 theo declared class scope, per-client mean, class recall/precision, transfer rescue/break, acceptance and UNKNOWN rates, old-class forgetting, native route accuracy, total bytes and inference time.

Average forgetting cần task accuracy matrix. Chỉ eval final Task5 không đủ để tính forgetting đúng; protocol phải có task-boundary historical rows của matrix, không dùng final-only metrics làm average forgetting.

## 12. Related work và claim

[FedPAC](https://arxiv.org/abs/2306.11867) nghiên cứu feature alignment và collaboration giữa personalized classifier heads; [FedWeIT](https://proceedings.mlr.press/v139/yoon21b.html) nghiên cứu weighted transfer của sparse task-specific parameters. Vì vậy classifier-head averaging/masks đơn lẻ chưa đủ novelty.

Claim cần chứng minh của APPLIANCE là **functional-gap-driven direct head transfer trong live decentralized continual graph, receiver-specific masked integration, evidence/version-aware commit và single-local-model inference**, với contribution vượt Availability-only/Mask-only/native DeNICE và full cost accounting.

Không gọi weights donor là universal class module. Không tuyên bố no replay tự suy ra strict protection, 0 FP tự chứng minh population FAR, hoặc tiny row tự suy ra communication-efficient whole system.

## 13. Ước lượng công việc và việc bắt đầu trước

Ước lượng engineering, không gồm chờ dataset/GPU: P0–P2 khoảng **2–3 ngày làm việc** nếu reused loaders/authority phù hợp; P3–P4 thêm **1–2 ngày** khi feasibility pass. Đây không phải cam kết thời gian train full; cần runtime đo từ smoke để ước tính full training/test.

**Việc đầu tiên:** implement codec + masked absolute-weight aggregation + shadow apply/mask-only control, rồi khóa và chạy đúng bounded P2. Chưa sửa notebook full, chưa chạy training lại từ Task0, chưa thêm imported route, bridge hoặc donor optimizer.
