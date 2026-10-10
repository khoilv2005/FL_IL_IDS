# APPLIANCE V3 — Contract nghiên cứu Decentralized Masked Knowledge Transfer

Ngày: 2026-10-10. Trạng thái: **DESIGN / CONTRACT — chưa triển khai V3, chưa benchmark, chưa được cấp phép full training.**

Tên phương pháp: **APPLIANCE — Functional-Gap-Aware Decentralized Masked Knowledge Aggregation**. Cấu hình nghiên cứu chính đề xuất: **APPLIANCE-Sparse**. V2 readout-only vẫn [CLOSED / NO-GO](APPLIANCE_V2_NO_GO_20261010_VI.md); V3 là redesign mới, không đổi nhãn các kết quả V2 thành kết quả V3.

## 1. Kiến trúc và phạm vi

Nền: DeNICE legacy, `mode=decentralized`, `denice_clustering_mode=paper`, ξ=0,8. Mỗi receiver giữ model/router riêng, chỉ trao đổi trên live positive-alpha collaboration edges. Không global model, không central aggregation server, không CGoFed/CME, không imported route, không donor model trong inference.

Implementation hiện tại của DeNICE là simulation trên một runner. Runner nắm state của các client để mô phỏng communication/aggregation, không phải deployment P2P thật. V3 phải dùng endpoint/transport interfaces và ledger, không được coi truy cập Python object của peer trong simulation là một network message miễn phí.

```text
Local training
→ Native DeNICE graph aggregation + age merge
→ Native router refresh
→ Current metadata announcements
→ Availability-only functional-gap probe
→ Receiver chốt donors + capacity allocation + common reference
→ Donors học sparse updates trên own current BASE
→ Receiver masked aggregation + protected integration trên shadow
→ Khóa final candidate
→ Endpoint CAL-HOLDOUT verification
→ Empirical commit hoặc rollback
→ Checkpoint
```

Các class trong cùng round không được commit riêng rồi mặc nhiên coi tổ hợp an toàn. Model cuối sau tất cả updates/integration/router changes mới là function được verification và commit.

## 2. Ba masks và semantics

Với 34 classes, mỗi mask bit-packed dài 5 bytes. Ba masks dài 15 bytes; đây chỉ là bit payload, chưa tính headers, versions, ownership counts, maturity hay evidence summaries.

| Mask | Semantics |
|---|---|
| `data_mask_current[c]` | Có current owned BASE hợp lệ của class c, đúng role/task; không suy ownership từ inherited router memory |
| `availability_mask[c]` | Class được native classifier/router cho phép dùng; ghi kèm task-context mapping, không chỉ một bit |
| `request_mask[c]` | Functional gap đã được đo, đủ current evidence và đủ precheck để xét transfer |

`request_mask != 1 - data_mask_current`. Announcement lịch sử phải có provenance riêng; historical ownership không cho phép đọc lại raw historical BASE/CAL trong donor update generation. Chỉ target class còn current được request trong prototype prospective đầu tiên. Future classes không được dùng để chọn candidate/capacity.

`registration_gap`: xét availability repair riêng, không tạo weights update. `unknown_gap`: skip. `functional_gap`: mới xét học update. Native donor accuracy không phải final transfer eligibility; competence cần đánh giá function trong receiver coordinates.

## 3. Live graph và lựa chọn donor

\[
\mathcal N_i^{t,r}=\{j\ne i:\alpha_{ij}^{t,r}>0\}.
\]

Receiver hỏi metadata tất cả eligible neighbors, nhưng chỉ truyền reference tới donors được chọn bằng development protocol khóa trước. Ghi `task`, `round`, `graph_hash`, receiver/donor IDs và alpha provenance.

Eligibility: graph edge, current BASE ownership, native maturity/provenance, receiver capacity/shape compatibility, sufficient FIT positive evidence, và không bỏ qua negative-risk veto đã quan sát cho cùng function. Alpha là graph prior cho aggregation, không phải bằng chứng donor phân loại tốt.

Quality `q[j,c]` phải finite, nonnegative, có function hash/role binding. Không lấy recall Availability-only thấp của receiver làm quality donor: số đó mô tả gap. Đề xuất quality của **receiver-aligned donor proposal trên CAL-FIT**, tách khỏi BASE fitting và khóa trước HOLDOUT. Uniform `q=1` là control bắt buộc. Chi tiết estimator/selection và hyperparameters phải preregister trước prototype; không dùng HOLDOUT để quyết định q hoặc chọn lại donor.

## 4. Receiver-aligned reference

Một common reference bao gồm:

- Named parameters, architecture/shapes và preprocessing version.
- BN running stats/counters và các buffers ảnh hưởng forward.
- Connection/bias masks, neuron ages, task-freeze state.
- Router memory/parameters, availability và active-context policy.
- Capacity allocation contract, cùng ordering của coordinates trên mọi donor clone.

`reference_hash` là fingerprint của toàn bộ function/algorithm state có liên quan, không chỉ `fc2.weight` hoặc encoder weights.

Receiver lập reference **sau native aggregation và router refresh**. Mọi donors clone cùng reference và cùng allocation. Donor training không sửa model thật của receiver/donor.

\[
\Delta_{j\rightarrow i,c}=\theta_i^{(j,c)}-\theta_i^{ref}.
\]

Không average heads native của donors đã drift. Không mix packets từ reference hashes khác nhau. Nếu receiver đổi weights/masks/BN/router sau reference, reject stale transaction hoặc khởi tạo transaction mới; không âm thầm rebase delta.

## 5. Sparse capacity: điều V3 phải bổ sung thật sự

V2 đã học receiver-aligned target readout. Chỉ thêm donor thứ hai vào weighted mean của cùng readout **chưa chứng minh redesign V3 giải quyết giới hạn của V2**. APPLIANCE-Sparse phải cho thấy contribution từ capacity/features ngoài target readout.

NICE/DeNICE dùng ranks: `0=reserve`, `1=learner`, `>=2=mature`. Trong `forward_output`, FC1 reserve có thể bị `MaskedOutYoung` loại khỏi classifier training. Vì vậy chỉ gửi free-mask rồi train không đủ: capacity phải được **cấp phát/promote trên staging** trước donor training và cùng như nhau trên mọi clone.

### Phạm vi prototype đầu tiên đề xuất

- Target `fc2` row/bias được cấp phát, chưa mature/owned trái protocol.
- Một tập nhỏ FC1 reserve rows được receiver cấp phát bằng metadata/capacity rule khóa trước; học row weights/bias và target-readout connections được phép.
- Conv/GRU, BN affine/running stats/counters, mọi FC1 row khác và mọi non-target classifier row giữ nguyên.
- Native router parameters, memories và feature-selection state giữ nguyên trong prototype đầu tiên; chỉ xét đăng ký target availability theo cùng policy ở controls. Features đi vào router vẫn có thể đổi khi FC1 đổi, nên phải đo native route reachability/risk; không gọi frozen router parameters là frozen routing behavior. Không post-hoc refit bằng historical raw data để cứu candidate.
- Không tự mở micro-adapter hoặc shared encoder mới. Nếu cần, đó là variant riêng với protocol mới.
- Không cập nhật layer đang bị `task_freeze_layers` chặn. Không có legal reserve path thì trả `NO_ELIGIBLE_CAPACITY`; không tự mở mature neurons.

Số FC1 rows và exact sparse coordinate budget phải được chọn bằng shape/capacity trước prediction, ghi trong locked prototype protocol. `s=10.000` trong bảng communication của specification là minh họa, chưa phải measured/configured update size.

Receiver phải kiểm tra read paths từ reserve rows tới existing mature readouts và router activation features. Freeze old readout weights không đảm bảo logits hoặc routes cũ giữ nguyên nếu features thay đổi. Không zero các connections của mature rows để tạo capacity rồi tuyên bố mature parameters không đổi.

### Allowed update mask

\[
A_{i,c}=M^{allocated}_{i,c}\land M^{not\_mature}_{i}
\land M^{legal\_connections}_{i}\land M^{not\_task\_frozen}_{i}.
\]

Donor chỉ sửa subset của `A`. Receiver kiểm tra lại, không tin donor mask để cấp thêm quyền. Parameter/buffer không nằm trong allowlist được giữ nguyên; untracked state mặc định không được sửa. Không dùng trực tiếp `build_compatible_mask()` làm contract V3: native helper có trường hợp untracked tensors được phép update và native pair mask có label-overlap restrictions khác mục tiêu missing-class transfer.

Zero gradients ngoài mask chưa đủ với Adam momentum/weight decay. Cần masked optimizer state hoặc post-step restore/project và kiểm tra byte equality của protected parameters/buffers. Allocation state cũng thuộc transaction, rollback cùng weights.

## 6. Packet và masked aggregation

Packet tối thiểu:

```text
schema_version, task, round, graph_hash, receiver, donor, target_class
architecture_hash, preprocessing_hash, reference_hash, allocation_hash
parameter_schema_hash, named-tensor coordinates, coordinate_mask
FP32 delta values, quality_FIT_binding, payload_bytes, checksum
```

Coordinates phải hợp lệ, unique, trong allowed mask, đúng shape/dtype, values finite. Không double-count duplicate packets hoặc duplicate coordinates. Receiver không tự sửa clipping/selection sau HOLDOUT. Tuân thủ hard byte/coordinate budgets trước accept payload.

Đặt `w[j,c]=alpha[i,j]*q[j,c]` và `B[j,c,p]` là packet mask sau khi intersect allowed mask. Normalize riêng mỗi coordinate:

\[
Z_{i,c,p}=\sum_{j\in\mathcal D_{i,c}}w_{j,c}B_{j,c,p},
\]

\[
\overline\Delta_{i,c,p}=
\begin{cases}
\dfrac{\sum_j w_{j,c}B_{j,c,p}\Delta_{j\rightarrow i,c,p}}{Z_{i,c,p}},&Z_{i,c,p}>0,\\
0,&Z_{i,c,p}=0.
\end{cases}
\]

Numerical implementation dùng safe denominator để tránh divide-by-zero; không làm loãng coordinate chỉ có một contributor bằng weights của donors không gửi coordinate đó. Với một contributor positive-weight, output chính là delta của contributor. Quality 0 không đóng góp. Native `age_aware_aggregate()` hiện dùng masked weighted sum, không tự cung cấp masked-coordinate normalization này; V3 cần implementation riêng.

\[
\theta_i^{stage}=\theta_i^{ref}+\eta A_{i,c}\odot\overline\Delta_{i,c}.
\]

Receiver integration trên own current BASE, chỉ sửa allowed coordinates. Preservation teacher là reference trước transfer; proximal anchor là staged aggregate và phải ghi rõ trong objective:

\[
\mathcal L=\mathcal L_{current}+\lambda\mathcal L_{preservation}
+\mu\lVert A\odot(\theta-\theta^{stage})\rVert_2^2.
\]

Nếu receiver không có target positive current BASE, không giả lập target CE bằng validation labels. Target learning diễn ra ở donor; receiver integration dùng current anchors và preservation/proximal terms. Coefficients, steps, seeds và teacher behavior khóa trước experiment.

## 7. Acceptance và data roles

Kế thừa lựa chọn **empirical acceptance** người dùng đã chốt trước đây, không tự quay lại strict broad-scope certification. Đây không làm giảm các numerical gates của prototype:

- CAL-FIT: gap/quality/selection; BASE: donor fitting + receiver integration.
- Khóa final combined candidate, masks, ranks, router và thresholds trước CAL-HOLDOUT.
- Native label-blind routing, cùng checkpoint/test protocol; không dùng true task/class để mở class lúc inference.
- Target recall ≥95%, observed FAR ≤0,1% theo owner/class đã khóa, negative break=0, đủ unique current CAL support (min 32 positive, min 32 receiver negatives là empirical minimum, không phải population certification).
- Có target benefit và net rescue dương so với Availability-only trên cùng rows. Báo count/denominators từng owner; không pooled-average để che failure tại owner được gate.
- Dùng all already-observed negative vetoes cho cùng function. Updated/combined function cần receipts mới, không dùng certificate/score của một version cũ.

Current gate fail → rollback. Current gate pass nhưng independent cumulative old-risk thiếu → **CURRENT_EMPIRICAL_PASS / OLD_RISK_UNKNOWN**, không ghi `STRICT_SAFE` hoặc cumulative certification. Commit under empirical policy phải lưu đầy đủ unverified old-class coverage và risk assumptions; metadata UNKNOWN không được biến thành old-risk PASS. V3 chưa triển khai verifier/commit path, chưa có empirical installation nào.

Không đọc raw historical CAL hoặc lấy provenance summaries làm CAL. Nếu chuyển sang strict policy trong một nghiên cứu khác, old-risk UNKNOWN phải fail-closed; đó là contract khác, không thay ngầm để tạo số PASS.

## 8. Reference caching và communication accounting

Cache chỉ hợp lệ khi peer có đúng complete reference/allocation hash. Trong current runner, native aggregation lấy snapshots sau local training để tính states mới; **không đảm bảo donors đã nhận model receiver sau aggregation/router refresh**. Vì vậy không thể mặc định piggyback reference là 0 bytes.

Ba case cần log riêng:

1. Cold reference: gửi đủ required snapshot, tính toàn setup bytes.
2. Exact cache hit: chỉ gửi reference hash/request, nhưng tính prior setup/amortization theo protocol.
3. Stale cached reference: gửi exact changed tensors/buffers/metadata nếu delta synchronization đã được implement và xác nhận hash; nếu không thì full refresh. Đo refresh bytes, không áp dụng stale update.

Donors có common reference có thể reconstruct final combined candidate từ sparse deltas + integration deltas + allocation metadata. Đây là optimization đề xuất, chưa implementation; verification messages và any refresh đều phải vào ledger.

\[
Comm_{total}=Comm_{DeNICE}+Comm_{APPLIANCE}.
\]

APPLIANCE overhead gồm metadata, reference setup/refresh, proposals, quality/verification queries/responses, allocation/mask metadata, accepted và rejected packets, retries nếu protocol cho phép. Không dùng total bytes của Python model trong RAM làm native measured wire cost.

Ví dụ dùng P=512.654 parameters, 800 directed transmissions, 200 packets, s=10.000, FP32+uint32 index: 3 masks=0,012 MB; sparse payloads=16 MB; 200 receiver capsules 6.685.240 bytes/capsule=1.337,048 MB. Các số này là **scenario assumptions**, không phải V3 measurements hoặc proof tiết kiệm. Model parameters P và actual serialized state phải đo từ đúng configured model.

Giữ complexity đúng phạm vi: masked metadata O(REC) bits; sparse payload O(RTs); cold references O(RQP) cộng state overhead; whole system vẫn có native O(REP) nếu native full-state exchange. V3 là overhead bổ sung cho DeNICE; sparse packets không tự giảm baseline native communication. Muốn tuyên bố thay full exchange cần variant thay đổi native protocol và benchmark riêng.

## 9. Feasibility protocol trước runner integration

Chỉ một receiver, hai donors đã khóa bằng metadata, một target current. Không dùng V2 trials fail làm bằng chứng V3 sẽ pass. Old fixture có thể là development sandbox, không independent confirmation.

Các controls cần có:

| Control | Câu hỏi |
|---|---|
| Native DeNICE baseline | Function ban đầu |
| Availability-only / APPLIANCE-Mask | Knowledge đã có bị registration che? |
| Capacity-only, chưa học delta | Allocation/mask/rank changes tự tạo gain không? |
| Receiver-aligned single-donor Head | Reproduce giới hạn readout-only, historical control |
| Two-donor Head aggregation | Gain do donor diversity hay sparse features? |
| Single-donor Sparse | Feature/capacity update có đóng góp? |
| Two-donor APPLIANCE-Sparse | Receiver-specific masked aggregation có đóng góp thêm? |

Đánh giá positive recall, precision/FP, rescue/break, native routing reachability, current and old-risk evidence coverage, protected-state equality, exact model/version restore, sparse coordinates, byte ledger, compute time. Cùng allocation giữa các variants phù hợp; Capacity-only quan trọng vì rank promotion có thể đổi function trước donor learning.

Khóa trước training: checkpoint/graph/roles, receiver+donors+target, capacity/masks, sparse budget, donor steps, eta/lambda/mu, quality estimator, splits/content dedup, tie policy, acceptance gates, time/byte/attempt budget. **Chưa khóa numeric learning configuration hoặc cases cho V3 trong tài liệu này; chưa chạy prototype.** Không mặc định kế thừa V2 steps/lr chỉ để gọi một experiment đã bắt đầu.

Điểm dừng: sparse transfer không vượt Availability-only và Capacity-only, hoặc current acceptance fail, hoặc communication vượt budget → NO-GO cho cấu hình prototype đã khóa. Không mở thêm guard/donor/hyperparameter sau HOLDOUT. Nếu feasibility đạt empirical gates, mới native survival, smoke hai task, rồi quyết định full một seed; không tự chuyển production sau một metric feasibility.

Variants NoFreeze/NoReg/NoGapCheck/NoSelection là ablations sau feasibility, không chạy đồng loạt trước. Shared frozen encoder là variant riêng vì thay giả định training và coordinate space của DeNICE.

## 10. Novelty và related work

Mask/sparsity/weighted transfer đơn lẻ chưa đủ novelty. [FedWeIT, ICML 2021](https://proceedings.mlr.press/v139/yoon21b.html) đã dùng global federated parameters và sparse task-specific parameters với weighted inter-client transfer. [FedPAC của Xu/Tong/Huang](https://arxiv.org/abs/2306.11867) nghiên cứu feature alignment và classifier collaboration. [FedProto, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20819) trao đổi class prototypes và server-aggregated global prototypes.

Claim đề xuất cần kiểm chứng của APPLIANCE là **functional-gap-triggered, receiver-aligned class transfer trên live decentralized graph, local mask-normalized aggregation và verification vượt Availability-only**, với single-local-model inference và total-cost accounting. Chưa tuyên bố APPLIANCE vượt related methods hoặc có strict safety.

## 11. Deliverables implementation tiếp theo

Sau khi khóa prototype protocol:

- `appliance/v3_contract.py`: reference/allocation/packet schemas và validation.
- `appliance/v3_sparse_transfer.py`: capacity planning, donor restricted training, sparse delta generation.
- `appliance/v3_masked_aggregation.py`: coordinate normalization và protected integration.
- `tools/run_appliance_v3_feasibility.py`: bounded local endpoint experiment + baseline controls + full ledger.

Tên file trên là **planned**, chưa phải code callable. Không sửa full training notebook hoặc enable APPLIANCE V3 trong `decentralized_denice_il.py` tại mốc contract.
