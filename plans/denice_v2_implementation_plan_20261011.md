# Kế hoạch implementation DeNICE V2

> Cập nhật 2026-10-13: khóa main Conv2 coordinate + binary-cosine, hoàn thành
> P3 runner/checkpoint/evaluator và chuyển execution sang notebook Kaggle.
> Quality NO-GO của các pilot dưới đây là lịch sử, không chặn engineering full run.
> Xem `docs/DENICE_V2_FULL_NOTEBOOK_20261013_VI.md`. Không chạy thêm local.

## 0. Mục tiêu và trạng thái

Specification gốc: [DeNICE V2 Cluster Protocol](../docs/DENICE_V2_CLUSTER_PROTOCOL_20261011_VI.md).
Đây là kế hoạch implementation, chưa phải runner V2 đã chạy thành công.
P0 đã triển khai và đạt 44/44 structural unit tests ngày 11/10/2026;
xem [báo cáo P0](../docs/DENICE_V2_P0_IMPLEMENTATION_20261011_VI.md).
P1 pure aggregation engine đã đạt 30/30 tests; cùng P0 là 74/74 PASS.
Xem [báo cáo P1](../docs/DENICE_V2_P1_AGGREGATION_20261011_VI.md).
Native P1 worker/moment collector đã nối vào DeNICEClient/Trainer và PASS
one-round smoke trên BASE thật; tổng regression 89/89 PASS.
Xem [báo cáo native P1](../docs/DENICE_V2_NATIVE_P1_SMOKE_20261011_VI.md).
P2 primitives/shadow contracts đã có; regression tổng 103/103 PASS ngày
12/10/2026. Preflight trên BASE thật: **NO-GO cho whole-tensor routing closure**
vì khóa 100% backbone. Xem [báo cáo P2 capacity](../docs/DENICE_V2_P2_ROUTER_CAPACITY_20261012_VI.md).
Native two-task router smoke chưa chạy; P2 chưa PASS. Full runner/P3–P4 chưa triển khai.

Kết quả cần đạt: mỗi cluster có **một shared model + một shared router + một
temporary server**, local training giữ Non-IID, no historical raw replay,
class-wise masked aggregation và cumulative full-test evaluation.

**K tự động từ clustering ở đầu task — người dùng đã chốt.** Không set K=5.
Membership giữ cố định trong task, không re-cluster mỗi round. ξ ban đầu 0,8.
Coordinator election độc lập với aggregation weights. Không CGoFed, CME,
APPLIANCE, generative/mimic data hoặc inter-cluster aggregation trong MVP.

## 1. Khoảng cách giữa code hiện tại và V2

| Có thể tái sử dụng | Cần thay/thêm |
|---|---|
| `DeNICEModel`, `DeNICEClient`, `DeNICETrainer` | Runner theo cluster state, không loop aggregate riêng cho từng receiver |
| Clean roles, data loader, task mapping | Metadata local knowledge theo thời gian, không dùng future roles để chọn leader/partition |
| Paper context similarity trong `denice_clustering.py` | Disjoint partitioner; `paper_context_cluster()` hiện labels toàn 0, K=1 sentinel |
| Các helper masks/maturity và context activations | Mask-aware normalization theo từng contributor/parameter; common topology |
| ContextDetector, memory snapshot/restore | Router space/version, shared thresholds, compatible cluster memory |
| `denice_eval.py` cho một predictor | Evaluator mỗi cluster chạy cùng FULL cumulative TEST, mean-model summaries |
| Delta checkpoint và ZIP archive helpers | Schema cluster state, partition/lineage, optimizer/scaler resume |

Không đổi behavior mặc định của V1. Tạo runner/namespace V2 riêng; V1 giữ làm
control. Các working-tree changes của clean/CGoFed training không thuộc work V2.

## 2. Cấu trúc module dự kiến

```
fed_learning/strategies/decentralized/denice_v2/
    contracts.py       # state/message schema, shape/version checks
    partition.py       # auto K, fixed membership within task
    knowledge.py       # Class Knowledge Matrix, ownership provenance
    election.py        # temporary server, deterministic tie-break
    aggregation.py    # common-reference class-wise masked updates
    normalization.py  # shared BN state, protected channels
    router.py         # shared feature space, sketches and drift checks
    state.py          # atomic broadcast, lineage, snapshot/restore
    communication.py  # serialized application message ledger
fed_learning/training/
    decentralized_denice_v2_il.py
    denice_v2_eval.py
    denice_v2_checkpoint.py
tools/
    train_denice_v2.py
    eval_denice_v2_full_test.py
    eval_denice_v1_full_test.py
    build_denice_v2_notebook.py
train_denice_v2_kaggle.py
train_denice_v2_kaggle.ipynb
```

Các đường dẫn trên là deliverables theo kế hoạch; `contracts.py`, `partition.py`,
`knowledge.py`, `election.py`, `state.py` đã có ở P0; `aggregation.py` và
`normalization.py` có ở P1 pure engine; `native.py` và
`tools/run_denice_v2_native_smoke.py` có ở native P1; `router.py` và
`tools/probe_denice_v2_router_capacity.py` có ở P2 shadow/preflight. Các phần còn lại chưa triển khai. Ưu tiên
module nhỏ, reuse helper sau khi xác minh semantics; không copy toàn runner V1.

## 3. P0 — Partition, knowledge và cluster-state contract

### P0.1 — Auto K

Đề xuất MVP: **deterministic complete-linkage threshold partition** trên
paper context similarity, ξ=0,8; đây là partition rule mới của V2, không gọi
là neighborhood formulation nguyên gốc của V1.

1. Chỉ dùng current BASE/metadata đã được phép tại task t và state hiện có.
2. Mỗi active client bắt đầu là một singleton.
3. Score merge hai groups bằng minimum pair similarity giữa hai groups.
4. Merge group pair có score cao nhất nếu score ≥ ξ và state compatibility
   cho phép; tie-break theo sorted client IDs.
5. Dừng khi không còn merge hợp lệ; K_t là số groups cuối cùng.

Complete linkage tránh connected-components gộp A và C chỉ vì cùng nối với B
trong khi A–C không đạt threshold. Không ép số cluster, minimum group size,
top-k hoặc ưu tiên test accuracy để cứu partition. K=1 hoặc nhiều singleton
là kết quả hợp lệ; báo phân bố và cost. Main partition algorithm là đề xuất
engineering cần ghi vào protocol lock trước smoke, chưa có benchmark.

Inactive clients không được xem là đã tham gia local training của task.
Lưu state/ownership của họ; khi tham gia lại, xử lý bằng cluster-state contract
V2 thay vì tự gọi catch-up của V1.
Population evaluation MVP là các cluster trong current task partition, được
khóa trước local training; dormant states vẫn persist nhưng không âm thầm
thêm/bớt predictors theo test performance. Báo active/dormant counts mỗi task.

### P0.2 — Knowledge Matrix và election

Giữ riêng:

- `local_data_counts[c,t]`: current/historical counts metadata hợp lệ.
- `locally_trained_classes`: provenance từ các lượt local training thực sự;
  client không có steps hoặc bị lỗi không được đánh dấu đã học.
- `cluster_available_classes`: registration của shared cluster predictor.

Không suy local ownership từ weights broadcast hoặc inherited availability.
Score leader = số unique locally trained classes tính đến hiện tại; tie-break
client ID nhỏ nhất. Task đầu chưa ai học thì dùng tie-break, không giả định
future class coverage. Chọn trong members; giữ coordinator trong task trừ
failover, tránh election dao động mỗi round. Failover không đổi cluster state.

### P0.3 — Canonical ClusterState

Schema tối thiểu:

```
schema_version, task, round, cluster_uid, parent_lineage
members, coordinator_id, ownership_metadata
model_state, unit_ranks, weight_masks, bias_masks, freeze_masks
BN_state, task/allocation_state, adapter_state_if_present
router_state, thresholds, routing_mask, routing_space_hash
episode_classes, feature/architecture/class_mapping_versions
training_rng_state, local_optimizer/scaler_policy
```

`routing_space_hash` chỉ bind đúng dependency state tạo routing activations,
không bao gồm FC2 thay đổi không ảnh hưởng router. Hash kiến trúc/kích thước
riêng không đủ chứng minh compatibility.

Mỗi round broadcast cùng canonical state; local workers chỉ tạo training
clones. Không duy trì 98 divergent persistent predictors rồi chỉ đổi tên
chúng thành cluster model.

### P0.4 — Handoff giữa tasks

Task0 groups bắt đầu từ cùng initialization seed. Task sau giữ persistent
cluster UID/lineage; regroup phải có explicit parent mapping:

- Split: child clone toàn bộ parent state/router; không mất historical entries.
- Merge: chỉ cho phép khi protected model state, historical routing space,
  thresholds và class/adapter schemas tương thích. Không average encoder cũ
  rồi concatenate historical sketches.
- Không tương thích: partitioner giữ các groups riêng và log rejected merge;
  không reset router, bỏ task history hoặc chọn tùy ý một parent model.
- Client đổi group nhận state của destination theo handoff đã khóa; ownership
  metadata đi theo client, nhưng không tự nhập local/model/router history vào
  destination. Không dùng membership mới làm bằng chứng class đã transfer.

Do no inter-cluster aggregation trong MVP, merge khác lineage với protected
weights xung đột là DEFERRED. Đây là ràng buộc của **auto partition có state
compatibility**, phải báo K_t và tỷ lệ merge bị chặn. Không trình bày như
unconstrained clustering. Nếu rule này khiến MVP không khả thi, giải quyết
handoff trước full run, không thêm inter-cluster transfer âm thầm.

**Gate P0:** partition exhaustive/disjoint trên active clients; deterministic
election; clones/broadcast/restore khớp full state; lineage và no-future-data
authority rõ. Không dùng test labels cho bất kỳ bước trên.

## 4. P1 — Common-reference class-wise masked aggregation

### P1.1 — Local training và topology

Clients trong cluster cùng bắt đầu từ θ_k^r. Giữ optimizer family, local epoch
budget, batch sampling và data split của V1. Native prepare/CANC per-client
hiện có không được tự thay neuron layout rồi aggregate theo cùng chỉ số.

Cluster quyết định shared learner layout/masks từ current-task metadata và
activation summaries hợp lệ; workers được hạn chế participation, không được
tự rewiring shared topology. Consolidation/maturity thực hiện một lần ở
cluster task boundary theo contract; không tăng tuổi thêm một lần cho mỗi
member. Metadata mature không thay thế knowledge weights hoặc competence.

### P1.2 — Formula phải implement

\[
\Delta_j=\theta_j^{local}-\theta_k^r,
\quad
\bar\Delta_{k,p}=
\begin{cases}
\frac{\sum_{j\in C_k}w_{j,p}M_{j,p}\Delta_{j,p}}
{\sum_{j\in C_k}w_{j,p}M_{j,p}}, & \text{denominator}>0,\\
0, & \text{otherwise},
\end{cases}
\]

\[
\theta_{k,p}^{r+1}=\theta_{k,p}^{r}+\eta_{agg}\bar\Delta_{k,p}.
\]

Với FC2 row c, chỉ client có current training contribution hợp lệ cho c tham
gia, weighted theo số current BASE samples của c. Backbone dùng sample-count
weights trong cluster trên từng parameter hợp lệ. Leader không có multiplier.
Đây là weighting MVP đề xuất; khác reserve-weighted V1, cần tên/config rõ và
ablation nếu claim class-wise weighting đóng góp.

Mature reference parameters và routing dependency closure bị chặn. Tọa độ
không có contributor giữ original weight/mask, không ghi zero. Maturity,
topology, connection mask và architecture compatibility kiểm tra trước apply.
Bias phải được xử lý tương ứng. Sau aggregate, tất cả clients nhận cùng state.

### P1.3 — BN và masks

Không average tùy ý local BN running variances hoặc lấy BN của leader.
Đề xuất canonical BN: protected/routing channels giữ reference buffers;
mutable channels recalibrate từ current BASE bằng distributed activation
moments trên **cùng post-aggregation model**. Truyền counts/sums/squared sums,
không raw inputs. Thiết kế pass theo layer để downstream moments dùng cùng
upstream normalization; đo additional forwards/bytes.

Mask topology do cluster quyết định; không OR local masks để vô tình mở
connections vào mature neurons. Consolidation sử dụng current summaries và
giữ prior protected state. Nếu không đủ căn cứ merge BN/masks/maturity thì
reject transaction, không broadcast mixed state.

**Gate P1:** shared-reference hashes khớp; permissions/mature values không
bị vi phạm; zero-contributor identity đúng; đổi leader không đổi aggregation
khi contributions giữ nguyên; shared model/masks/BN sau broadcast identical.

## 5. P2 — Một cluster router tương thích, replay-free

MVP đề xuất giữ `binary_cosine` làm router đầu tiên để tách thay đổi cluster
architecture khỏi router LR. `multiclass_balanced` là ablation sau khi shared
memory contract chạy đúng; không lấy kết quả LR làm kết quả binary gốc.

### P2.1 — Tạo current-task memory đúng version

1. Aggregate model và canonical BN/masks trước.
2. Các members dùng cùng encoder version để extract current BASE activations.
3. Khi thresholds chưa có, trao đổi moments để tạo chung first-task thresholds;
   broadcast thresholds rồi tạo binary sketches. Không binarize bằng local
   thresholds khác nhau rồi concatenate.
4. Packet bind task/member/class mapping, encoder/threshold/mask hash và số rows.
5. Router nhận compatible sketches; episode-class registration là union class
   có actual current training contribution trong cluster, giữ old entries.
6. Xử lý empty classes/bank/singleton bằng explicit deterministic policy.

Không đăng ký tất cả global classes nếu cluster chưa có đóng góp; các class
ngoài coverage vẫn nằm trong full test và được tính lỗi như protocol đã chốt.

### P2.2 — Bảo vệ historical feature space

Đề xuất main path: bảo vệ routing subspace trong native encoder bằng
**dependency closure**, gồm upstream weights, connection masks, recurrent
dependencies và BN affine/running buffers. Freeze selected output rows đơn
thuần chưa đủ. Feature selection/thresholds khóa tại lần tạo canonical memory;
không mở rộng routing dimensions qua task mà không có migration contract.

Sau mỗi local update/aggregation/BN pass, kiểm tra dependency hash và activation
consistency trên current witness hợp lệ (FP32 eval); numerical tolerances khóa
trước smoke. Lưu summaries/receipts, không giữ raw witness qua task.
Không có raw historical witness thì không tuyên bố đã empirically re-test old
activations; structural dependency preservation là bằng chứng riêng.

Nếu closure gần toàn backbone hoặc làm capacity không đủ học task mới, P2
**NO-GO cho router design này**. Chưa tự thêm frozen auxiliary encoder, mapper,
replay, generative/mimic data để cứu. Cần redesign riêng và update protocol.

**Gate P2:** compatible sketches, không historical raw đọc lại; per-sample
label-blind routing; dependency/BN protection; task transition và resume giữ
memory/classes. Đo route accuracy/margins trên development và capacity cost,
không yêu cầu thuật toán phải đạt một test accuracy chọn sau khi xem kết quả.

## 6. P3 — Runner, full-test evaluator và checkpoint

### P3.1 — Native V2 runner

Orchestrator mô phỏng clients/temporary servers trong một process Kaggle;
không tuyên bố physical P2P network đã triển khai. Coordinator xử lý state
riêng từng cluster, không tạo federation-wide trained global model.

Round transaction: shared-state load → local clones → updates/moments →
aggregate → router commit → broadcast → checkpoint. Invalid/stale packets
không làm mất committed state; log reject/rollback và reason. Checkpoints
chỉ đánh dấu round complete sau toàn bộ cluster transactions đã ghi receipts.

### P3.2 — Evaluation

Tại Task t, mỗi cluster predictor chạy cùng toàn cumulative test của global
seen classes; T5 đủ 34 classes. Không disjoint shards, không local-support
filter, không majority/ensemble giữa cluster outputs. Chỉ đưa x vào prediction.

Report:

- `mean_cluster_accuracy`, `mean_cluster_macro_f1` là headline.
- Per-cluster confusion/accuracy/F1/recall, size, leader, UID/lineage.
- `unique_test_rows=N_t`, `prediction_events=K_t*N_t` tách riêng.
- Optional client-weighted metric đặt tên riêng.
- Fixed global seen-class universe cho macro-F1; không tính macro-F1 từ
  confusion matrix cộng K models rồi gọi đó là mean macro-F1.
- Nếu cluster thiếu state hoặc không đủ N_t predictions: incomplete run,
  không âm thầm loại cluster khỏi mean.

Evaluation ở round cuối mỗi task để đo CL; fresh full run 6 tasks × 20 rounds.
Không full test mỗi round mặc định. Snapshot/freeze predictor **trong lúc
evaluation** để tránh mutation; đây không phải freeze toàn bộ training.

Task forgetting: so task-boundary performance theo persistent cluster lineage
và cùng task test rows. Split/merge cần explicit metric mapping; nếu lineage
không đủ, ghi NOT_IDENTIFIED thay vì gọi biến động mean theo K là forgetting.

V1 control: đánh giá mỗi original local model trên cùng full cumulative test,
báo mean-local-model metrics/cost và cluster-matched summary nếu cần. Các số
31,24% / 37,15% từ disjoint-shard audit giữ làm historical evidence.

### P3.3 — Checkpoint mỗi round và Kaggle storage

Lưu canonical model/router **một lần mỗi cluster**, không lặp lại cho mỗi
member. Tách client ownership, local optimizer/scaler/RNG state; khóa reset
hoặc carry-optimizer policy qua broadcast trước run. Nếu carry optimizer,
resume phải bind đúng cluster reference/version và local moment state.

Task base + round deltas, ZIP mỗi round; cuối task seal archive self-contained.
Reuse archive helpers sau khi schema V2 được xác minh. Manifest có parent
base hashes, partition/lineage và checksum. Chỉ xóa PT/ZIP parts sau archive
mới được đóng/verify. Tính cả peak temporary bytes khi seal; zip theo task
không tự bảo đảm quota. Budget fail giữ checkpoint hợp lệ và báo cách resume.

Delta format không được đổi precision âm thầm; đòi exact restart equivalence
phải lưu lossless algorithm/training state. Eval chỉ lưu confusion/receipts
mặc định, không dump K*N prediction CSV để tránh nổ output. Resume evaluation
theo cluster/batch với row counts/checksums; không chạy lại predictors hoàn tất.

Notebook nhỏ clone GitHub tại pinned commit; không nhúng runtime source hoặc
locked ZIP lớn. Data path env/config rõ; không phụ thuộc Downloads trên Windows.

## 7. P4 — Bounded smoke rồi full benchmark

Thứ tự gate, không mở chuỗi audit vô hạn:

| Mốc | Scope tối đa trước khi mở rộng | Bằng chứng bắt buộc |
|---|---|---|
| Structural integration | 4–6 clients, auto partition, ít nhất 2 groups nếu data similarity cho phép | Leader, broadcast, class-wise aggregation, rollback và resume |
| Native survival | 3 rounds, một task | Protected weights/BN/router, shared final state, communication ledger |
| Task transition | Hai task liên tiếp | Auto K/handoff, current-only data authority, retained historical sketches |
| Evaluation pilot | Cùng checkpoint, full-test executor với output receipts | Mỗi predictor cùng rows, mean/F1 đúng, label-blind, no silent exclusions |
| First full run | Fresh T0–T5, 20 rounds/task, seed42 | Task-boundary full tests, all-round checkpoints, completion/artifact receipts |
| Benchmark | ≥3 independent training seeds sau config lock | V1/V2 cùng data/training budget, mean±std/CI và tổng cost |

Smoke cần fixture có split/merge-compatible và merge-reject branches; nếu auto
K=1 ở một task thì không ép K=2 để báo pass multi-cluster. Controlled fixture
kiểm tra branch riêng, phân biệt với natural partition result.

Full run chỉ sau gates P0–P3. Không chuyển old divergent V1 checkpoints thành
V2 rồi tuyên bố fresh V2 training; chúng chỉ dùng engineering fixtures.

## 8. Configuration dự kiến

```text
method = denice_v2
cluster_count_mode = auto
partition_algorithm = complete_linkage_threshold   # MVP proposal
similarity_threshold = 0.8
membership_update = task_start
leader_policy = cumulative_locally_trained_class_count
leader_tie_break = smallest_client_id
leader_weight_bonus = false
aggregation = common_reference_masked_delta
fc2_weighting = per_class_current_sample_count
router = binary_cosine                             # MVP proposal
memory = compatible_binary_sketches
historical_raw_replay = false
inter_cluster_aggregation = false
eval_scope = full_global_seen_test_per_cluster
headline_accuracy = mean_cluster
headline_macro_f1 = mean_cluster
eval_cadence = task_end_last_round
checkpoint_every_round = true
seal_archive_every_task = true
```

User-locked: auto K, within-task membership, local class-count coordinator,
Non-IID/current-only training, masks/maturity protection, shared model/router,
full-test-per-cluster và mean-cluster metrics. Các lựa chọn đánh dấu proposal
và BN/topology/lineage policy cần ghi rõ version trong implementation lock;
không gọi chúng là thuật toán đã benchmark.

## 9. Definition of Done trước notebook full

Một command từ clean repo dựng/run được V2 smoke; full cluster state đồng bộ,
router replay-free hợp lệ, class-wise aggregate không vi phạm permissions,
checkpoint restore/resume nhất quán, full-test evaluation đúng scope, logs
flush tiến độ và cost/ZIP quota có receipts. V1 launchers không bị đổi.

P0 contracts/state và P1 pure engine đã PASS. Native P1 worker/moment collector
đã PASS one-round real-BASE smoke, cùng 89/89 regression tests. Deliverable
tiếp theo là **giải quyết capacity gate P2 shared cluster router**; full runner/training chưa bắt đầu.
Native P1 hiện dùng plain legacy model, một phase bounded, shared class-block
allocation; chưa có historical router/consolidation hoặc benchmark accuracy.
P2 whole-tensor closure bị NO-GO: 215.916/215.916 backbone parameters locked.
103/103 structural/regression tests không thay native two-task PASS. Chưa P3.

## 10. P2-Shallow status — 2026-10-12

Native pilot `conv1_only_v1` đã chạy Task 0 → Task 1 bằng optimizer thật.
Capacity PASS: fresh Conv1/BN1 closure 384 parameters; historical memory,
thresholds, dependency hash và restore PASS. Discrimination **NO-GO**:
T1 recall 8/213 = 3,76%, balanced task accuracy 41,58%; không tune lại.
123/123 regression tests PASS không thay thế gate chất lượng.
Xem [báo cáo](../docs/DENICE_V2_P2_SHALLOW_RESULTS_20261012_VI.md).
P0/P1 giữ nguyên. Chưa P3/full training; chỉ xem xét coordinate-level closure
sau khi có contract upstream/BN/GRU recurrent protection hợp lệ.

## 11. Routerless variant — 2026-10-12

Theo hướng mới, tạm dừng coordinate-level router protection. `routerless_v1`
dùng một cluster model và cumulative class mask; không Task Router hoặc activation
memory. Fresh native T0→T1 dùng cùng budget/seed với shallow pilot đã chạy.
Technical PASS, quality NO-GO: old accuracy55,08%→0%, new94,37%, cumulative34,24%.
146/146 regression tests PASS, không thay quality gate. Chưa thêm distillation
hoặc mở full training. Xem [báo cáo](../docs/DENICE_V2_ROUTERLESS_RESULTS_20261012_VI.md).

## 12. Old-only diagnostic — 2026-10-12

Inference checkpoint T1 hiện có: old-only273/374 =72,99%; cumulative0/374.
Baseline T0 từ original receipt206/374 =55,08%, không re-execute T0 vì file T0
chưa persist ở run trước. Runner đã thêm lưu checkpoint T0 cho các run sau.
Không train lại/fit calibration trong audit. Old class2 recall90%→0%; mean FC2
norm ratioold/new1,004 không chứng minh norm-only alignment là lời giải.
Ưu tiên logit competition và class-wise preservation; chưa distillation/full run.
Xem [diagnostic](../docs/DENICE_V2_ROUTERLESS_OLD_LOGIT_AUDIT_20261012_VI.md).

## 13. CNN coordinate router — 2026-10-12

`cnn_coordinate_router_v1`: Conv2 trước, conditional fresh Conv3 fallback đúng
protocol. Cumulative CE/full forward giống routerless, cùng BASE/VAL budget/seed.
Capacity và activation/memory stability PASS;179/179 tests PASS. Routing empirical
NO-GO: balanced47,70%/50,34%; actual class Macro-F1 thấp hơn own cumulative.
Oracle77,34%/79,22% chỉ diagnostic trên từng checkpoint; class2 recall vẫn0%.
Xem [báo cáo](../docs/DENICE_V2_CNN_COORDINATE_RESULTS_20261012_VI.md).
Theo decision tree, dừng mở family này; quay routerless calibration/preservation
với protocol mới. Chưa P3/full training, chưa áp dụng distillation/aligner.
