# Kế hoạch debug toàn diện DeNICE: routing, mask và continual classifier

Ngày tạo: 13/08/2026  
Phạm vi: checkpoint DeNICE task 5 / round 19, context router, adapter routing,
class mask, protocol evaluation và continual classifier.

## 1. Mục tiêu và nguyên tắc

Mục tiêu không phải chỉ tăng accuracy bằng cách tắt mask. Cần xác định, định
lượng và sửa đúng từng nguồn suy giảm:

1. Sai protocol evaluation hoặc checkpoint reconstruction.
2. Client/router không có coverage cho episode hoặc adapter cần thiết.
3. Router học sai, calibration sai, hoặc representation drift theo task.
4. Hard class mask biến routing error thành classification error chắc chắn.
5. Adapter hoặc backbone/classifier bị forgetting/negative transfer.

Mỗi experiment phải cố định checkpoint SHA, config SHA, seed test indices và
ghi `evaluation_mode`. Không dùng stress protocol làm metric để so sánh paper.

## 2. Evidence hiện có (checkpoint task 5, round 19)

Checkpoint: `checkpoint_task_5_round_19.pt`, DeNICE decentralized, router
`multiclass`, 20 rounds/task, seed 42.

| Protocol hiện tại | Accuracy | Macro-F1 | Loss | Route accuracy |
| --- | ---: | ---: | ---: | ---: |
| `partitioned_local + hard` | 11.91% | 6.87% | 96.08 | 18.25% |
| `partitioned_local + nomask` | 32.06% | 13.36% | 22.89 | 18.25% |

Điều cần diễn giải thận trọng:

- `partitioned_local` giao ngẫu nhiên global cumulative test 34 class cho 98
  client. Đây là **stress/debug protocol**, không phải personalized metric.
- Chỉ 45/98 client có đầy đủ episode router `0..5`; 53/98 client thiếu ít nhất
  một episode. Router local thiếu coverage không thể dự đoán episode chưa học.
- `nomask` vẫn chạy adapter của predicted episode, nên không phải classifier
  ceiling thuần.
- Router collapse về episode 5: recall episode 0=0.01%, 1=9.09%, 2=12.25%,
  3=38.23%, 4=42.43%, 5=90.13%.

Kết luận tạm: hard mask/routing là nguồn suy giảm lớn, nhưng còn phải tách
coverage mismatch, feature drift và classifier forgetting trước khi sửa thuật
toán.

## 3. Ma trận decomposition bắt buộc

| ID | Router chọn episode | Adapter | Mask | Câu hỏi trả lời |
| --- | --- | --- | --- | --- |
| E0 | Không dùng | Không dùng | seen-class only | Backbone/classifier thuần tốt đến đâu? |
| E1 | Predicted | Predicted | none | Hại do predicted adapter là bao nhiêu? |
| E2 | Oracle task episode | Oracle | none | Trần adapter/classifier nếu context đúng |
| E3 | Oracle task episode | Oracle | hard | Mapping episode-to-class/mask có đúng? |
| E4 | Predicted | Predicted | hard | Pipeline DeNICE hiện tại |
| E5 | Predicted | Predicted top-1 | top-k 2/3 union | Hard mask có quá brittle? |
| E6 | Predicted + confidence | Predicted | adaptive fallback | Fallback an toàn có giảm lỗi mask? |

Suy luận:

- E0 thấp: classifier/backbone/continual learning là bottleneck.
- E2 < E0: adapter gây negative transfer.
- E1 << E2: routing chọn adapter sai.
- E3 < E2: mask map hoặc implementation sai.
- E4 << E3: routing là nguyên nhân chính.
- E5/E6 > E4: hard mask quá giòn và có thể dùng fallback.

## 4. Kế hoạch theo thứ tự thực hiện

### P0 — Khóa evaluator và protocol

Files dự kiến:

- `eval_checkpoint.py`
- `fed_learning/training/denice_eval.py`
- `tests/test_denice.py`
- `eval_denice_kaggle.ipynb`

Thực hiện:

1. Thêm evaluation modes E0-E6; oracle episode lấy từ task/class mapping,
   không lấy prediction.
2. Thêm `global_stress`, `coverage_aware_local`, `representative_global`.
3. Với `coverage_aware_local`, chỉ assign sample tới client có cả episode,
   router memory và adapter context phù hợp; log unsupported samples rõ ràng.
4. Ghi sample-index hash, client/episode coverage, adapter coverage và số
   sample unsupported.

Pass/fail:

- [ ] Cùng checkpoint + seed cho đúng cùng indices và cùng metric.
- [ ] Không sample nào bị đánh giá bởi client không support true episode trong
      `coverage_aware_local`.
- [ ] Oracle hard luôn giữ true class trong allowed class set.
- [ ] Không fallback silently sang latest episode khi router/client thiếu state.
- [ ] `global_stress` được ghi nhãn debug-only.

Commit: `test(eval): add DeNICE coverage and oracle invariants`  
Commit: `feat(eval): decompose DeNICE router adapter and mask effects`

### P1 — Audit router trên checkpoint hiện có

Files dự kiến:

- `eval_checkpoint.py`
- `fed_learning/servers/nice_server.py`
- `tests/test_denice.py`

Thực hiện:

1. Đánh giá multiclass router trên held-out split từ `activation_memory` từng
   client: balanced accuracy, per-episode recall, entropy/confidence.
2. Đánh giá router bằng current-model activation của test/reference data và so
   sánh với stored-memory result.
3. Log per-layer bit rate, threshold, Hamming/cosine drift theo episode.
4. Log `episode_classes`, `activation_memory`, LR classes, calibration
   signature và adapter contexts cho từng client dạng histogram/top-worst, không
   dump toàn bộ prediction per-sample.

Pass/fail:

- [ ] Router stored-memory balanced accuracy >= 80% trên full-coverage client.
- [ ] Không episode có recall < 50% mà không được ghi lý do coverage/drift.
- [ ] Prediction distribution không collapse >70% vào một episode trên set có
      ground-truth distribution khác.
- [ ] Phân biệt được: fitting failure vs feature drift vs missing coverage.

Commit: `feat(debug): report DeNICE router coverage calibration and drift`

### P2 — Audit hard mask và adapter availability

Files dự kiến:

- `fed_learning/training/denice_eval.py`
- `fed_learning/models/denice_model.py`
- `tests/test_denice.py`

Thực hiện:

1. Assert `true_class in allowed_classes(oracle_episode)`.
2. Với predicted episode thiếu adapter, log `missing_adapter` và thử two
   explicit policies cho ablation: clear adapter vs fallback backbone; không
   che giấu policy.
3. Add top-k 2/3 đúng theo episode probability và adaptive confidence fallback:
   high confidence=hard, medium=top-k, low/unsupported=nomask.
4. Đo mask violation count, missing-adapter count và hard-to-nomask gap theo
   task/client coverage.

Pass/fail:

- [ ] E3 không có mask violation.
- [ ] E5/E6 không thấp hơn E4 về macro-F1 trong cùng checkpoint/indices.
- [ ] Loss hard không còn tăng do true class bị mask ở oracle route.

Commit: `fix(denice): make routed mask coverage-safe`  
Commit: `feat(mask): add confidence-aware DeNICE fallback`

### P3 — Chọn targeted router fix

Chỉ triển khai sau P1/P2.

| Diagnosis | Fix ưu tiên |
| --- | --- |
| Stored memory cũng thấp | balance memory; train/validation router; tune LR C; compare centroid/Hamming router |
| Stored memory cao, current activation thấp | frozen router encoder; anchor replay/re-encode; fixed calibration |
| Missing local coverage | cluster-level compatible pooled router; representative router; coverage-aware selection |
| Confidence thấp nhưng hard mask hại | adaptive top-k/nomask fallback |

Pass/fail:

- [ ] Router fix phải cải thiện P1 metric mà không làm E0/E2 giảm.
- [ ] Checkpoint snapshot/restore bảo toàn router state và calibration.
- [ ] Regression test cover missing episode, missing adapter và low confidence.

Commit tùy diagnosis: `fix(router): ...`

### P4 — Audit continual classifier sau khi loại router confounder

Files dự kiến:

- `fed_learning/training/decentralized_denice_il.py`
- `fed_learning/strategies/decentralized/denice_aggregation.py`
- `fed_learning/training/denice_eval.py`

Thực hiện:

1. Báo E0/E2 theo task, class, client join task và coverage cohort.
2. Đo forgetting task 0..4 từ checkpoint/round history.
3. Đo adapter benefit `E2-E0`, capacity/free ratio, class-head update và
   aggregate participation theo client.
4. Chỉ điều chỉnh CANC/aggregation/replay nếu E0/E2 xác nhận classifier là
   bottleneck còn lại.

Pass/fail:

- [ ] Không class có support đáng kể mà accuracy 0% vì lỗi mapping/code.
- [ ] Oracle routing cải thiện rõ predicted routing.
- [ ] Mọi thay đổi classifier được so trên cùng protocol, 1 seed smoke trước.

### P5 — Smoke retrain có instrument

Config: 3 task, 10–20 client, 3 rounds/task, 1 seed.

Log mỗi task:

- [ ] router coverage histogram, stored/current router accuracy, confusion,
      confidence/entropy;
- [ ] per-layer drift/calibration bit-rate;
- [ ] E0..E6 compact metrics;
- [ ] cluster validity/self-only rate;
- [ ] capacity free/mature ratio, adapter availability;
- [ ] per-task forgetting.

Pass/fail:

- [ ] Không router collapse về latest episode.
- [ ] `pred_hard` cách `oracle_hard` <=10 points trong smoke.
- [ ] Không mask violation; full test suite pass.

### P6 — Full validation

Thứ tự:

1. Full rerun seed 42 với best configuration từ P5.
2. Chạy E0..E6 với protocol `coverage_aware_local` và
   `representative_global`.
3. Nếu pass, chạy seed 42/43/44 (tối thiểu 3 seed).
4. Báo mean ± std: accuracy, macro/weighted F1, route balanced accuracy,
   oracle gap, forgetting, cluster validity, capacity và inference cost.

Pass/fail:

- [ ] Adaptive/predicted hard gap tới oracle hard <=5 points hoặc báo rõ giới
      hạn còn lại.
- [ ] No episode collapse >70% khi ground truth không tương ứng.
- [ ] Macro-F1 tăng cùng accuracy, không chỉ lợi ích class imbalance.
- [ ] Protocol final, SHA, config và seed được lưu artifact.

## 5. Checklist trạng thái tổng hợp

- [x] Baseline checkpoint routing/mask evidence được lưu.
- [x] Xác nhận 45/98 client có full router episode coverage.
- [x] P0 — Evaluation decomposition và coverage-aware protocol.
- [x] P1 — Router stored-memory/current-feature/drift audit.
- [x] P2 — Hard-mask invariant và adaptive fallback ablation.
- [x] P3 — Targeted router repair theo diagnosis.
- [x] P4 — Continual classifier audit/fix nếu còn bottleneck.
- [x] P5 — Instrumented smoke retrain pass.
- [ ] P6 — Full rerun, >=3 training seed và final report.

## 5.1 Evidence from P1 (13/08/2026)

- `router_memory_audit_task5_round19.json`: 98/98 clients are eligible for the
  held-out audit; persisted router balanced accuracy is 99.998% and refit
  held-out balanced accuracy is 97.281%.
- `router_current_feature_audit_task5_round19.json`: 45 clients have all six
  stored router episodes. On a deterministic balanced global-test subset
  (256 samples x 6 episodes), the first 10 full-coverage clients average only
  31.549% current-feature router balanced accuracy (range 22.201%-41.536%).
- Diagnosis: router fitting and checkpoint restoration are not the primary
  failure. Final-model context features drift substantially from stored router
  memory, causing a strong late-episode prediction bias. P2 will separately
  quantify the hard-mask damage and make fallback behavior explicit.

## 5.2 Evidence from P2/P3 smoke (13/08/2026)

- `decomposition_smoke_e0.json` and `decomposition_smoke_e3.json` use the
  same 20,000-sample, seed-42 `coverage_aware_local` assignment. E0
  backbone/seen-class accuracy is 36.510% (macro-F1 15.617%); E3 oracle-hard
  is 67.705% (macro-F1 38.332%). Therefore the classifier remains a secondary
  bottleneck, but routing/masking has a much larger recoverable gap.
- E3 has `oracle_mask_violation_count=0`: the class-to-episode map and hard
  mask implementation are correct. Predicted routing is only 18.050% on this
  assignment, so hard masking converts routing errors into deterministic class
  errors.
- Only 2,078/20,000 smoke samples activated an adapter; 17,922 fell back to
  backbone because the selected context had no adapter. The evaluator now logs
  this explicitly and has adaptive hard/top-k/nomask ablation counters.
- Targeted repair: each client now retains its tiny router reference inputs and
  re-encodes all episode sketches after every decentralized aggregation, then
  refits its router. This aligns checkpoint-time router memory with the final
  model feature space. Old checkpoints are evaluated unchanged; the new policy
  is enabled only for fresh training via
  `denice_refresh_router_memory_after_aggregation=True`.

## 5.3 Evidence from P4/P5 stratified smoke (13/08/2026)

- The old task-5 checkpoint's E0/E2 values (36.510%/15.617% versus
  37.690%/16.306% accuracy/macro-F1) show that its sparse adapters provide
  negligible benefit. The remaining classifier/backbone quality is therefore
  a separate bottleneck, not evidence that the router/mask implementation is
  incorrect.
- A first 3-task/10-client/3-round smoke with uniform client subsampling
  failed its router criterion because rare non-IID classes were omitted from
  the tiny local sample. The smoke-only data limit is now deterministic and
  stratified by locally present class; production full data is unchanged.
- The corrected smoke (`router_refresh_stratified_smoke_*`) passes router
  memory quality: persisted balanced accuracy 95.746%, refit held-out 80.349%.
  Its final-feature router accuracy is 63.527% on a balanced test subset, with
  prototype cosine drift mean 0.953 (minimum 0.833), rather than the old
  checkpoint's severe old-episode drift.
- On the same 20,000-sample smoke partition: E3 oracle-hard=18.130%, E4
  predicted-hard=17.835% (0.295-point gap), route accuracy=69.830%, and
  `oracle_mask_violation_count=0`. Therefore P5 routing/mask pass criteria
  are met. Classifier quality is intentionally not accepted as a full-run
  result; it must be re-measured in P6.

## 6. Quy tắc dừng

Không chạy full retrain hoặc tune hyperparameter router trước khi P0-P2 xác
nhận nguyên nhân. Nếu P1 chứng minh router local không có coverage episode,
không coi `partitioned_local/global_stress` là thất bại router algorithm;
chuyển sang representative/pooled-router hoặc coverage-aware protocol để đo.
