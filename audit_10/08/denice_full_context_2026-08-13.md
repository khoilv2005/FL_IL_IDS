# DeNICE/CANDLE — full audit context and current debug state

Date: 13/08/2026  
Repository branch: `main`  
Recovery-plan commits through: `321a023`

This document consolidates the DeNICE/CANDLE audit timeline, code changes, artifacts, measurements, root-cause diagnosis, and next debug work. It is a handoff document and does not rely on chat history.

## 1. Experiment setting

The repository calls the decentralized CANDLE-like method **DeNICE**. It combines NICE-style class-incremental learning, a `ContextDetector` router, affinity-propagation clustering of client capsules, and age-aware peer aggregation. CANC/adapters are intended to help with novelty and capacity.

The 100-client split has 34 classes and 6 tasks:

| Task | Classes | Active clients |
| ---: | --- | ---: |
| 0 | 0–5 | 50 |
| 1 | 6–11 | 60 |
| 2 | 12–17 | 70 |
| 3 | 18–23 | 80 |
| 4 | 24–29 | 89 |
| 5 | 30–33 | 98 |

Primary artifacts reviewed:

- `C:\Users\khoak\Downloads\results.zip`: old run before recovery.
- `C:\Users\khoak\Downloads\results (1).zip`: full seed-42 recovery run.
- `data/federated_splits/100-clients`: source federated split.

## 2. Original high-severity failure: DeNICE was local-only

The old archive completed 114 rounds: task 0–4 fully and task 5 through round 13. It looked decentralized but had no effective peer aggregation.

| Signal | Observation |
| --- | --- |
| Effective K | Equal to active-client count in all 114 rounds |
| Aggregation policy | `self_only` in 114/114 rounds |
| Raw AP clustering | Usually found 10–25 clusters |
| Raw silhouette | min/median/max `0.2365 / 0.3194 / 0.4221` |
| Threshold then | `denice_cluster_theta_s = 0.50` |

The `0.50` validity threshold rejected every raw AP cluster and converted every round into singleton groups. The old checkpoint must not be resumed for an official result because all earlier tasks learned under local-only dynamics.

The same old run rebuilt/refit routers every round. Context time rose from 0.9 s/task-round in task 0 to 351.9 s in task 5, while local training was about 14 s. That was a separate runtime bottleneck.

## 3. Recovery implementation already committed and pushed

| Commit | Change |
| --- | --- |
| `bc8862e` | Formal recovery plan in `audit_10/08/26.md` |
| `5a051cd` | Complete timing and router freshness metadata |
| `0ee07f4` | Router refresh only at task end |
| `63b795c` | Vectorized/batched reference encoding and separate quota |
| `4885504` | Compact JSONL debug and checkpoint cadence 5 |
| `74cd0cc` | Calibrate cluster validity to `theta_s = 0.20` |
| `2199360` | Collaboration guard and fail-fast on self-only streak |
| `4a755cc` | Numeric test: peer deltas change receiver parameters |
| `321a023` | Record recovery runtime/clustering evidence |

Production configuration in `train.ipynb` and `train_incremental_kaggle.py`:

```text
TRAIN_PHASE = 5                 # train task 0–5 from scratch
seed = 42
denice_router_update_schedule = task_end
denice_router_reference_per_class = 20
denice_router_refresh_batch_size = 2048
denice_cluster_theta_s = 0.20
denice_cluster_edge_top_k = 40
denice_cluster_edge_quantile = 0.25
denice_cluster_min_signal_std = 0.02
denice_collaboration_guard_mode = error
denice_max_consecutive_self_only_rounds = 2
denice_min_mean_peer_alpha = 0.05
round_checkpoint_every = 5
```

The project regression suite after these changes: `317 passed, 1 skipped`.

## 4. Recovery smoke: runtime and collaboration gates passed

Fixed smoke: seed 42, tasks 0–2, 10 clients, 3 rounds/task, max 300 training samples/client.

| Mode | Router schedule | Reference/class | Context + router | Total round time |
| --- | --- | ---: | ---: | ---: |
| T0 | `every_round` | 50 | 30.139 s | 118.675 s |
| T1 | `task_end` | 50 | 9.323 s | 90.679 s |
| T2 | `task_end` | 20 | 5.492 s | 87.483 s |

Results:

- T1 gives 3.23× and T2 gives 5.49× context/router speedup relative to T0.
- Every one of 9 smoke rounds had `effective_K_t < active_clients`.
- Mean collaboration group size: 3.933–4.133.
- Mean peer alpha: 0.5696–0.5757; at least 8/10 clients aggregated a peer each round.
- No guard triggered; task-end routers were fresh for all clients.
- Current-feature router audit: 50/class = 89.00%, 20/class = 90.83%.
- Timing accounting differed from wall-clock only 0.08–0.25%.

Thus the recovery does not merely change logs: it establishes genuine peer aggregation and removes router refitting from ordinary rounds.

## 5. Full recovery run: `results (1).zip`

The full seed-42 run completed correctly from task 0 through task 5:

| Item | Value |
| --- | --- |
| Rounds | 120 (20 for each of 6 tasks) |
| Final checkpoint | `checkpoint_task_5_round_19.pt` |
| Valid clusters | 120/120 |
| Self-only rounds | 0/120 |
| Mean effective K | 19.625 |
| Mean group size | 6.577 |
| Mean peer alpha | 0.638 |
| Minimum peer-aggregated clients/round | 47 |
| Max self-only guard streak | 0 |

Final round task 5 / round 19:

```text
98 active clients
raw/effective K = 20 / 20
silhouette = 0.5344
mean group size = 9.5714
peer alpha mean = 0.6711
peer-aggregated clients = 91
router freshness = 98 fresh / 0 stale
```

The final router refresh costs 152.3 s, of which 150.0 s is logistic-regression fitting on 33,703 reference samples. This happens once at task end by design, not every round.

## 6. Evaluation protocols: do not mix the metrics

### Training-time quick post-task metric

`final_report.json` evaluates only 3 full-coverage clients (`1, 3, 6`) on an internally balanced 50,000-sample subset and averages client metrics:

```text
accuracy = 26.84%
macro-F1 = 18.55%
route accuracy = 50.91%
```

This is an operational diagnostic, not an all-98-client benchmark.

### E0–E6 final-checkpoint ablation

E0–E6 use the same checkpoint, seed and 50,000 global test samples allocated disjointly to all 98 clients under `coverage_aware_local`. Each sample is evaluated once by a client whose router covers its episode.

| ID | Inference policy | Accuracy | Macro-F1 | Meaning |
| --- | --- | ---: | ---: | --- |
| E0 | backbone, no mask | 10.83% | 6.17% | plain backbone |
| E1 | predicted adapter, no mask | 11.85% | 6.83% | adapter from router |
| E2 | oracle adapter, no mask | 12.00% | 6.90% | correct adapter alone |
| E3 | oracle hard mask | 53.35% | 31.82% | task-incremental classifier upper bound |
| E4 | predicted hard mask | 42.21% | 24.51% | actual routed prediction |
| E5 | top-k=2 | 26.23% | 14.11% | unsuitable here |
| E6 | adaptive | 41.74% | 24.07% | close to E4 |

E3/E4 and macro/per-class recall are the correct core diagnostic metrics for the next changes. Do not directly compare 26.84% with 42.21% because their client selection and aggregation protocols differ.

## 7. Root-cause diagnosis

### 7.1 Closed: clustering no longer causes bad metrics

The old local-only logic failure is fixed. All 120 full-run rounds have valid clusters, peer aggregation, and guard pass. Do not lower `theta_s` below 0.20 to chase the current metric problem.

### 7.2 Confirmed: router generalization is poor

Final router current-feature audit:

| Metric | Value |
| --- | ---: |
| Balanced accuracy | 49.88% |
| Mean confidence | 80.16% |
| Mean entropy | 0.514 |
| Persisted-memory balanced accuracy | 98.55% |
| Refit-holdout balanced accuracy | 68.08% |

The router fits its reference memory but generalizes poorly to final-model/global test features, and is overconfident when wrong. Episode current-feature recalls:

| Episode | Recall |
| ---: | ---: |
| 0 | 20.3% |
| 1 | 68.7% |
| 2 | 48.0% |
| 3 | 60.8% |
| 4 | 60.1% |
| 5 | 41.3% |

The routing penalty is directly measurable:

```text
E3 oracle hard 53.35% -> E4 predicted hard 42.21%
loss attributable to routing = 11.14 percentage points
```

### 7.3 Confirmed: class-incremental global classifier is weak without task mask

Correct adapter without a class mask yields only 12.00% (E2), whereas a true episode mask yields 53.35% (E3). NICE `forward_output` zeros non-learner output rows through `LetLearner`, then freezes mature rows between tasks. Therefore it learns task-routed classification, not a calibrated direct 34-way classifier.

This is a method limitation/missing calibration mechanism, not a clustering failure. Unmasked 34-way accuracy must not be treated as the same benchmark as the task-routed DeNICE target.

### 7.4 Confirmed: data imbalance is extreme

The train data itself is extremely imbalanced:

| Task | Largest class | Rarest class |
| ---: | ---: | ---: |
| 0 | class 1: 735,961 | class 0: 2,155 |
| 1 | class 6: 4,825,281 | class 10: 15,680 |
| 2 | class 14: 3,626,719 | class 16: 8,765 |
| 3 | class 21: 2,224,126 | class 18: 48,159 |
| 4 | class 24: 596,886 | class 28: 1,513 |
| 5 | class 32: 250,308 | class 31: 837 |

In task 5, class 32 is 97.3% of samples. NICE currently uses normal unweighted cross entropy and standard distributional mini-batches. There is no class-balanced sampler, clipped class-weighted loss, focal loss, or balanced classifier calibration. This directly explains weak rare-class recall and low macro-F1 even under oracle routing.

### 7.5 Confirmed: capacity is restricted in final task

At task 5 entry, mature ratios were approximately conv1 88.8%, conv2 89.2%, conv3 83.1%, GRU 89.2%, and fc1 60.3%. The runner still reserves 10% free capacity on the terminal task, despite there being no future task. This reduces available adaptation for classes 30–33. It should be ablated, not blindly removed, because old-task retention also matters.

### 7.6 Confirmed: adapter coverage is low

E2 has active adapter on only 4,915/50,000 samples. On 45,085/50,000 samples (90.17%) no matching adapter is active. Adapters cannot currently compensate for the frozen backbone/classifier at scale.

### 7.7 Plausible but not yet proven: peer aggregation may dilute rare classes

Clients need only one shared label to aggregate. After that, current code can average every still-plastic `fc2` row. A peer sharing a dominant class but not a receiver's rare class may dilute that receiver's rare-class classifier update.

Evidence: mean peer alpha 0.638, mean group size 6.577, severe class imbalance, and no per-row label-support gate for `fc2`. This is not yet a confirmed bug; it needs controlled self-only and label-aware aggregation ablations.

## 8. Relevant code locations

| Concern | File |
| --- | --- |
| Main DeNICE loop, reserve, cluster and aggregate | `fed_learning/training/decentralized_denice_il.py` |
| NICE CE loss | `fed_learning/strategies/incremental/nice.py` |
| NICE phase loop / batches | `fed_learning/clients/nice_client.py` |
| Output masking and neuron ages | `fed_learning/models/nice_model.py`, `fed_learning/models/denice_model.py` |
| Context router LR fit | `fed_learning/servers/nice_server.py` |
| Age-aware aggregation | `fed_learning/strategies/decentralized/denice_aggregation.py` |
| Cluster similarity | `fed_learning/strategies/decentralized/denice_clustering.py` |
| E0–E6 evaluation | `eval_checkpoint.py`, `fed_learning/training/denice_eval.py` |
| Kaggle run configuration | `train.ipynb`, `train_incremental_kaggle.py` |

## 9. Next debug experiments, in mandatory order

### D0 — Standardize evaluation

Build a reusable all-client, coverage-aware, class-balanced protocol with fixed per-class support. Always save E0/E2/E3/E4/E6, per-class recall, per-episode router recall, macro and weighted metrics. Pass condition: same checkpoint and seed reproduce exact supports and metrics.

### D1 — Does peer aggregation help?

On a fixed 2–3-task 10–20-client smoke, compare:

1. Current peer aggregation.
2. Self-only ablation (guard in warn/disabled mode only for this ablation).
3. Lower peer strength, e.g. self floor 0.50 or aggregation eta 0.5.

Read E3/E4, rare-class macro recall, old/new task accuracy, group size and peer alpha. If self-only materially wins, investigate peer negative transfer before changing router/classifier.

### D2 — Label-aware classifier aggregation

Only if D1 shows negative transfer: for `fc2.weight` and `fc2.bias`, accept a peer row only when its capsule label histogram has positive support for that class. Retain the receiver's local row otherwise. Keep encoder aggregation unchanged. Pass: improved rare-class E3 recall without dominant-class collapse.

### D3 — Correct training imbalance

Compare one change at a time:

1. Class-balanced local mini-batches.
2. Smoothed/clipped effective-number or inverse-frequency CE weights.
3. A small bounded balanced classifier-only calibration phase.

Do not use raw unlimited inverse-frequency weights: 2–5 sample classes would create unstable gradients. Primary gate is E3 macro recall/F1.

### D4 — Terminal-task reserve ablation

Keep the 10% reserve for tasks 0–4, compare it with reserve 0 only for task 5. Measure task-5 E3 per-class recall and old-task E3 accuracy.

### D5 — Router calibration/reference ablation

After the E3 classifier path is stable: sweep 20/50/100 router references per class, separate calibration references from fit references, and calibrate adaptive confidence thresholds using held-out data. Improve E4 toward E3; the current gap is 11.14 points. Keep `task_end` schedule for runtime.

### D6 — Full retrain only after a smoke winner

Train seed 42 from task 0 only after D1–D5 select a configuration. Then save commit/config/checkpoint hashes, rerun E0–E6, recheck 120-round peer gates, and only then run seeds 43 and 44 for mean ± standard deviation.

## 10. Guardrails and non-decisions

- Do not resume old `results.zip` for an official result.
- Do not lower cluster threshold below 0.20 merely to change accuracy.
- Do not claim peer aggregation is harmful before D1.
- Do not compare the 3-client quick metric directly with all-client E0–E6.
- Do not call unmasked 34-way metrics a direct substitute for task-routed DeNICE without defining a new method/protocol.

## 11. Workspace incident

During diagnosis on 13/08/2026, `D:\Project\FL_IL_IDS` unexpectedly became empty, including `.git`. No delete/reset command was issued by this audit. The ZIP archive in Downloads remains intact. A clean temporary clone at commit `321a023` is being used only for debugging and documentation:

```text
C:\Users\khoak\AppData\Local\Temp\fl_il_ids_debug_20260813_2252
```

Do not overwrite the empty original workspace until the user decides whether to attempt recovery of uncommitted files. All substantive DeNICE code through `321a023` is already pushed to `origin/main` and can be restored by cloning.

## 12. Status checklist

- [x] Diagnose and fix original `self_only` clustering collapse.
- [x] Recover router runtime and validate peer aggregation by smoke.
- [x] Complete full seed-42 run with 120 genuine peer rounds.
- [x] Run E0–E6 and router audits on final checkpoint.
- [x] Rank root causes: router, imbalance, capacity, adapter coverage; aggregation negative transfer remains an ablation question.
- [ ] D0 standardized eval.
- [ ] D1 self-only / peer-strength ablation.
- [ ] D2 label-aware `fc2` aggregation decision.
- [ ] D3/D4 imbalance and terminal-reserve ablations.
- [ ] D5 router calibration/reference sweep.
- [ ] D6 multi-seed final retraining.

