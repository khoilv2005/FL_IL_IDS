# DeNICE / CANDLE — current algorithm specification and AI audit brief

**Repository:** `FL_IL_IDS`  
**Branch state when written:** `main`, commits through `fce3e3f` locally  
**Date:** 14/08/2026  
**Scope:** the current implemented decentralized CANDLE-like method called
**DeNICE** in this repository. This is an implementation-aware specification,
not a claim that every component is theoretically validated.

---

## 1. What the method is intended to solve

DeNICE is a decentralized, federated class-incremental learning method for an
IDS dataset. The complete problem has 34 classes introduced in six sequential
tasks. Each task activates a non-IID subset of 100 clients. A client trains a
local CNN–GRU/NICE model, produces a compact context capsule, is grouped with
similar peers, and receives age-compatible peer updates; there is no central
FedAvg model in this path.

The intended ingredients are:

1. **NICE:** allocate/free/freeze neurons over incremental tasks to reduce
   forgetting.
2. **Context routing:** predict the episode/task of a test sample from binary
   internal activations, then restrict class logits to the predicted episode.
3. **CANC:** decide whether capacity pressure/novelty requires micro-adapters
   or high-layer-only adaptation.
4. **DeNICE capsules + dynamic affinity-propagation clustering:** identify
   compatible peers without sharing raw training data.
5. **Age-aware decentralized aggregation:** merge only receiver-plastic
   parameters and compatible adapters; preserve mature receiver knowledge.

The code should therefore be assessed against two distinct objectives:

- **Task-incremental / routed classification:** the router identifies an
  episode and the classifier chooses among its classes. This is the current
  primary DeNICE objective.
- **Class-incremental 34-way classification without a task mask:** this is a
  different, harder objective. The current NICE output mechanism was not
  designed or calibrated as a direct 34-way classifier, so its unmasked scores
  must not be presented as equivalent to routed scores.

---

## 2. Data and task protocol

The local dataset is `Dataset/2023/federated_splits/100-clients`.

| Task | New classes | Active clients in full split |
|---:|---|---:|
| 0 | 0–5 | 50 |
| 1 | 6–11 | 60 |
| 2 | 12–17 | 70 |
| 3 | 18–23 | 80 |
| 4 | 24–29 | 89 |
| 5 | 30–33 | 98 |

The distribution is extremely imbalanced. Examples from the full training
split: task 5 has class 32 with 250,308 samples (97.3% of that task) but class
31 with only 837. Similar large disparities exist in prior tasks. Local
training currently uses natural sample order/distribution and ordinary,
unweighted cross entropy.

Relevant code:

- Dataset metadata and client `.npz`: `Dataset/2023/federated_splits/100-clients/`.
- Data loader: `fed_learning/data/incremental_loader.py`.
- Production config: `train_incremental_kaggle.py`.

---

## 3. Model and NICE state

### 3.1 Backbone

`DeNICEModel` subclasses `NICEModel`:

```text
input sequence
  ├─ CNN path: conv1 → BN/ReLU/pool → conv2 → BN/ReLU/pool → conv3 → BN/ReLU/pool
  ├─ GRU path: GRU, last hidden state
  └─ concatenate CNN flatten + GRU state → fc1 → dropout → fc2 (34 logits)
```

The model maintains `unit_ranks` for `conv1`, `conv2`, `conv3`, `gru`, `fc1`,
and `fc2`:

| Rank | Meaning | Training consequence |
|---:|---|---|
| `0` | free/young | capacity available for selection |
| `1` | learner | active for the current episode |
| `>=2` | mature | gradient updates protected |
| `<0` | retired | excluded until optional recycling |

There are connection masks and bias masks. When connections are dropped, their
weights are physically zeroed. Mature gradients are reset after backpropagation.

Important NICE behavior:

- `forward_output()` masks young `fc1` activations and applies `LetLearner` to
  retain only output rows whose `fc2` rank is learner (`==1`).
- The local loss is `F.cross_entropy(output, target.long())` on this full,
  masked output. It currently has no per-class weighting.
- `forward_inference()` exposes all rows; DeNICE evaluation then applies an
  explicit seen-class or episode-class logit mask.

This explains why correct episode masking can work while raw 34-way inference
is poorly calibrated: training deliberately suppresses non-learner output rows
and later freezes mature rows.

Relevant code:

- `fed_learning/models/nice_model.py`: masks, age state, forward paths and
  gradient protection.
- `fed_learning/strategies/incremental/nice.py`: learner selection, phase
  operations, standard CE loss.
- `fed_learning/clients/nice_client.py`: phase loop, fresh Adam per phase,
  `select_learner_units`, pruning/growth, backward and frozen-gradient reset.

### 3.2 Task preparation and reserve

At task start the runner assigns each new class's `fc2` row to learner rank,
sets the context detector's episode-to-class map, samples per-class references,
computes novelty/capacity/consumption, obtains a CANC plan, and activates any
planned adapters. During and after local training, the runner enforces a
minimum free-capacity ratio (production default 10%) by releasing only neurons
selected during the current task, rather than unfreezing old knowledge.

The terminal task still uses the same reserve by default. This is intentionally
not assumed correct; it is an open D4 ablation.

---

## 4. CANC capacity and novelty controller

### 4.1 Capacity quantities

For layer `l` and client `i`:

```text
rho0_i,l = number of free units / total units
rhom_i,l = number of mature units / total units
u_i,l    = free units before task that became selected learners / free units before task
```

The runner also computes a local old-reference validation-loss increase:

```text
Delta L_val = max(0, CE(current model on local old references)
                     - CE(baseline model on the same references))
```

Novelty is computed by `NoveltyEstimator` from activation-pattern changes. On
task 0 there is no history, therefore novelty is zero and CANC always selects
NICE-only with no adapter.

### 4.2 Pressure and actions

The implementation uses:

```text
kappa_i,l = alpha * (1 - rho0_i,l)
          + beta  * u_i,l
          + gamma * Delta L_val
          + delta * novelty_i
```

Default coefficients in `CANCConfig`: `alpha=0.45`, `beta=0.25`,
`gamma=0.15`, `delta=0.30`. Layer decisions are:

| Condition | Action |
|---|---|
| depleted low layer + high novelty | emergency low-layer adapter |
| `kappa >= kappa_adapter`, or depleted layer with novelty/consumption | add adapter |
| depleted low layer but low novelty | high-layer-only |
| otherwise | NICE-only |

Production enables adapters only at `fc1`, `gru`, `conv3`. Optional graceful
recycling exists but is disabled in the production configuration.

Relevant code:

- `fed_learning/strategies/incremental/denice_capacity.py`.
- `fed_learning/strategies/incremental/denice_novelty.py`.
- task preparation in `fed_learning/training/decentralized_denice_il.py`.

---

## 5. Context router and adapters

### 5.1 Router features and memory

`ContextDetector` computes per-sample activations from `conv1`, `conv2`,
`conv3`, and GRU. These are binarized using layer thresholds derived from the
first episode (`mean + std` in the original NICE-style mechanism). It stores:

- `activation_memory[episode]`: binary activation sketches;
- `reference_input_memory[episode]`: a small client-local raw reference bank;
- `episode_classes[episode]`: classes introduced by each task;
- fitted logistic-regression router state and freshness metadata.

After model aggregation, old binary sketches no longer belong to the current
encoder. The current implementation re-encodes the raw reference bank and
refits the router at **task end**, rather than refitting every round. This is a
deliberate runtime/correctness tradeoff.

### 5.2 Router model

The code supports:

- `chained`: the original sequence of binary logistic regressions; and
- `multiclass`: one multinomial `LogisticRegression(class_weight="balanced")`.

Production DeNICE uses `multiclass` because the chained form tends to default
old examples to later episodes. A router can be marked stale after aggregation;
the checkpoint records freshness status and refresh task/round.

### 5.3 Micro-adapters

For a context/episode `t` and layer `l`, a micro-adapter is a low-rank residual:

```text
A_l(h) = U_l sigmoid(V_l h)
```

`U_l` starts at zero, so a new adapter initially has exactly zero residual. An
adapter is keyed by `(context_id, layer, rank, architecture_version)`. At
inference, a predicted or oracle context activates only matching adapters;
missing adapters simply disable the adapter on that layer.

The final full seed-42 checkpoint showed very low effective adapter coverage:
only 4,915 / 50,000 samples had a matching active adapter under the tested
protocol. This is a confirmed limitation, not yet isolated as a code bug.

Relevant code:

- `fed_learning/servers/nice_server.py` (`ContextDetector`).
- `fed_learning/models/denice_model.py` (`MicroAdapter`, registry, activation).
- `fed_learning/training/denice_eval.py` (routing and adapter inference paths).

---

## 6. Context capsule and peer similarity

Each active client creates a serializable `ContextCapsule`; it contains no raw
training tensors. Main fields are:

- class-balanced activation prototypes and per-class activation prototypes;
- binary age masks;
- activation-derived neuron importance;
- per-layer free/learner/mature capacity histogram;
- label histogram and label set;
- sample count and scalar reliability `1 / (1 + local loss)`;
- context-router summary and adapter registry;
- optional update summary.

For two capsules, the base similarity components are:

```text
prototype:    shared-class prototype cosine similarity
age:          Jaccard similarity of age masks
importance:   cosine similarity
label:        Jaccard overlap of label sets
capacity:     normalized capacity-histogram compatibility
reliability:  reliability of candidate peer j
update:       L2 distance of normalized update summaries
```

The directional score is:

```text
s_ij = 0.35 * prototype + 0.10 * age + 0.15 * importance
     + 0.25 * label     + 0.05 * capacity + 0.10 * reliability
     - 0.10 * update_distance
```

Components with near-zero across-pair signal can be suppressed and remaining
weights rescaled. The full expression is implementation heuristic, not a
learned/calibrated metric.

Relevant code:

- `fed_learning/strategies/decentralized/denice_capsule.py`.
- `fed_learning/strategies/decentralized/denice_clustering.py`.

---

## 7. Dynamic clustering and collaboration groups

1. Build pairwise context-similarity matrix and sparse mutual top-k / quantile
   context graph.
2. Run affinity propagation (AP) dynamically to produce raw cluster labels.
3. Compute silhouette score from the similarity/distance structure.
4. Treat the raw result as valid only when the signal/validity conditions pass.
5. If invalid, reuse a compatible previous valid assignment; if unavailable,
   make every client self-only.
6. For every receiver, restrict its cluster group to graph neighbors, then to
   peers sharing at least one label and (when needed) inside centroid distance
   threshold `0.75`.

Production cluster parameters are:

```text
theta_s = 0.20
edge_top_k = 40
edge_quantile = 0.25
min_signal_std = 0.02
invalid_policy = previous_valid_or_self_only
```

Historical failure and repair:

- The old run used `theta_s=0.50`; every raw AP result was invalid and training
  silently became local-only.
- Current runner uses the effective fallback assignment, logs raw/effective K,
  and has a collaboration guard. In the full recovery run, all 120 rounds were
  valid and none were self-only.

The guard checks effective K < active clients, mean group size > 1, positive
peer contribution, and mean peer alpha above a configured threshold. Production
mode is `error` after two consecutive collapsed rounds.

---

## 8. Age-aware decentralized aggregation

### 8.1 Peer weights

For receiver `i` and group member `j`:

```text
w_ij = max(s_ij, 0) * transform(n_j) * max(reliability_j, 0)
alpha_ij = w_ij / sum_k w_ik
```

`transform(n)` defaults to `log(1+n)`. A self-floor ensures a lower bound for
`alpha_ii`; production default is `0.25`. The D1 low-peer control sets it to
`0.50`.

### 8.2 Parameter aggregation

All clients start a round from their own parameter state. For a receiver:

```text
Delta theta_j = theta_j_after_local_train - theta_i_before_aggregation
theta_i_new = theta_i_before + eta * M_i ⊙ sum_j alpha_ij * Delta theta_j
```

`M_i` is receiver-defined. Rows linked to mature receiver units are zeroed, so
a peer cannot overwrite the receiver's mature knowledge. The code supports
weighted mean (default), coordinate median, and trimmed mean. `eta=1` in the
production config.

### 8.3 Adapter and age aggregation

- An adapter is averaged only when its exact key and parameter shapes match.
  The receiver contributes with self alpha; peers contribute with their peer
  alphas.
- Neuron ages use **consensus**, not elementwise maximum. A receiver is promoted
  only when peers marking that unit mature carry at least configured consensus
  weight (default 0.5). Existing receiver maturity is never reduced.

### 8.4 D1 ablation control

`denice_aggregation_mode` accepts:

- `peer`: normal decentralized aggregation;
- `self_only`: preserve local train/capsule/raw clustering but force every
  group to `[receiver]`, preventing peer parameter, adapter, and age merging.

This is intentionally not implemented as `eta=0`, because `eta=0` would still
allow separate adapter/age merge logic. It is a test-only control; production
remains `peer`.

Important unresolved point: label overlap only determines whether a peer is in
a group. Once admitted, the current code can aggregate all plastic `fc2` rows.
There is no per-class row-support condition yet. This is a hypothesis requiring
evidence, not a confirmed bug.

Relevant code:

- `fed_learning/strategies/decentralized/denice_aggregation.py`.
- `_aggregate_round` in `fed_learning/training/decentralized_denice_il.py`.

---

## 9. End-to-end training loop

For each task and active client:

```text
1. Load client task data; optional deterministic stratified sample cap.
2. Create or bootstrap a client model.
   - Task-0 newcomers use the shared initial template.
   - Later newcomers clone a compatible existing client model/state.
3. Prepare task: NICE classes, references, novelty, CANC, adapters, capacity plan.
4. Repeat R rounds:
   a. Local one-phase NICE training.
   b. Enforce free-capacity reserve; update freeze/BN state.
   c. Update local context memory only if schedule requires it.
   d. Build context capsules.
   e. AP clustering, group gates, age-aware aggregation.
   f. Mark routers stale after encoder change.
   g. At task end, refresh router memory/retrain routers.
   h. Log timing, raw/effective K, group sizes, alphas, guard, capacity, freshness.
   i. Save checkpoint according to cadence.
5. End-of-task reporting/checkpoint/continuation-state handling.
```

The runner stores full or delta round checkpoints, cluster history, router
freshness, round metrics, task metrics, adapter registry, capsule snapshots,
and debug JSONL. Main implementation:
`fed_learning/training/decentralized_denice_il.py`.

---

## 10. Inference and evaluation policies

### 10.1 Routed logit path

For each test batch, DeNICE can:

1. predict episode with the local context detector;
2. activate adapters for that episode if present;
3. calculate logits;
4. apply a mask to classes admitted by one predicted episode (`hard`), a union
   of top-k episodes (`topk`), or adaptive confidence logic;
5. argmax the masked logits.

Oracle paths map a true class to its episode only for diagnosis. They are never
deployable inference policies.

### 10.2 E0–E6 diagnostic matrix

| ID | Policy | Purpose |
|---|---|---|
| E0 | backbone, no adapter, no mask | raw backbone/classifier |
| E1 | predicted adapter, no mask | router-selected adapter effect |
| E2 | oracle adapter, no mask | adapter ceiling independent of router |
| E3 | oracle hard episode mask | task-routed classifier ceiling |
| E4 | predicted hard episode mask | deployed routed prediction |
| E5 | top-k episode union | ambiguity fallback diagnostic |
| E6 | adaptive routing | confidence-based fallback diagnostic |

`coverage_aware_local` assigns each test item exactly once to a client whose
router covers the true episode. It is a personalized/distributed protocol, not
a central ensemble. Metrics concatenate all predictions before computing global
accuracy and F1.

### 10.3 Standardized class-balanced evaluation (D0)

`eval_checkpoint.py` and `run_denice_p6_eval.py` now support
`--samples-per-class N`.

- Selects exactly N examples from every observed class with a deterministic
  seed before coverage-aware partitioning.
- Fails if a class has too few examples unless
  `--class-balanced-with-replacement` is explicitly set.
- Records selected and source support by class, unique source sample count,
  seed, and SHA-256 of selected indices.
- Saves per-class recall and per-episode router recall in compact summaries.

Do not mix these metrics with the older quick post-task diagnostic, which used
only three full-coverage clients and a separate 50,000-sample subset.

Relevant code:

- `fed_learning/training/denice_eval.py`.
- `eval_checkpoint.py`.
- `run_denice_p6_eval.py`.

---

## 11. Current production configuration

The relevant default values in `train_incremental_kaggle.py` are:

```text
seed = 42
TRAIN_PHASE = 5              # task 0–5 from scratch
rounds_per_task = 20
batch_size = 2048
learning_rate = 0.001
nice_phase_epochs = 1

denice_adapter_layers = [fc1, gru, conv3]
denice_router_mode = multiclass
denice_router_update_schedule = task_end
denice_router_reference_per_class = 20
denice_router_refresh_batch_size = 2048

denice_cluster_theta_s = 0.20
denice_cluster_edge_top_k = 40
denice_cluster_edge_quantile = 0.25
denice_cluster_min_signal_std = 0.02
denice_cluster_invalid_policy = previous_valid_or_self_only

denice_aggregation_mode = peer
denice_aggregation_method = weighted_mean
denice_aggregation_self_floor = 0.25
denice_collab_use_context_edges = True
denice_require_label_overlap = True
denice_centroid_gate_threshold = 0.75
denice_age_merge_policy = consensus
denice_age_merge_consensus_threshold = 0.5

denice_min_free_capacity_ratio = 0.10
denice_collaboration_guard_mode = error
denice_max_consecutive_self_only_rounds = 2
denice_min_mean_peer_alpha = 0.05
round_checkpoint_every = 5
```

`DENICE_CONFIG_OVERRIDES` can supply a JSON object to
`train_incremental_kaggle.py` for controlled experiments without editing the
production configuration.

---

## 12. Evidence to date

### 12.1 Fixed collaboration/runtime failure

The old archive had 114/114 self-only rounds because silhouette validity
threshold `0.50` rejected all observed raw cluster scores. The recovery full
run (seed 42, six tasks × 20 rounds) showed:

| Evidence | Result |
|---|---:|
| rounds | 120 |
| valid clusters | 120/120 |
| self-only rounds | 0/120 |
| mean effective K | 19.625 |
| mean group size | 6.577 |
| mean peer alpha | 0.638 |
| minimum peer-aggregated clients/round | 47 |

Thus current poor metrics cannot be attributed to the previously known
local-only clustering collapse.

### 12.2 Final seed-42 component isolation, prior full run

| Policy | Accuracy | Macro-F1 |
|---|---:|---:|
| E0 backbone/no mask | 10.83% | 6.17% |
| E1 predicted adapter/no mask | 11.85% | 6.83% |
| E2 oracle adapter/no mask | 12.00% | 6.90% |
| E3 oracle hard mask | 53.35% | 31.82% |
| E4 predicted hard mask | 42.21% | 24.51% |
| E5 top-k=2 | 26.23% | 14.11% |
| E6 adaptive | 41.74% | 24.07% |

Interpretation:

- The task-routed classifier has usable within-episode discrimination (E3).
- Correct adapter alone does not create a direct global classifier (E2).
- Router error costs `53.35 - 42.21 = 11.14` percentage points (E3→E4).

Router audits in that run: final-current-feature balanced accuracy 49.88%,
persisted-memory balanced accuracy 98.55%, refit-memory-holdout accuracy
68.08%, and high mean confidence 80.16%. This is overfitting/mismatch between
tiny reference memory and final-model/global-test features, not a completed
root-cause fix.

### 12.3 D1 local smoke, 14/08/2026

Protocol: seed 42, tasks 0–2, 3 rounds/task, first 10 active clients, max 300
stratified train samples/client, all-client coverage-aware evaluation with 100
samples/class (1,800 total; sampling with replacement allowed explicitly).

| Variant | mean peer alpha | E3 acc. | E3 macro-F1 | E4 acc. | E4 macro-F1 |
|---|---:|---:|---:|---:|---:|
| peer default (self floor .25) | .509 | 19.28% | 9.88% | 10.06% | 5.24% |
| self-only | .000 | 18.67% | **10.89%** | **10.61%** | **6.87%** |
| peer, self floor .50 | .401 | **19.33%** | 10.31% | 9.33% | 5.17% |

All 9 raw clusters were valid. The observed differences are too small and the
budget too short to prove peer aggregation harmful. **Do not implement
label-aware `fc2` aggregation (D2) based only on this smoke.** Detailed local
artifact: `d1_local/D1_results_2026-08-14.md`.

---

## 13. Confirmed facts, open hypotheses, and non-conclusions

### Confirmed

1. The historical effective-local-only clustering bug is fixed.
2. The router is a major final-metric bottleneck.
3. Direct unmasked 34-way classifier performance is weak because of NICE's
   learner/mask/freeze training design; it is not comparable to E3 without
   defining a new class-incremental method.
4. Extreme training imbalance with unweighted CE is directly incompatible with
   strong rare-class macro recall.
5. Router reference-memory performance does not generalize reliably to final
   model/global test features.
6. Final-task reserve and adapter coverage are restrictive; both need ablation.

### Plausible but not proven

1. Peer aggregation dilutes rare `fc2` rows after label-overlap admission.
2. CANC pressure thresholds or adapter placement do not match actual task shift.
3. The fixed 10% capacity reserve hurts task-5 adaptation more than it helps
   retention.
4. Router reference count/calibration/activation representation can close much
   of E3→E4 gap.

### Do not conclude

- Do not lower `theta_s` below 0.20 to chase accuracy.
- Do not call peer aggregation harmful from D1 alone.
- Do not compare the quick 3-client post-task metric with all-client E0–E6.
- Do not call E0/E2 a substitute for the defined task-routed target.
- Do not claim a full-method improvement before a larger controlled ablation
  and at least multiple training seeds.

---

## 14. Recommended next research/debug sequence

1. **Repeat D1 at a decision-capable budget:** e.g. 20 clients, tasks 0–2,
   5–10 rounds/task, fixed balanced support. Compare peer/default, self-only,
   self-floor .50. Check E3 accuracy and macro-F1, E4, old/new task recall,
   peer alpha, group sizes, and per-class recalls.
2. **D3 imbalance correction:** one change per run: class-balanced local
   batches, then clipped/smoothed effective-number or inverse-frequency CE,
   then possibly a bounded classifier-only calibration. Primary gate: E3
   macro-F1/rare-class recall; monitor retention.
3. **D4 terminal reserve:** keep reserve for tasks 0–4, compare terminal-task
   reserve .10 versus 0.0; report task-5 and old-task E3 outcomes.
4. **D5 router:** after E3 is stable, sweep 20/50/100 reference examples per
   class, separate fit and calibration references, and calibrate adaptive
   thresholds. Primary gate: reduce E3→E4 gap while retaining task-end schedule.
5. Only if larger D1 establishes negative transfer, implement and test **D2**:
   a peer may update a receiver `fc2` row only with positive capsule support
   for that class; preserve receiver's local row otherwise.
6. Retrain full seed 42 only after selecting a smoke winner, then seeds 43/44.

---

## 15. Prompt for an independent AI algorithm audit

Copy the prompt below, attach this file and give the auditor read access to the
repository. Ask it to inspect source before accepting any claim.

```text
You are conducting an adversarial, implementation-aware research audit of the
DeNICE/CANDLE algorithm in the FL_IL_IDS repository. Read
denice_algorithm_current_2026-08-14.md first, then inspect the cited source
files and tests. Do not assume the document is correct; verify all material
claims against code and artifacts.

Goal
----
Determine whether the current DeNICE method has (a) remaining implementation
bugs, (b) internal algorithmic inconsistencies, (c) invalid or misleading
evaluation claims, or (d) justified design limitations that explain low metrics.
Separate verified facts from hypotheses. Do not recommend tuning merely because
it might raise accuracy.

Required code paths
-------------------
1. `fed_learning/training/decentralized_denice_il.py`
   - task preparation, reserve enforcement, bootstrap/catch-up,
     `_aggregate_round`, clustering fallback, collaboration guard, router
     freshness, checkpoints, and task transitions.
2. `fed_learning/models/nice_model.py` and
   `fed_learning/strategies/incremental/nice.py`
   - unit rank semantics, output masking, training vs inference path, loss,
     freeze masks, and whether class-incremental claims are justified.
3. `fed_learning/models/denice_model.py`,
   `fed_learning/servers/nice_server.py`, and
   `fed_learning/training/denice_eval.py`
   - adapter activation, router feature extraction, router refitting,
     calibration/freshness, hard/top-k/adaptive masking.
4. `fed_learning/strategies/incremental/denice_capacity.py` and novelty code
   - CANC signals, units/scales, action thresholds, and whether the decisions
     actually control behavior as intended.
5. `fed_learning/strategies/decentralized/denice_capsule.py`,
   `denice_clustering.py`, and `denice_aggregation.py`
   - information leakage, similarity directionality, AP validity, group gates,
     alpha normalization/self floor, parameter delta reference, mature-row
     protection, adapter aggregation, and neuron-age merging.
6. `eval_checkpoint.py`, `run_denice_p6_eval.py`, and `tests/test_denice.py`
   - E0–E6 semantic validity, coverage-aware partitioning, class-balanced
     support/replacement, metrics aggregation, and regression coverage.

Questions to answer
-------------------
A. Precisely state the actual mathematical objective optimized during local
   training. Is it compatible with the reported E3 task-routed metric and with
   any unmasked 34-way metric?
B. Trace one sample through task preparation, local train, capsule, cluster,
   aggregation, router refresh, and routed inference. Identify every point at
   which its class-specific information can be lost, masked, frozen, diluted,
   or routed incorrectly.
C. Trace one `fc2` row for a rare class across two peers where only one peer
   has that class. Establish from code whether the other peer can influence the
   row, and whether this is a bug, an intended assumption, or unproven risk.
D. Verify that `self_only` D1 blocks all peer parameter, adapter, and age
   transfers, and does not accidentally change unrelated training behavior.
E. Assess whether the CANC formulas are numerically meaningful with their
   current feature scales and thresholds. Identify dead branches, impossible
   branches, or signals that are always zero/stale.
F. Assess router memory leakage/overfitting and test whether the router is
   trained and evaluated on comparable feature distributions. Explain the
   E3→E4 gap without conflating it with classifier accuracy.
G. Audit whether each evaluation policy uses oracle information only where
   labelled as oracle, and whether coverage-aware assignment changes what can
   legitimately be claimed.
H. Review reproducibility: seed handling, client ordering, sampling,
   checkpoint state, router serialization, and evaluation index provenance.
I. Rank all findings by severity and confidence. For every alleged issue,
   supply: exact files/functions/lines, a minimal reproduction or test,
   affected metric/objective, and the smallest safe repair.

Deliverable format
------------------
1. Executive verdict: no more than 10 bullets.
2. Table: finding | category (bug/design/protocol/hypothesis) | confidence |
   evidence | likely effect | smallest verification/fix.
3. A concise end-to-end data/control-flow diagram.
4. A list of claims in the supplied document you confirmed, refuted, or could
   not verify.
5. A dependency-ordered experiment plan. Do not recommend a full retrain until
   you specify the smoke gate that chooses a configuration.
6. New regression tests required before accepting any algorithm change.

Audit standards
---------------
- Be adversarial: prefer source evidence over comments/docstrings.
- Distinguish “metric low because the method is not designed for this target”
  from “metric low due to an implementation defect.”
- Never infer causality from one seed or the short D1 smoke.
- Preserve prior fixes: do not propose removing collaboration guards, validity
  fallbacks, router freshness tracking, or class-balanced provenance merely to
  increase a metric.
- If information is missing, state exactly which artifact/test would resolve it.
```

---

## 16. Supporting documents and artifacts

- `denice_full_context_2026-08-13.md`: chronological recovery/debug handoff.
- `audit_10/08/26.md`: original audit and recovery plan.
- `d1_local/D1_results_2026-08-14.md`: current local D1 evidence.
- `tests/test_denice.py`: primary executable regression evidence.
- `run_denice_d1_kaggle.py`: controlled D1 launcher for a CUDA Kaggle run.

