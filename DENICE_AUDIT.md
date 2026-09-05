# DeNICE forensic audit — 2026-09-05

Scope: historical **100-client** results, newer FULL_D2 runs and source at
`4406d78` plus the targeted fixes below. Initial worktree was clean. The current
**500-client/task-5 Kaggle continuation is a different experiment**; its Drive
URL, task selection, client count and checkpoint frequency were not changed.

Status: artifact/code audit and small causal diagnostics completed. The original
audit lacked the dataset; the subsequently supplied `Downloads/archive.zip` has
now been inspected (see dataset addendum in section 11). Complete real-data
forgetting and transition-level experiments remain unmeasured. This is **not** a
claim that the entire historical accuracy collapse has been causally resolved.

## 1. Executive summary

Ranked findings, distinguishing demonstrated mechanisms from unknown impact:

1. **Confirmed checkpoint/bootstrap fidelity bug.** Connection masks were omitted
   from snapshots and representative clones. Weights alone do not define this
   masked network. Save/load changed logits by up to **16.596607** in a controlled
   reproduction; the patch restores exact tested predictions. The actual seed-42
   terminal checkpoint contains masks for **0/98 clients**. Its metrics describe
   a reconstructed network whose identity with the training network is uncertified.
   Real-data accuracy impact is unmeasured. Fixed for new snapshots and bootstrap.
2. **Weak reconstructed representation/class scoring, not just routing.** Across
   FULL_D2 seeds 42–46, E4 predicted-hard accuracy is **28.05%**, versus **38.08%**
   with oracle adapter and true-episode mask. Oracle adapter without episode mask
   scores only **8.56%**. Perfecting the routing policy alone does not produce a
   strong classifier. These are paired inference effects, not additive training
   root-cause percentages.
3. **Confirmed functional protection gap in GRU.** Changing only plastic gate rows
   changes mature features by **0.592447**, despite unchanged mature parameters.
   Isolating plastic-to-mature recurrent/upper-layer edges in a diagnostic control
   yields **0** change. This demonstrates a forgetting mechanism, not its share of
   the historical collapse. No production architecture change was made.
4. **Router features move with the backbone.** A separate train-reference-only
   probe changes heldout route accuracy **100 → 93.75 → 100%** after a backbone
   perturbation and reference refresh. FULL_D2 already refreshes at task boundaries,
   so stale memory alone cannot be asserted to explain its final 52.51% routing.
5. **Provenance and method drift.** The old “5-seed” CSV varies evaluation splits,
   not training. Separately, ZIPs 42–46 corroborate five different training runs.
   P6 incorrectly wrote evaluation RNG as training seed; fixed. Current single-model
   capsule DeNICE is not the original multi-bank protocol, and bootstrap copies
   raw training references across simulated clients.

Not demonstrated: order-dependent peer averaging, complete capacity exhaustion at
task 2, all clients becoming self-only, or a numerical 100-client forgetting rate.
Do not immediately spend compute on a full five-training-seed rerun.

## 2. Current metric diagnosis

Historical `denice_eval_5seed_summary.csv`, final communication round 19; percentages:

| Task | Active clients | K | Accuracy | Macro-F1 | Weighted-F1 | Route accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 50 | 16 | 74.84 | 29.59 | 71.46 | 100.00 |
| 1 | 60 | 14 | 72.26 | 30.30 | 67.43 | 96.39 |
| 2 | 70 | 22 | 42.01 | 18.91 | 33.00 | 62.75 |
| 3 | 80 | 25 | 25.97 | 12.72 | 17.97 | 41.65 |
| 4 | 89 | 18 | 23.14 | 10.85 | 16.57 | 33.19 |
| 5 | 98 | 35 | 22.49 | 9.13 | 15.72 | 35.82 |

The break begins at task 1→2: accuracy −30.25 points, routing −33.64 points.
Early accuracy is misleadingly reassuring: task-0 macro-F1 is only 29.59%.
All **120/120** aggregate rows have zero training-loss standard deviation.

New evidence: `C:/Users/khoak/Downloads/{42,43,44,45,46}.zip`, members under
`denice_full_d2_seed_<seed>/terminal_task_5/`. The actual terminal checkpoint bytes
were streamed to SHA-256. All five hashes differ and each matches every policy's
recorded hash. Saved configs independently record training seeds 42–46.

| Training seed | First-round training loss | Terminal byte hash agrees with all policies |
|---:|---:|---|
| 42 | 1.5882173628 | Yes |
| 43 | 1.5788498099 | Yes |
| 44 | 1.5278138269 | Yes |
| 45 | 1.4840098121 | Yes |
| 46 | 1.4894527827 | Yes |

FULL_D2 uses 3,400 balanced draws, 100 per class, with replacement. Mean ± sample
standard deviation across five runs:

| Policy | Inference intervention | Accuracy % | Macro-F1 % |
|---|---|---:|---:|
| E0 | Backbone, no episode mask | 7.97 ± 0.32 | 6.55 ± 0.35 |
| E1 | Predicted adapter, no episode mask | 8.50 ± 0.42 | 7.02 ± 0.37 |
| E2 | Oracle adapter, no episode mask | 8.56 ± 0.41 | 7.09 ± 0.36 |
| E3 | Oracle adapter + true-episode hard mask | 38.08 ± 1.39 | 34.02 ± 1.54 |
| E3b | No adapter + true-episode hard mask | 37.35 ± 1.47 | 33.26 ± 1.62 |
| E4 | Predicted adapter + predicted hard mask | 28.05 ± 0.39 | 25.71 ± 0.52 |
| E5 top-2 | Top-2 routed policy | 18.64 ± 0.62 | 16.15 ± 0.72 |
| E5 top-3 | Top-3 routed policy | 14.62 ± 0.64 | 12.34 ± 0.77 |
| E6 | Adaptive routed policy | 27.75 ± 0.29 | 25.70 ± 0.39 |

Weighted-F1 equals macro-F1 here because supports are equal. Recomputed macro-F1
from saved target/prediction traces differs by at most `5.55e-17`. The low scores
are not an arithmetic metric-aggregation mistake. They validate the evaluated
reconstructed model, not its identity with a missing-mask training state.

## 3. Evaluation validity

### Clients and seeds

`prepare_data/step2_federated_splits.py::ClientAllocator._allocate` (line 176)
selects 50%, 60%, …, 100% of the configured population independently per task;
sets are not guaranteed nested. `IncrementalDataLoader.get_client_data` (121)
and the DeNICE runner exclude empty current-task allocations. Thus 100 means
population, not 100 nonempty participants each task; 89/98 are compatible with
empty allocations. Persistent model ownership, current participation, and episode
coverage are distinct quantities.

`eval_5.ipynb::evaluate_one_checkpoint` (notebook JSON near line 577) and its outer
`EVAL_SPLIT_SEEDS` loop reuse checkpoint paths. They shuffle and equally partition
the cumulative test set across clients, then concatenate predictions for global,
sample-weighted metrics—not an unweighted mean of client F1. Each client gets a
subset, not the entire global test set. Samples can be assigned to clients whose
router lacks their episode. This proxy can depress results relative to eligible
placement, and its variance is assignment variance, not optimization variance.

FULL_D2 configs, different first losses and different actual checkpoint hashes
corroborate **five independent training runs**. They reuse a prebuilt dataset;
changing training RNG does not regenerate the data split/Dirichlet allocation.
Initialization and training RNG change. Evaluation RNG changes too: selected
test-panel seeds are 47–51 (`eval_seed + task_id`, final task 5). Reported std
therefore mixes training and sampled-test variation. Within each run all nine
policies have the same source-index hash; paired contrasts are supported. Future
training-seed comparison should use one fixed heldout panel across runs.

### Coverage, labels and masks

`eval_checkpoint.py::_build_coverage_aware_partitions` (around line 550) uses
true test label→episode mapping to place samples on eligible clients. Eligibility
intersects declared episode classes, activation memory and fitted router coverage.
All FULL_D2 policies report 3,400 assigned, zero unsupported, 100% coverage.
This is a **label-conditioned assignment benchmark**, not arbitrary task-agnostic
deployment. True episode is not passed into E4 inference, but placement itself
uses privileged information. Coverage also does not prove local training support
for every class within an episode. This can remove unsupported-client penalties
but potentially inflate results relative to unconstrained deployment.

`IncrementalDataLoader._parse_task_classes/get_test_data` (94/169) uses persistent
global labels and cumulative metadata classes: six labels per task 0–4, four at
task 5. Task 2 has no special class-count increase. No per-task relabel-to-zero
was found in this path. `denice_eval.py::_denice_routed_logits_with_episodes`
(129) uses eval/no-grad, seen-class restriction and context-selected adapters.
E2/E3/E3b intentionally use true episode and are diagnostic oracles, not deployable
scores. Their returned route accuracy still describes the *predicted* detector;
52.51% there does not mean oracle selection is only 52.51% correct. E0's zero
route metric is a no-router sentinel.

Macro-F1 normally uses sklearn's union of observed target/predicted labels. All
34 classes have support 100 in FULL_D2, so its denominator is correct. Missing
labels can change the denominator on smaller panels; always report supports.
Finite −100 masking is not unconditional exclusion for extreme logits (section 6).

### Contamination and environment limits

Data preparation splits before scaling (lines 499/511), fitting the scaler on
training data. `_prepare_client_task` and `_sample_reference_with_labels` use
client train tensors. Context refresh uses stored train references, not test data.
CANC's “validation loss delta” is on retained old training references, not a
clean heldout validation set. No test-to-context-memory flow was found.

Dataset absence prevents raw duplicate detection, checking NPZ feature/label
alignment and all task/client histograms. Split-before-scale alone cannot prove
raw flow independence. Representative bootstrap additionally copies raw train
reference inputs across clients: not test leakage, but a separate protocol issue.

The seed-42 checkpoint was unpickled only for schema inspection in the audit
environment. Its LogisticRegression reports sklearn 1.6.1 versus audit sklearn
1.9.0; no new real-checkpoint inference score from that cross-version load is
claimed. Reevaluate using a compatible pinned environment or legitimately refit
routers on authorized training references.

## 4. Actual DeNICE execution flow

Entry: `train_incremental_kaggle.py` →
`fed_learning/training/decentralized_denice_il.py::run_decentralized_denice_il`.
Below, `runner` means that module. DeNICE does not execute the centralized
NICEServer training loop or DFCA's model-bank runner.

| Stage / source | Input → output; retained/reset state | Local/shared and label use |
|---|---|---|
| Load, `IncrementalDataLoader`, 121/169 | Metadata/NPZ → current train/cumulative test tensors | Fixed global labels; split train/test |
| Active set, runner | Nonempty current clients; persistent model IDs | Orchestrator sees allocation; stable IDs preserved |
| Bootstrap, `_bootstrap_source_client/_bootstrap_denice_model`, 343/371 | Template or representative model → separately owned model; rejoin path catches up plastic state | Shares model/detector; raw-reference caveat |
| Prepare, `_prepare_client_task`, near 869 after patch | New class output ages=1, episode→classes, novelty/capacity/consumption/loss plan | Local train labels and task ID; no test |
| CANC/adapters, `CapacityController.plan_task`, `DeNICEModel.add_adapter` | Optional adapters/recycling; clear active adapters, retain registry/weights | Current training task becomes context ID |
| Local training, `NICEClient.train`, 93 | One phase per round via offset/override, learner selection, new Adam per phase | Current local train examples; no optimizer state across phases |
| Prune/grow, `strategies/incremental/nice.py` | Mutate masks; physically zero pruned weights; reopen young incoming links | Age-based topology; no internal GRU column isolation |
| Optimize, `NICEClient.train`, `DeNICEClient.train` | LetLearner CE, gradient masking/clipping, active adapters in optimizer | Current train labels; default zero weight decay |
| Reserve, `_enforce_minimum_free_capacity`, 193 | Release eligible learners originally free, preserve old mature ranks | FULL_D2 .1 through task 4, .0 terminal task |
| Router, `ContextDetector.push_activations/refresh_activation_memory`, 182/225 | Train inputs → binary CNN/GRU sketches → episode classifier | Persistent local references; intermediate rounds may be stale |
| Capsule, `_build_round_capsule`, `denice_capsule.build_context_capsule` | Class-aware activations, labels/histogram, ages, importance, capacity, count/reliability | Summaries shared; train labels only |
| Similarity/AP, `denice_clustering.build_similarity_matrix/dynamic_ap_cluster`, 230/431 | Normalized components, sparse graph → Dynamic-K assignment/validity | Simulator uses all active capsules |
| Fallback/group, `_effective_cluster_assignment`, 239 | Raw valid, matching previous-valid, or self-only | Previous assignment requires exact ordered IDs; resets per task |
| Aggregate, `_aggregate_round`, near 1191 after patch | Immutable pre-aggregation snapshots → personalized receiver states | Separate compute/apply sweeps; no global model |
| Ages/adapters/BN, `merge_neuron_ages/aggregate_adapters`, 229/294 | Consensus mature ages, same-key adapter averaging, refresh freeze/BN | Shared coordinates need not mean aligned features |
| End task, runner near 2800 | Reserve, one rank increase, freeze BN, novelty/reference baseline, clear adapters | Persistent state; learners become mature once per task |
| Evaluate/save, runner + `checkpoint_state.py` | Round, task and explicit continuation payloads | Test for reporting; round-19 precedes task-boundary aging |

Default non-robust aggregation is approximately
`theta_i' = theta_i + eta*M_i*sum_j alpha_ij*(theta_j-theta_i)`.
Differences are between **post-local-training model states and the receiver**,
not separately recorded SGD updates. Self delta is zero. Weights use effective
similarity × transformed sample count (`log1p` default) × reliability, normalized
with self floor .25. Counts are not multiplied twice. Mature receiver rows are
protected. FULL_D2's `denice_selective_fc2_peer_rows=True` additionally restricts
plastic output-class rows to peers whose capsules support that class, renormalizing
eligible weights. Do not assume this flag also describes the 500-client config.

Continuation retains client IDs, model/algorithm state, novelty, previous ages,
old reference banks/loss baselines, last-active task, histories and Python/NumPy/
Torch RNG. New masks are now included. `_load_denice_continuation_state` rejects
raw round checkpoints as full continuation. It resumes at a task boundary, not
an arbitrary mid-round optimizer state. Round-19 and post-task checkpoint states
are not equivalent. `save_resume_after_task` is used truthily in this runner,
not as an exclusive task-number filter.

## 5. Spec vs implementation drift

| Component | `denice_protocol.md` | Micro-adapter plan | Actual code | Risk |
|---|---|---|---|---|
| Ownership | Per-client multi-cluster bank | One backbone/context adapters | One personalized model per client | Not original bank-based method |
| Assignment | Local loss plus age/class compatibility | Capsule neighborhoods | Simulator-wide AP on active capsules | Coordination/communication assumptions need specification |
| Aggregation | DFCA running average | Masked delta collaboration | Synchronous masked state interpolation | Distinguish local SGD delta from model differences |
| Ages | Max-style propagation | Capacity-aware protection | Consensus default, max ablation | Different experimental variants |
| Routing | Chained binary episode learners | Context chooses adapters/mask | Configurable; FULL_D2 multiclass | Different router model |
| Capacity | NICE lifecycle | Novelty-driven CANC/recycling | CANC plus deterministic reserve/final override | Reserve is part of the method |
| Protection | Mature knowledge retained | Frozen backbone + adapters | Fixed rows, dense GRU recurrence; old adapters still mixed | Parameter preservation ≠ feature preservation |
| Privacy/memory | No raw-data exchange | Local reference machinery | Raw train references copied in bootstrap | No-raw-sharing claim is not satisfied |

`fed_learning/dfca/client.py::DFCANode` implements the separate cluster bank and
`aggregate_received_messages`; `dfca/aggregator.py` implements its running average.
`servers/dfca_server.py::_run_decentralized_aggregation` (473) snapshots messages
before aggregation. Those semantics are not automatically inherited by DeNICE.
Current DeNICE is closer to the adapter plan, with further experimental changes.
There is no centrally trained global model, but global simulator coordination
and source-reference transfer still require an explicit decentralized contract.

## 6. Findings

### [P0] Checkpoint and bootstrap lose connection topology — repaired for new states

Evidence:

- File/function/lines: `training/checkpoint_state.py::snapshot_denice_state` (129)
  and `restore_denice_state`; `eval_checkpoint.py::_make_denice_client_model` (404);
  `training/decentralized_denice_il.py::_bootstrap_denice_model` (371).
- Artifact/log: `audit_denice/invariant_evidence.json::historical_checkpoint_state`:
  actual seed-42 final file, 98 clients, zero connection-mask/frozen-BN-cache fields.
- Observed behavior: four policy roundtrips failed before patch; max logit error
  16.596607 backbone, 12.891 in routed/oracle reproduction. Separate late-client
  clone reproduction failed on dense initialized masks.

Expected behavior: identical effective network after serialization or cloning.

Actual behavior: new models initialize masks to ones; state_dict and ranks cannot
reconstruct historical connection topology. Aggregation can leave nonzero stored
parameters behind a zero mask, so physical pruning is not sufficient. Even stored
zeros do not prevent reopened links from changing subsequent training. Active
adapter and frozen-BN Python state were also omitted.

Why this matters: the exact problematic operation is reconstruction/clone. Both
evaluation identity and continuation can be affected.

Estimated impact: fidelity violation proven; real-data accuracy magnitude unknown.
P0 means experiment identity is uncertified, not that every old score has a known error.

Minimal reproduction: zero an fc2 connection mask while stored row weights=2;
serialize weights/algorithm state; load into a fresh model; compare identical inputs.

Recommended fix: include connection masks, active adapters and frozen-BN state;
reuse the common restore path in evaluator and bootstrap. Implemented. Missing
legacy masks are not guessed from ages or zeros; historical ZIPs remain unchanged.

Validation test: four-policy `test_masks_and_router_roundtrip_preserves_logits`,
standalone-loader mask test and `test_late_join_bootstrap_preserves_source_connection_topology`
in `tests/test_denice_forensic.py`. New tested masks/logits/routes match exactly.

### [P0] Evaluation RNG falsely labeled training provenance — repaired

Evidence:

- File/function/lines: `eval_5.ipynb::evaluate_one_checkpoint` near 577 and split-seed
  loop; `run_denice_p6_eval.py::main/_compact`; `summarize_denice_p6.py::main`.
- Artifact/log: 120/120 old train-loss std values zero; independently verified
  FULL_D2 configs/hashes/first losses in `artifact_evidence.json`.
- Observed behavior: P6 wrote `training_seed=args.seed`, where seed is evaluation
  RNG. Summarizer trusted it and required obsolete `e3_oracle_hard`, omitting E3b.

Expected behavior: distinct saved training seed and evaluation RNG; repeated
evaluation of one checkpoint never counts as independent training.

Actual behavior: misleading provenance and incompatible current policy schema.

Why this matters: false scientific error bars and final-validation gates.

Estimated impact: invalidates the old CSV's training-seed interpretation, not the
separately corroborated existence of the five FULL_D2 training runs.

Minimal reproduction: evaluate training seed 42 with evaluation seed 99; old P6
reported training seed 99.

Recommended fix: obtain training seed from checkpoint config; emit evaluation seed
separately and provenance field. Summarizer now validates policy/summary seed and
same-checkpoint consistency, rejects duplicate hashes between claimed runs and
missing provenance, supports E3/E3b and local-only protocol. Implemented. Old
metadata needs regeneration or external audit, not silent certification. Old
summary markdown/generator now explicitly say evaluation-split seeds; numbers unchanged.

Validation test: `test_p6_records_training_seed_from_checkpoint_not_evaluation_rng`,
`test_p6_validator_accepts_current_local_policies_and_rejects_false_provenance`
and updated three-verified-seed integration fixture. The validator checks recorded
metadata; the artifact script separately verifies actual checkpoint byte hashes.

### [P1] Mature GRU parameter protection does not isolate old features — demonstrated

Evidence:

- File/function/lines: `models/nice_model.py::NICEModel.__init__` (GRU 144),
  `_forward_backbone`, `reset_frozen_gradients`; `strategies/incremental/nice.py::drop_young_to_learner` (118).
- Artifact/log: `invariant_evidence.json::gru_isolation`.
- Observed behavior: unchanged mature gate rows, mature feature delta .5924468;
  isolated recurrent/upper-layer control delta 0.

Expected behavior: fixed mature features require excluding plastic hidden influence.

Actual behavior: output masks and frozen gate rows leave dense recurrent columns
and second-layer hidden inputs. Plastic units influence mature computations over time.

Why this matters: fixed classifier rows can receive changing old features; checking
only mature parameter equality misses a potential forgetting mechanism.

Estimated impact: direct mechanism proven on the actual two-layer architecture;
its 100-client accuracy contribution remains unmeasured.

Minimal reproduction: change only the latter 50 hidden units' gate rows by +.2;
control first zeros plastic→mature recurrent/upper-layer links, then applies the same change.

Recommended fix: measure old-example activations before/after local updates,
aggregation and aging. A production isolated-GRU design needs explicit recurrent
mask semantics and serialization; no silent architecture change was made.

Validation test: `test_mature_gru_features_need_structural_isolation_not_only_frozen_rows`
documents current leakage and the control; it does not claim production isolation is fixed.

### [P1] Cross-episode score competition and routing are separate bottlenecks

Evidence:

- File/function/lines: `nice_model.py::LetLearner` (53), `NICEClient.train` (93),
  `denice_eval.py::_denice_routed_logits_with_episodes` (129), P6 `POLICIES` (25).
- Artifact/log: nine paired policies and prediction traces in all five FULL_D2 ZIPs.
- Observed behavior: E2→E3 +29.51 accuracy points, E4→E3 +10.03; E3 only 38.08%.

Expected behavior: seen-class scores must discriminate episodes, or a reliable
router must supply the missing episode decision.

Actual behavior: LetLearner zeros nonlearner logits and their gradients before CE.
New classes are not optimized against actual mature-class logits. At inference
unmasked real logits compete across episodes; a true-episode mask helps greatly.
This loss choice is intentional NICE behavior, not evidence of an accidental detach.

Why this matters: adding top-k episode classes can worsen competing-logit errors;
even a perfect route cannot repair weak within-episode recognition.

Estimated impact: paired inference effects measured; this training mechanism is
consistent with them but not uniquely proven to explain the entire weak representation.

Minimal reproduction: same-checkpoint/hash E2/E3 and E3/E4; no new training required.

Recommended fix: matched local-only/peer/feature-isolation diagnostics before changing
loss/calibration. Do not deploy the oracle as an accuracy fix.

Validation test: existing oracle-mask/no-adapter tests and artifact trace recalculation.

### [P1] Moving backbone can invalidate router sketches; final refresh is not enough evidence of quality

Evidence:

- File/function/lines: `nice_server.py::ContextDetector.push_activations` (182),
  `refresh_activation_memory` (225), `_train_multiclass_router` (351);
  `NICEModel.get_context_activations_per_sample` (418).
- Artifact/log: `invariant_evidence.json::stale_router`, FULL_D2 config/confusion.
- Observed behavior: heldout route 100→93.75→100%; old sketch bits change 10.146%.
  Real final route is 52.51% with task-end refresh enabled.

Expected behavior: stored and current features must share an encoding space.

Actual behavior: CNN/GRU feature space moves; initial scalar binarization thresholds
remain fixed. Stored train inputs permit refresh, but task-end scheduling leaves
intermediate rounds potentially stale and tiny references may remain inadequate.

Why this matters: wrong hard routes normally exclude the true class entirely.

Estimated impact: synthetic staleness causal effect proven; final real route failure
cannot be assigned solely to staleness because refresh is already present.

Minimal reproduction: 80 train references plus separate 80-example holdout; change
conv1 by +.5; score before/after train-reference refresh. Never fit on holdout.

Recommended fix: transition-level routing/feature diagnostics and reference/calibration
ablation; no unsupported threshold/memory tuning performed.

Validation test: existing reference-refresh/router-state tests and `stale_router_probe`.

### [P1] Raw reference transfer during bootstrap violates a strict no-raw-sharing protocol

Evidence:

- File/function/lines: runner's new-client branch near 2070 snapshots source detector
  then restores into new detector; `snapshot_context_detector` includes `reference_input_memory`.
- Artifact/log: inspected first seed-42 client stores 292 raw reference inputs.
- Observed behavior: bootstrap transfers those inputs, not only coefficients/sketches.

Expected behavior: if original no-raw-sharing protocol is required, references stay local.

Actual behavior: source raw train input bank is copied across simulated clients.

Why this matters: no test leakage, but communication/privacy assumptions differ;
classifier replay-free does not mean raw-memory-free.

Estimated impact: protocol mismatch confirmed; accuracy direction unknown.

Minimal reproduction: trace the `reference_input_memory` snapshot field into bootstrap restore.

Recommended fix: explicitly define authorized bootstrap knowledge and reconstruct
references only from permitted local data, or disclose a different protocol.
Blindly dropping references breaks refresh/coverage; no silent change applied.

Validation test: source-provenance/no-cross-client-raw-memory tests required once
the intended contract is implemented; current serialization tests do not establish privacy.

### [P2] Adapter support and secondary contracts limit interpretation

Evidence:

- File/function/lines: `DeNICEModel.add_adapter/set_active_adapter` (131/169),
  `CapacityController.plan_task`, `denice_aggregation.py::aggregate_adapters` (294),
  `denice_eval.py::_mask_logits_to_classes` (84), runner task/round freeze refresh.
- Artifact/log: E3 activates adapters on only 216/314/353/210/235 of 3,400 samples
  for seeds 42–46 (6.18–10.38%); E3−E3b gain .72 accuracy points.
- Observed behavior: CANC adapters are optional; old inactive adapters still mix
  by matching key. Custom-rank creation is allowed, but inference recomputes default
  rank. Finite −100 mask can lose to allowed logits below −100. CANC low-layer
  freeze override is overwritten by normal per-round rank-derived mask refresh.

Expected behavior: interpret gains on actual activation support; specify old-adapter
protection, custom-rank lookup, unconditional class masks and task-level freeze scope.

Actual behavior: most samples have backbone fallback, not necessarily failed
restoration. Shape/key compatibility does not prove aligned personalized features.
Several API/control-flow guarantees are weaker than their names suggest.

Why this matters: small global adapter gain does not prove broken gradients or
ineffectiveness on active samples. Old adapters can drift without local gradients.

Estimated impact: support/contribution quantified; other real-data effects unmeasured.

Minimal reproduction: compare E3/E3b and activation counts; nondefault add then
`set_active_context`; logits `[−101,0]` allowing only class 0; set all-low freeze
with young ranks then call `update_freeze_masks`.

Recommended fix: active-only and old-adapter-drift diagnostics; explicit mask/loss
contracts before changing finite exclusion to −infinity (which changes wrong-route
CE/nonfinite validation). Defer these from minimal confirmed P0/P1 repairs.

Validation test: active-U gradient and lifecycle tests pass; custom-rank,
pathological-logit and CANC-override persistence need dedicated follow-up tests.

## 7. Failure attribution

### Paired inference decomposition

| Comparison | Accuracy change (pp) | Subsystem isolated | Not established |
|---|---:|---|---|
| E1−E0 | +0.53 | Predicted-adapter activation without episode mask | Adapter learning independent of routing/support |
| E2−E1 | +0.06 | Oracle vs predicted adapter selection, no mask | Routing is irrelevant with hard masks |
| E3−E2 | +29.51 | True-episode mask with same oracle adapter | Deployable gain or unique training cause |
| E3−E3b | +0.72 | Adapter contribution with fixed true-episode mask | Active-sample-only adapter contribution |
| E3−E4 | +10.03 | Joint oracle adapter selection and hard class mask | Independent additive fraction of total failure |
| E5 top-2−E4 | −9.41 | Current top-k vs hard policy | Benefit from routing coverage alone |
| E6−E4 | −0.29 | Current adaptive vs hard policy | All soft routing must fail |

E3−E4 macro-F1 gap is 8.31 points; E3−E2 gap is 26.92. No arbitrary attribution
percentages are assigned. The oracle system remains weak: representation/training,
within-episode learning and potentially forgetting matter beyond routing. Missing
checkpoint topology limits extrapolation to the original live training state.

### Forgetting versus failure to learn

Historical cumulative accuracy is not `A[t,k]`. FULL_D2 traces provide the final
row, averaged across five runs:

| Evaluation task k | E4 A[5,k] accuracy % | E3 A[5,k] accuracy % | E4 global-class macro-F1 % | Route recall % |
|---|---:|---:|---:|---:|
| 0 | 17.17 | 22.97 | 14.42 | 30.23 |
| 1 | 43.47 | 48.83 | 44.89 | 67.17 |
| 2 | 32.33 | 48.20 | 31.54 | 47.30 |
| 3 | 38.73 | 48.60 | 36.48 | 67.30 |
| 4 | 20.83 | 30.07 | 15.10 | 57.37 |
| 5 | 9.60 | 25.65 | 4.91 | 42.25 |

The F1 column averages global per-class F1 over each task's classes, including
false positives from other tasks. It is not F1 silently recomputed on a task-only
subset. Task-5 oracle accuracy is near four-class balanced chance: **newest-task
learning is weak too**, so pure old-task forgetting is not the whole explanation.

Matrix availability:

| After task | k0 | k1 | k2 | k3 | k4 | k5 |
|---|---|---|---|---|---|---|
| 0 | Unknown | — | — | — | — | — |
| 1 | Unknown | Unknown | — | — | — | — |
| 2 | Unknown | Unknown | Unknown | — | — | — |
| 3 | Unknown | Unknown | Unknown | Unknown | — | — |
| 4 | Unknown | Unknown | Unknown | Unknown | Unknown | — |
| 5 | Available | Available | Available | Available | Available | Available |

Per-task forgetting `max_{u<5} A[u,k]−A[5,k]`, mean forgetting and BWT
`mean_k(A[5,k]−A[k,k])` cannot be estimated from these available scores. Do not
substitute cumulative scores or training reference-bank accuracy. All 34 final
per-class recalls/F1s and concrete wrong-prediction examples are saved in
`artifact_evidence.json::full_d2::<seed>::trace_audits`.

### Collaboration and capacity

Historical `.tmp_denice2_json/cluster_history.json` has only 8/120 raw-valid flags
but groups larger than one; it lacks effective-policy/peer-alpha telemetry.
It is invalid to infer 112 self-only rounds using today's fallback code.

FULL_D2 has **597/600 raw-valid rounds**; the three invalid rounds are seed 44,
task 4, using `previous_valid`. No effective self-only rounds were recorded.
Seed-42 per-task means:

| Task | K | Group size | Singleton clusters | Clients getting peers % | Peer alpha |
|---|---:|---:|---:|---:|---:|
| 0 | 12.60 | 5.78 | 2.95 | 94.10 | .656 |
| 1 | 18.05 | 4.89 | 5.35 | 91.08 | .635 |
| 2 | 23.05 | 5.20 | 10.30 | 85.29 | .598 |
| 3 | 27.45 | 4.57 | 11.10 | 86.13 | .598 |
| 4 | 22.10 | 7.59 | 6.95 | 92.19 | .634 |
| 5 | 21.00 | 8.51 | 6.60 | 93.27 | .666 |

This does not resemble 98 independent local learners. It does not prove beneficial
sharing either: sizable peer weights and differently specialized coordinates can
cause negative transfer. Class-aware similarity and supported-output-row protection
do not establish functional alignment. Same-task cluster purity is trivial when
all active clients follow the same scheduled task; class-distribution purity needs
local histograms/capsules. Receiver group size is not identical to AP cluster size.
Raw archives contain per-round min/max groups and per-client alphas; the derived
JSON summarizes distributions without dumping every tensor.

The synchronous invariant is verified: weights 2 and 6 with equal weights become
4 and 4 regardless of client iteration order; another aggregation preserves 4/4.
The hand-computed mature mask returns `[[10],[26]]` from target `[[10],[20]]`,
peer delta `[[4],[8]]`, ages `[2,1]`, peer weight .75. These rule out sequential
mutated-peer reads and wrong mature-row updates in the tested path, not semantic
compatibility of non-IID peers. Default consensus age promotion likewise cannot
prove that an interpolated coordinate represents the peer's mature function.

Seed-42 round-19 capacity before task-boundary aging:

| Task | GRU young % | GRU mature % | FC1 young % | FC1 mature % |
|---|---:|---:|---:|---:|
| 0 | 73.06 | 0.00 | 88.69 | 0.00 |
| 1 | 57.05 | 26.75 | 81.60 | 9.08 |
| 2 | 37.73 | 46.53 | 72.32 | 19.29 |
| 3 | 22.84 | 65.49 | 61.94 | 29.78 |
| 4 | 11.79 | 85.04 | 44.17 | 49.71 |
| 5 | 0.00 | 89.43 | 0.00 | 67.17 |

Retired counts are zero; remainder is learner capacity. Full recorded backbone
layer free/learner/mature/total distributions are in the derived JSON's
`clusters::<task>::last_capacity`. Source debug excludes fc2; empty fc2 summaries
mean unknown, not zero. Mature row fraction approximates row freezing, not all
tensor-element freezing or functional invariance. No universal task-2 exhaustion
is demonstrated. Terminal young=0 is expected when all remaining units become
learners; it does not mean zero plastic capacity. FULL_D2's reserve override is
different from some other configurations.

### Data and routing

Detailed `denice_log.json` task-start histograms are available only for task 0:
50 clients, 32–95,440 samples each, mean 19,118.96; mean missing-class fraction
30%. Class 1 has 735,961 examples versus 2,155 for class 0. Strong imbalance and
heterogeneity are factual; a complete six-task data audit is unavailable. P6
source test supports range from 359 to 2,067,978 in seed 42. Balanced replacement
sampling changes this deployment prior. Task 2 may introduce harder distributions,
but does not introduce more classes than tasks 0/1; causal difficulty remains unknown.

Seed-42 routing confusion, rows=true episode, columns=predicted episode:

| True / predicted | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 176 | 34 | 62 | 40 | 145 | 143 |
| 1 | 36 | 387 | 77 | 88 | 8 | 4 |
| 2 | 45 | 31 | 283 | 111 | 78 | 52 |
| 3 | 23 | 37 | 117 | 400 | 19 | 4 |
| 4 | 51 | 0 | 52 | 17 | 338 | 142 |
| 5 | 55 | 0 | 26 | 0 | 143 | 176 |

Task 0 often routes to 4/5 and those contexts confuse each other. This is not
everything predicting the latest task. Saved traces contain targets, predictions,
client IDs and subset indices, not per-sample router probabilities. Confidence
distributions, low-confidence fractions and probability-level wrong-route examples
require new diagnostic evaluation; none is invented here.

## 8. Ablation results

Invariants/interventions were specified before running these small probes.

| Experiment | Expected invariant/intervention | Before / observation | After/control |
|---|---|---|---|
| Four-policy roundtrip | Identical network/router | Four failing cases; max logit error 16.596607 | Exact tested logits/routes |
| Standalone loader | Preserve topology | Missing masks | Preserved masks |
| Late-client bootstrap | Clone source topology | Dense clone reproduction fails | Exact masks/logits |
| Local Adam | Mature forbidden entries and BN protected; learners/adapters train | Mature entries unchanged, U gradient nonzero | Pass |
| Hand aggregation | Normalization/self floor/mature protection | Expected `[[10],[26]]`, self floor .25 | Pass |
| Two-client reverse sweep | Synchronous snapshots/identical fixed point | 2,6→4,4 either order; 4,4 unchanged | Pass |
| GRU intervention | Change plastic rows only | Mature feature delta .592447 | Isolated control 0 |
| Router refresh | Train-only reference intervention | Holdout 100→93.75% after backbone change | 100% after refresh |
| One client, two tasks | Age once per task, not per round | `[2,2,0,0]` then `[3,3,2,2]` | Pass |

One-client fixture: four synthetic classes, 16 train and 16 disjoint holdout
examples/class, two rounds/task, fixed adapters (not CANC), diagnostic 25% task-0
reserve. Oracle accuracy matrix `[[100%, —], [100%, 81.25%]]`; no task-0 forgetting
on this easy fixture. It does not reproduce the full collapse and is not a new
DeNICE benchmark. No real-data five-/ten-client or full training was performed.

`denice_context_2026-08-17.md` documents previous D6 peer/self-only and D2
supported-class protection diagnostics; these are historical notes, **not freshly
rerun paired raw-data evidence here**. Same-protocol local-only NICE, DFCA and
no-aggregation training ablations remain unavailable. The P6 table consists of
inference ablations, not substitutes for those training ablations.

Validation results:

- Before edits: existing relevant suite **145 passed**.
- After edits: DeNICE/NICE/resume plus forensic suite **158 passed**.
- Whole repository: **350 passed, 1 skipped, 1 failed**.
- The failure, `tests/test_plexus.py::test_decentralized_plexus_il_writes_fed_il_output_contract`,
  expects a metric key set without `precision_weighted`/`recall_weighted`.
  It also fails identically on a separate clean archive of baseline **4406d78**.
  Plexus was not modified; this is a pre-existing failure.
- New forensic tests: 13 cases including four parameterized inference policies.

The measured improvement is restoration of state/prediction invariants, not a
claimed real 100-client accuracy improvement.

## 9. Recommended fixes

| Priority/action | Confidence / impact | Risk | Compute | Status |
|---|---|---|---|---|
| Complete checkpoint/bootstrap state | Confirmed fidelity violation | Low, additive fields/shared restore | Unit tests | Implemented |
| Separate seeds, validate current P6 matrix | Confirmed reporting bug | Low; legacy metadata fails closed | Unit/JSON tests | Implemented |
| Live/saved identity on real small run | Prerequisite for valid comparisons | Low diagnostic risk | Small multi-task run | Next |
| Define raw-reference bootstrap contract | Confirmed spec mismatch | Medium, affects refresh/coverage | Small ablation | Pending protocol choice |
| Instrument old-feature/GRU drift | Mechanism confirmed, magnitude unknown | Low instrumentation; high architecture risk | 1/2/5-client probes | Next |
| Matched peer/self-only and old-adapter ablation | Negative transfer plausible | Low diagnostic risk | 5–10 clients, 2–3 tasks | Pending dataset |
| Fixed heldout panel + A[t,k] | Required for forgetting attribution | Low | Evaluation/small training | Pending dataset |
| Router reference/calibration adequacy | Final route quality weak despite refresh | Medium | Paired diagnostic | Not tuned |

Changes are limited to checkpoint state, DeNICE bootstrap, evaluation/P6 reporting,
summary labels/generator, regression tests and audit tools/evidence/report. No
FedAvg substitution, oracle training, classifier replay, GRU architecture change,
dataset modification or 500-client Kaggle continuation configuration edit occurred.

**Do not immediately rerun 100 clients × six tasks × five training seeds.** First
run a small fresh two-/three-task experiment with complete saved masks and exact
live/reload checks. Resolve method/privacy specification, measure old-feature and
route changes at each operation, then freeze the method and perform a fresh
independent five-training-seed experiment. Reevaluation cannot recreate masks
omitted from old checkpoints; resuming only task 5 cannot repair an earlier
altered training trajectory.

## 10. Reproduction commands

PowerShell from `E:/School/UIT/FL_IL_IDS`. The separate audit environment uses
Python 3.14, Torch 2.14.0+cpu, sklearn 1.9.0, not the original Kaggle environment.

```powershell
git status --short
python -m venv .venv-audit
.venv-audit\Scripts\python.exe -m pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm pytest
python tools/audit_denice_artifacts.py --downloads 'C:\Users\khoak\Downloads'
.venv-audit\Scripts\python.exe tools/audit_denice_invariants.py --trusted-seed42-archive 'C:\Users\khoak\Downloads\42.zip'
.venv-audit\Scripts\python.exe -m pytest tests/test_denice_forensic.py tests/test_denice.py tests/test_nice.py tests/test_resume_state.py -q
.venv-audit\Scripts\python.exe -m pytest tests -q
git diff --check
```

The artifact script reads source archives without extraction/modification and
hashes actual bytes without unpickling. The invariant script's optional trusted
archive flag **unpickles a trusted local checkpoint** for schema inspection;
never use it on untrusted files. Without it, only synthetic probes run and the
generated invariant JSON omits historical-checkpoint metadata.

Evaluation-only example when the dataset and compatible environment are available;
use a fresh complete-state checkpoint for fidelity, and fixed evaluation RNG across
training runs. This command does not train:

```bash
python run_denice_p6_eval.py \
  --checkpoint /kaggle/working/audit_fresh/checkpoint_task_5_round_19.pt \
  --data-dir /kaggle/input/datasets/khoilv2005/100-clients/100-clients \
  --output-dir /kaggle/working/audit_p6_seed42 \
  --device cuda --seed 2026 --protocols coverage_aware_local \
  --samples-per-class 100 --class-balanced-with-replacement

python summarize_denice_p6.py \
  --run-dirs /kaggle/working/audit_p6_seed42 /kaggle/working/audit_p6_seed43 \
             /kaggle/working/audit_p6_seed44 /kaggle/working/audit_p6_seed45 \
             /kaggle/working/audit_p6_seed46 \
  --output /kaggle/working/audit_p6_five_training_seeds.json
```

For forgetting, preselect a fixed per-class heldout index panel. The current
cumulative sampler uses `eval_seed + task_id`, so repeated per-task calls do not
automatically hold old examples fixed. Score seen-task subsets at every checkpoint
under both E4/E3, keep prediction traces and consistent assignment semantics.
Record train/reference/test indices and score immediately before local update,
after local update, aggregation, reserve/aging and fresh-process reload. This
is the missing experiment for identifying the historical temporal failure.

## 11. Remaining uncertainties

1. Missing 100-client NPZ data prevents complete client/task histograms, content-level
   leakage checks, full A[t,k], and heldout before/after-fix metrics.
2. Legacy masks cannot be uniquely reconstructed. Seed-42's 0/98 state-field
   absence is directly inspected; all five actual hashes are verified, but the
   other four checkpoint objects were not separately unpickled for their schemas.
3. The exact operation that first reduced historical task-0 real-data accuracy
   is unknown. Save/load/clone and GRU transitions are demonstrated mechanisms,
   not proof of the precise task-2 causal chain.
4. Synchronous mathematical aggregation passes tests; peer semantic compatibility,
   old-adapter drift, reserve effects and age/weight functional alignment remain open.
5. FULL_D2, old imbalanced proxy and current 500-client settings are different
   experiments. Current code must not be retroactively assigned to every old log.
6. Final routing confusion is available; per-sample confidence and temporal feature
   traces are not. Calibration and operation-level routing losses remain unmeasured.
7. New CPU tests do not establish deterministic reproduction of historical Kaggle
   execution; pin original dependencies and dataset identity for real reevaluation.
8. Original multi-bank/no-raw-sharing and newer capsule/adapter variants need explicit
   naming and protocol specification before an expensive comparative experiment.

Evidence ledger: E01 old eval-only seeds; E02 five actual FULL_D2 hashes/configs;
E03 P6 provenance fix; E04 checkpoint/bootstrap reproduction and fix; E05 synchronous
aggregation proof; E06 GRU leakage/control; E07 missing dataset/matrix; E08 train-only
reference refresh probe; E09 trace-recomputed global/per-class metrics; E10 raw-reference
bootstrap drift. Machine-readable evidence: `audit_denice/artifact_evidence.json`
and `audit_denice/invariant_evidence.json`. Findings distinguish measured facts,
implemented repairs and hypotheses requiring further data/experiments.

### Dataset addendum — archive supplied after the initial audit

The previous missing-dataset statements describe the initial audit, not the
current availability. `C:/Users/khoak/Downloads/archive.zip` now provides:

- All client IDs 0–99, `metadata.json`, `global_test_data.npz`, scaler pickle and
  a distribution image. The scaler pickle was not loaded.
- 31,513,463 training examples, agreeing with metadata; 13,505,771 test examples.
- Feature shape `(39,1)`, 34 global class IDs, six tasks.
- Archive SHA-256: `33e102797169a0d49b2a7e347d179ae1a832aa5703a79df5cf8900103e62319e`.
- All 34 test class supports exactly match the saved seed-42 FULL_D2 evaluation;
  task-0 training class totals exactly match the historical log. This is strong
  dataset-consistency evidence, not a historical byte-for-byte identity proof
  because the old experiment did not record the dataset archive hash.

`tools/audit_denice_dataset.py` read all 100 complete training label arrays and
the test labels, checked feature headers/row alignment, label ranges and feature
shapes without materializing full feature arrays or extracting the archive.

| Task | Nonempty clients | Train samples | Min/max samples per active client | Assigned but empty |
|---|---:|---:|---|---|
| 0 | 50 | 955,948 | 32 / 95,440 | None |
| 1 | 60 | 13,011,358 | 5,910 / 1,033,488 | None |
| 2 | 70 | 9,682,499 | 204 / 989,211 | None |
| 3 | 80 | 6,590,913 | 1,447 / 361,808 | None |
| 4 | 89 | 1,015,491 | 191 / 65,141 | 47 |
| 5 | 98 | 257,254 | 1 / 22,117 | 19, 81 |

Mean missing-class fraction among active clients is 30% for tasks 0–3, 30.15%
for task 4 and 32.40% for task 5. No client has all six classes in **both** tasks
0 and 1. For example, client 34 supports five task-0 classes and all six task-1
classes; client 64 supports six then five. A one-client diagnostic must report
locally supported-class scores separately from the complete task/global scores;
otherwise absent local training classes can be mislabeled as forgetting.

The dataset preparation seed is null in saved metadata. Preserve this supplied
split and its hash rather than regenerate it to reconstruct historical runs.
Full feature-content integrity/nonfinite checks, train/test duplicate overlap,
real-data training probes and the complete forgetting matrix are not yet run.
The supplied data removes the availability blocker, not these measurement gaps.

Reproduce the new profile:

```powershell
.venv-audit\Scripts\python.exe tools/audit_denice_dataset.py --archive 'C:\Users\khoak\Downloads\archive.zip'
```

Detailed per-client/per-task histograms and compatibility checks are stored in
`audit_denice/dataset_evidence.json`. No source archive, production configuration
or historical checkpoint was modified by this inspection.

### Real-data transition probe — one client, two tasks

Completed using `tools/diagnose_denice_real_client.py` around the **production
DeNICE runner**, with its actual CANC, local trainer, reserve and aging operations.
This addendum supersedes the initial report's statement that no real-data probe
has been run; the full 100-client experiment/matrix remains outstanding.

Protocol: client 34; training seed 42; tasks 0 and 1 of the original six-task
schedule (task 1 is not treated as terminal); three communication rounds/task;
one local epoch/round; batch 32; 10% free-capacity reserve; self-only aggregation.
Initialization/config derive from the archived seed-42 FULL_D2 config, with
explicit diagnostic overrides in `transitions.json`. Training uses up to 64
random samples per available class, without replacement, selection seed 2026.
Test uses 32 examples per class for classes 0–11, selection seed 2027, only from
the global test file. Selected indices are retained.

Actual train support: five task-0 classes (246 examples) and six task-1 classes
(384 examples), 630 total. Test has 384 examples; task-0 locally supported score
uses 160 examples, full task-0 score uses 192. Sampling is deliberately small and
stratified, not the original data prior. This is not a 100-client benchmark.

Main results, accuracy on **locally supported classes**:

| Model boundary | Old task 0, backbone/no mask | Old task 0, oracle hard | Old task 0, predicted hard | New task 1, predicted hard |
|---|---:|---:|---:|---:|
| After task 0 | 63.125% | 63.125% | 63.125% | — |
| Task 1, after local round 0 | 59.375% | 63.125% | 0% (router not yet refit) | Interim |
| Task 1, after local round 1 | 7.500% | 63.125% | 0% (router not yet refit) | Interim |
| Task 1, after local round 2 | 0% | 62.500% | 0% (before task-end refresh) | Interim |
| After task 1 boundary/refresh | 0% | 62.500% | 60.000% | 94.271% |

After task 1, oracle new-task accuracy is 97.917%. Final old-task route recall on
local support is 95%; new-task route recall is 95.833%. Scores on **all** task-0
classes are lower because one class had no local training examples: initial
52.604%, final oracle 52.083%, final predicted-hard 50.000%.

Small-panel forgetting/BWT on the two-task supported-class accuracy matrix:

- Oracle: `[[63.125%, —], [62.500%, 97.917%]]`; old-task forgetting .625 pp,
  backward transfer −.625 pp.
- Predicted hard: `[[63.125%, —], [60.000%, 94.271%]]`; old-task forgetting
  3.125 pp, backward transfer −3.125 pp.
- These quantify this client/panel/two-task probe only, not historical six-task forgetting.

**Operation attribution supported by the probe:**

1. Large unmasked old-task loss occurs during local training on task 1, especially
   its second/third rounds. With true-episode masking the old discrimination mostly
   survives. Fresh checkpoint inference confirms **all 192 old-task test examples
   are predicted into task-1 classes** without episode masking. Median new-class
   maximum logit minus old-class maximum is **+1.668585** on those old examples.
   This directly demonstrates cross-episode score competition in a real-data,
   no-peer run. It does not uniquely establish the entire 100-client root cause.
2. Predicted-hard old accuracy temporarily becomes 0 immediately after task-1
   preparation, *before* its first optimizer update. `ContextDetector._predict_multiclass_scores`
   falls back to the latest declared episode when `multiclass_router is None`;
   a one-episode detector has no fitted multiclass classifier yet. Task-end
   scheduling defers the new fit; after refresh the old score recovers to 60%.
   Intermediate routing failure must not be labeled irreversible forgetting or
   confused with a final checkpoint score. No router fallback fix was applied here.
3. Self-only aggregation, reserve and rank-aging events do not change the measured
   classification scores in this probe. This does not test harmful peer transfer.
4. At each tested aging boundary and final boundary, weight/algorithm serialization
   into a fresh model has **zero maximum logit discrepancy** for E0/E3/E4.
   A separate process loading the actual saved task checkpoint through the standalone
   evaluator reproduces the live final accuracies.

The observer evaluates deep copies and restores RNG. A second run without
intermediate observations uses identical sample indices and yields **bitwise
identical final model tensors**, with zero E0/E3/E4 logit difference. This rules
out observer-induced optimization changes for this run. No historical checkpoint
or production training configuration was changed by this diagnostic.

Evidence:

- `audit_denice/real_client34_probe2/transitions.json`: stage-by-stage accuracy,
  explicitly task-restricted macro-F1, routing, ranks/capacity and GRU feature deltas.
- `audit_denice/real_client34_probe2/verification.json`: observer neutrality,
  fresh-process checks and new-vs-old score dominance.
- `audit_denice/real_client34_control/`: independent no-observation control and
  saved small sampled panel. Actual task/round/continuation files exist under each
  run's `training/` directory.
- `real_client34_probe1` is an incomplete attempt that stopped on Windows console
  encoding **before training**; it is not used as experimental evidence.

Reproduction (choose new output directories; the harness refuses overwrite):

```powershell
.venv-audit\Scripts\python.exe -X utf8 tools/diagnose_denice_real_client.py --archive 'C:\Users\khoak\Downloads\archive.zip' --output-dir audit_denice/reproduce_observed
.venv-audit\Scripts\python.exe -X utf8 tools/diagnose_denice_real_client.py --archive 'C:\Users\khoak\Downloads\archive.zip' --output-dir audit_denice/reproduce_control --no-observation
.venv-audit\Scripts\python.exe -X utf8 tools/verify_denice_real_probe.py --observed audit_denice/reproduce_observed --control audit_denice/reproduce_control
.venv-audit\Scripts\python.exe -m pytest tests/test_denice_real_probe.py tests/test_denice_forensic.py -q
```

Next justified experiment: matched peer versus self-only on a small multi-client
panel through task 2, with identical sampled train/test data and seed, preserving
the operation-level observations and supported/all-class score distinction. The
one-client two-task test did **not** reproduce the severe final routed-system
collapse, so it cannot clear peer aggregation, task-2 difficulty or later capacity
effects. No parameter tuning or full five-seed rerun is warranted from it alone.

### Three-step multi-client diagnostic cycle — completed 2026-09-06

This completes the requested bounded cycle: (1) matched peer/self-only runs,
(2) component and operation isolation, and (3) repeat/resume/checkpoint/regression
verification. It does **not** establish a complete cure for the historical
100-client experiment. No new production algorithm default was changed.

#### Step 1 — five clients through task 2, two paired training seeds

Clients `[15,26,34,78,98]` have data in all three tasks; the cohort was selected
using training-support metadata, not test performance. All runs use the exact
same cached train/test arrays and source indices, 64 train examples/class/client
at most and 16 test examples/class. Some rare local classes have fewer train
examples. Totals: 1,413 task-0 train, 1,536 task-1 train, 1,633 task-2 train;
288 global test examples across 18 classes. This stratified diagnostic is not
the full local data distribution or a random population sample.

Each of seeds 42 and 43 is trained once self-only and once with peer aggregation,
three rounds/task, one local epoch/round, 10% reserve. Original six-task semantics
remain in force; task 2 is not treated as terminal. The full production runner,
including AP, CANC, selective fc2 support and consensus age merging, is used.
The collaboration guard is disabled only because self-only is an intentional
control, with actual peer participation still recorded explicitly.

Final predicted-hard accuracy on locally supported classes, weighted by evaluated
sample count across the same five clients:

| Seed / mode | Task 0 | Task 1 | Task 2 |
|---|---:|---:|---:|
| 42 self-only | 48.66% | 88.02% | 51.68% |
| 42 peer | 52.68% | 86.72% | 58.41% |
| 43 self-only | 51.12% | 89.06% | 56.97% |
| 43 peer | 50.22% | 86.72% | 52.64% |

Peer-minus-self differences in percentage points: seed 42 `[+4.02, −1.30, +6.73]`;
seed 43 `[−0.89, −2.34, −4.33]`. **The sign is not consistent across seeds/tasks.**
Turning off collaboration globally is not justified by this experiment.

Final oracle-hard supported-class accuracies:

| Seed / mode | Task 0 | Task 1 | Task 2 |
|---|---:|---:|---:|
| 42 self-only | 55.58% | 95.57% | 64.42% |
| 42 peer | 59.60% | 95.05% | 68.75% |
| 43 self-only | 56.25% | 96.61% | 73.08% |
| 43 peer | 55.58% | 92.71% | 66.11% |

Predicted-hard matrices for the two peer runs (rows=after task, columns=test task):

- Seed 42: `[[58.48%, —, —], [55.58%, 93.23%, —], [52.68%, 86.72%, 58.41%]]`.
- Seed 43: `[[56.03%, —, —], [54.24%, 90.36%, —], [50.22%, 86.72%, 52.64%]]`.

Both all-class and supported-class matrices for all four inference policies and
four runs are in `audit_denice/group_probe_v1/analysis.json`. Per-client/task
macro-F1 in raw transitions explicitly fixes the task label set; it is not a
pooled global macro-F1 or an unweighted mean over clients.

Collaboration is real: seed 42 has 35/45 client-round events receiving peers,
mean peer alpha .46295; seed 43 has 38/45, mean alpha .47497. All nine rounds/run
have raw-valid assignments. Self-only controls have zero peer events/alpha.
No self-only-degeneration explanation applies to the peer runs.

#### Step 2 — locate operations and isolate components

At each local update, aggregation and task-aging boundary, evaluation uses deep
copies and restores RNG. Observed immediate aggregation effects on the current
task's oracle accuracy average **−3.19 pp** (seed 42) and **−2.91 pp** (seed 43)
over client-round events. This demonstrates short-term negative transfer in
these runs, but the complete training trajectories give mixed final effects.
Immediate task-0 oracle effects during later-task aggregation average only
approximately +.03/+ .11 pp. These event means are not final accuracy gains/losses.

Aggregation never directly changes entries protected by the receiver's pre-merge
mature-row mask: maximum measured storage change is **0** in both seeds.
Nevertheless mature GRU activations can change: maxima .49928/.23856 during peer
aggregation, and .53585/.38297 during local steps. Self-only also has mature GRU
feature drift during local training (.61493/.31318). Thus parameter checks alone
remain insufficient, but feature drift is not automatically harmful accuracy drift.

A repeated seed-42 peer run evaluates three counterfactuals on **copies** after
each task-1/task-2 aggregation (30 client-round events). Training itself is unchanged:

| Copy-only intervention | Mean old-task-0 predicted-hard change | Mean current-task predicted-hard change | Oracle classification effect |
|---|---:|---:|---|
| Refresh router from stored training references, same weights | +25.78 pp | +22.56 pp | 0 pp |
| Restore only pre-aggregation GRU weights | −0.03 pp | +0.08 pp | Approximately zero mean effect |
| Restore only pre-aggregation old-context adapters | 0 pp | 0 pp | 0 pp on this panel |

These are **intermediate-event** improvements, not a +25-point gain on final
checkpoints. The task-end schedule already refreshes before final checkpointing.
The test/reference separation is preserved; no heldout examples are used to fit
the router. GRU drift is real, but this particular GRU-weight reversal does not
explain the large current-task aggregation loss. No observed adapter accuracy
effect does not mean adapters are absent or their logits identical: seed-42 final
registries contain adapters for clients 15 and 78 with nonzero U weights.

**Additional protection finding, causally localized:** the original transition
observer saw mature-row storage changes up to .05817 after local training in peer
mode (none in self-only). A task-1 replay from the exact saved task-0 continuation
instrumented pruning and every Adam step separately:

| Operation | Calls | Max protected storage change | Max effective masked-weight change |
|---|---:|---:|---:|
| Pruning, consensus age merge | 15 | .058173 | .028270 |
| Adam steps, consensus age merge | 144 | 0 | 0 |
| Pruning, diagnostic `age_merge_policy=none` | 15 | .058173 | 0 |
| Adam steps, diagnostic `age_merge_policy=none` | 144 | 0 | 0 |

At round 0, pruning only removes nonzero storage already hidden behind zero masks:
effective masked weights do not change. At later rounds under consensus, pruning
also changes effective connections into coordinates newly marked mature by peers.
Disabling only peer age merging from the same task-0 continuation removes this
second effect. The causal chain is **peer maturity transfer → later local pruning
of young inputs → altered effective connections on mature-labeled rows**, not
optimizer momentum overriding gradient freezes.

Both replay variants finish task 1 with identical supported-class accuracies:
oracle `[59.15%,95.57%]`, predicted hard `[55.58%,93.23%]`. The counterfactual
removes the measured protection issue but does not improve this heldout accuracy.
It changes maturity-sharing semantics and is therefore retained as a diagnostic,
**not silently promoted to the production default**. A general algorithmic repair
requires a topology/age compatibility contract, not just disabling a component.

#### Step 3 — verification and fix decision

- Re-running seed-42 peer with extra copy-only counterfactuals produces identical
  tensor hashes at **all three task boundaries**. The measurements did not change
  its optimization trajectory.
- A fresh process reloads all 75 task/client models across the four primary runs
  and the repeated diagnostic run. **1,200 task/subset/policy accuracy comparisons**
  exactly match live observations. Every sampled-panel file hash is checked.
- Task-1 consensus replay matches uninterrupted seed-42 task-1 model tensors
  exactly (maximum difference 0), validating this task-boundary continuation.
- Newly added regression tests cover source-index alignment, deterministic sampling,
  task-2 inclusion/future-class exclusion, sample-count weighting, and the
  maturity-then-pruning functional-protection counterexample.
- Final targeted regression run: **163 passed** across `test_denice_real_probe.py`,
  `test_denice_forensic.py`, `test_denice.py`, `test_nice.py`, and
  `test_resume_state.py`. This is not a full-suite pass; the previously documented
  baseline Plexus failure remains outside this diagnostic cycle.
- Earlier checkpoint/bootstrap/provenance repairs remain in place. This cycle did
  not identify evidence supporting a new low-risk default algorithm change that
  improves final accuracy. No blanket self-only switch, GRU rewrite, oracle
  deployment, or speculative loss tuning was implemented.

The requested three-step **diagnostic cycle is complete**. The severe final
100-client routed collapse is not fully reproduced by the selected five-client,
three-round stratified panel. Do not interpret these stronger small-panel scores
as repaired 100-client performance. The next research decision is a matched,
more representative participation/data-budget experiment (and explicit age/topology
contract), not an immediate expensive five-seed full rerun based on these results.

Reproduction, preserving existing outputs:

```powershell
.venv-audit\Scripts\python.exe -X utf8 tools/diagnose_denice_group.py --archive 'C:\Users\khoak\Downloads\archive.zip' --output-root audit_denice/group_reproduce
.venv-audit\Scripts\python.exe -X utf8 tools/analyze_denice_group_probe.py --root audit_denice/group_reproduce
.venv-audit\Scripts\python.exe -X utf8 tools/verify_denice_group_probe.py --root audit_denice/group_reproduce
.venv-audit\Scripts\python.exe -X utf8 tools/trace_denice_mask_cleanup.py --root audit_denice/group_reproduce --output audit_denice/mask_reproduce
.venv-audit\Scripts\python.exe -X utf8 tools/trace_denice_mask_cleanup.py --root audit_denice/group_reproduce --output audit_denice/mask_no_age_reproduce --age-merge-policy none
```

The additional counterfactual repeat uses the same cached panel:

```powershell
.venv-audit\Scripts\python.exe -X utf8 -c "from pathlib import Path; import torch; from tools.diagnose_denice_group import run; torch.set_num_threads(1); run(Path('audit_denice/group_reproduce/panel'),Path('audit_denice/group_reproduce/42_peer_counterfactual'),42,'peer',3,counterfactual=True)"
```

Rerun the analyzer/verifier after that repeat to include its copy-only effects
and tensor-identity check. Existing evidence is in `group_probe_v1/analysis.json`,
`group_probe_v1/verification.json`, per-run `transitions.json`, and the two
`group_mask_trace*/mask_trace.json` files. These tools unpickle only trusted
locally generated diagnostic checkpoints, never arbitrary downloaded pickle files.
