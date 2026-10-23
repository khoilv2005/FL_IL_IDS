# CANDLE V4 native coverage-aware full development — seed 42

Status: prospectively registered before the first full run. Supersedes the
two-ACTIVE launch prerequisite for this protocol only. Earlier smoke outcomes
remain unchanged. No further algorithm search is authorized by this run.

## Frozen method and execution

- Private V1 legacy runtime, native paper neighborhoods xi=.8, binary_cosine,
  same personalized models/local learner, native optimizer and participation.
- All benchmark clients (up to 100), fresh T0..T5, 20 native rounds/task, seed42.
  Native batch2048, one GPU and AMP, as V1. No full training locally.
- Existing content-disjoint V1 roles: split seed20261006, BASE .80, FIT .08,
  CAL .02, validation .10. Native learner keeps its original local BASE validation
  split. This is not a claim to train on 100% original TRAIN.
- Expert 39→64→1, ReLU, raw preprocessed inputs, CPU FP32, Adam .001, clip5.
- After each native aggregation, discover current-task missing ownership classes
  from current positive-alpha neighbors only. Metadata precheck: donor BASE,
  FIT and CAL counts >=32; smallest eligible donor ID. No result-based ranking.
  This is a metadata opportunity, not proof of a functional gap.
- Positive gradients: fixed primary donor. Negatives: receiver and donor plus
  one minimum-ID current eligible auxiliary owner per zero-covered negative
  class, with >=32 current BASE samples. No hardcoded class pairs.
- Deduplicate/cap expert training views at4096/class as in development; native
  training retains the V1 budget. No historical training/CAL raw replay.
- 128 positive+128 negative presentations/update; class-balanced objective;
  44 updates per available round, pending quota, catch-up at most88/round.
- Three expert arms: coverage-aware, no auxiliary negatives, donor-only.
  Donor-only uses the same class-balanced objective with donor samples only;
  it is a source ablation, not the old unbalanced standalone implementation.
  Same candidates, initialization and optimizer/sample update count. If any arm
  reaches its training resource cap, all three stop that candidate together.
  A donor-only arm needs at least one current negative class at its donor;
  otherwise all three arms defer updates for that opportunity.
- V1 is a shared, isolated native predictor, not retrained separately per arm.
  Experts never enter native modules, masks, optimizer or graph construction.

## Acceptance and retention

One gate at task end, fixed donor FIT recall95 order statistic, then current
CAL only; no tuning/retry/subset search from CAL outcomes. Required current
scope: receiver and every active positive-alpha peer at the gate. Unreachable
primary donor, no training, insufficient unique FIT/CAL or negative evidence
produces DEFERRED, never PASS.

All pending old experts and current drafts of a receiver compete in one bank.
Fixed normalized-threshold-excess arbitration, smallest class ID for ties.
Require >=32 unique donor CAL positives, recall>=95%, >=32 non-bank negatives
per owner, observed per-owner/per-negative-class FAR<=.1%, break=0, positive
net rescue for new installations. Validate the final combined predictions.
Any failure rejects new drafts atomically and suspends pending old experts.
No search for a passing subset. Quality rejection does not stop native training.

At task transition, old ACTIVE becomes PENDING_NEW_TASK, weights/thresholds
freeze, old BASE/FIT/CAL APIs close. Current-negative checking can restore ACTIVE
using unchanged historical expert receipts; it cannot establish universal
historical safety. Unobserved historical risk remains UNKNOWN. This is an
empirical development deployment policy, not a strict global safety certificate.

## Resources and persistence

Serialized application-wire cap:128MiB/expert/arm and64GiB/arm for the run;
reserve10% of the latter from gradient training for discovery/verification.
Byte accounting includes discovery, reference/gradients, verification and
receipts. Native dense-model payload upper bounds are reported separately;
their sum with expert bytes is an estimate, not measured physical network use.
No assertion that sparse payload alone is total communication.

No arbitrary two-class cap: at most34 experts/receiver/arm, at most10200 MLPs
over100 receivers/three arms. FP32 expert parameters alone <=107.1MB;
Adam moments approximately another214.2MB (excludes objects/native models).
Only one live expert branch per arm; current BASE/role caches expire by task.
Every native round saves atomic full continuation: draft/ACTIVE state, Adam,
pending quota, ledger, native state and native RNG. Task archives include the
continuation; retain run lock and threshold locks for resume.12GiB native
output guard remains a correctness/resource stop, never a quality gate.
Atomic gate journals replay committed decisions after a round resume without
reopening CAL. An interrupted lock without a committed receipt defers that
bank; it never authorizes a CAL retry. Keep gate journals with run artifacts.

## Evaluation and scientific interpretation

This first run is development. No automatic TEST access. At each task end,
report cumulative held-out VALIDATION for V1 and three arms, each receiver on
its own disjoint validation shard, pooled metrics. This permits forgetting
measurement without reusing historical data for training or CAL acceptance.
Do not present these as full-TEST or equal-mean-cluster metrics.

Export accuracy, Macro-F1 (all seen labels), per-class recall, per-task accuracy,
average forgetting, per-client accuracy, expert statuses/steps/hashes, observed
CAL per-owner/class FAR, rescue/break, collisions, bytes, expert runtime and
incremental inference latency. Record source SHA, requested config, roles hash,
seed and environment before training. TEST requires a separately locked final
evaluation protocol; it is not consulted for architecture/hyperparameter choice.

## Launch decision

GO after controlled technical regression of gradients, actual graph causality,
native isolation, optimizer/checkpoint restore, task transition and simulated
ACTIVE/rejection. No real CAL is opened by regression. Two natural ACTIVE
experts are not required before full. Runtime NaN/state mismatch/leakage or
broken checkpoints still stop the run. Rejections/zero gain must be reported.
