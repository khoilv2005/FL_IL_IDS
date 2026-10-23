# CANDLE V4 — original TRAIN protocol v1

This prospectively supersedes the four-role full-run data protocol. Fresh
seed42 only; the role-preparation Kaggle session must be stopped manually.
Old role-based checkpoints cannot resume this run. Earlier pilot results and
their BASE/FIT/CAL decisions remain historical, not evidence for this protocol.

## Training and prediction system

- Use native IncrementalDataLoader on original client_{id}_train.npz.
- Filter only the current task's classes; no historical raw replay.
- Keep the existing native local validation split (fraction .1, native seed).
  Native learner and V4 experts see the same remaining current training view.
  No external BASE/FIT/CAL/VALIDATION, role indices, role manifest, merging old
  role files, or global re-partition step.
- Native batch2048/AMP, Adam .001, local epoch1, xi=.8/paper neighbors,
  binary_cosine, six tasks x20 rounds, benchmark participation, seed42.
- One native V1 backbone per client plus one coverage-aware expert bank.
  No shadow V1 copy, parallel no-aux bank, or donor-only bank in the default run.
  Temporary expert references used by peers for gradients are small MLP copies,
  not additional CNN-GRU predictors.
- Expert 39→64→1, Adam .001, clip5, 44 updates/round, max88 catch-up;
  128 positive+128 negative presentations/update. Sample from the entire current
  native training view; remove pilot4096/class cap. Class-balanced gradients,
  same-reference packets, graph-causal primary/auxiliary selection unchanged.
- Select donors from current neighbors by current training support>=32,
  minimum donor ID. No FIT/CAL counts. Negative auxiliary selection still uses
  zero-covered classes and minimum owner ID with>=32 current training samples.

## Fixed native-validation acceptance (new evidence contract)

Threshold .5 is fixed before this run; do not derive it from TRAIN, TEST, a
target-positive quantile or repeated validation searches. Read the native
X_validation/y_validation already created for each current client; no new split.

At task end, deduplicate validation evidence within an owner and exclude content
also present in any current active client's native training view. These are
gate eligibility filters, not new training datasets or persisted role indices.
Do not put excluded samples back into training or search for a passing subset.
Native validation already participates in V1 model selection/capsules: it is
development evidence, not an independent CAL or statistical safety certificate.

Keep target recall>=95%, observed per-owner/per-negative-class FAR<=.1%, break=0,
net rescue>0 for new installs; minimum32 unique donor positives and32 negatives
outside the combined bank per required current owner. Missing evidence gives
DEFERRED/SUSPENDED, never ACTIVE. Check the combined old-pending/new-draft bank
atomically, with fixed normalized-excess arbitration. No TEST-based activation.

Close current validation/training caches at task transition; freeze old expert
weights and thresholds. Current-negative checking may restore a pending expert
using unchanged historical receipts, under the existing empirical policy.
Unobserved historical risk remains UNKNOWN; no universal old-class safety claim.

## Evaluation, comparisons and persistence

Gate/report use current native local validation only. Do not replay historical
validation to manufacture cumulative development metrics. Forgetting is not
measured under this final-TEST-only protocol; report the limitation explicitly.

Full original TEST is opened only after T5/R19, consolidation, gate and archive.
Use the same V1 disjoint receiver assignment for every reported predictor, with
all34 labels and no sample cap. Native V1 counterfactual uses the SAME trained
backbone with experts bypassed, not a second stored/trained model. The native
state/RNG isolation regression must pass. For a separate baseline run, use
--variant native-only with a fresh output directory and the same source/config.
Do not compare directly against historical V1 trained with80/2/8/10 roles.

Every round retains atomic continuation, experts/Adam/RNG/quota/ledger; task ZIPs
retain full continuation and gate journals. Original-TRAIN version/config binding
rejects old role-run resume. Keep128MiB/expert and64GiB incremental wire cap,
12GiB output guard. Full training remains Kaggle-only; no local full run.
