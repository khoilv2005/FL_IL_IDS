# CANDLE V1/V4 full current-task TRAIN — provisional activation v3

Registered before fresh seed42. Supersedes validation-based original-TRAIN v1
and empirical-validation v2 for the new notebooks. No mid-run source/protocol
change or resume from previous validation/role checkpoints.

## Data and native training

- Native IncrementalDataLoader, original client TRAIN, current-task classes only.
- denice_validation_fraction=0.0; all current-task rows enter native training.
- No BASE/FIT/CAL/validation role creation, additional splits or historical replay.
- Expert sampling uses that SAME complete current-task TRAIN view. Mini-batch
  sampling remains128positive+128negative/update,44updates/round,max88catch-up;
  it does not mean every expert visits every sample in an epoch.
- Keep donor TRAIN support>=32, live graph eligibility, missing-negative-class
  auxiliary selection and class-balanced same-reference gradient aggregation.
- One personalized native model/client. V1 native-only creates zero experts;
  V4 adds one coverage bank. No shadow native model or parallel control banks.
- Capsules use the native TRAIN/reference fallback when validation is empty.
  No assumption that TRAIN capsule reliability constitutes quality certification.
- Otherwise keep native architecture, masks, maturity, router, graph xi=.8,
  optimizer, batch2048/AMP, one GPU, 6tasks x20rounds, seed42.

## Training-qualified provisional activation

At the end of a target's birth task, require actual expert updates>0, matching
step/provenance records, authorized donor/graph at the rounds that supplied
gradients, matching input/version/class metadata and finite weights.
The donor need not remain a neighbor at task end if its earlier updates were
authorized. Never use future graph information or historical raw data.

Qualified expert: status ACTIVE, quality_status UNVERIFIED,
acceptance_type training_only_unverified. Fixed threshold0.5 and existing
multi-expert arbitration/native fallback. No recall/FAR/break/net-rescue gate.
No TRAIN-based surrogate for validation, no fabricated quality metrics.
Untrained or inconsistent/nonfinite current drafts are DEFERRED.
Old ACTIVE experts keep frozen weights/thresholds and remain usable across
tasks; mutated historical functions/receipts are correctness errors.

Receipts bind model hash, input/version, declared donor, authorized training
rounds and update count. They are training provenance, not safety certificates.
An activated expert may cause false positives or negative transfer; those are
outcomes to measure after training, not facts certified by this policy.

## Evaluation and persistence

Per-task validation: NOT_MEASURED, rows0, metric values null. No historical
validation replay; forgetting NOT_MEASURED. No intermediate TEST.
Full original34-class TEST after T5/R19, consolidation and checkpoint/archive.
Keep the V1 disjoint receiver partition/seed and pooled metrics. Report native
counterfactual and provisional V4 on identical rows, including rescue/break,
class recall/F1, confusion and inference overhead. Never use TEST to activate,
select thresholds or retroactively label receipts VERIFIED.

Provide separate V1 native-only and V4 notebooks with the SAME shared runtime
and full-TRAIN configuration. Distinct output directories, fresh runs. Historical
80/2/8/10 V1 numbers are not matched baselines. Round continuation, optimizer/RNG,
quota, ledger and task archives remain. Previous protocol checkpoints rejected.
Full training is Kaggle-only; local regression uses synthetic data, no real TEST.
