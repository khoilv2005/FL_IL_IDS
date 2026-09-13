# CANDLE PDF logic audit — 2026-09-13

Source: `Neurocomputing_Anh_Khýi.pdf`, printed pages 7–8, equations (12)–(26).
This audit supersedes the earlier recommendation to use `class_blocks` as the
main allocation policy. That policy remains available for old runs and ablations.

## Changes applied

| Paper reference | Logic repair and verification |
| --- | --- |
| Eq. (12) | Allocate a fixed integer budget per locally new class, capped by available reserve. Global label IDs do not change the budget. A locally reobserved class neither reopens mature output rows nor allocates new capacity. |
| Eqs. (15)–(16) | Build penultimate class prototypes from the local validation split when available. Capsule `reliability` holds reserve fraction in paper mode. Removed an unnecessary validation forward that could mutate BN statistics in training mode. |
| Fisher / Step 7 | Sample classes round-robin for the empirical Fisher estimate instead of taking the first samples from a class-grouped bank. Recompute final prototypes/Fisher after the final aggregation and retain Fisher for mature coordinates from preceding tasks. |
| Eqs. (18)–(19) | Keep the direct similarity-threshold neighborhood in the weight denominator. Block disjoint-label sender updates across the entire backbone through the pair mask. Preserve local BN buffers, including integer counters, when applying local deltas. |
| Eq. (20) | Measure Euclidean distance between the means of prototypes for the same shared classes in consecutive tasks. Report `defined=false` if no shared class exists. |
| Eqs. (21)–(22) | Add an explicit paper CANC mode with Reuse/Expand/Recycle branches, global capacity utilization, domain drift and preceding-task consumption. Plan before consolidation, apply at the following task boundary. |
| Step 6 recycling | Select low-Fisher mature units, protect the routing anchor, bound tie cases by the requested percentile, reset selected parameters and clear their stale Fisher. Exclude classifier output rows. |
| Continuation | Persist per-class budgets, prototypes, accumulated Fisher, consumption and the pending CANC plan. Reject resume across incompatible allocation/controller modes. A three-task synthetic run and split/resumed run produce exactly equal final model tensors. |

## Explicit implementation choices and remaining differences

This is closer algorithm alignment, not a claim of an exact reproduction of every
equation or of the paper's reported results.

- The code tracks capacity at neuron granularity; the paper describes a parameter
  mask. Capsule reserve fraction counts all age-tracked units, while CANC uses
  hidden-layer capacity, excluding the fixed classifier label space.
- The PDF does not specify numeric values for the controller thresholds or
  activation/recycling budgets. Current settings are `theta1=0.8`, `theta2=0.35`,
  extra expansion of one unit per layer, and a recycling percentile of 2.
  The default per-class budget is `ceil(layer_width / total_classes)`.
  These are implementation settings, not measured optimal values or paper constants.
- Disjoint class-incremental tasks do not define Eq. (20)'s shared-class drift.
  The controller uses a neutral zero fallback with an explicit undefined flag;
  it does not interpret that fallback as evidence of no domain shift.
- Comparable old validation loss is unavailable under the sketch-only memory
  policy. Its pressure term is marked undefined and zero; the main configuration
  uses `denice_gamma=0`.
- Strict freezing keeps the mature-coordinate displacement in Eq. (14) zero
  during ordinary local training; no nonzero soft penalty is claimed. Recycling
  is a separate, explicit task-boundary operation.
- Existing micro-adapters remain `U sigmoid(V h)` residuals on layer outputs.
  They are not the linear adapter on layer input written in Eq. (23). This
  architectural difference remains; the disjoint-class smoke tests do not
  establish correctness or improved metrics for the domain-shift adapter path.
- Step 6/Step 7 ordering and a carried Reuse mask are not fully specified together
  in the PDF. This implementation saves the task-end decision before consolidation
  and applies it before the next task's new-class activation. With no new local
  class it carries the mask without allocation, expansion or recycling.
- Recycling excludes the first-task routing anchor so stored sketches remain
  usable without old raw traffic. This is an additional implementation constraint.

## Running the repaired version on Kaggle

`train_incremental_kaggle.py` selects `fixed_per_class`, paper capsules, paper
threshold clustering, local-delta aggregation and paper CANC. Upload the updated
`fed_learning` directory together with the script, or set `DENICE_CODE_DIR` to the
directory containing that updated package. Running only a copied script can fall
back to cloning GitHub `main`, which does not include unpushed local changes.

Start fresh from task 0 with this controller. The default phase 1 trains tasks
0–1; set `DENICE_TRAIN_PHASE=5` to train all six tasks in one run. Subsequent
phases must resume a continuation checkpoint produced by this same configuration.
The legacy `denice_enable_recycling` flag does not disable the paper controller's
Recycle branch.

## Validation and metric limits

Regression coverage includes fixed allocation, reobserved classes, disjoint-label
aggregation and its denominator, BN preservation, Fisher sampling and retention,
all three controller branches, router-anchor recycling, and exact continuation
equivalence. Tests run entirely inside the workspace sandbox using temporary
directories with inherited Windows permissions; no elevated test execution is
required.

Final result: **194 passed, 9 warnings** in 88.53 seconds across
`test_candle_pdf_logic`, `test_candle_repairs`, `test_denice_metric_repairs`,
`test_denice_forensic`, `test_denice`, `test_nice`, and `test_resume_state`.
All changed Python modules pass `py_compile`; `git diff --check` reports no
whitespace errors.

No new full CICIoT2023 training result has been measured in this audit. Accuracy,
macro-F1, per-class recall and forgetting must be compared on the same real split,
seed and evaluation protocol before claiming an improvement. Test success alone
does not establish an increase in those metrics.
