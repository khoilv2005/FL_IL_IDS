# Clean run results (8): stopped at class-28 reachability gate

Inputs: `C:\Users\khoak\Downloads\results (8).zip` and `fl-il-lk-na (7).log`.
This audit reads saved artifacts and source code; it does not execute inference on the original dataset, which is not available locally.

## Actual run configuration and progress

- Source commit: `879ad72399f0f6e6176bafb374d093703e0a319c`.
- Recorded similarity threshold **0.5**, seed 42, `paper` clustering, local CGoFed with peer projection off.
- Natural batch sampling and no class weights; batch size 2048, one phase epoch.
- All six tasks and all 120 rounds completed. The checkpoint index has 126 entries: six bases and 120 rounds. All six sealed task ZIPs are present.
- Final active receiver count is 98. Backbone evaluation was deliberately disabled; task metrics contain null test accuracy and `final_test_evaluated=false`.
- Multiclass fitting and expert evidence collection for calibration/fit/validation completed. Gate MLP fitting reached its configured 50 iterations.
- At log time 27177.5 seconds (about 7 h 33 min), Class Meta preflight stopped. The MLP convergence warning is not the exception responsible for this stop.
- `clean_class_meta/completion.json` has `completed=false`, `stage=fit`. There is no frozen pipeline, validation policy result or final-test prediction artifact. **No final accuracy is available.**

## Why Class Meta stopped

The explicit 34-class guard checks that each true fitting class has at least one correct prediction among the queried self + 16 normal-routed peer experts. Only class 28 fails.

| Quantity | Class 28 |
|---|---:|
| Globally unique BASE contents | 1,208 |
| Globally unique calibration contents | 30 |
| Globally unique fit contents | 121 |
| Globally unique validation contents | 151 |
| Locally supported donors in origin audit | 58 |
| Sampled calibration / fit / validation occurrences | 271 / 1,316 / 685 |
| Distinct sampled calibration / fit / validation contents | 29 / 118 / 143 |
| Receivers with class-28 fit occurrences | 98 |
| Fit occurrences with any correct self/16-peer prediction | **0 / 1,316** |

The 1,316 occurrences are shared across receivers, not 1,316 independent inputs. The fit data contains class 28 and almost all its globally unique fitting contents. Thus the failure is not simply an omitted fit label or missing data role.

Across all classes the fit role has 50,176 occurrences; 38,578 (76.8854%) have at least one correct queried expert. This is a **label-assisted candidate bound on fit data**, not Gate/Meta accuracy and not a test result. The other 33 classes have nonzero reachable fitting targets.

This does not prove that every expert is incapable of class 28, nor that the router alone is responsible. The recorded coverage is the union of the selected 17 experts' *normal-routed* predictions, not an oracle-task evaluation of all 98 models.

## Interpretation and next diagnostic

The guard worked as designed: it prevents silently fitting a 33-class estimator and reporting it as complete 34-class support. Removing the guard or inventing a correct target does not create a competent class-28 action under the fixed action restriction.

Keep the completed backbone and original role lock. Before any fresh full training, inspect class 28 using only locked META-fit/VALIDATION data:

1. Compare normal Multiclass routing with forced true task 4, using each donor's original local mask and adapters. Record whether task/class masks allow class 28, normal route confusion, class prediction confusion and rank/margin of the class-28 logit.
2. Query all recorded eligible donors as a labeled **diagnostic bound**, while separately reporting the existing self + 16 candidate set. If only the larger pool has correct normal predictions, candidate discovery is implicated; this does not authorize changing the deployable pool based on test labels.
3. Compare exact FP32 terminal checkpoints at task 4 and task 5 on the same held-out contents, separating acquisition failure from later forgetting. Refit the router from each checkpoint's own BASE memory, without updating backbone weights.
4. If oracle-task predictions also fail, audit rare-class acquisition/retention and aggregation. BASE has only 1,208 unique class-28 contents versus 437,230 for class 24; natural sampling/no weighting is a plausible contributor, not an established cause. Existing class-balanced sampling and effective-number weighting can be separate controlled training ablations after diagnosis.

No raw input transmission is needed for the offline audit. Do not use final-test labels or tune the selector on final test. Keep class 28 in the requested full 34-class final benchmark.

## Recovery

The six backbone tasks do **not** need to be repeated just to diagnose this stop. Preserve:

- `denice_clean_roles/` including role manifest, lock and indices.
- `checkpoint_task_4_all_rounds.zip` and `checkpoint_task_5_all_rounds.zip` (keep all task archives for backup).
- The original 100-client dataset at its declared mount.
- The partial `clean_class_meta/` audit artifacts and log.

Once the coverage issue has been understood and a justified fitting protocol is selected, the standalone `python -m tools.train_denice_clean_meta` entry can rerun selector fitting from the saved terminal checkpoint into a new output directory. This version does not save the fitted Gate or expert evidence before the coverage guard, so selector/evidence stages must be recomputed; the backbone remains reusable. The original archive is approximately 5.35 GiB.
