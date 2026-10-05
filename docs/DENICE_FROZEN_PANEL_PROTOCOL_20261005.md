# Frozen remaining-test confirmation protocol

## Collector lifecycle correction and verification

The run after `d27fd1f` passed the first donor's reference/fresh computations and
weight checks, then crashed in its progress log: `inputs` had already been deleted
at the end of the stage loop. This was an implementation error in the batch-splitting
change, unrelated to data/model quality. The progress log now uses persistent scalar
`stage_counts` recorded before cleanup, and reports reference/fresh counts separately.

Preparation verification performed locally:

- Replayed the actual collector control flow for **98 donors / 196 stages** using
  real saved V2 feature arrays. Raw model inference and fingerprint lookup were
  replaced with cache-backed providers for this operational replay.
- Passed **850,000 reference expert-sample comparisons**, wrote all **1,666** fresh
  receiver-donor cache files for a 391-row cached subset, and continued through
  the complete `evaluate()` path and all metric/CSV/JSON writes.
- All replayed feature arrays and the four policies' predictions exactly match
  their saved V2/meta counterparts. No fitting occurred.
- Injected a margin difference: the unchanged reference guard rejected it, saved
  actual/expected failure arrays and diagnostics, and stopped before fresh cache writes.
- Separately reconstructed the **real original delta checkpoint** and restored all
  **98 real models/routers** on CPU. Every model/mask fingerprint matches the frozen
  manifest; saved router coefficients/intercepts restore exactly; zero fit calls.

This verifies collector lifetime/offsets, logging, serialization, model restoration
and CPU meta decisions. Full raw-input inference on Kaggle GPU still requires the
original dataset and is not claimed to have run locally. The numerical reference
guard, sampling seeds, frozen artifacts and final policy remain unchanged.

Notebook retries now preserve earlier output directories and automatically choose
a fresh suffixed directory. Restart the kernel, then Run All in the same Kaggle
session to reuse downloaded ZIPs. A loaded old runner module triggers an explicit
kernel-restart message. The latest run is packaged under the usual output ZIP name.

## Graph CSV precision correction

The first Kaggle attempt stopped before drawing the panel with `Graph alpha changed`.
The graph was unchanged. Default pandas CSV parsing differed from checkpoint floats
on 7,458/8,320 positive edges, by at most 9.9882e-17 absolute (9.3267e-13 relative).
The old relative-only tolerance of 1e-14 falsely rejected 3,361 edges.

Read the graph a second time with `float_precision='round_trip'` **for verification**:
all 8,320 positive edge IDs/weights match the original checkpoint exactly, with
zero mismatches. `graph_alpha_validation.json` records the result and parser deltas.
The original default-parsed `source.alphas` remain unchanged for feature creation,
preserving the frozen V2/meta pipeline. No model, graph weight, seed, scaler or policy
is changed; exact graph validation is retained rather than removed or weakened.

## User-approved scope

Stop tuning after the 50.732% development result. The user chose **remaining test
rows from the original dataset, reporting missing classes**. This is not an
additional original-training seed or a new full-34-class result.

The original sampler initially took up to 1,470 rows per class. Classes 0, 28, 31
and 33 had fewer development rows and therefore exhausted their original test
pools. Other classes may also be exhausted or have all remaining content excluded.
The fresh run records exact availability; it cannot create unseen rows for them.
Fresh accuracy cannot be compared directly with full-34-class 50.732%. To establish
a new full-label-space >=50% claim, new data covering exhausted classes is needed.

## Run

Notebook: `eval_denice_frozen_panel_kaggle.ipynb`.
Attach the original 100-clients dataset, enable Internet and GPU, use a fresh session.
The historical dataset path remains preferred, with mounted-dataset auto-discovery.

Configured Drive inputs:

- Results (4): `1BEjP4iGJbPT0uFcHZ7WXX4oM_1vyvx0M`.
- Gate V2: `1gweD4iZ4_NYmEsQyGDOlInTITwcxHrnP`.
- Class meta: `1YA8ixsHaf0_oHFMvE0R3VNIBE6CzPZdM`.

Local overrides: `DENICE_RESULTS4_ZIP`, `DENICE_GATE_V2_ZIP`, `DENICE_CLASS_META_ZIP`.
Pin sklearn 1.6.1 before imports. Checkpoint dependencies are extracted outside
the output directory to avoid duplicating them into the diagnostic ZIP.

The repo includes `artifacts/denice_multiclass_routers_03b9b53.zip` (1.14 MB):
98 fitted snapshots copied **byte for byte** from the successful Multiclass
integration artifact. All snapshots were inspected: zero raw reference rows.
They contain learned routers, thresholds, masks, episode metadata and historical
binary descriptors. The manifest records snapshot hashes, full-model/mask signatures
and checkpoint checksum. Packaging never fits/regenerates a router.

## Freeze and safeguards

Preserve checkpoint task 5 round 19, 03b9b53, recorded candidate ordering seed 42,
k=16 plus self, graph eligibility, saved V2 MLP_top1, ClassLR_C0.1 and its scaler,
feature schemas, calibration priors and action/tie/fallback policy.

Required audited hashes:

- Gate: `3aebd1d68c4342d1e9d3addb38c53ec71364a69df0abb1d1d6f49611e47f9633`.
- Meta: `786477d5c66d46cf5cf7db263b567bcb0e406ad9e11a6f5ad8613b7414c7b771`.

Verify checkpoint checksum against the original protocol/router manifest, every
restored model/mask signature and every router snapshot hash. Runtime blocks
estimator/scaler/Pipeline fitting and ContextDetector fitting, activation pushes
and reference re-encoding. Any accidental fit aborts. There is no policy selection.

Write/hash protocol and lock before reading dataset labels: metrics, sampling,
exclusions, model hashes and decision policy are predeclared. Labels are used for
declared stratification and eventual metrics, never as prediction features.

All old rows per receiver are verification-only references. Recompute all
17 experts' features with the saved routers and compare against V2 caches:
prediction/task/support/mask fields exact; continuous fields rtol 1e-4 / atol 1e-6.
Any discrepancy aborts without refitting. Fresh inputs receive newly queried expert
predictions; old cache predictions never supply their decisions.

The earlier truncated 16-row references were concatenated with fresh inputs. That
did not preserve V2's original GPU batch boundaries or route-group shapes. A run
stopped on receiver 12 / donor 30 / class_margin; the old error did not record the
actual difference, so its magnitude could not be determined from that log alone.
Full old receiver streams now execute separately, in original receiver order at
batch 512, matching the original V2 notebook default. Fresh inference executes
separately after each donor's reference check. The check's tolerances are unchanged.
If it still fails, save actual/expected arrays, failing row count and maximum
absolute error in `reference_reproduction.json` and a `reference_failure_*.npz`.
Do not weaken the guard or fit a replacement router using fresh panel data.

Reference work now adds 850,000 expert-sample evaluations, in addition to 850,000
fresh evaluations. `expert_runtime.csv` distinguishes reference/fresh stages.
This adds verification time but does not alter the logical 17-expert deployed
decision or the frozen selection/scaler/feature policy.

## Panel selection

Size 50,000, panel seed 20261006, partition seed 523687, original 98-receiver
ordering and class-stratified receiver partitioning.

Exclude globally:

1. All 50,000 old `global_test_row` IDs in cumulative T5 coordinates.
2. Content hashes of **all** old panel inputs, not only reference rows.
3. All V2 calibration/fit/validation provenance input hashes, including fitting
   rows excluded by the constrained meta training filter.
4. Repeated content within the new panel, across any receivers/classes.

Use original float32 shape-aware SHA256. Randomize remaining IDs by class using
the fixed seed, accept safe unique content up to initial class quotas, redistribute
shortages round-robin. If fewer than 50,000 safe unique inputs exist, fail rather
than relaxing exclusions/changing size after seeing metrics.

Check old IDs/labels and total coordinate count against the dataset; record the
test NPZ checksum. Export `panel_manifest.csv`, `panel_sampling_audit.csv` and
`panel_scope.json`, including selected IDs/hashes/labels, receiver assignment,
counts, rejection reasons, missing classes and zero-overlap evidence.
This excludes known gate roles, not every original backbone training input; the
dataset's original train/test construction is not independently re-proven.

## Metrics

Run exactly self, same-pool majority, frozen V2 and frozen class meta, plus the
actual-routed oracle as a label-assisted diagnostic bound. Report:

- Pooled and client-mean accuracy.
- Macro-F1 over 34 classes (missing-class terms zero), and over present classes.
- Macro recall over present classes; per-task/class recall and missing-class list.
- Paired receiver-bootstrap gain vs V2/majority, 10,000 draws, seed 20261006.
- Old self/majority/V2/meta metrics restricted to present classes, explicitly unpaired
  with potentially different frequencies; no paired old/new CI.
- Expert runtime and logical 17 queries/fresh sample. Model restore, reference
  inference and IO are additional work.

This remains one checkpoint and shared retrospective inference. Models run locally;
no raw inputs are sent to remote peers. It does not establish an unchanged private
decentralized deployment or a streaming replay-free method.

Output: `/kaggle/working/denice_frozen_panel_diagnostics.zip`.
Inspect `frozen_panel_completion.json`; failure also saves incomplete archives.
Do not revise the selector using partial results. Independent training seeds and
new full-class test data are separate later experiments.

## Preparation status

The router asset was produced and source snapshots inspected locally. Code/notebook
syntax and static no-fit structure were reviewed. The fresh panel requires the
complete original global test dataset on Kaggle and has not run locally. No new
accuracy is predicted or claimed.
