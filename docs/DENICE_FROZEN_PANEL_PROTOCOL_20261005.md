# Frozen remaining-test confirmation protocol

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

First 16 old rows per receiver are verification-only references. Recompute all
17 experts' features with the saved routers and compare against V2 caches:
prediction/task/support/mask fields exact; continuous fields rtol 1e-4 / atol 1e-6.
Any discrepancy aborts without refitting. Fresh inputs receive newly queried expert
predictions; old cache predictions never supply their decisions.

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
