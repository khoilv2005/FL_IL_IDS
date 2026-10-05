# Frozen remaining-test panel: completed results (2026-10-05)

## Decision

The frozen class meta-ensemble achieved **58.358% accuracy** on a new, nonoverlapping 50,000-sample panel containing **28 classes**. Its improvement over Gate V2 persists on new samples without fitting or selecting a new policy. Stop selector tuning; move to replication with independent training seeds and a test source with full class coverage for a full 34-class claim.

This is **self + 16 frozen peer experts + a frozen class meta-ensemble**, not standalone DeNICE accuracy. The experiment remains a shared, retrospective evaluation from one training seed.

## Source and frozen policy

- Input: `C:\Users\khoak\Downloads\denice_frozen_panel_diagnostics (1).zip`.
- Evaluation commit: `5b6b29978f248ec2ab2762389461c01bd29a4fb9`.
- Training checkpoint: `03b9b534f4c3f12b1072e8eafa3802d50490f25f`, task 5, round 19.
- Checkpoint file SHA256: `016def2430638ea910ee0aabe042a3e4155ef145a067796997fa9b154326be02`.
- Policy: StandardScaler + `ClassLR_C0.1`, k=16 + self, candidate seed 42. Gate V2, router states, class feature schema, action restriction and fallback remain frozen.
- Panel seed: 20261006; receiver partition seed: 523687; batch size: 512.
- 98 receivers, 50,000 samples. New inputs received new expert predictions; old predictions were used exclusively for reference verification.

## Results on the same new panel

| Policy | Pooled accuracy | Macro-F1, present 28 classes |
|---|---:|---:|
| Self multiclass | 32.146% | 31.338% |
| Majority | 50.326% | 46.235% |
| Gate V2 | 54.184% | 51.361% |
| **Frozen class meta** | **58.358%** | **57.202%** |

The actual-routed candidate oracle is 90.064%. This uses labels to ask whether any queried expert predicted correctly; it is not a deployable selector.

Paired receiver bootstrap, 10,000 resamples, 95% intervals:

| Class meta comparison | Mean receiver gain | 95% CI | Improved receivers |
|---|---:|---:|---:|
| vs Gate V2 | +4.174 pp | [+3.810, +4.538] pp | 97/98 |
| vs majority | +8.032 pp | [+7.523, +8.567] pp | 98/98 |

These intervals describe receiver-level variability within this run, not variability across independent training seeds.

## Class scope and interpretation

Missing true classes: **0, 3, 28, 30, 31, 33**. Class 2 has only 118 samples; the other 27 present classes have 1,847 or 1,848 samples each. T5 contains only class 32.

The old full 34-class development accuracy of 50.732% is not directly comparable with 58.358% here. Restricting the old panel to these 28 classes gives 56.501% meta accuracy on 43,902 samples, but even that comparison has different class frequencies and is unpaired. Do not attribute the old/new difference to an algorithm change: the algorithm was frozen.

Macro-F1 calculated with all 34 labels is 47.107%, while macro-F1 over the 28 present labels is 57.202%. Including missing classes in a metric does not establish performance on their absent test examples.

| Task | New rows | Class meta accuracy |
|---|---:|---:|
| T0 | 5,662 | 66.125% |
| T1 | 11,088 | 87.933% |
| T2 | 11,086 | 51.470% |
| T3 | 11,082 | 54.232% |
| T4 | 9,235 | 38.885% |
| T5: class 32 only | 1,847 | 20.466% |

Class 32 remains weak: majority recall is 58.148%, meta recall 20.466%, and the candidate oracle 96.914%. Record this limitation without changing the frozen selector using this confirmation panel.

## Integrity audit

The completed archive reports:

- `completed=true`, zero fit calls, unchanged frozen artifacts.
- 850,000 original expert/sample reference comparisons passed; discrete fields exact, continuous fields at declared rtol=1e-4 and atol=1e-6. Original batch-512 reference streams were evaluated separately from fresh streams.
- 8,320 positive graph edges matched checkpoint membership and round-trip alpha values exactly; historical feature parsing remained unchanged.
- Zero old-panel row overlap, zero old-panel content overlap, zero gate calibration/fit/validation content overlap, and 50,000 unique content hashes.
- No raw-sample network queries; model inference ran locally in the evaluation runtime.

Independent local recomputation from the supplied archives verified:

- Protocol file checksum matches its lock; Gate V2 and class-meta payload checksums match the frozen declaration.
- Every prediction row matches the panel manifest's receiver, row ID and label. No old row ID or gate-role content hash appears in the new manifest.
- Replayed all 98 receivers using 1,666 fresh expert-feature files and the original frozen estimators: **zero prediction mismatch** for self, majority, Gate V2 and class meta.
- Candidate oracle and action restriction reproduced: every final predicted class appears among that sample's queried experts' normal predictions.
- Pooled accuracy, macro-F1 over 34 and 28 labels, and all paired bootstrap intervals reproduce.

Local estimator replay used scikit-learn 1.9.0 with version warnings suppressed; serialized estimators and the Kaggle notebook use 1.6.1. Despite that local version difference, all discrete outputs reproduced exactly. This audit replays saved features; it does not independently rerun GPU inference or reconstruct content hashes from the unavailable local raw test dataset. Raw content exclusions and GPU reference checks are supported by the archived runtime reports.

## Next experiment

Keep the selector protocol frozen. Replicate with at least three independent original training seeds, fitting gate/meta artifacts only on each seed's declared training/calibration/validation roles. Prepare an untouched test source covering all 34 classes for a full-scope final claim. Report model caching, 17 logical expert evaluations per sample, memory and communication assumptions explicitly; the single-self baseline itself requires only one expert evaluation.
