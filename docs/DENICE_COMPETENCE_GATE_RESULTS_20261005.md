# Frozen competence gate results — 2026-10-05

Input: `C:\Users\khoak\Downloads\denice_competence_gate_diagnostics.zip`.
Training checkpoint `03b9b534f4c3f12b1072e8eafa3802d50490f25f`, task 5 round 19.
Evaluation source `1fed8f54dd2e260a086ba6da55c75829962e0330`.

## Integrity checks

- Completion marker is true; 98 clients, 50,000 unique test row IDs, 98 donor records in each runtime stage.
- Persisted gate SHA256 matches `gate_lock.json`: `f87624194b362c0d4c79b631d411fe120957d6d7e6e193905661924e62d14856`.
- Selected policies reproduce validation ordering: MLP_top1 at k=4, MLP_top4 at k=8, MLP_top1 at k=16. The evaluator persists/restores gates before test expert execution.
- All pooled accuracies and macro-F1 scores independently recompute from saved test predictions.
- Reconstructed features and gate decisions from frozen gate plus test expert caches: 900,000 comparisons, zero prediction mismatches across both families, three decision policies and three budgets.
- Same-seed random majority exactly matches the previous peer voting run: zero mismatches at k=4/8/16. Self reproduces 27.7681444% client mean, 27.772% pooled.
- Local sklearn is 1.9.0, while the saved estimators are 1.6.1; loading emits version warnings. Despite this difference, every tested output reproduces. Future reruns should record/use the training estimator version for reliable persistence.

## Primary results: validation-selected policy

All accuracy/F1 columns below are pooled percentages on the original class-stratified 50k panel. The majority comparator is **random seed 42**, not the five-seed mean from the previous summary.

| k | Selected policy | Validation accuracy | Test gate accuracy | Same-pool majority | Gate macro-F1 | Majority macro-F1 | Actual-routed oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | MLP_top1 | 58.279 | 37.516 | 36.202 | 34.510 | 32.735 | 65.066 |
| 8 | MLP_top4 | 59.686 | 39.916 | 40.236 | 36.715 | 35.702 | 74.294 |
| 16 | MLP_top1 | 60.654 | 42.358 | 42.850 | 38.817 | 37.400 | 81.026 |

Paired client-mean differences versus majority:

- k=4: +1.314 percentage points, 95% bootstrap CI [0.535, 2.131].
- k=8: −0.321 points, CI [−1.037, 0.440].
- k=16: −0.492 points, CI [−1.155, 0.144].

There is a clear small gain at k=4. At k=8/16 the observed accuracy is lower, but the CIs include zero; do not call it a statistically established degradation. Macro-F1 is higher descriptively at all three budgets; no F1 CI was computed here.

The criterion of >=45% at k<=8, and the 50% overall target, are not achieved. The gate should not replace majority as the accuracy baseline at larger budgets.

## Secondary gate policies

MLP_top4 observed test accuracy is 37.810 / 39.916 / 42.586% at k=4/8/16. The slightly better observed top4 test result at k=16 must not replace validation-selected top1 retrospectively. LR top1 scores 34.044 / 35.898 / 38.466%; LR top4 scores 35.922 / 38.172 / 40.020%.

Top2 weighted votes exactly match top1 predictions for both families in this run. Two different predicted classes receive only their individual gate scores, so the higher-scoring expert wins; top2 therefore provides little additional consensus. This is expected policy behavior, not a cache or selector implementation fault.

## Coverage-conditioned diagnosis

Joining the previous local-mask coverage audit on unique global row IDs gives 31,957 locally covered samples and 18,043 uncovered samples. Coverage here is a **ground-truth diagnostic bucket**, not an inference feature or deployable switch.

| k | Covered majority | Covered gate | Uncovered majority | Uncovered gate |
| --- | ---: | ---: | ---: | ---: |
| 4 | 42.504 | 47.386 | 25.040 | 20.035 |
| 8 | 43.305 | 48.512 | 34.800 | 24.691 |
| 16 | 43.868 | 49.679 | 41.046 | 29.391 |

At k=16, gate improves covered accuracy by 5.811 points but loses 11.655 points on uncovered samples; after weighting by bucket sizes, these gains/losses explain the overall −0.492 point difference. It rescues 4,846 majority errors but loses 5,092 previously correct predictions.

The gate sampling code restricts calibration/fit/validation labels to receiver-local supported classes with recorded participation. Consequently, these partitions do not represent the receiver's out-of-local-support target examples, while such examples are 36.086% of the panel. A shared gate may have seen those classes on other receivers, but the receiver-specific calibration condition is different. This is a strong **distribution/support mismatch hypothesis**, supported by the conditional results; it is not a controlled causal ablation yet.

Validation accuracy of 58–61% must therefore not be interpreted as a reliable forecast of the cumulative global test panel. Original backbone may also have seen these gate validation rows; this is not an independent backbone holdout.

Additional observations:

- All receiver roles reach their declared caps: 12,544 calibration, 50,176 fit and 25,088 validation rows. Unique content counts are 12,459 / 49,611 / 24,630. No content hash spans more than one role. Exact panel content exclusion is enforced by the sampling code; this audit does not re-read original raw shards to repeat that check.
- Receiver-donor-task competence rate has the largest absolute standardized LR coefficient at every budget (about +1.4); agreement features also matter. This is descriptive, not proof that removing one feature fixes transfer.
- `task_supported` and `class_supported` are constant in all fitted gates: they describe the donor's own normal routed prediction, which already obeys its mask. They do not describe whether the receiver supports the donor's predicted class/task.
- MLP runs use all 50 fixed training iterations. LR uses 32/26/30 iterations. More iterations are not an established remedy.
- At k=16, mean top expert MLP score is 60.431%, while top1 correctness is 42.358%. Treat these scores as uncalibrated; do not choose a score threshold using the test labels.
- Gate-data expert inference processes 1,492,736 samples in 145.83 seconds; test expert inference processes 850,000 samples in 85.02 seconds. Totals exclude data preparation, gate fitting and model loading, and are not distributed deployment latency.

## Next experiment

Keep the frozen checkpoint, fixed legitimate candidate graph, and majority baseline. Do not increase peer budgets or backbone training solely to repair this gate.

1. Correct the gate **training/validation support**, using a declared peer-supported historical train/validation mixture that includes receiver-unseen classes where a legitimate candidate has local participation evidence. Do not use test labels or inherit training access from binary memories. Fix sampling/weighting using training/validation information rather than matching the observed 36.086% test fraction.
2. Separately retain donor-only/global competence priors and receiver-conditioned priors as validation ablations. Candidate prediction support relative to the receiver is a valid inference feature; the sample's true class coverage is not.
3. Evaluate shared-class-balanced and peer-supported validation, including majority on those exact validation inputs, before locking the final gate. Persist validation labels/predictions and sampling provenance to permit artifact-level recomputation without rereading raw data.
4. Choose policy and any fallback using validation only. A switch using the true covered/uncovered bucket would be an oracle and must not be reported as inference.
5. Keep the shared retrospective gate and model-local evaluation deployment caveats explicit. Confirm a final method on an untouched test panel and independent training seeds after design is fixed.

No new gate variant has been implemented or evaluated in this result audit.
