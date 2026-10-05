# Gate V2 peer-supported diagnostic results — 2026-10-05

Input: `C:\Users\khoak\Downloads\denice_competence_gate_v2_diagnostics.zip`.
Checkpoint: `03b9b534f4c3f12b1072e8eafa3802d50490f25f`, task 5 round 19.
Evaluation source: `f9baa030d9fcc8cca8558874d4b88959efa99daf`.
The same 50,000 class-stratified samples are a **development/diagnostic panel**, not an untouched final test set.

## Artifact verification

- Run completion is true. All 98 receivers and 50,000 unique test row IDs are present.
- Gate SHA256 matches the lock: `3aebd1d68c4342d1e9d3addb38c53ec71364a69df0abb1d1d6f49611e47f9633`.
- Validation selection independently agrees with the lock: MLP_top1 at all three budgets. Test-best alternatives are not substituted.
- Recomputed 2,027,376 learned-gate/prior decisions from frozen models and expert features across test and validation: zero mismatches.
- All test and validation pooled accuracies and macro-F1 scores independently recompute without discrepancies. Recomputed calibration count tables exactly match the persisted priors.
- Same-pool majority predictions exactly match V1 at k=4/8/16; self remains 27.772% pooled, 27.7681444% client mean.
- No provenance origin is outside its receiver's fixed maximum-budget candidates/positive-alpha graph, and every contributed label is in its origin's recorded training support audit.
- No input hash crosses gate roles. Full raw-data panel exclusion and original backbone participation are not re-audited here from the original train files/checkpoint; those guards are implemented in the evaluator and recorded in artifacts.
- Saved sklearn version is 1.6.1; local audit uses 1.9.0. Despite the persistence version difference, all audited predictions reproduce exactly.

## Primary policies selected by validation

All accuracy/F1 values below are pooled percentages. Majority uses the exact same fixed random-seed-42 candidate pool, rather than the prior five-seed random mean.

| k | V2 validation accuracy | V1 primary test accuracy | V2 primary test accuracy | Majority test accuracy | V2 macro-F1 | Majority macro-F1 | Actual-routed oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 40.621 | 37.516 | 43.006 | 36.202 | 39.254 | 32.735 | 65.066 |
| 8 | 43.363 | 39.916 | 45.192 | 40.236 | 40.763 | 35.702 | 74.294 |
| 16 | 44.683 | 42.358 | 46.388 | 42.850 | 42.029 | 37.400 | 81.026 |

V2 meets the user-declared >=45% at k<=8 criterion. The 50% target is not met: the primary k=16 result is 3.612 points short. Inference still queries k+1 models before selecting one; top1 output does not mean one model was queried.

Paired client-mean V2 gains over majority (artifact 2,000-resample bootstrap):

- k=4: +6.805 points, 95% CI [5.986, 7.734].
- k=8: +4.956 points, CI [4.354, 5.570].
- k=16: +3.538 points, CI [3.146, 3.957].

Paired V2 gains over the frozen V1 primary policies (independent 10,000-resample bootstrap, seed 20261005):

- k=4: +5.491 points, CI [4.621, 6.419].
- k=8: +5.277 points, CI [4.397, 6.216].
- k=16: +4.031 points, CI [3.384, 4.704].

MLP_top4 has observed test accuracy 46.632% at k=16, but validation chose top1. Keep 46.388% as the primary result; do not promote top4 after inspecting test labels. Macro-F1 gains are descriptive here; no F1 confidence intervals were computed.

## Pure competence priors and LR

| k | Global donor prior | Global donor-task prior | Receiver-donor-task prior | LR top1 | LR top4 | MLP selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 37.366 | 35.522 | 35.572 | 40.558 | 40.098 | 43.006 |
| 8 | 38.942 | 35.390 | 35.338 | 42.616 | 42.950 | 45.192 |
| 16 | 39.890 | 33.726 | 34.284 | 44.132 | 44.568 | 46.388 |

Pure priors are not sufficient substitutes for learned selection. At larger budgets they lose to majority; MLP and LR improve over the same-pool majority. V2 changes the sampling/data-cleaning bundle (including conflicting training-content exclusion), so this result supports the support-mismatch hypothesis but does not isolate class-support expansion as the only causal factor.

## Support coverage and V1 failure mode

All receiver roles reach caps: calibration 12,544 rows, fitting 50,176, validation 25,088. Unique content counts are 10,342 / 39,491 / 19,247. Inputs can repeat across receivers within the same role but cannot cross roles. About 94.24% of sampled rows originate from peers rather than self.

Receiver-uncovered rows are now 4,501 calibration / 17,924 fit / 8,967 validation. Validation therefore includes about 35.74% receiver-uncovered examples; the development panel has 36.086%. The test proportion was not used to set sampling weights. Only client 10's calibration lacks one supported-union class (class 0); all fitting/validation receiver pools represent their supported union.

| k | Covered majority | Covered V1 | Covered V2 | Uncovered majority | Uncovered V1 | Uncovered V2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 42.504 | 47.386 | 44.682 | 25.040 | 20.035 | 40.038 |
| 8 | 43.305 | 48.512 | 44.998 | 34.800 | 24.691 | 45.536 |
| 16 | 43.868 | 49.679 | 46.106 | 41.046 | 29.391 | 46.888 |

V2 improves uncovered samples substantially and beats majority in both buckets at every budget. Some V1 covered accuracy is traded away, but the overall result improves clearly. Ground-truth coverage is used only for this analysis; it is not a deployable gate/fallback condition.

The original validation-to-test drop of roughly 18–21 points is replaced by much closer validation/test values. This remains retrospective gate validation; the backbone may have trained on those historical inputs.

## Remaining task/class weaknesses

At k=16, gate rescues 4,935 majority errors but loses 3,166 previously correct predictions.

| True sample task | Majority accuracy | V2 primary accuracy | Actual-routed oracle |
| --- | ---: | ---: | ---: |
| T0 | 34.786 | 35.277 | 63.087 |
| T1 | 77.530 | 87.266 | 98.034 |
| T2 | 28.412 | 46.599 | 85.395 |
| T3 | 51.764 | 46.312 | 98.246 |
| T4 | 30.399 | 32.355 | 67.291 |
| T5 | 21.259 | 9.067 | 61.290 |

These are accuracies on samples belonging to each task after final training, not accuracies at each training task endpoint. Overall improvement does not imply every task improves.

Class 32 is a major selector loss: majority recall is 61.391%, gate recall 17.486%, while actual-routed oracle coverage is 96.873% (1,567 samples). Classes 19/21/22 also lose recall. Class 28 has zero actual-routed candidate coverage on its 648 panel samples at k=16; no selector restricted to these normal argmax predictions can recover those samples.

The source sampler scanned 1,946,031 rows and excluded 41,871 distinct input hashes with conflicting training labels. This is a count of ambiguous input identities, not a percentage of dataset rows, and not proof of an incorrect dataset. Its class composition and effect on transfer require a separate audit before attributing class-32/T5 losses to that filter.

Gate-data expert inference totals 1,492,736 samples / 150.46 seconds; test inference totals 850,000 / 83.54 seconds. These exclude gate fitting, data preparation and loading, and are not deployment latency.

## Decision and next step

Retain V2 as the frozen learned-selector baseline; the predeclared failure condition of remaining at 42–43% is not triggered. Use k=8 and k=16 as reported cost/accuracy comparisons, without choosing a deployment budget from this panel. Do not retrain the backbone solely because the result is below 50%.

The next useful controlled refinement is class-level/meta-ensemble selection learned on the existing peer-supported training/validation roles, with per-class/task validation metrics and the fixed V2/majority baselines. Audit the ambiguous training-content class distribution and task-5 validation failures first. Select any weighting, calibration or consensus policy using validation, never a true-task/test-label switch. If a new meta-classifier predicts classes outside the existing expert argmax action set, recompute its diagnostic bound rather than calling 81.026% its unchanged upper bound.

After design is fixed, evaluate on a newly locked untouched panel and independent training seeds. The current 46.388% is a development diagnostic result, not final deployment performance or a replay-free decentralized gate claim. No refinement is implemented in this artifact audit.
