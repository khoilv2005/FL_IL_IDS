# Frozen DeNICE class meta-ensemble results — 2026-10-05

Input: `C:\Users\khoak\Downloads\denice_class_meta_diagnostics.zip`.
Source Gate V2 archive: `denice_competence_gate_v2_diagnostics.zip`.
Checkpoint: `03b9b534f4c3f12b1072e8eafa3802d50490f25f`, task 5 round 19.
The existing 50,000-row panel is a **development diagnostic**, not untouched final evidence.

## Integrity audit

- Completion is true; frozen meta SHA256 matches the lock:
  `786477d5c66d46cf5cf7db263b567bcb0e406ad9e11a6f5ad8613b7414c7b771`.
- Source Gate V2 SHA matches the previously audited artifact:
  `3aebd1d68c4342d1e9d3addb38c53ec71364a69df0abb1d1d6f49611e47f9633`.
- Independently recomputing validation selection picks **ClassLR_C0.1 at all budgets**.
- All budgets contain 98 receivers and the same 50,000 distinct global test row IDs;
  no duplicates. Labels and row IDs match the source V2 CSVs exactly.
- Self, majority and Gate V2 predictions reproduce the source V2 archive exactly.
- Recomputed pooled test accuracy and macro-F1 for all policies agree to floating-point
  precision (maximum absolute difference below 1e-15). All validation accuracies agree.
- Checked every test decision of the five class models, GateAllVote and the selected
  policy against the actual queried expert predictions: **zero action-set violations**.
  Recomputed actual-routed oracle flags exactly match the saved flags.
- Replayed the primary frozen LR and source feature construction for all validation/test
  roles and budgets: **225,264 decisions, zero prediction mismatches**.
- Saved sklearn version is 1.6.1; independent local replay used 1.9.0. Persistence-version
  warnings were suppressed during this audit only; production notebook still requires 1.6.1.
- Saved validation restore mismatch count is zero. The lock flags match the implementation
  sequence: models/selection are saved before test-role access. Flags alone do not prove a
  previously untouched panel; the development-panel qualification remains essential.

## Primary validation-selected results

All accuracy and F1 numbers below are pooled percentages. CI refers to paired
client-bootstrap gain against Gate V2, as saved in the artifact (2,000 replicates).

| Peer budget k | Selected validation accuracy | Majority test | Gate V2 test | Class meta test | Class meta macro-F1 | Gain vs V2, pp [95% CI] | Actual-routed oracle |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 4 | 43.9613 | 36.202 | 43.006 | **45.618** | 42.197 | +2.612 [2.232, 3.024] | 65.066 |
| 8 | 47.3055 | 40.236 | 45.192 | **48.620** | 45.293 | +3.428 [3.044, 3.828] | 74.294 |
| 16 | 49.0234 | 42.850 | 46.388 | **50.732** | **47.534** | **+4.344 [4.015, 4.678]** | 81.026 |

The primary k=16 policy is **StandardScaler + ClassLR_C0.1**, selected using
validation before reading test records. Its client-mean accuracy is 50.7320267%.
Macro-F1 increases from Gate V2's 42.0286% to 47.5340%, a gain of 5.5054 pp.
Compared with same-pool majority, pooled accuracy gains 7.882 pp.

Independent paired-bootstrap recalculation with 10,000 client draws gives k=16
gain vs V2 CI [4.010, 4.674] pp, and gain vs majority CI [7.411, 8.369] pp.
The corresponding descriptive client-mean accuracy interval is [50.404, 51.061]%. 
These intervals describe resampling the recorded receivers; shared peers/overlapping
historical fitting data and repeated use of the panel limit generalization claims.

At k=16, meta rescues 5,943 previously incorrect V2 decisions and loses 3,771
previously correct ones: net **2,172 additional correct samples**. It improves
accuracy on **97/98 receivers**.

The selected k=16 meta is also the highest observed test candidate, so there is no
posthoc policy substitution. At k=4, C=1 scores 45.646%, slightly above C=.1,
but the reported primary remains validation-selected C=.1 at 45.618%.

## Alternatives

| k=16 policy | Test accuracy % | Macro-F1 % |
| --- | ---: | ---: |
| **ClassLR_C0.1 (primary)** | **50.732** | **47.534** |
| ClassLR_C1 | 50.386 | 47.190 |
| ClassLR_C1_balanced | 49.950 | 47.182 |
| ClassMLP | 49.686 | 46.373 |
| ClassMLP_balanced | 49.094 | 45.952 |
| GateAllVote | 46.442 | 41.209 |
| Gate V2 | 46.388 | 42.029 |

Simply summing all V2 competence votes does not reproduce the class-model gain:
GateAllVote is essentially unchanged at k=16 and worse at k=4/8. Evidence supports
learning a class decision from the aggregate rather than merely enlarging top1 voting.
This is not an ablation isolating which of the 578 features supplies the gain.

## Task/class behavior at k=16

| True task | Rows | Majority % | Gate V2 % | Class meta % | Routed oracle % |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 | 8,745 | 34.786 | 35.277 | 38.857 | 63.087 |
| T1 | 9,408 | 77.530 | 87.266 | 87.893 | 98.034 |
| T2 | 9,408 | 28.412 | 46.599 | 50.904 | 85.395 |
| T3 | 9,408 | 51.764 | 46.312 | 53.306 | 98.246 |
| T4 | 8,487 | 30.399 | 32.355 | 37.080 | 67.291 |
| T5 | 4,544 | 21.259 | 9.067 | 16.461 | 61.290 |

All tasks improve relative to Gate V2, though T5 still falls below majority.
Validation already shows T3 rising from 47.083% to 55.749% and T5 from 8.382%
to 14.322%; the observed improvement is not solely an unexplained test-only shift.

Relevant per-class test recalls:

| Class | Gate V2 % | Class meta % | Majority % | Routed oracle % |
| --- | ---: | ---: | ---: | ---: |
| 19 | 18.622 | 34.056 | 35.395 | 98.597 |
| 21 | 30.485 | 54.401 | 62.946 | 99.745 |
| 22 | 50.957 | 56.569 | 92.028 | 99.490 |
| 30 | 3.052 | 13.006 | 0.000 | 34.240 |
| 31 | 2.786 | 8.914 | 0.000 | 23.677 |
| 32 | 17.486 | 20.677 | 61.391 | 96.873 |
| 33 | 7.381 | 17.642 | 0.360 | 59.946 |

Class 32 remains weak. Reaching 50% did not require recovering its full bound;
gains are distributed across tasks/classes. Some classes regress (e.g. 1, 11, 14,
20), so the method is not uniformly better class by class. Class 28 remains 0/648
with zero actual-routed expert coverage; this restricted meta cannot recover it.

## Fitting eligibility

Every budget starts with 50,176 fitting rows. Reachable rows used for meta fitting:

| k | Reachable fitting rows | Excluded unreachable fitting rows |
| --- | ---: | ---: |
| 4 | 30,690 | 19,486 |
| 8 | 35,269 | 14,907 |
| 16 | 38,812 | 11,364 |

All 25,088 validation rows and all 50,000 test rows remain in metric denominators.
The exclusion is only a fitting eligibility policy for a constrained action set.
Original Gate V2 fitting scores are in-sample; this experiment is not cross-fitted.

## Decision and next experiment

The user-defined **>=50% frozen-checkpoint diagnostic target is met**. Preserve:

- Backbone checkpoint 03b9b53, original donor router/mask/adapter behavior.
- Fixed random-seed-42 candidate ordering, k=16 plus self.
- Frozen Gate V2 feature/prior artifact and primary class LR C=.1 artifact.
- Exact class feature schema, scaling, action restriction and tie/fallback policy.

Stop selector tuning on this panel. Do not retrain backbone as a response to this
result. Next validate the frozen method using a **new, non-overlapping test panel**,
with exclusion of the existing global row IDs and relevant gate-role content hashes.
Draw the new panel and declare metrics before inspecting its labels/results. Query
the original models on those new inputs; old test-prediction caches cannot be used
as predictions for new samples. Reuse the fitted gate/meta weights, without fitting
or selection on the new panel. Report class-stratified and population-weighted metrics
separately if both sampling protocols are evaluated.

This 50.732% is **self + 16 peer expert inference with a shared retrospective
class meta-ensemble**, not the standalone DeNICE client accuracy. It is not yet
an unchanged decentralized/private deployment result or a streaming replay-free
claim. The run itself loads only caches and makes no additional model queries,
but deployment logically requires 17 expert predictions per sample. A future model
cache/communication protocol needs explicit resource accounting.
