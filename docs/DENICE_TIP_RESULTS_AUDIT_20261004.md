# Frozen router results audit — 2026-10-04

Source: `C:\Users\khoak\Downloads\denice_tip_diagnostics.zip` (305 entries).
Training checkpoint `03b9b53`, task 5 round 19; evaluator `1298fda`.
Analysis reads saved predictions/statistics; no training or new test inference.

## Main result

All 98 clients completed. P0 mean-client accuracy 22.787718% matches baseline.
There are 50,000 unique global test row IDs and no duplicate client/shard keys.
Profile manifests contain 205,949 fit and 51,437 validation row selections with
no fit/validation row overlap within a client/task. This does not check duplicate
flow contents or backbone exposure. Every client has some validation data.

| Policy | Mean-client accuracy % | Pooled macro-F1 % | Global task accuracy % | Accuracy gain pp, 95% client bootstrap CI |
| --- | ---: | ---: | ---: | --- |
| Legacy | 22.7877 | 20.4190 | 36.466 | baseline |
| Centroid | 20.7982 | 19.2629 | 32.624 | -1.9896 [-3.3842, -0.6745] |
| Mahalanobis | 28.8858 | 26.7038 | 43.032 | +6.0981 [4.3998, 7.8345] |
| TIP | 23.5879 | 22.3667 | 35.704 | +0.8002 [-0.6791, 2.1872] |
| Multiclass | 27.7681 | 25.7204 | 41.574 | +4.9804 [4.1805, 5.7917] |
| OracleLocal | 50.9362 | 47.0386 | supplied task | diagnostic |
| OracleGlobal | 46.8696 | 41.9576 | supplied task | diagnostic |
| AllClasses | 12.5119 | 11.1384 | none | diagnostic |

Applying the predeclared numerical thresholds to each challenger: Mahalanobis
and multiclass meet accuracy gain, pooled macro-F1 and positive CI requirements;
TIP does not. The saved gate.json explicitly evaluates TIP only. Batch probes
ran on 8 samples/client for continuous methods; this is not exhaustive invariance
verification or a new verification of the existing multiclass inference path.

Mahalanobis improves accuracy on 82 clients, loses on 16; multiclass improves
on 87, ties on 3 and loses on 8. TIP improves on 64, ties on 4, loses on 30.
TIP's mean-client macro-F1 actually decreases from 13.9058% to 13.7865%, despite
the increase in pooled macro-F1. These averaging conventions must stay separate.

## Mahalanobis versus multiclass

Mahalanobis minus multiclass mean-client accuracy: +1.117629 pp. A paired
bootstrap over clients (20,000 draws, NumPy RNG seed 42) yields 95% CI
[-1.120110, 3.244512] pp; Mahalanobis wins on 59/98 clients. This panel does
not establish a clear winner between those two methods. Their data access
also differs: historical continuous refit versus saved binary memory.

Multiclass has higher mean-client macro-F1 (19.0382% versus 18.1081%), while
Mahalanobis has higher pooled macro-F1 (26.7038% versus 25.7204%). A single
unqualified claim that one has higher F1 would therefore be misleading.

## Coverage is a separate bottleneck

- Local class mapping covers 31,957/50,000 samples (63.914%). All four learned
  challenger policies and legacy have zero correct predictions on the other
  18,043 samples. No learned method has a class-correct/task-wrong prediction
  in this panel. Hard local masks impose a substantial structural restriction.
- Legacy/multiclass have task entries covering 42,990 samples (85.98%).
  Continuous profiles cover only 36,417 (72.834%): local historical participation
  evidence restricts fitting. Only 13 clients have six fitted tasks; task-count
  distribution: 1:1 client, 2:1, 3:10, 4:34, 5:39, 6:13.
- 13,583 samples have no continuous profile for their true task; all continuous
  policies are wrong there. This includes 6,541 samples whose true class is in
  the original local mapping. Missing profile coverage is not solely missing
  class support. The data does not distinguish legitimately absent historical
  access from conservative exclusion due to missing provenance.
- On the same profile-available subset (36,417 samples), classification accuracy
  is legacy 26.20%, Mahalanobis 39.66%, multiclass 30.86%, TIP 32.39%. This is
  diagnostic conditioning on true task availability, not a deployable selection
  rule and not a replacement for all-sample accuracy.
- Route accuracy on locally covered classes: legacy 47.03%, Mahalanobis 57.76%,
  multiclass 55.87%, TIP 46.95%. Do not confuse these with global task accuracy.

## OracleLocal is not a pure all-sample routing ceiling

The existing evaluator falls back to all seen classes when the requested task
has no local allowed-class list. OracleLocal therefore changes behavior for
unavailable tasks as well as supplying the true ID. It correctly predicts 15.43%
of locally uncovered samples; learned routing correctly predicts none there.

For covered samples OracleLocal accuracy is 70.99%, versus OracleGlobal 36.41%:
the narrower local mask removes competing classes. For uncovered samples the
figures reverse to 15.43% versus 65.38%. This explains why overall OracleLocal
50.94% exceeds OracleGlobal 46.87%; it is not evidence of a new deployable model.

A next diagnostic should report Oracle on a shared task/class-supported subset,
and a strict unsupported-task policy explicitly. Keep the existing fallback
result separately for reproduction. Do not promise 50% from the current router.

## Task-level classification accuracy (%)

| Test task | Legacy | Mahalanobis | Multiclass | TIP |
| --- | ---: | ---: | ---: | ---: |
| T0 | 16.05 | 14.07 | 24.30 | 11.06 |
| T1 | 40.84 | 35.75 | 43.47 | 33.15 |
| T2 | 17.79 | 29.49 | 23.49 | 21.82 |
| T3 | 31.57 | 40.11 | 34.94 | 35.57 |
| T4 | 8.68 | 26.16 | 18.55 | 13.97 |
| T5 | 16.90 | 23.81 | 13.20 | 24.71 |

The tradeoff is clear: multiclass improves old tasks T0/T1, but loses T5;
Mahalanobis improves T2–T5 but regresses T0/T1. A combined router is a hypothesis,
not established by choosing the better method per true task on this test set.

## Numerical and TIP diagnostics

Across all TIP candidate/task fits, 5,298 decompositions used NumPy SVD and 6
used the scaled scipy gesvd fallback; none required the Gram fallback. Thus the
fallback was actually exercised in this Kaggle run. Selected configurations:
60 independent banks, 38 residual banks; 30 independent/RMS/rank32 and 30
independent/mean/rank32. Of 440 selected task bases, 281 have rank 3–5.

Mean off-diagonal cosine between normalized selected reference vectors is
0.951418 (pooled over ordered task pairs across clients). References remain
similar, consistent with limited separation. This is not directly comparable
to the earlier mean-per-client binary-prototype cosine 0.982793: representation
and averaging differ. Rank alone does not establish why TIP fails.

## Recommended next step

1. Keep checkpoint 03b9b53. Stop promoting TIP to streaming based on this run.
2. Prioritize the existing binary multiclass router for a controlled integration:
   it uses already retained memory and has broad client gains. Fit from memory
   at each relevant refresh, without revisiting historical raw train data.
   This remains to be tested across the actual stream, including encoder drift.
3. Keep Mahalanobis as the strongest retrospective continuous challenger.
   Audit missing-task provenance and coverage before changing its bank or adding
   peer support. Do not invent profiles for tasks a client never observed.
4. Use training/validation data to design any calibration/fallback between the
   two, then evaluate on a fresh locked panel or independent seed. No test-task
   oracle selection; do not tune repeatedly against these 50,000 test rows.
5. Report the current panel as class-stratified: 50,000 of 13,505,771 test rows.
   Accuracy here is not an estimate under the full test set's natural class
   frequencies without an appropriate sampling/weighting analysis.

No training settings or inference implementation are changed by this audit.
