# Frozen class-level meta-ensemble: validation audit and protocol

## Input and baseline

Run `eval_denice_class_meta_kaggle.ipynb` on Kaggle with Internet enabled.
The notebook downloads the completed Gate V2 archive from the configured
[Drive link](https://drive.google.com/file/d/1gweD4iZ4_NYmEsQyGDOlInTITwcxHrnP/view?usp=sharing).
Alternatively set `DENICE_GATE_V2_ZIP` to a mounted archive path.
No original dataset attachment, checkpoint download, GPU, backbone training,
router refitting or expert model inference is needed.

The caches belong to checkpoint `03b9b53`, task 5 round 19, 98 receivers,
fixed random candidate seed 42, budgets k=4/8/16 (self is additional).
The source V2 gate checksum must match its lock and sklearn must be 1.6.1.
Source V2 primary k=16 accuracy remains **46.388% pooled**; 46.632% top4
was not validation-selected and is not substituted for the baseline.

## Validation findings motivating the experiment

These findings use the **V2 validation partition**, rather than fitting to test
errors. They corroborate the already observed development-panel failure mode.
At k=16:

| Class | Validation rows | Majority % | V2 top1 % | Actual-routed oracle % | Mean correct expert votes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 19 | 689 | 34.11 | 18.14 | 98.84 | 4.79 |
| 21 | 688 | 66.28 | 33.72 | 99.13 | 7.30 |
| 22 | 688 | 92.88 | 51.31 | 99.56 | 7.55 |
| 30 | 686 | 0.15 | 1.90 | 33.67 | 0.42 |
| 31 | 686 | 0.00 | 3.94 | 24.78 | 0.29 |
| 32 | 686 | 56.71 | 17.93 | 95.63 | 5.64 |
| 33 | 686 | 0.44 | 9.77 | 58.02 | 0.97 |

T3 validation: majority 51.61%, V2 47.08%, routed oracle 98.38%.
T5 validation: majority 14.32%, V2 8.38%, routed oracle 53.02%.
For class 32, several normal peer predictions are correct on many samples,
but selecting the single highest competence score loses that evidence.
Classes 30/31 have a much smaller candidate bound; the same explanation
does not apply uniformly to all T5 classes.

The evaluator exports **all classes and all tasks**, at every budget, to
`validation_class_audit_k*.csv` and `validation_task_audit_k*.csv`.
There is no class-32 override, T5 special rule or test-frequency adjustment.

## Class evidence and permitted decisions

Use the already frozen V2 MLP and calibration priors. For each candidate class,
aggregate 17 features across experts predicting that class:

- Vote fraction, normalized competence sum, max/mean competence and self vote.
- Mean router confidence, class confidence, margin, entropy, log mask size,
  three calibration competence priors and graph alpha.
- Maximum router confidence, class confidence and margin.

This gives 578 features for 34 classes. Class identities use a fixed coordinate
order; donor order cannot change the aggregate, provided self stays identified.
Features contain no true class, true task or receiver-covered test indicator.

The output class is restricted to the **union of normal argmax predictions of
the queried experts**, including self. Thus the original actual-routed oracle
is still an upper bound for this experiment. This is narrower than a classifier
allowed to output any class supported by a peer. Zero available model probability
falls back to majority; exact probability ties prefer self, then lowest class ID.

Fitting rows whose true class is absent from this union are excluded from meta
training because the constrained decision cannot output that target. Their counts
are saved by class/budget in `fit_candidate_coverage.csv`. **All validation and
test rows remain in every metric denominator**, including unreachable samples.
Balanced variants use only reachable fitting-label counts.

## Fixed candidates and selection

For each original budget, compare:

| Candidate | Configuration |
| --- | --- |
| ClassLR_C0.1 | StandardScaler + multinomial LR, C=0.1, max_iter=1000 |
| ClassLR_C1 | Same, C=1 |
| ClassLR_C1_balanced | Same C=1, class_weight=balanced |
| ClassMLP | StandardScaler + MLP(64,32), alpha=.001, max_iter=50, batch=1024 |
| ClassMLP_balanced | Same MLP, deterministic class-balanced fitting resampling |
| GateAllVote | Sum frozen V2 competence scores over **all** expert predictions per class |

Seed is fixed at 20261005. Baselines are self, same-pool majority and the
original validation-selected V2 gate. Select among the six new policies using
pooled validation accuracy, then macro-F1 for ties, then policy name.
No test result chooses the primary policy. All alternatives remain ablations.

Before any test-role access, persist `frozen_class_meta.joblib` and
`class_meta_lock.json`. Restore models and compare their validation predictions
against saved predictions; any mismatch aborts. Then reproduce original V2 and
majority test predictions from their cache, evaluate the new policies and assert
no correct prediction occurs outside the original routed-oracle action set.

## Output and interpretation

Output ZIP: `/kaggle/working/denice_class_meta_diagnostics.zip`.
Inspect `class_meta_completion.json`: partial archives are also saved on failure.
Use a fresh output directory/session to avoid mixing artifacts from two runs.

Artifacts include validation audits, fit reachability counts, validation predictions,
selection lock, model artifact, test predictions, per-client metrics, pooled/client-mean
accuracy, pooled macro-F1, paired client-bootstrap gain CIs against V2, and an
accuracy-vs-budget PNG/PDF. Logical expert queries remain k+1 per sample in the
underlying protocol; cached evaluation itself makes zero expert queries.

V2 competence scores on fitting rows are in-sample outputs of the previously fitted
gate. This experiment does not add cross-fitting; validation and test partition
labels are not used to fit the class models. It is a shared retrospective stacker,
not a streaming replay-free claim. The backbone may already have seen historical
gate validation data during original training.

The existing 50k panel has guided development and must not be presented as untouched
final evidence. Freeze the design before independent final evaluation. No accuracy
improvement or 50% result is claimed before the notebook runs.

User decision thresholds: >=50% stops this frozen-checkpoint line; 48-49% justifies
one refinement; approximately 46-47% motivates revisiting training/aggregation.
