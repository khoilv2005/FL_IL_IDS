# LDC-inspired T4-to-T5 feasibility

Use fresh-run results19 Task4 continuation and results20/21 finalTask5. Extract
only verified Task4 continuation; no old raw TRAIN replay. Choose first3 client
IDs with at least2 historical task memories, Task5 memory, equal selected indices,
and byte-equal old historical feature arrays carried into final memory.
Require actual local Task5 TRAIN count>1280 in native debug receipt; router
memory alone may originate from bootstrap and does not prove local TRAIN exists.
An initial metadata preflight without this condition stopped before fitting/TEST
when client0 lacked current TRAIN. The fixture was corrected by data availability,
not evaluation quality.

At each client read original TRAIN and filter strictly classes30..33. Seed42,
up to256 rows/class for fitting and next64/class for assessment. Same inputs
pass frozen old/new encoders; StandardScaler fitted on paired TRAIN features.
Fit affine ridge(lambda1, intercept unpenalized) and MLP(d->128->d, ReLU,
Adam.001,100 full-batch steps,seed42). No search. These are LDC-inspired
prototypes, not exact ECCV2024 code reproduction.

Measure current-domain displacement and held-out current TRAIN mapping error.
Map only historical saved memory; keep finalTask5 memory unchanged, refit same
class LR. Controls: saved production baseline, memory-only refit with no mapping.
All fitting ends before opening class-balanced TEST diagnostic128/class,
seed20261107. No true label/task used to choose predictions.

Per receiver evaluate one task's adapter/mask/experts at a time, unchanged expert
threshold. Report route correct, rescue/break, cumulative and old/current class
accuracy and MacroF1. Verify native/service hashes. V1 and training not modified.

Limitations: only3clients, diagnostic repeated TEST development, sklearn local
version differs from original; refit control required. Memory from earlier tasks
may already be stale byT4. Mapping trained on current classes may extrapolate
poorly to old classes. Old/new model copies are temporary diagnostic references,
not a new production shadow predictor. No claim of zero cost or live integration.
