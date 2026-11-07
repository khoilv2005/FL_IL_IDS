# Task versus background routing pilot

Following zero observed T4->T5 feature drift in the bounded LDC pilot, test
discriminative task/background scores. Inspired by task/OOD separation in MORE,
not its transformer architecture, replay protocol or complete algorithm.
Source:https://proceedings.mlr.press/v199/kim22a.html

Two fixed variants, same saved receiver TRAIN activation memory:
- OVR task: binary LR for each task versus other remembered tasks.
- OVR class: binary LR per remembered class versus other classes; max per task.

Existing scaler, C1, lbfgs, max_iter1000, class_weight balanced, seed0.
Compare decision scores only inside original two task candidates; tie preserves
production Top1. No calibrated posterior or OOD guarantee. Historical raw inputs
not read. Scores from separately balanced classifiers may be incomparable; test
end-to-end rather than assume probabilities share a scale.

All100clients, same204800-row fixed TEST development panel as earlier audits.
Every fit finishes before TEST opens. Exactly one selected task scopes adapter,
class mask and experts. Same threshold. Shared evaluator reuses two task-specific
predictions for all variants. Exact production baseline CM parity and native/
service hashes required. No threshold/hyperparameter sweep or production change.

This learns binary classification against observed background, not general unseen
OOD detection. Missing classes/tasks and historical feature mismatch remain
unresolved. No external peer bytes; model fitting/storage/inference costs grow.
Report all outcomes even if both fail, without searching fixture by TEST quality.
