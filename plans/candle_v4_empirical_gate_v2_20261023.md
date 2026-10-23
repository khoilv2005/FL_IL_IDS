# CANDLE V4 empirical native-validation gate v2

Prospective protocol: `candle_v4_original_train_empirical_gate_v2`.
Supersedes only the validation count prerequisites of original-TRAIN v1.
Fresh seed42 with a newly pinned source. No mid-run source change or resume
from v1/role-based runs. Original TRAIN, current-task filtering, native local
validation, one coverage bank, no shadow V1 model and final full TEST unchanged.

## Frozen gate

- Donor selection: keep >=32 current TRAIN positives, same graph/ID rule.
- Remove both minimum32 donor validation positives and minimum32 negatives/owner.
- Require at least one unique eligible positive at the declared primary donor
  for a new expert, and at least one negative for each expert across the current
  declared owner scope. Negatives include other imported classes. If absent,
  DEFERRED, never divide by zero or infer PASS from empty data.
- An individual owner with zero negative samples is UNKNOWN for FAR, not zero
  FAR. It does not veto evidence that exists elsewhere in the declared scope.
  Do not claim that owner/class was protected. This defines the zero-evidence
  scope explicitly after removing the per-owner sample-count prerequisite.
- Threshold stays .5. Observed donor recall>=.95, per-owner/per-observed-negative-
  class FAR<=.001, break=0 and positive net rescue remain unchanged.
- 10/10 positive hits can pass recall; 9/10 fails. One FP on ten negatives still
  fails FAR. No override, threshold search, retry or passing-subset search.
- Keep unchanged historical expert receipts and current-negative retention
  policy; no historical raw validation replay. Unobserved historical risk UNKNOWN.
- TEST never supplies evidence for activation, threshold selection or training.

## Mandatory reporting

Record empirical validation acceptance, not statistical safety certification:
positive/hit/negative/FP counts, recall/FAR or null when absent, counts per owner,
observed and unobserved negative classes, future unseen classes and explicit
UNKNOWN owner pools. ACTIVE does not guarantee population FAR<=.1%; even 0 FP
on a small panel cannot support that statement. Native validation is development
evidence, not independent CAL confirmation.

No change to gradient objective, positive/negative sampling, optimizer, native
training, graph, expert architecture, threshold, class splitting or final TEST.
