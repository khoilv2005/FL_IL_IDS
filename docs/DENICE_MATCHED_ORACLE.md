# Matched Oracle and BestAllowedRoute (P1)

Run `eval_denice_matched_oracle_kaggle.ipynb` on Kaggle with Internet, the original
100-clients dataset and **the previous denice_tip_diagnostics.zip** attached as a
dataset. An automatically extracted diagnostic folder is also supported. The
checkpoint still comes from the old results (4).zip Drive link. No training or
router fitting is performed. GPU is optional; execution uses one device.

Before measuring, the evaluator checks sample IDs, labels, encoder fingerprints,
legacy route predictions and every method's replayed class predictions against
the prior diagnostic output. A mismatch stops evaluation rather than producing
an incomparable number. The diagnostic ZIP must come from the completed 98-client
run on checkpoint 03b9b53.

For each method (legacy, multiclass, Mahalanobis, TIP):

- **Allowed tasks** are its actual nonempty binary memory tasks or its previously
  fitted continuous task bank, not all dataset tasks.
- **OracleMatched** substitutes true task only when it is an allowed task.
  Otherwise it retains the original method's selected route. Class mask,
  empty-mask fallback, adapter policy and classifier remain the existing ones.
  Both all-sample and task/class-supported conditional metrics are saved.
- **BestAllowedRoute** runs every allowed task action and asks whether any such
  action predicts the label correctly. This uses ground-truth labels and is not
  a deployable inference policy. It bounds route selection only for these fixed
  actions, weights, masks, adapters, fallback and sampled panel. It does not bound
  future soft routing, peer inference, a different readout or the full dataset.

Error partition: correct predictions first; then errors due to unavailable true
task, unavailable true class, wrong selected task, and classifier error with
correct task/class support. Correct predictions with unsupported true tasks
remain in the correct bucket. Independent availability flags are also exported;
these buckets are an accounting partition, not additive recoverable accuracy.

Outputs:

- `matched_summary.csv`: normal/matched/best-allowed accuracy, pooled and
  mean-client, conditional support metrics and error counts.
- `matched_per_client.csv`, `matched_predictions.csv`: audit trail.
- `profile_lifecycle_audit.csv`: memory/profile/class support and existing
  provenance. Missing provenance is labelled for investigation, not as proof of
  failed persistence or a client having trained that task.
- `matched_definition.json`, `protocol.json`: exact scope and definitions.
- `denice_matched_oracle_diagnostics.zip`: packaged outputs, also generated on
  failure (partial outputs must not be reported as final metrics).

This step implements P1 only. Multiclass integration and streaming/peer/training
changes remain gated on this diagnostic. No local tests or Kaggle execution have
been performed for the new evaluator yet.
