# Opt-in multiclass integration (P2)

## Run the reproduction gate

Upload `eval_denice_multiclass_integration_kaggle.ipynb` to Kaggle. Attach the
original 100-clients dataset and enable Internet. The notebook downloads the
original results (4).zip and prior TIP diagnostics using the configured Drive
links. No additional dataset upload is needed.

The gate restores checkpoint 03b9b53, checks original sample IDs and model
fingerprints, reproduces legacy predictions, fits the integrated router using
saved binary memory, and runs **normal pred_hard inference**. No oracle task IDs
are supplied. It serializes/restores each ContextDetector with the existing
checkpoint helpers and checks exact post-restore predictions.

Expected mean-client accuracy: 0.2776814444 (27.7681%). Gate tolerance: 0.1
percentage point versus the original diagnostic. Per-sample class/task mismatch
counts are exported even when aggregate accuracy meets the tolerance. Review
those counts before promoting a default. Original legacy replay and serialization
restore must match exactly.

Output: `denice_multiclass_integration_diagnostics.zip`, containing
`integration_gate.json`, per-client metrics, paired predictions and serialized
router states. A failure still packages partial output; it is not a completed
result. This notebook has not yet been run locally or on Kaggle.

## Normal pipeline API

New mode: `denice_router_mode="multiclass_balanced"`. The existing binary_cosine
default and legacy multiclass mode are unchanged. This mode uses all nonempty
saved binary episode memories without extra feature masking, balanced logistic
regression, C=1, lbfgs, max_iter=1000, random_state=0. A one-task bank predicts
that task explicitly; empty/incompatible banks and fit errors fail visibly.
It uses sklearn.predict for task IDs to match the original frozen diagnostic.

The normal ContextDetector.train_models refresh path fits this router. Existing
multiclass_router/multiclass_episodes snapshot fields persist the estimator,
including when the normal training checkpoint stores the detector. No optimizer,
classifier, adapter, class mask, graph or aggregation behavior is modified by
this opt-in setting. Encoder drift remains a limitation to assess in streaming.

For a later authorized training comparison, merge
`configs/denice_multiclass_router_override.json` into the existing config, or set
`DENICE_CONFIG_OVERRIDES` to `{"denice_router_mode":"multiclass_balanced"}` in
the current Kaggle training launcher. This is not an instruction to start full
training before the frozen reproduction gate passes.

The generic `eval_checkpoint.py --router-mode multiclass_balanced` entry point
also accepts this mode. Use the dedicated notebook for the exact 98-client,
50,000-sample reproduction protocol; other CLI evaluation modes may use different
sampling or aggregation conventions.
