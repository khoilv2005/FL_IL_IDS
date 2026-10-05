# Retrospective competence gate on frozen experts

Run `eval_denice_competence_gate_kaggle.ipynb` with the original 100-clients Kaggle dataset and Internet enabled. Existing Drive links download results (4), checkpoint `03b9b53`, and the original TIP reference. No peer-voting ZIP upload or new Drive URL is required.

## Declared design

- Directed candidates must have positive recorded graph alpha at task 5 round 19; self is included. Random candidate seed 42 is fixed, with nested k=4/8/16 prefixes. This seed is not the best previously observed seed.
- Each receiver supplies historical rows only from its own train file and tasks with recorded local participation (CGoFed completed-task or last local router-refresh evidence), intersected with its local class support. Inherited binary memory does not authorize historical task access.
- Per receiver, caps are 128 calibration rows, 512 gate-fit rows and 256 validation rows. A deterministic content hash assigns roles globally across client files; duplicate input content cannot cross roles. Exact float32 input duplicates of the 50k test panel are excluded, without using test labels. This is not an audit of overlap with the entire 13.5M test pool or original backbone fitting.
- Sampling scans a bounded, class-balanced set of original local rows. Counts can be below caps; a role with fewer than 32 usable rows fails explicitly. Provenance includes client-local row IDs and content hashes.
- Frozen donor-local balanced multiclass routers, adapters and masks produce predictions and confidence features. Empty-mask fallback is refused. Model fingerprints match the existing reference and are checked after inference.

## Gate inputs and targets

Features include predicted task/class one-hots, task-router probability, masked classifier top probability/margin/normalized entropy, mask size, task/class support, agreement with self and candidate consensus, aggregation alpha, self indicator, and smoothed competence estimates. No raw input vector, true task ID or sample label is an inference feature. Candidate donor IDs are used for calibration lookup and deterministic ties, not as learned one-hot identity features.

Competence estimates are obtained only from the calibration partition. Global donor-task rates shrink toward global donor accuracy (strength 20); receiver-donor-task rates shrink toward global donor-task rates (strength 10). They are fixed before gate fitting/evaluation. The fit target is whether the donor's ordinary routed prediction equals the gate-fit label.

The shared gate benchmarks fixed Logistic Regression (C=1, max_iter=1000) and MLP (32/16 hidden units, alpha=.001, batch=1024, 50 fixed iterations, seed 20261005), preceded by standard scaling. No parameter search on test data. The MLP may emit a convergence warning at its fixed iteration limit; the saved model is still evaluated, and the warning is not a successful convergence claim.

For each family, evaluate top-1 selection and competence-weighted top-2/top-4 votes. Selection-score ties favor self, then smaller donor ID. Vote ties follow the existing self-first rule. Query cost is still k+1 donor models, even if only one or a few predictions are selected after all confidence features have been collected.

The primary `ValidationSelected` policy for each budget is chosen by validation pooled accuracy, then macro-F1, then alphabetical policy. Both gate families and all predeclared policies remain reported. Gates, calibration tables, feature schema and validation selections are persisted in `frozen_gate.joblib`; `gate_lock.json` records its SHA256 before any test-expert inference. Test labels are used only afterward for metrics and actual-routed oracle coverage. Self test predictions must exactly reproduce the prior reference.

## Scope and deployment assumptions

This is a **shared gate fitted centrally in a retrospective diagnostic runtime**. It revisits historical training shards and aggregates calibration/fitting records across receivers. Validation rows are held out from gate fitting, but may have been used to train the frozen backbone. It is not a streaming replay-free gate, independent backbone holdout, or implemented decentralized gate training protocol. No remote peer is contacted with a raw sample: frozen donor models are loaded and run locally in the notebook. Receiver-side caching of peer model weights/adapters is a possible deployment design, whose memory/communication costs remain unmeasured.

The panel has been used for prior research diagnostics. A final paper performance claim should be confirmed on an untouched evaluation split and independent training seeds. Do not tune this gate, remove donors, or pick its deployment budget using panel outcomes.

## Output

`denice_competence_gate_diagnostics.zip` includes gate provenance/split audit, fixed graph/candidate protocol, calibration/fit/validation and test expert feature caches (no raw inputs), runtime records, validation metrics, frozen gate and lock hash, per-client test predictions, summary accuracy/macro-F1, paired client-bootstrap CI versus same-budget majority, actual-routed oracle coverage, and PNG/PDF accuracy-versus-budget figures. `gate_completion.json` must be true before interpreting a run as complete. Partial archives are written when execution fails.

Success references are at least 45% at k<=8, or 50% overall, with comparison against the same fixed candidate pool and majority. These are evaluation criteria, not promised results. Backbone weights, router behavior, masks and graph are unchanged.

## Local verification

Unit checks cover content-role stability, calibration counts, feature ordering and batch invariance, label-free selections and ties. A synthetic end-to-end gate pipeline uses mocked expert outputs, fits both LR and MLP, restores the persisted gate and creates CSV/JSON/PNG/PDF artifacts. Changing all test labels leaves fitted gate predictions, calibration priors and validation selections unchanged. This does not replace running the original checkpoint and dataset on Kaggle.
