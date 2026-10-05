# Gate V2: peer-supported calibration, fitting and validation

Run `eval_denice_competence_gate_v2_kaggle.ipynb` on Kaggle with Internet and the original 100-clients dataset. Existing Drive links download results (4), checkpoint `03b9b53` task 5 round 19, and the original TIP reference. No V1 gate ZIP upload is required.

## Controlled change

V1 improved locally supported samples but lost accuracy on receiver-uncovered samples. V2 changes **the gate data support**, keeping:

- The same directed positive-alpha graph and fixed random candidate seed 42.
- Budgets k=4/8/16 plus self, using nested prefixes.
- Frozen donor weights, local adapters, local masks and balanced multiclass routing.
- The same 56 gate features, LR/MLP architecture, regularization, training iterations, seeds and six learned decision policies.
- Primary learned policy selection by validation accuracy, then macro-F1, then alphabetical policy, before test inference.

No alpha tuning, donor exclusion, additional budget or backbone training is introduced.

## Data support and provenance

For each receiver, the permitted training origins are **self + its fixed k=16 candidate prefix**. This maximum-budget origin bank is reused across the three budget experiments; k=4/8 gate features/actions still involve only their respective candidate prefixes. Additional origins supply labeled historical gate-training examples, not extra inference predictions. This is disclosed as a shared retrospective diagnostic, not a budget-limited online data exchange protocol.

Each origin's classes must belong to a task recorded in its CGoFed completed-task state or its last local router refresh, and intersect that origin's own mask. A receiver's inherited memory cannot grant an origin permission to contribute data from an untrained task.

Each origin train NPZ is loaded once into bounded per-role/per-class banks. Scan cap remains max(20,000, 20 × total receiver row cap). Bank caps per class are min(role receiver cap, max(32, 4 × ceil(role receiver cap / origin supported class count))). This is a deterministic memory/IO limit, not tuned on metrics. Coverage artifacts disclose missing classes and shortfalls.

Receiver calibration/fit/validation sampling cycles uniformly over available classes in the training-supported candidate-origin union, with randomized choice within class. Receiver-local mask membership is recorded for diagnosis but never used as a ground-truth-dependent gate switch. The observed 36.09% uncovered test fraction is not used to choose a mixture ratio. No test label or class-frequency estimate informs sampling.

The per-receiver caps remain 128 calibration / 512 fit / 256 validation. Exact float32 input hashes fix roles globally across client shards. Equal input content cannot cross roles; duplicates within a receiver are skipped. Exact development-panel inputs are excluded without reading their labels. Training content with conflicting labels in the scanned banks is excluded and counted. Sampling fails when fewer than 32 usable rows remain in any receiver role.

`gate_data_provenance.csv` records receiver, origin, original row ID, role ordinal, label/task, input hash, origin task evidence and receiver local coverage. `gate_origin_audit.json` records supported origin classes/evidence and source-bank counts. `gate_support_coverage.csv` records represented/missing classes and local-covered/uncovered counts in every receiver role.

## Validation comparisons

All methods use the same new validation inputs and fixed inference candidate pool:

| Method | Decision |
| --- | --- |
| Self | Receiver prediction |
| Majority | Equal one-hot votes |
| GlobalDonorPrior | Highest calibration donor accuracy with smoothing |
| GlobalTaskPrior | Highest global donor competence for its predicted task |
| ReceiverTaskPrior | Highest receiver-donor competence for its predicted task |
| LR/MLP top1/top2/top4 | Same learned policies as V1 |

Prior baselines use only the separate calibration partition and predicted tasks. No true task ID is an inference input. They share the existing fixed smoothing strengths (20 global-task toward donor; 10 receiver-task toward global-task). They are reported separately and do not replace the primary learned gate after looking at test outcomes.

Role targets and validation predictions are persisted so metrics, calibration tables and gate decisions can be recomputed from artifacts. `frozen_gate.joblib` and `gate_lock.json` record schema, selected policies, sklearn version and artifact hash before test expert inference; the saved artifact is restored for evaluation. The primary gate is selected among the original six learned policies, so majority remains an explicit comparator even when it wins validation.

## Evaluation and decision

The output is `denice_competence_gate_v2_diagnostics.zip`. It includes the existing gate outputs plus provenance, coverage, role targets, validation predictions, validation coverage-conditioned metrics, prior baselines and PNG/PDF budget figures. `gate_completion.json` must be true. The exact prior self baseline is required on 98 clients and the same 50k development panel.

Use the user's declared interpretation: >=45% makes the selector direction worth retaining; 47–48% motivates a subsequent refinement; >=50% meets the frozen-checkpoint development target. If V2 remains about 42–43%, stop refining the same simple gate architecture and consider a different meta-ensemble or training changes. Do not promote the test-best policy or budget retrospectively.

This panel is now a diagnostic/development panel. After fixing the design, lock a new untouched evaluation panel and confirm across independent training seeds for final paper claims. Gate calibration/validation rows may have been used by the original backbone; V2 is not streaming/replay-free learning. All raw data and expert models reside in the same diagnostic runtime; receiver-side model caching and its communication/memory costs remain deployment work.

## Local verification

Synthetic tests exercise peer-origin sampling, exclusion of inherited-only task access, exact-panel content exclusion, conflicting training labels, globally disjoint content roles and rejection of illegal origins. Changing all test labels leaves selected gate-data provenance/labels unchanged. A mocked-expert end-to-end V2 run fits both unchanged model families, locks/restores the gate, verifies validation comparisons from saved predictions and emits complete artifacts. V1 tests continue to cover gate-fitting independence from test labels. This does not establish accuracy on the original Kaggle dataset; the full V2 experiment remains to be run.
