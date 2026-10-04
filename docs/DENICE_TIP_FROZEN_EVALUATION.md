# Frozen router experiment

## Run on Kaggle

Upload **`eval_denice_tip_router_kaggle.ipynb`**, attach the existing 100-clients
dataset, enable Internet, and run all cells. The notebook clones latest main and
downloads the original `results (4).zip` from the configured Google Drive URL.
It restores task 5 round 19 of training commit `03b9b53`. No training runs.

The original full-training launcher is not the launcher for this experiment.
Download `denice_tip_diagnostics.zip` after completion. Checkpoints themselves
are excluded from this output ZIP. Per-client progress and partial outputs remain
under `/kaggle/working/denice_tip_03b9b53` if a later client fails.

## Protocol

1. P0 restores the 98 original evaluation clients and original 50,000 stratified
   test shards. Mean-client accuracy must be within 0.1 percentage point of
   22.79%. A mismatch stops execution before fitting a new router.
2. P1 uses recorded local CGoFed tasks plus the last local router refresh as
   participation evidence, intersected with local class support. It loads only
   that client's original train file, splits per class, caps fit samples at 512
   per task and validation at 128, and captures adapter-free fc1 features.
   Missing participation evidence fails explicitly. Profile row IDs are saved.
3. Continuous centroid, pooled shrinkage Mahalanobis, and TIP use the same
   features and split. Mahalanobis lambda is selected from 0.1/0.01/0.5; TIP rank
   from 32/16/64, independent/residual bases and RMS/mean reference variants.
   Selection is per-client task-balanced validation accuracy. Ties favor the
   first/default candidate. Without validation rows defaults are retained and
   the manifest records null validation scores.
4. P2 evaluates the selected routers with unchanged adapter/local-mask behavior.
   The existing oracle executor is reused with **predicted** task IDs for learned
   routers; no ground-truth task enters their prediction. True IDs are passed
   only for the two explicit Oracle policies.

Multiclass uses saved binary memory and therefore has a different fitting data
budget; it is a secondary representation baseline, not the controlled comparison
between the three continuous routers. It fits balanced logistic regression with
fixed C=1 and max_iter=1000, without test-set tuning.

Historical train data may already have been seen by the frozen backbone. The
validation split is a router holdout only. This experiment is a retrospective
router refit, **not a replay-free streaming result**. Duplicate flows can occur
across row-based splits; the manifest permits auditing this limitation.

## Source relationship

Reviewed upstream `seohyeon-cha/FedProTIP` commit
`54193fa2d44f6203f39299a0ac3845097559a440`:

- [client/client_fedprotip.py](https://github.com/seohyeon-cha/FedProTIP/blob/54193fa2d44f6203f39299a0ac3845097559a440/client/client_fedprotip.py):
  activation SVD and `compute_references`.
- [server/server_fedprotip.py](https://github.com/seohyeon-cha/FedProTIP/blob/54193fa2d44f6203f39299a0ac3845097559a440/server/server_fedprotip.py):
  subspace relevance and reference matching in `_eval_cnn`.

This is an independent adaptation, not copied FedProTIP training. Upstream
computes a task for a batch and includes client voting and a special two-task
rule. Here each sample has its own relevance vector, all available task columns
are retained, references are local, and no peer voting is enabled. Independent
bases and RMS moment references are explicit ablations. SVD uses the feature
columns of a sample-by-feature matrix; no upstream gradient projection replaces
local CGoFed. Upstream CIFAR task-prediction numbers are not expected IDS scores.

## Outputs and interpretation

- `protocol.json`: commits, checkpoint hash, data access, masks, feature policy.
- `baseline_guard.json`: P0 observed accuracy and pass/fail.
- `profile_manifest.json`: participation evidence, fit/validation row IDs and
  validation candidate scores selected before that client's test evaluation.
- `per_client_metrics.csv`: raw routing numerators/denominators, class coverage,
  task availability, classification metrics and error decomposition.
- `predictions.csv`: paired sample predictions, actual global test row IDs and
  predicted task IDs (Oracle IDs are intentionally not counted as learned routing).
- `summary.csv`, `per_task_metrics.csv`, `route_confusion.csv`: final comparisons.
- `gate.json`: mean-client accuracy gain >=2pp, pooled macro-F1 drop <=0.5pp,
  paired client bootstrap CI lower endpoint >0; separate TIP-vs-continuous CIs.
- `router_diagnostics.json`, `profiles/`: selected settings, rank, timing, moments,
  means, bases, references, scores and model fingerprint. Profiles are external
  diagnostic artifacts, not injected into training checkpoints.

Runtime checks reject non-finite features, mutated model state, and prediction
changes on batch-size/order probes. These checks do not establish invariance on
every possible input. Empty/zero relevance ties choose the smallest fitted task
ID deterministically. Model/BN/mask fingerprint changes invalidate the bank.

P0/P1/P2 are implemented here. Streaming encoders, graph changes and peer TIP
remain future gated work. Full Kaggle accuracy has not been measured by this
implementation until the notebook completes on the mounted dataset.
