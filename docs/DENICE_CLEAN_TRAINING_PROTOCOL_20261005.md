# Clean DeNICE + CGoFed + peer Class Meta on Kaggle

## Entry point

Upload **`train_denice_cgofed_clean_kaggle.ipynb`** to a new Kaggle notebook. Enable GPU and Internet, and attach the original 100-client dataset. The equivalent standalone entry point is `train_denice_cgofed_clean_kaggle.py`.

The notebook clones latest GitHub main without a pinned commit. Default dataset path is `/kaggle/input/datasets/khoilv2005/100-clients/100-clients`; a unique matching dataset is discovered if Kaggle changes the mount path. Set `DENICE_DATA_DIR` explicitly if multiple datasets match.

Default run: seed 42, fresh backbone, tasks 0..5, 20 rounds/task, followed automatically by clean Gate V2/Class Meta fitting and whole-test evaluation. First complete and inspect this seed before starting the replication campaign. The notebook installs scikit-learn 1.6.1 before importing the fitting code.

## Fixed training formulation

- `denice_clustering_mode=paper`, configurable similarity threshold, beta 0.5. The updated launcher defaults to **0.8**, preserving the requested local edit; the previous baseline used 0.5.
- Internal K=1 is a grouping sentinel; global cluster K is reported as N/A.
- Peer neighborhoods remain threshold-driven and dynamic. AP and top-k graph pruning are disabled for this main configuration.

Set `SIMILARITY_THRESHOLD` near the top of the launcher/notebook, or set `DENICE_SIMILARITY_THRESHOLD` before execution. This single value drives config and logging. The validator accepts finite values in [0,1] and checks before the expensive role split; it no longer forces 0.5. Regenerate the notebook after editing the Python entry point. Resume must retain its original xi; a different xi requires fresh training. Clean role indices may be reused across xi experiments with the same dataset/split seed.
- Preserve existing CANC/allocation/adapter configuration, local CGoFed projection and AMP fix; mature-head peer projection remains disabled.
- Training router remains the existing binary-cosine configuration. Clean balanced-multiclass/Gate/Meta inference is a subsequent fitting stage, not a hidden backbone change.
- Meta budget 16 is recorded separately as a candidate pending clean validation, not a training neighborhood size or an already-selected new-seed policy.

New round diagnostics include peer mean/median/P90/max, fraction of receivers without peers, edge density, positive-alpha peer counts and a dense state-dict transfer-byte estimate. Candidate-neighbor counts and positive-alpha counts are separate. The transfer estimate excludes compression, masks, capsules and transport overhead; it is not measured network traffic.

## Data roles created before backbone training

`tools/prepare_denice_clean_roles.py` scans only original `client_<id>_train.npz` shards and assigns globally content-disjoint roles, stratified by class:

| Role | Requested unique-content fraction | Use |
|---|---:|---|
| BASE | 80% | Backbone training and existing BASE-local calibration |
| META calibration | 2% | Future competence priors |
| META fit | 8% | Future Gate/Class Meta fitting |
| VALIDATION | 10% | Evaluation and later policy selection |

Small-class quotas are adjusted to retain at least one unique input in each role, so ratios are approximate for rare classes. Equal float32 input content across clients always gets one role. Inputs with conflicting labels are excluded. Raw files are not copied; compressed row-index views and source checksums are persisted in `/kaggle/working/denice_clean_roles`.

The existing 10% local validation split *inside BASE* is retained for paper capsule/CANC behavior. It is separate from the external 10% VALIDATION role; META/VALIDATION rows never enter backbone clients or their context memory.

Preflight stops before training if any of classes 0..33 lacks:

- Unambiguous content in BASE, META calibration, META fit or VALIDATION.
- A scheduled client with BASE support for that class.
- A locally BASE-supported origin supplying each holdout role for that class.

This is input and scheduling coverage, not proof an expert will correctly predict each class. Reachable-candidate coverage, including class 28, must be checked on clean validation after backbone training and before the multi-seed campaign. A class with zero reachable fitting targets must not silently disappear from the new selector.

The temporary SQLite content registry is deleted after successful role locking to save storage. On preparation failure it remains for diagnosis. Incomplete role directories are not silently reused; use a new role directory after correcting the dataset/split issue.

## Evaluation timing and scope (updated: full original test requested)

Backbone round evaluation, post-task evaluation and local validation evaluation are disabled. After **task 5 / round 19 and task-end consolidation**, the launcher loads the exact FP32 terminal checkpoint, fits the selectors, audits validation and locks artifacts. Only then does the final pass open **every row of the original `global_test_data.npz`**. All 34 source classes are required. Backbone task metrics record `final_test_evaluated=false`; the authoritative final result is `clean_class_meta/completion.json`, not the skipped backbone metrics.

The test set is split into disjoint, approximately equally sized receiver shards. Class-stratified round-robin assignment preserves the **original global class counts**, without class balancing/subsampling of the source test. Every row is assigned once. Feature shards are lazy indexed views of the shared test tensor: only one batch is materialized for a receiver, avoiding a second full feature copy. Test labels enter benchmark partitioning and metrics, not the predictor. Both global test features and row indices still need CPU RAM; this is not an out-of-core NPZ reader.

The role-preparation manifest's `final_test_read=false` applies only to preparation of BASE/META/VALIDATION, which does not read test. Selector fitting also does not read test. No prior development-panel exclusions or 50k cap are applied: this run evaluates the whole supplied original test source.

## Automatically fitted frozen inference method

The recipe follows the historical method that obtained **58.358% on a 28-class confirmation panel**. New training, clean fitting roles and the full 34-class test change the experiment; 58.358% is not an expected or guaranteed new result.

1. Restore every terminal expert with its own weights, adapters and local class masks. Fit `multiclass_balanced` from its BASE-derived binary context memory, without changing model weights.
2. Read the actual task-5/round-19 directed graph. Only positive-alpha peers are eligible; include self and take 16 peers in the fixed random ordering `seed=42 + receiver_id*1009`. Require all 17 experts for each receiver. Alpha remains a feature, not the deciding vote weight.
3. Build class-balanced peer-supported calibration/fit/validation pools, capped respectively at 128/512/256 rows per receiver. Read only the locked role indices. Origins must have recorded local participation and BASE class support; inherited binary memory alone does not authorize historical data access. Persist origin and row/content provenance.
4. Estimate competence priors on calibration only. Fit Gate V2 on fit only: `StandardScaler + MLP(32,16)`, alpha 0.001, batch 1024, 50 iterations, seed 20261005, top-1 decision.
5. Fit class-level evidence using the frozen gate: `StandardScaler + LogisticRegression(C=0.1)`, 1000 iterations, seed 20261005. Only reachable fit targets are used, as in the historical recipe. **Require reachable fitting targets for every one of 34 classes**, including class 28, or stop before test with a coverage artifact. Gate and Meta share the dedicated fit role; this is not cross-fitting.
6. Audit Self/majority/Gate/Meta on clean validation. The primary recipe is predeclared; validation does not promote a different test winner. Save and reload `frozen_pipeline.joblib`, then write `pipeline_lock.json` with checkpoint/role/artifact checksums, schemas, candidate ordering and versions before opening test.
7. Query self + 16 cached experts for each test batch, then run the frozen label-blind selector. `predict_records()` accepts expert evidence and the frozen bundle; it has no sample-label or true-task argument. Truth is read by scoring code after predictions. The final class is restricted to classes actually predicted by those 17 experts; self wins exact ties, then smallest class, with majority fallback.

All four policies are scored in the same final pass with the same candidates, without refitting or test selection. Report pooled accuracy, 34-class macro-F1, per-class metrics, per-client metrics, confusion matrices and actual-routed oracle as a diagnostic. The method is **DeNICE/CGoFed + self/16-peer frozen Class Meta**, not single-client DeNICE accuracy.

Expert models are restored and cached locally; this implementation does not send raw samples over the network. It bounds GPU residency to 17 models and writes compressed final predictions incrementally, avoiding a full test expert-feature cache. Model query cost is 17 per sample; the existing evidence collector also performs routing/confidence computation. Full test features and row indices still require host RAM, and the whole 13.5M-row test can take substantially longer than a 50k panel. This script uses one CUDA device; it does not distribute work across both T4 GPUs.

Because the original test dataset has already supplied development panels, it is not a newly untouched test source. A separate fresh source remains necessary for that scientific claim.

## Checkpoint storage

- Save a delta checkpoint **every round**: all 120 round snapshots over six tasks.
- Immediately compress each PT into a separate atomic ZIP. Verify SHA256/CRC before removing that PT.
- At each task boundary, create `checkpoint_task_<t>_all_rounds.zip`, containing task base, all 20 round deltas, the full terminal model checkpoint and the full continuation state.
- Stream the merge, verify all members, atomically seal the task ZIP, then remove the already-verified individual archives and terminal PTs. Never delete a delta dependency without putting it in the sealed task archive.
- Existing delta snapshots use fp16 bases/deltas; this existing compact format is approximate. Full terminal checkpoints remain fp32. Loading a sealed task ZIP defaults to its full terminal checkpoint, suitable for subsequent clean selector fitting. Specify a round member to reconstruct a particular compact round snapshot.
- A 12 GiB results-directory budget raises a clear error after safe compression if exceeded, retaining stored checkpoints. This is a configurable safety budget, not a guarantee about Kaggle quota or compression ratio. Raw checkpoint writes and merging need temporary headroom. Role indices and other working-directory files are additional storage.

Historical output: task-5 round PTs were about 424 MiB each, versus roughly 72 MiB ZIP-compressed. This supports immediate compression; a new training run may compress differently.

Replacing the old four saved rounds/task with twenty, while assuming the old compression ratios for every file, projects about **6.41 GiB** for the historical whole output. This is only a sizing estimate, not a promised size of the clean run.

`checkpoint_index.json` includes the archive/member location for each round. A task archive is self-contained for its saved snapshots. An interrupted task leaves individual verified base/round ZIPs.

Exact training continuation is supported at **completed task boundaries**, using the full lifecycle/RNG continuation state. Round snapshots are available for model recovery/evaluation; this change does not implement exact mid-task continuation from a round delta alone.

## Output and recovery

Default outputs:

```text
/kaggle/working/denice_clean_roles/
    role_manifest.json
    role_lock.json
    indices/client_<id>.npz
/kaggle/working/results_denice_cgofed_clean_seed_42/
    checkpoint_task_0_all_rounds.zip
    ...
    checkpoint_task_5_all_rounds.zip
    checkpoint_index.json
    config.json / metrics and debug history
    clean_class_meta/
        frozen_pipeline.joblib / pipeline_lock.json
        meta_provenance.csv.gz / origin_support.json
        calibration_coverage.json / fit_coverage.json / validation_coverage.json
        reachable_fit_coverage.json / validation_metrics.json
        final_predictions.csv.gz / confusion_matrices.npz
        client_metrics.json / completion.json
```

Keep both role artifacts and all task ZIPs. To resume after a completed task, attach that task ZIP and the original locked role folder, then set `DENICE_CLEAN_RESUME_ARCHIVE`, `DENICE_CLEAN_ROLES_DIR`, and a new `DENICE_OUTPUT_DIR` before the notebook cell. Resume verifies the same role-manifest checksum and training seed; legacy checkpoints are rejected. The original source dataset path must remain available.

For independent fresh seed 43 or 44, set `DENICE_SEED` before execution; role split seed remains 20261006 for all seeds. Do not start those replications until clean selector fitting/coverage and the final evaluation protocol have been settled.

If selector fitting fails after backbone completion, retain the clean role folder and task-5 archive. After resolving the reported pre-test coverage issue, selector fitting can be invoked without repeating backbone training, using a new output directory:

```bash
python -m tools.train_denice_clean_meta --checkpoint-archive /path/checkpoint_task_5_all_rounds.zip --role-dir /path/denice_clean_roles --output-dir /path/new_clean_class_meta
```

Use the same original dataset mount and scikit-learn 1.6.1. Do not interpret a partial ZIP/directory as a result: `completion.json` must say `completed=true`. The checkpoint budget covers checkpoint storage; role artifacts, frozen selectors and compressed predictions additionally consume working-directory space.

## Validation performed during preparation of this change

Python syntax/AST checks passed for the new modules and modified runner/loader, and the notebook was generated from the entry point. No full GPU training or functional dataset run was executed locally; the complete raw federated dataset is not available in this workspace. Runtime role/coverage checks are therefore mandatory before the Kaggle training loop starts.
