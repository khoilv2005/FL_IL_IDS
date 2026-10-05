# Clean DeNICE + CGoFed backbone training on Kaggle

## Entry point

Upload **`train_denice_cgofed_clean_kaggle.ipynb`** to a new Kaggle notebook. Enable GPU and Internet, and attach the original 100-client dataset. The equivalent standalone entry point is `train_denice_cgofed_clean_kaggle.py`.

The notebook clones latest GitHub main without a pinned commit. Default dataset path is `/kaggle/input/datasets/khoilv2005/100-clients/100-clients`; a unique matching dataset is discovered if Kaggle changes the mount path. Set `DENICE_DATA_DIR` explicitly if multiple datasets match.

Default run: seed 42, fresh backbone, tasks 0..5, 20 rounds/task. This is full backbone training, not a smoke run and not fitting the old frozen selector. First complete and inspect this seed before starting the replication campaign.

## Fixed training formulation

- `denice_clustering_mode=paper`, similarity threshold **0.5**, beta 0.5.
- Internal K=1 is a grouping sentinel; global cluster K is reported as N/A.
- Peer neighborhoods remain threshold-driven and dynamic. AP and top-k graph pruning are disabled for this main configuration.
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

## Evaluation timing and scope

Only **task 5 / final round 19** triggers evaluation, after task-end consolidation. This avoids evaluating a pre-consolidation state and then reporting a different terminal checkpoint. Earlier tasks/rounds write checkpoints without evaluation. The final pass uses clean **VALIDATION**, with up to 50k stratified rows and 34-class coverage asserted by the role loader. The terminal checkpoint and task metrics identify `evaluation_data_role=validation`; legacy metric filenames/aliases remain for compatibility.

The original `global_test_data.npz` is **not opened** by this launcher/role loader. Final test coverage is marked unverified. Gate/Meta must be fitted on the reserved roles, selected using validation and frozen before a separate label-blind final test. This notebook does not yet perform that subsequent fitting stage. Old retrospective Gate/Meta artifacts must not be reused as the new clean-trained result.

A fresh independent full-34-class final test source is still needed for an untouched final claim. Do not reinterpret the original development/28-class confirmation panels as a new untouched final test.

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
```

Keep both role artifacts and all task ZIPs. To resume after a completed task, attach that task ZIP and the original locked role folder, then set `DENICE_CLEAN_RESUME_ARCHIVE`, `DENICE_CLEAN_ROLES_DIR`, and a new `DENICE_OUTPUT_DIR` before the notebook cell. Resume verifies the same role-manifest checksum and training seed; legacy checkpoints are rejected. The original source dataset path must remain available.

For independent fresh seed 43 or 44, set `DENICE_SEED` before execution; role split seed remains 20261006 for all seeds. Do not start those replications until clean selector fitting/coverage and the final evaluation protocol have been settled.

## Validation performed during preparation of this change

Python syntax/AST checks passed for the new modules and modified runner/loader, and the notebook was generated from the entry point. No full GPU training or functional dataset run was executed locally; the complete raw federated dataset is not available in this workspace. Runtime role/coverage checks are therefore mandatory before the Kaggle training loop starts.
