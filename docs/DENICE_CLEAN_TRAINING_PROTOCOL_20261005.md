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

## Evaluation timing and scope (updated: full original test requested)

Only **task 5 / final round 19** triggers evaluation, after task-end consolidation. This avoids evaluating a pre-consolidation state and then reporting a different terminal checkpoint. Earlier tasks/rounds write checkpoints without evaluation. At the user's explicit request, the final pass uses **every row of the original `global_test_data.npz`**, with `denice_eval_max_samples=None`. The role loader asserts that the full source test contains all 34 declared classes. The terminal checkpoint and task metrics identify `evaluation_data_role=test` and `final_test_evaluated=true`.

The test set is split into disjoint, approximately equally sized receiver shards. Class-stratified round-robin assignment preserves the **original global class counts**, without class balancing/subsampling of the source test. Every row is assigned once. Feature shards are lazy indexed views of the shared test tensor: only one batch is materialized for a receiver, avoiding a second full feature copy. Test labels enter benchmark partitioning and metrics, not the predictor. Both global test features and row indices still need CPU RAM; this is not an out-of-core NPZ reader.

The role-preparation manifest's `final_test_read=false` applies only to preparation of BASE/META/VALIDATION, which does not read test. The actual terminal evaluation opens test and records its role in task metrics. No prior development-panel exclusions or 50k cap are applied: this run evaluates the whole supplied original test source.

META and VALIDATION remain reserved. This notebook does not fit Gate/Class Meta or use test outcomes to select a policy. Subsequent clean selector fitting must use its dedicated roles and validation; old retrospective artifacts must not be reused as the new clean-trained result.

This evaluates the original dataset's test set for the backbone. Because that dataset has already supplied development panels, it is not a newly untouched test source. A separate fresh source remains necessary for that scientific claim.

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
