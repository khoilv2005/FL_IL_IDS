# Class-28 checkpoint diagnostic

Upload `debug_denice_class28_kaggle.ipynb` to Kaggle; enable GPU and Internet and attach the original 100-client dataset. Set `RESULTS_DRIVE_URL` to the share link for **results (8).zip**, or use the environment variable `DENICE_RESULTS8_DRIVE_URL`. The link must allow download. The notebook clones latest GitHub main and installs scikit-learn 1.6.1.

If the source ZIP is already attached, set `DENICE_RESULTS8_ZIP` to its path instead of downloading it again. The original dataset path is retained, with unique-mount autodetection and optional `DENICE_DATA_DIR` override. Relocated mounts are validated against original metadata and train-shard checksums; the locked role manifest is not rewritten.

The script extracts only role artifacts, provenance, origin support and task 4/5 archives. It does not extract the other four task archives. The downloaded outer ZIP and selected archives together need roughly 9 GB of disk, plus small diagnostic output. Start in a fresh session; existing extraction/output folders are not overwritten.

## Questions answered

- **Routing:** compare normal balanced-Multiclass predictions and label-assisted forced task-4 predictions, preserving donor-local class masks and adapters.
- **Candidate discovery:** compare self + up to 16 fixed random eligible peers against all recorded positive-alpha peers. Task 5 reproduces the existing 17-expert candidate definition; task 4 may have fewer peers and does not invent missing late-joining clients.
- **Acquisition/retention:** compare task 4 and task 5 exact FP32 terminal experts on identical saved META-fit/validation contents. The paired donor table uses only donors present in both checkpoints; donor-local mask changes are recorded separately.

There is no backbone training, no Gate/Class Meta training and no final-test read. Each checkpoint's multiclass router is fitted from its own BASE-derived binary memory. Oracle uses the true task because this is explicitly a diagnostic; it is not deployable accuracy. Backbone fingerprints are checked before and after inference.

## Outputs

`/kaggle/working/denice_class28_diagnostics.zip` contains:

- `completion.json`: completion flag, unique input counts, actual training xi and coverage by checkpoint/role/budget.
- `sample_provenance.csv`: locked role/origin/source row/content hash and the corresponding prediction-pool index.
- `donor_class28.csv`: route-task-4 counts, normal/oracle correct counts, task/class availability, oracle logit rank and margin.
- `receiver_coverage.csv`: label-assisted candidate coverage on the original saved receiver occurrences. Repeated content across receivers is not independent data.
- `paired_task4_task5.csv`: oracle successes lost/gained by common donors on the same unique inputs.
- `expert_predictions.npz`: predictions, routes, ranks and margins aligned by donor ID and pool index.

Interpretation:

- Normal fails but forced-task-4 succeeds: routing is implicated on these inputs.
- Normal succeeds only in the larger legitimate peer pool: candidate discovery is implicated.
- Oracle succeeds at task 4 and loses the same inputs at task 5: later knowledge/mask/adapter deterioration is implicated; inspect mask changes before attributing all loss to weights.
- Oracle is already poor at task 4: acquisition or representational/classifier difficulty is implicated; this audit alone does not identify a unique cause.

The scope is only saved class-28 fit/validation contents, not final accuracy or a universal ceiling. Read the result only when `completion.json` says `completed=true`. A failure still packages any partial artifacts available after the audit starts.

Preparation checks: Python/notebook syntax and source alignment only. No complete Kaggle/data/GPU execution was performed locally.
