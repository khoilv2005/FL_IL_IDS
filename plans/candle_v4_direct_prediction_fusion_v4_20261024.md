# CANDLE V4: direct prediction fusion, no validation or acceptance gate

Protocol `candle_v4_direct_prediction_fusion_v4`. Fresh seed42, newly pinned
source. Previous v3 already used100% TRAIN and training-only receipts; this
revision removes the task-end acceptance callback from the current V4 path and
makes trained experts available directly after a round's authorized updates.

100% original current-task TRAIN → native V1 and coverage-aware expert learning
→ direct fuse(native, experts, threshold0.5) → final full34-class TEST.

- Validation fraction0.0; no BASE/FIT/CAL, split preparation, shadow predictor
  or quality checks of recall/FAR/break before expert use.
- Keep live-graph authorization, current-task data checks, actual optimizer
  steps/provenance and finite weights. Untrained/invalid experts cannot infer;
  missing validation is never a deferral reason.
- After each native round and expert updates, activate trained experts directly
  and refresh their model-hash receipts before round checkpointing. An already
  active current-task expert may keep learning; its receipt must follow the
  latest weights. Old-task experts stay frozen and usable, no historical replay.
- ACTIVE is training_only_unverified, quality UNVERIFIED, not a safety approval.
  Pre-TEST recall/FAR/break remain NOT_MEASURED/null. No TRAIN surrogate metrics.
- Keep the existing fuse() arbitration and threshold>=0.5; use native prediction
  when no expert triggers. Never merge MLP parameters into the CNN-GRU model.
- Current V4 does not call prepare_validation(), inherited legacy _gate(), or
  base finish_task() acceptance/journal flow. Legacy audit implementations remain
  in Git/base modules for historical protocols, not the current execution path.
- Keep backbone/router/graph, LR/optimizer, sampling budget and communication
  accounting. V1 native-only has no experts. Both use identical full TRAIN and
  final TEST assignment; full TEST only after T5/R19 and consolidation.
- Native/expert states, Adam, RNG, quota, current activation receipts and training
  history persist every round. Reject older protocol/source resume; do not modify
  a live training job. Run the new notebook fresh.

No pre-inference negative-transfer guarantee. Full TEST measures actual accuracy,
recall, false activation and rescue/break without retroactive acceptance or tuning.
