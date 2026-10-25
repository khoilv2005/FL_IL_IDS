# Frozen CANDLE V4 + multiclass scoped-fusion debug

Purpose: compare V4 expert contribution against the completed V1 multiclass checkpoint readout, without retraining or another threshold search. This is development diagnostics on an already accessed TEST set, not independent confirmation.

Inputs: user's `v4.zip` final Task5 continuation; original TEST from `archive (1).zip`; completed 100-client V1 `router_comparison.json`. Source Python is the original run's pinned `2ec3d0e43b4cccd4ebc56a9ad78ddacdd9f41a20` package. Checkpoint, TEST and reference SHA256 values are locked in output JSON.

Fixed conditions: 100 clients, all 13,505,771 TEST rows assigned once by seed523687, batch512, no data training, no threshold fitting, no new validation split, no acceptance gate. Multiclass router is fitted only on saved activation memory. FC2, encoder, masks, adapters and expert weights remain frozen. Experts compete only when birth_task equals input-predicted task; threshold0.5, max-score/smallest-class-ID arbitration and native fallback are unchanged.

Reuse the previously verified native multiclass confusion matrix. Recompute routing on every input and require its task confusion matrix to match the native reference exactly. Execute native classification only on rows where scoped experts trigger. Update the confusion matrix by subtracting the native decisions on these rows and adding the expert decisions. The existing forced-route executor is passed the model-predicted routes, never true task labels. Stop if native subset cells exceed the reference, routing differs, model weights change, or row/count identities fail.

Outputs: pooled Accuracy/Macro-F1, per-client and per-class changes, rescue/break, per-expert attribution, immutable-input hashes, saved per-client overrides and resumable aggregate results in `audit_denice/candle_v4_multiclass_scoped_20261024/`. Console progress every10 clients; no per-sample prints.

Only after the complete result should a follow-up implementation decision be made. This audit does not silently change the current notebook's binary-router contract or launch full training.
