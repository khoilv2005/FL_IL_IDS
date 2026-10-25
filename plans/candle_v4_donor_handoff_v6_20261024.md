# CANDLE V4 current-graph donor handoff — protocol v6

Fix the starvation observed for receivers9/71, class12: donor10 left their graph after round0; the experts remained at44 updates despite alternate current-task donors. Do not hardcode these IDs/classes or select donors using future graph/TEST outcomes.

At each native round, keep the current donor if it is still an active positive-alpha peer with ≥32 positive current-TRAIN samples and an authorized receiver+donor negative pool. Otherwise choose the smallest currently eligible donor ID. If no donor is eligible, pause training and accumulate the existing quota; do not train with an absent peer. Do not switch just because a lower-ID donor becomes available.

Handoff retains the receiver-owned expert model, Adam objects/moments/step counters, pending quota, byte counters and input/preprocessing reference. Only the selected donor in the engine spec changes. Both donors teach the same model coordinates. The existing44 updates/round and88 catch-up cap remain; no extra training budget, replay or quality gate is added. Past-task experts are frozen and never handed off using new-task data.

Each entry stores initial_donor plus chronological handoff lineage (round/from/to/current graph SHA256). Each round receipt records actual donor authorization and counts. Training proof follows that lineage and accumulates updates per real donor, rather than falsely requiring every old update to come from the final donor. Checkpoint/restore recreates the engine with its latest donor. Receipts and inference require matching donor/target/task/input/weight metadata. Historical graph snapshots are not used to choose live donors.

Version `candle_v4_current_graph_handoff_v6` rejects earlier protocol checkpoints for training resume. The launcher supports binary_cosine and multiclass_balanced explicitly and binds router choice to the saved config/spec. The new V4 notebook selects multiclass_balanced, matching the stronger checkpoint development readout. V1 notebook remains pinned to its historical binary baseline. This router selection is development-informed, not independent scientific confirmation.

Keep100% original current-task TRAIN, validation0%, no BASE/FIT/CAL, no shadow native model, scoped fusion at threshold0.5 and native fallback. No full training or new TEST readout is run for this implementation.

Logging remains compact: V4 console summary on round0 and every5 rounds, including cumulative handoffs for the task. No per-expert/per-sample messages. Handoff details and per-round authorization go to the existing checkpoint/training-history JSON. Metadata counts are cached; handoff selection does not repeatedly scan training tensors.

Technical regression: graph departure/replacement, retained donor/no churn, missing/too-small alternate, pauses and88-step catch-up, exact optimizer/weight/quota/provenance resume before and after handoff (wall-clock timing excluded), old-task freeze, tampered lineage/graph/count rejection, native isolation, streaming multiclass fit+restore, locked router config, and existing activation/fusion/terminal tests. All use synthetic data.

The prior42.91% Accuracy /23.04% Macro-F1 was measured with frozen existing experts and multiclass-scoped inference before handoff training. It is not a result of this new implementation. Handoff may improve learning continuity; its accuracy effect is unmeasured. The45.52% routing ceiling of that checkpoint remains a separate bottleneck.
