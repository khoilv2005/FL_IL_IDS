# CANDLE V4: predicted-task-scoped fusion (protocol v5)

Fix the mismatch between task-conditional expert learning and unconditional expert overrides. Native V1 weights, binary_cosine router, local episode class masks, graph and training remain unchanged. Expert learning still uses 100% original current-task TRAIN and coverage-aware gradients; no validation split, acceptance gate or historical data replay.

During a single native inference pass, retain both its class prediction and input-predicted task. Only ACTIVE experts whose stored birth_task equals that predicted task may compete. Keep threshold 0.5 and existing normalized-excess arbitration with sorted class-ID tie-breaking. If there is no eligible trigger, return native prediction unchanged. A native router with no profile yields task sentinel -1 and cannot enable an expert. Never derive routing decisions from labels, caller-selected true tasks or the current training task.

The expert target, birth_task, receipt version, preprocessing identity and weight hash must agree. Malformed route arrays or modified expert state raise errors. Experts from past tasks remain frozen and ACTIVE, but participate only for input routed to their birth task. Model/router/optimizer state must not change during inference.

Version: `candle_v4_predicted_task_scoped_v5`. Fresh training only for the new notebook: old direct-fusion-v4 checkpoints are rejected, rather than silently relabeled. This is an inference design change, not an accuracy claim. It sacrifices unconditional router bypass. No new real TEST evaluation or full training is authorized as part of this implementation.

Console logging: one compact V4 summary at round 0 and rounds 4, 9, 14, 19; no per-sample or per-expert prints. Terminal progress every 10 clients and at completion. Terminal JSON keeps aggregate rows, eligible triggers, multiple triggers and actual overrides. Existing detailed receipts remain in files, not console. Critical correctness failures remain visible. Native V1 logs are unchanged.

Regression uses synthetic inputs only: outside-task score 1.0 cannot override; matching-task arbitration, no-trigger/no-router fallback, malformed routes, metadata/hash errors, immutable inference, real optimizer updates, old-expert retention, checkpoint/resume and native-only terminal evaluation. No real dataset is opened for this fix.
