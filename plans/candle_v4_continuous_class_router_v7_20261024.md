# CANDLE V4 continuous_class_v1 — isolated variant

## Constraints to preserve

The user explicitly requires: do not modify CANDLE V1; keep training/evaluation comparable with the other algorithms. Keep V1 source, config and notebook intact. Do not change dataset splits, current-task filtering, backbone loss, optimizer, epochs/steps, batch, collaboration graph, expert gradients/quotas or evaluation frequency to favor this variant.

Use100% original current-task TRAIN, validation0%, no BASE/FIT/CAL, no old raw-data replay, no shadow native model, no acceptance gate. Same6tasks×20rounds, seed42, original34-class TEST, finalT5/R19 only, same disjoint receiver partition and pooled metrics. Do not average control policies into headline accuracy. No new benchmark is authorized by technical regression.

## Isolated design

Variant name `continuous_class_v1`, V4 service protocol `candle_v4_continuous_class_router_v7`. Existing v6 multiclass notebook remains default and unchanged. A separate notebook/file/output is provided for the new variant; no automatic binary checkpoint conversion or old-protocol resume.

The original ContextDetector, native TRAIN sampler, routed-inference executor and native checkpoint helpers remain byte-for-byte unchanged. V4-only adapters wrap the existing native sampler at its two V4 orchestration call sites. They capture activations from forward calls the native sampler already performs. Class labels follow the existing class-chunk order/counts; no second sample draw, no extra TRAIN rows, no extra encoder forward.

Native binary memory, multiclass control, capsules and graph remain unchanged. A separate receiver-local feature memory stores selected FP32 activations and class/task identity, capped by the same router_reference_per_class. It keeps no raw input. Historical feature matrices are retained; only current-task matrices are replaced. Stable feature indices/layout are locked. No extra freezing or network capacity is allocated.

Router fitting: StandardScaler + class-balanced LogisticRegression, C1/max_iter1000/lbfgs/random_state0, fit saved feature memory only. It predicts class then maps to task. Per-task compatibility scores are normalized maxima of class probabilities, NOT calibrated task posteriors. This represents a task as a union of class regions while preserving activation amplitude. Router fitting adds CPU computation; native CNN/GRU optimizer training is unchanged. Fit time is included in the existing context-fit accounting.

Continuous feature memory/scaler/classifier arrays are serialized only inside V4 service state, not native V1 checkpoint state. Missing/corrupt continuous state or changed feature coordinates raise errors. Binary sketches cannot recreate continuous historical activations.

Inference uses a V4-only detector view on the same local model. The unmodified native routed executor receives continuous features/task predictions from that view; no true task label is provided. Native V1 multiclass remains an unchanged control. The continuous-native readout is recorded separately from expert fusion so router gain is not falsely attributed to expert learning. All policies use identical TEST rows; no intermediate TEST is added. Timing separates control forward, continuous-native forward and expert overhead.

## Validation scope

Synthetic-only regression covers amplitude distinctions lost by binarization, class-to-task multimodality, caps/no raw retention, historical-memory preservation, current labels/finite values/coordinates, exact V4 checkpoint restore, corrupted-state rejection, unchanged native Torch RNG/binary memory/graph capsule summary, real CNN-GRU capture/routed inference, unchanged V1 dispatch, and identical terminal TEST rows/control metrics. Separate hashes confirm protected V1/shared files and V1 notebooks unchanged.

No accuracy gain is claimed. Prior audits showed stable selected features only for the measured T3→T5 panel; early-task function drift and late-class information limits remain unverified. The first-task feature subspace and missing task memory can still limit this variant. No TEST fitting or recovery from historical raw TRAIN is allowed to fill missing continuous memory.

## Logging

No per-sample/per-class router console logs. Keep V4 summary on round0/every5 rounds and terminal progress every10 clients. Persist selected-feature count, rows/classes and local feature bytes in the V4 report. These are local memory bytes, not network traffic. Checkpoint/resume keeps the full feature-memory contract.
