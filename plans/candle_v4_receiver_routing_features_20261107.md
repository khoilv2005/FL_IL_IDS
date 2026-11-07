# Receiver-space routing feature feasibility

One metadata-selected fixture at Task5/R19 final checkpoint (results20/21).
Select lexicographically receiver/class/donor from ACTIVE Task5 experts whose
class is absent from receiver router memory, Task5 itself exists, and whose donor
received authorized positive expert updates at round19. No quality-based selection.

Send serialized exact receiver function + feature indices to donor. Donor samples
20 (existing per-class cap) target current-task TRAIN rows, seed42; runs receiver
function, returns selected continuous routing activations and provenance.
No historical raw-data replay. All raw data access stays in the simulated donor
phase. Same-process execution simulates a wire protocol, not a real network run.

Fit an isolated router copy using existing TRAIN memory plus new features,
same existing LR settings. Compare baseline, memory-only refit control,
augmented selection restricted to original Top2, and augmented unrestricted Top1.
Each prediction uses exactly one task's adapter/class mask/expert scope.

Fit completes before opening TEST. Fixed diagnostic panel: up to128 TEST samples
per class, seed20261107, 34 classes. This class-balanced panel is not the official
pooled benchmark; TEST is development-only given preceding iterations.

Report target recall, target routing, cumulative accuracy/Macro-F1, false positives,
rescue/break, actual serialized capsule/feature bytes. Verify base model and service
hashes unchanged. Native V1/training/default notebooks remain untouched.

This final-checkpoint fixture uses the final encoder after consolidation. It cannot
prove the reference would have been identical during earlier round training or
that features remain valid after a future task. No claim of native integration,
historical retention, generalization across receivers, or full-run improvement.
