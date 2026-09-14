# CANDLE / DeNICE logic repairs — 2026-09-14

Reference: `Neurocomputing_Anh_Khýi.pdf`, Eq. (23), Step 5 aggregation,
and Step 7 deployment. This continues the 2026-09-13 audit.

## Repaired behavior

1. **Linear adapter on layer input.** `denice_adapter_mode=linear_input` uses
   `U(V(x))` with no bias or sigmoid, instead of the legacy `U sigmoid(V(h))`
   on the layer output. U starts at zero, preserving the initial classifier.
2. **Explicit structural support.** New adapters capture their task's young
   output pool and the supported input features. These masks are saved, not
   recomputed from later neuron ages, so consolidation does not disable a
   trained adapter. Recycling removes references to reclaimed input features
   and reclaimed output units.
3. **Same embedding in classifier and capsule.** Both paths now call
   `penultimate_features`, avoiding an incorrect adapter input in prototype
   and Fisher calculations.
4. **Compatible peer updates.** Adapter aggregation additionally checks output
   support, along with input support, key and parameter shapes. Incompatible
   peers contribute zero without changing the neighborhood denominator.
5. **Evaluation no longer disables training adapters.** Routed evaluation
   previously cleared active adapters and left them cleared when training
   resumed. It now restores active adapters and each module's training flag,
   including early returns and exceptions. The combined logits/context helper
   also returns adapter-aware logits while retaining adapter-free routing
   activations.
6. **Architecture-aware continuation.** Registry metadata records mode, input
   dimension and version. Checkpoint restore, client cloning and catch-up
   reconstruct the proper adapter and masks. Old checkpoints retain v1's
   original logits. Changing adapter mode during continuation is rejected.

## CNN–GRU mapping choices

The paper gives a generic linear adapter equation rather than exact tensor
shapes for this repository's parallel CNN–GRU backbone. The v2 implementation
treats each convolution/BN/ReLU/pooling block as one `f_l`. A shared linear
projection of adjacent input positions matches its stride-two output length;
odd trailing positions are dropped as in the existing pooling path. The GRU
adapter projects the flattened input sequence to the last hidden vector, and
the fc1 adapter consumes the concatenated CNN–GRU feature vector.

These are explicit implementation choices. They satisfy the bias-free linear
residual form; they are not a claim that the paper specifies these particular
sequence projections. Earlier audit caveats concerning neuron-level capacity,
undefined drift between disjoint label sets, numeric thresholds and controller
timing still apply.

## Kaggle usage

The updated `train_incremental_kaggle.py` defaults to `linear_input`. Extract
`output/denice_candle_source_20260914.zip` and run its script with the packaged
`fed_learning` directory. Set `DENICE_CODE_DIR` to the extracted directory if
running from a different notebook location; check the printed `Training source`.

Start a fresh task-0 run for v2. Default phase 1 runs tasks 0–1; set
`DENICE_TRAIN_PHASE=5` for all six tasks. To continue an existing v1 run or perform
an adapter ablation with the repaired evaluation logic, set
`DENICE_ADAPTER_MODE=legacy_output` and retain the matching saved configuration.

The ZIP includes a SHA-256 source manifest so the actual uploaded modules can
be compared with this workspace. It contains source only, not datasets,
credentials, model checkpoints, or claimed benchmark results.

## Validation and metric interpretation

Result: **213 tests passed**: the full regression run passed 211 tests with
9 warnings in 211.20 seconds, followed by 2 additional peer-mask/evaluation
tests passing in 5.40 seconds. All execution remained in the workspace sandbox.
Changed Python files pass `py_compile`; `git diff --check` reports no whitespace
errors. The source archive is checked with ZIP CRC and SHA-256 for every file.

Tests exercise linearity, all five adapter locations, odd-length/multifeature
inputs, initial equivalence, nonzero gradients, mature-output masking, recycling
support, legacy and v2 checkpoint reconstruction, evaluation-state restoration,
incompatible peer masks, and preservation of old-context logits after training
a new context.

The three-task continuous-versus-resumed test also runs a deliberately injected
domain-shift signal to exercise creation, training, aggregation and continuation
of six adapters. This injection is confined to the test: it does not change the
production drift estimator and is not evidence of a metric gain.

No new CICIoT2023 Accuracy/macro-F1 measurement is claimed. The old logs found in
the workspace date from June/August and do not establish the user's latest
reported improvement. In particular, on strictly disjoint class tasks, the
current shared-class drift is undefined and may never activate adapters; these
adapter repairs cannot be assumed to raise metrics in that regime. Compare the
same real split, seed, training budget and routed evaluation protocol before
attributing an improvement to this revision.
