# CGoFed fixes after results (4).zip

Source run: `03b9b534f4c3f12b1072e8eafa3802d50490f25f`, seed 42.

## Implemented

- Keep the NICE client's AMP GradScaler across phases and round calls. Optimizers
  still restart per phase. Scaler lifetime is the client object's lifetime;
  scaler state is not restored from existing continuation checkpoints.
- Record initial/final scale, skipped-step fraction and nonfinite gradient names
  after clipping in `continual_losses[].amp`. These names are diagnostic and do
  not identify the first operation that overflowed. Print skipped-step events.
- Add `denice_amp_enabled` (default true) to the training runner. Setting it false
  uses FP32, with explicit rejection of nonfinite loss/gradients before a step.
- Peer CGoFed uses sender/receiver allocated fc2 class support instead of only
  current-task capsule labels. The receiver still applies its structural mask,
  historical projection and mature-row write restriction. Ordinary aggregation
  keeps its existing current-task label rules.
- Record mature peer correction norms before and after projection. A zero
  projection ratio alone does not distinguish zero input from an orthogonal update.
- Preserve client-mean benchmark metrics. Add `coverage_counts` and
  `pooled_diagnostics` with sample-weighted accuracy, router coverage, accuracy on
  covered classes, route accuracy, and accuracy conditional on a correct route.
  Undefined conditional ratios are null. Correct-route accuracy is not oracle accuracy.

## Next run

Use the existing Kaggle launcher after these source changes are pushed. To select
FP32 through its existing environment override mechanism, set:

```python
os.environ["DENICE_CONFIG_OVERRIDES"] = json.dumps({"denice_amp_enabled": False})
```

Set this before executing the launcher; import `os` and `json` first. FP32 may
increase training time and memory use. The old completed task-5 continuation does
not retroactively repair skipped updates; assess training fixes with fresh training.

Before a full run, inspect a short run for successful optimizer steps and nonzero
mature peer correction norms when historical support and peer updates exist.
Task 0 legitimately has no historical basis. No training or tests were run as part
of this change. Oracle/nomask checkpoint comparisons remain a separate diagnostic;
these changes do not enable extra inference passes or alter test shards.
