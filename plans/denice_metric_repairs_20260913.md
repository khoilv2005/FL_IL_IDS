# DeNICE logic repairs — 2026-09-13

Base: `922d618`. No real-data metric improvement has been measured in this
session: the 100-client dataset and latest run artifacts were not located in
the workspace. Historical CSV scores are not measurements of this patch.

## Confirmed defects and changes

1. **Fixed allocation rounded once per class, then exhausted late-task capacity.**
   `ceil(width / 34) * task_classes` repeatedly overallocated the first five tasks.
   Fresh runs now allocate disjoint blocks using global label boundaries
   `floor(class_id * width / 34)`. Clients that skip a task still use the same
   coordinates for a given later task. Frozen layers and mature units stay protected.

   Task-5 allocation after five six-class tasks, with no CANC layer freezes:

   | Layer | Previous | Repaired |
   | --- | ---: | ---: |
   | conv1 | 4 | 8 |
   | conv2 | 8 | 16 |
   | conv3 | 16 | 31 |
   | GRU | 10 | 12 |
   | fc1 | 16 | 31 |

   This redistributes a fixed total capacity; it does not enlarge the network.
   Earlier tasks receive slightly fewer units, so the accuracy tradeoff still
   needs a paired full-task experiment.

2. **Sketch thresholds included features excluded from the router.** Calibration
   now computes mean/std on the protected routing subspace. Changing only a
   masked reserve feature from 0 to 1000 previously changed the thresholds and
   erased discriminative bits in a controlled probe. It no longer affects them.
   Empty and singleton calibration subsets have finite thresholds. Existing
   checkpoint thresholds remain unchanged because stored sketches cannot be
   recalibrated without their source inputs.

3. **A fixed -100 sentinel did not exclude classes with extreme allowed logits.**
   The model could predict an unseen class even with the correct episode route.
   All DeNICE inference masks now share a finite floor that decreases when
   necessary. The regression uses allowed logits -200/-201 and forbidden logits
   +1000; allowed classes win and excluded softmax mass is below 1e-20.
   Oracle, top-k, adaptive, and ensemble callers share the corrected helper.

## Checkpoints and validation

The D3 configuration selector previously gated and ranked improvements using
oracle Macro-F1. It now uses predicted-hard traces, Macro-F1, and old-class
recall throughout selection. Oracle deltas remain diagnostic. A regression
rejects a candidate with improved oracle scores but degraded deployed predictions.

`denice_allocation_policy` is recorded in run config and `allocation_policy` in
model metadata. New runs default to `class_blocks`; checkpoints without this
field resume with `legacy_sequential`. Changing the policy during continuation
raises an error. Use a fresh task-0 run to measure the allocation/calibration fixes.

- Existing baseline suite: **142 passed**.
- Expanded regression suite (DeNICE, CANDLE, NICE, checkpoint restoration):
  **175 passed**, including eight new regressions (final run after all code edits).
- Final mask refinement: **29 passed**, excluding the unrelated P6 summary test
  already covered by the expanded suite.
- The two-task integration test exercises training, local-delta aggregation,
  checkpoint restoration, stable router features, and confusion-matrix accuracy.

The BatchNorm investigation did not establish a metric-degrading defect in this
session; no BatchNorm update policy was changed.

## Next real-data experiment

The existing `run_denice_d3_kaggle.py` runs a matched comparison of natural
batches, class-balanced batches, and effective-number weighted CE. Point
`DENICE_REPO_DIR` at this repaired source and `DENICE_DATA_DIR` at the dataset.
Use separate output roots, identical seeds and evaluation panels. Its small
default run is only a smoke diagnostic; 300 training samples with batch size
2048 give just one optimizer step per local epoch.

For a substantive comparison, set `D3_MAX_TRAIN_SAMPLES=0` (uncapped),
`D3_MAX_CLIENTS=100`, `D3_ROUNDS_PER_TASK=20`, and `D3_TASK_END=5`.
Compare predicted-hard Macro-F1, weighted-F1, route accuracy, and old-task recall;
oracle routing is diagnostic only. The repaired D3 analyzer gates on predicted-hard
Macro-F1 and old-class recall, then requests a confirmation seed for a winner.
No class balancing or learning-rate change was made the production default
without this evidence.
