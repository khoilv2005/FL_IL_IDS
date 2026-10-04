# DeNICE routing audit

Source: user-supplied `results (4).zip`, task 5 round 19, training commit
`03b9b534f4c3f12b1072e8eafa3802d50490f25f`. Diagnostics restored each saved
ContextDetector directly. Backbone weights and source artifacts were not modified.

## Metric semantics

`evaluate_denice_model` obtains its label-to-episode mapping from the client's
`episode_classes`, not the dataset-wide task map. Unknown labels do not enter
the route-accuracy denominator. The training runner reports the unweighted
mean of client route accuracies.

| Quantity | Value |
| --- | ---: |
| Test samples | 50,000 |
| Samples with locally mapped classes | 31,957 |
| Correct routes among those samples | 15,035 |
| Reported mean-client route accuracy | 47.415982% |
| Pooled accuracy on locally mapped classes | 47.047595% |
| Supported correct routes / all samples | 30.070000% |

The last quantity includes a class-coverage requirement. It is not pure global
task-identification accuracy: a router can identify the task of an unsupported
class without having that class in its output mask. Measuring global task
accuracy requires dataset task labels for every sample and the actual route
predictions, including samples excluded by the local mapping.

No duplicate class mappings were found across the 98 saved detectors.

## Actual routing rule

The run uses `binary_cosine`. Context activations from conv1/conv2/conv3/GRU are
binarized using the saved per-layer thresholds, then multiplied by the fixed
routing feature mask. Each episode's stored sketches are averaged to one
prototype. The router chooses the episode with maximum cosine similarity.
`train_models` returns immediately for this mode; it does not fit a logistic
classifier. Softmax scores preserve the cosine argmax and are not calibrated
probabilities of correctness.

The fixed mask comes from the client's first allocated task features. Saved
detectors expose 548 feature coordinates but retain only 17–102, mean 86.91.
This protects a stable feature subset but can restrict discrimination of later
tasks. The audit does not establish that widening the subset would improve test
accuracy; that requires a separate controlled experiment.

## Saved-memory diagnostic

- 39,176 stored binary sketches across 98 clients.
- Accuracy when routing those same stored sketches: **50.051052%**.
- No all-zero sketches and no exact maximum-score ties in this panel.
- Mean off-diagonal prototype cosine, averaged within client then across
  clients: **0.982793**.

The router struggles even with the stored feature representation used to build
its prototypes. This supports poor episode separability under the current
prototype rule; it does not prove encoder drift or isolate a single cause.

Covered-sample test route recall by task: T0 32.81%, T1 73.63%, T2 33.60%,
T3 65.64%, T4 29.40%, T5 55.64%. T0 is often sent to T5 (2,059 samples), and
T4 is often sent to T5 (1,544 samples).

No raw reference-input episodes are retained in these checkpoints. The fresh
flag is set after refreshing the current task's sketches; it is not evidence
that every old task has been re-encoded with the final backbone. Directly
measuring old-reference drift would require those raw references.

## Controlled sketch holdout

For each client/episode, randomly retain 20% of the stored sketches for scoring
and fit the two existing router modes on the remaining sketches. Seed is
42 + client ID; no hyperparameter search. Both methods use identical splits.

| Router | Accuracy on 7,734 held-out sketches |
| --- | ---: |
| binary_cosine | 46.211533% |
| multiclass logistic regression | 56.283941% |

This is a training-sketch diagnostic, not dataset test accuracy. Identical
binary patterns may occur across the split, and old sketches come from their
historical encoders. The next experiment should refit multiclass from saved
training sketches and compare both routers on the same frozen checkpoint and
test shards. No backbone retraining is needed for that experiment.

Detailed machine-readable evidence was saved in the user's Downloads folder:
`denice_router_mechanism_audit.json` and `denice_router_sketch_holdout.json`.
