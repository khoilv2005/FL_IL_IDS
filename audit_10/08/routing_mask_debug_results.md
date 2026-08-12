# DeNICE routing/mask debug results

Date: 13/08/2026. Checkpoint evidence: task 5 / round 19, DeNICE decentralized, multiclass router.

## Diagnosis

The poor result is not a class-to-episode mapping or checkpoint-restoration bug. The stored router is separable (97.281% refit held-out balanced accuracy), but its old binary memory was captured before later decentralized aggregation. The final model therefore emitted a different feature distribution and routed only 31.549% balanced accuracy on a test subset. Hard masking then turns each wrong episode into a deterministic wrong class.

| Same old-checkpoint smoke (20k, coverage-aware) | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| E0 backbone, no router/mask | 36.510% | 15.617% |
| E1 predicted adapter, no mask | 36.955% | 15.765% |
| E2 oracle adapter, no mask | 37.690% | 16.306% |
| E3 oracle adapter + hard mask | 67.705% | 38.332% |
| E4 predicted adapter + hard mask | 15.250% | 8.610% |
| E5 predicted top-k=2 mask | 32.120% | 15.580% |
| E6 adaptive mask fallback | 26.280% | 13.000% |

E3 has zero oracle mask violations. The task-5 router is 18.050% accurate; thus the E3-to-E4 gap identifies routing drift, not a mask-map error. Adapters are sparse: only 2,078/20,000 E3 samples activated one, so the old checkpoint's adapter benefit is negligible and classifier quality must be treated separately.

## Implemented repair

Each client now keeps only its small local router-reference inputs and, after every decentralized aggregation, re-encodes its episode sketches with the new model then refits the multiclass router. This is enabled in new training by `denice_refresh_router_memory_after_aggregation=True`. It does not rewrite old checkpoints. The evaluator also exposes E0–E6, coverage-aware assignment, router audits, adapter/mask diagnostics, top-k, and adaptive fallback.

## Verification on fresh stratified smoke

Three tasks, 10 clients, 3 rounds/task, 300 stratified train samples/client, 50 router references/class, seed 42:

- Stored router: 95.746% persisted / 80.349% refit-held-out balanced accuracy.
- Final-model feature router: 63.527% balanced accuracy; prototype cosine drift mean 0.953, minimum 0.833.
- E3 oracle-hard 18.130%, E4 predicted-hard 17.835% (0.295-point gap), route 69.830%, and zero mask violations.

The routing/mask regression is fixed in the smoke. The low absolute classifier score is expected from the deliberately tiny 3-task smoke and must be tested by P6 full training before claiming a final improvement.
