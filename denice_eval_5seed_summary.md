# DeNICE 5 Evaluation-Split Seeds Summary

Protocol: `eval_5.ipynb` proxy evaluation, global cumulative test split equally across active clients, aggregated over seeds `42,43,44,45,46`.

Audit clarification (2026-09-05): these are evaluation partition seeds over
saved checkpoints, not five independent training runs. Standard deviations
describe test-to-client assignment variability. Metrics below are preserved.
The separate FULL_D2 runs in Downloads/42.zip through 46.zip have independent
training-seed evidence; see `DENICE_AUDIT.md`.

Raw rows: `600`

Summary rows: `120`

Missing task-round-seed entries: `0`

## Final Round Per Task

| task | round | seeds | clients | K | accuracy | f1_weighted | f1_macro | route_acc | test_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 19 | 42,43,44,45,46 | 50/50 | 16 | 74.84 +/- 0.04 | 71.46 +/- 0.04 | 29.59 +/- 0.02 | 100.00 +/- 0.00 | 2.5043 +/- 0.0037 |
| 1 | 19 | 42,43,44,45,46 | 60/60 | 14 | 72.26 +/- 0.01 | 67.43 +/- 0.01 | 30.30 +/- 0.01 | 96.39 +/- 0.00 | 13.0722 +/- 0.0050 |
| 2 | 19 | 42,43,44,45,46 | 70/70 | 22 | 42.01 +/- 0.02 | 33.00 +/- 0.02 | 18.91 +/- 0.03 | 62.75 +/- 0.01 | 57.2518 +/- 0.0152 |
| 3 | 19 | 42,43,44,45,46 | 80/80 | 25 | 25.97 +/- 0.01 | 17.97 +/- 0.01 | 12.72 +/- 0.02 | 41.65 +/- 0.01 | 76.8916 +/- 0.0088 |
| 4 | 19 | 42,43,44,45,46 | 89/89 | 18 | 23.14 +/- 0.00 | 16.57 +/- 0.01 | 10.85 +/- 0.01 | 33.19 +/- 0.01 | 77.8795 +/- 0.0083 |
| 5 | 19 | 42,43,44,45,46 | 98/98 | 35 | 22.49 +/- 0.02 | 15.72 +/- 0.02 | 9.13 +/- 0.01 | 35.82 +/- 0.02 | 79.2690 +/- 0.0203 |

## Best Round Per Task

| task | best_f1_round | best_f1_weighted | best_acc_round | best_accuracy |
| --- | --- | --- | --- | --- |
| 0 | 4 | 77.24 +/- 0.05 | 4 | 80.34 +/- 0.04 |
| 1 | 15 | 68.00 +/- 0.01 | 15 | 72.93 +/- 0.01 |
| 2 | 16 | 33.06 +/- 0.02 | 11 | 42.69 +/- 0.02 |
| 3 | 19 | 17.97 +/- 0.01 | 14 | 25.99 +/- 0.01 |
| 4 | 1 | 17.75 +/- 0.01 | 1 | 24.85 +/- 0.01 |
| 5 | 1 | 15.97 +/- 0.01 | 1 | 23.10 +/- 0.02 |

## Input Coverage

| task_range | seed | rows | tasks |
| --- | --- | --- | --- |
| 0-3 | 42 | 80 | 0,1,2,3 |
| 0-3 | 43 | 80 | 0,1,2,3 |
| 0-3 | 44 | 80 | 0,1,2,3 |
| 0-3 | 45 | 80 | 0,1,2,3 |
| 0-3 | 46 | 80 | 0,1,2,3 |
| 4-5 | 42 | 40 | 4,5 |
| 4-5 | 43 | 40 | 4,5 |
| 4-5 | 44 | 40 | 4,5 |
| 4-5 | 45 | 40 | 4,5 |
| 4-5 | 46 | 40 | 4,5 |

## Missing Entries

_No rows._
