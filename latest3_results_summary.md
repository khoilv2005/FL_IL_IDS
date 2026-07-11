# Latest 3 CSV Result Summary

Source folder: `C:\Users\khoak\Downloads`

Newest 3 CSV files:

| algorithm | source_file | rows | tasks | raw_round_min | raw_round_max | round_norm_min | round_norm_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Plexus | 100 Clients(Plexus).csv | 120 | 0,1,2,3,4,5 | 0 | 119 | 0 | 19 |
| FedAvgEWC | 100 Clients(FedavgEWC).csv | 120 | 0,1,2,3,4,5 | 0 | 19 | 0 | 19 |
| FedAvgLWF | 100 Clients(FedavgLWF).csv | 120 | 0,1,2,3,4,5 | 0 | 19 | 0 | 19 |

Plexus normalization: `round_id = round_raw % 20`. Example: task 5 raw rounds `100..119` become normalized rounds `0..19`.

## Overall Final-Round Summary

| algorithm | rows | missing_task_rounds | final_avg_accuracy | final_avg_f1_weighted | final_avg_f1_macro | final_avg_recall_macro | task5_final_accuracy | task5_final_f1_weighted | best_avg_f1_weighted_by_task |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvgLWF | 120 | 0 | 48.10% | 38.99% | 11.75% | 16.93% | 17.84% | 6.58% | 52.82% |
| Plexus | 120 | 0 | 43.32% | 37.81% | 20.35% | 27.80% | 0.79% | 0.01% | 46.53% |
| FedAvgEWC | 120 | 0 | 43.08% | 37.54% | 20.75% | 27.42% | 2.67% | 1.33% | 39.09% |

## Final Round By Task

| algorithm | task_id | round_raw | round_id | accuracy | precision_macro | recall_macro | f1_macro | f1_weighted | test_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvgEWC | 0 | 19 | 19 | 98.96% | 65.42% | 51.23% | 52.33% | 98.50% | 0.0549 |
| FedAvgEWC | 1 | 19 | 19 | 93.05% | 39.66% | 47.71% | 42.79% | 90.03% | 1.5084 |
| FedAvgEWC | 2 | 19 | 19 | 40.87% | 12.20% | 29.39% | 15.80% | 27.43% | 12.5218 |
| FedAvgEWC | 3 | 19 | 19 | 20.16% | 8.88% | 21.15% | 10.16% | 7.65% | 24.5553 |
| FedAvgEWC | 4 | 19 | 19 | 2.76% | 1.52% | 9.16% | 1.29% | 0.29% | 28.1972 |
| FedAvgEWC | 5 | 19 | 19 | 2.67% | 1.60% | 5.85% | 2.10% | 1.33% | 50.8104 |
| FedAvgLWF | 0 | 19 | 19 | 76.99% | 12.83% | 16.67% | 14.50% | 66.98% | 1.9966 |
| FedAvgLWF | 1 | 19 | 19 | 95.46% | 24.45% | 24.85% | 23.87% | 94.85% | 1.1577 |
| FedAvgLWF | 2 | 19 | 19 | 60.46% | 23.81% | 22.51% | 17.62% | 50.99% | 1.9104 |
| FedAvgLWF | 3 | 19 | 19 | 19.89% | 12.01% | 15.33% | 6.40% | 7.97% | 2.2830 |
| FedAvgLWF | 4 | 19 | 19 | 17.95% | 2.52% | 11.59% | 4.03% | 6.55% | 2.9868 |
| FedAvgLWF | 5 | 19 | 19 | 17.84% | 2.99% | 10.60% | 4.09% | 6.58% | 3.1282 |
| Plexus | 0 | 19 | 19 | 98.96% | 65.53% | 50.89% | 51.90% | 98.49% | 0.0582 |
| Plexus | 1 | 39 | 19 | 93.05% | 36.94% | 47.70% | 39.77% | 90.42% | 1.1650 |
| Plexus | 2 | 59 | 19 | 40.88% | 11.84% | 29.75% | 15.42% | 26.92% | 3.9303 |
| Plexus | 3 | 79 | 19 | 21.51% | 7.94% | 24.25% | 11.37% | 8.76% | 5.2713 |
| Plexus | 4 | 99 | 19 | 4.75% | 2.61% | 11.26% | 3.63% | 2.24% | 1.7141 |
| Plexus | 5 | 119 | 19 | 0.79% | 2.80% | 2.94% | 0.05% | 0.01% | 1.2495 |

## Coverage

| algorithm | rows | missing_task_rounds | missing_preview |
| --- | --- | --- | --- |
| FedAvgEWC | 120 | 0 |  |
| FedAvgLWF | 120 | 0 |  |
| Plexus | 120 | 0 |  |

Output files:

- `latest3_results_normalized.csv`
- `latest3_results_summary.csv`
- `latest3_results_final_by_task.csv`
