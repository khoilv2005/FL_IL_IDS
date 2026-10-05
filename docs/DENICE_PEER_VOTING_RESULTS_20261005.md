# Frozen peer voting results — 2026-10-05

Input: `C:\Users\khoak\Downloads\denice_peer_voting_diagnostics.zip`.
Training checkpoint: `03b9b534f4c3f12b1072e8eafa3802d50490f25f`, task 5 round 19.
Evaluation source: `2a8382478669ed04005f9f7e3d3512e56387cb59`.

## Artifact audit

- `voting_completion.json` reports complete; all 98 receiver prediction files and 98 donor runtime rows are present.
- Exactly 50,000 distinct test row IDs; original test pool has 13,505,771 rows. This is the previous class-stratified panel, not natural-distribution accuracy.
- Self client-mean accuracy reproduces 27.7681444398%; pooled accuracy is 27.772%.
- All summary pooled accuracies independently recompute from receiver predictions without mismatch.
- Recomputed 5,400,000 label-free vote decisions from expert caches, recorded alpha and router confidence: zero mismatches. All cache row IDs align with receiver predictions; actual-routed and best-route oracle flags match recomputation.
- The archive contains 6,097 entries. Normal prediction plus confidence passes total 317.27 seconds; oracle route enumeration totals 1,816.05 seconds, excluding loading/setup. These are benchmark totals, not distributed inference latency.

## Top-alpha results (pooled percentages)

| Peer budget k | Majority | Alpha vote | Alpha × router confidence | Actual-routed oracle | Best-allowed-route oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 27.772 | 27.772 | 27.772 | 27.772 | 45.372 |
| 1 | 27.772 | 14.634 | 14.686 | 36.728 | 54.056 |
| 2 | 27.568 | 16.574 | 16.562 | 45.290 | 66.438 |
| 4 | 32.874 | 24.852 | 23.902 | 61.740 | 80.894 |
| 8 | 37.646 | 30.374 | 33.376 | 71.502 | 83.214 |
| 16 | 41.760 | 38.828 | 40.368 | 77.532 | 91.588 |

Single top-alpha peer at k=1 achieves 14.676%. Majority at k=1 is identically self by construction: two votes tie whenever they disagree, and self wins ties. Its unchanged score is not evidence of peer consensus quality.

Top-alpha majority at k=16 has pooled macro-F1 35.087%, versus self 25.720%. Its paired client-mean gain is +13.992 percentage points, bootstrap 95% CI [12.285, 15.688]. At k=8, gain is +9.877 points, CI [8.536, 11.268].

## Random candidates within the same legitimate graph

Five fixed seeds (42–46); table reports their mean, not their best result.

| k | Random majority accuracy | Mean pooled macro-F1 | Actual-routed oracle | Best-allowed-route oracle |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27.772 | 25.720 | 44.246 | 64.744 |
| 2 | 31.257 | 28.586 | 53.855 | 73.534 |
| 4 | 36.552 | 32.943 | 64.963 | 81.750 |
| 8 | 40.290 | 35.700 | 74.247 | 87.427 |
| 16 | 42.818 | 37.383 | 80.917 | 91.715 |

Random-majority k=16 spans 42.600–42.972% across selection seeds (standard deviation 0.161 percentage points). These are peer-selection seeds on one checkpoint, not independent training seeds.

At k=8, top-alpha majority minus within-client random mean is −2.644 points, CI [−3.158, −2.175]. At k=16 it is −1.058 points, CI [−1.436, −0.683]. Thus recorded-alpha ranking is inferior to random selection for majority in this panel. This does not test random peers outside the legitimate graph or imply the graph itself is useless. Alpha ranking does outperform random candidates for alpha-weighted policies at k=8/16, but both remain below majority.

An additional paired bootstrap of the random mean versus self (10,000 client resamples, seed 20261005) gives +12.521 points at k=8, CI [10.968, 14.107], and +15.049 points at k=16, CI [13.328, 16.755].

## Alpha concentration diagnostic

- Top-1 donor is client 70 for 94/98 receivers and client 9 for the other four.
- Top peer alpha exceeds self alpha for 97/98 receivers. With self plus one peer, the largest normalized weight has mean 92.552%, so alpha voting is usually dominated by the selected peer.
- Client 70's cached predictions across 48,468 queried samples score 14.513%, despite mean task-router confidence 0.606. It predicts tasks 0, 3, 4, 5 only in this panel; no predictions route to tasks 1 or 2.
- Client 9 scores 18.565% across its 47,449 cached samples, with mean confidence 0.443.

These donor scores are measured on their respective queried subsets, not a common full panel. They explain an observed failure mode; they must not become a test-label-based exclusion list. Aggregation alpha and router confidence are not established sample-level expert competence measures.

## Decision

1. Frozen, label-free peer inference gives a substantial observed gain without training or mask expansion. Majority is the strongest tested policy; random graph candidates outperform top-alpha candidates under majority.
2. With top-alpha k=4, actual-routed expert coverage already reaches 61.740%. Majority reaches only 32.874%, leaving 28.866 points between consensus and that label-assisted bound. At k=8, the gap is 33.856 points. Choosing or combining already-routed experts is a major remaining issue.
3. Best-allowed-route coverage exceeds 50% at k=1, but requires both expert and route oracle choices. Actual-routed coverage first exceeds 50% at top-alpha k=4 (random mean k=2). Neither result guarantees a learned gate can reach 50%.
4. Keep k=8/16 majority as frozen baselines. Do not increase budget or retrain solely to compensate for poor selection. Next, design a competence gate using separate train/validation data, retaining donor-local adapters/masks. Lock its configuration before evaluating this test panel; retrospective gate fitting must be labeled separately from streaming replay-free claims.
5. Do not use these test labels to remove client 70, optimize alpha transformations, calibrate probabilities, or select a production budget. No learned gate has yet been evaluated.
