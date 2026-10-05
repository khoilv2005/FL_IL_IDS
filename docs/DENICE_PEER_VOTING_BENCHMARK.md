# Frozen peer voting benchmark

Run `eval_denice_peer_voting_kaggle.ipynb` on Kaggle with Internet enabled and the original 100-clients dataset attached. The notebook downloads `results (4).zip` and the prior TIP diagnostic reference from the configured Google Drive links, then clones current GitHub main. It evaluates checkpoint `03b9b53`, task 5 round 19, on the same 98 receivers and 50,000 class-stratified test samples. It does not train a backbone, expand masks or change the graph.

## Fixed protocol

- Budgets are k = 0, 1, 2, 4, 8, 16 peers, with self additionally available.
- Candidates are directed peers with positive recorded aggregation alpha at task 5 round 19. Self is always allowed. Top-alpha ordering breaks ties by donor client ID.
- Random-k uses seeds 42, 43, 44, 45, 46. Each seed defines nested prefixes as k increases. Report their mean and spread, not the best seed.
- Each donor uses its own checkpoint weights, adapter, task router and local class masks. The balanced multiclass router fits only that donor's saved binary context sketches.
- Self predictions and task IDs must exactly reproduce the prior diagnostic panel. Model fingerprints must match and remain unchanged.
- Selection manifests are saved before expert inference. Normal selection and voting never inspect test labels; correctness flags are used only for metrics and explicitly labeled oracle diagnostics.

## Policies

| Policy | Decision |
| --- | --- |
| Self, k=0 | Receiver's balanced multiclass prediction |
| Single peer, k=1 | Highest-alpha peer, or first random peer |
| Majority | Equal one-hot votes from self and k peers |
| Alpha vote | Recorded aggregation alpha times each expert's one-hot vote |
| Alpha-router-confidence | Alpha times the donor's selected-task probability times its one-hot vote |

Ties favor self if its class shares the maximum score; otherwise the smallest class ID wins. All-zero weights fall back to self. Router probabilities are not calibrated across clients; the confidence multiplier is exploratory. A single-task router has confidence 1.

## Two oracle curves

`OracleActualRouted@k` measures whether any queried donor is correct using its own predicted route. `PeerExpertBestAllowed@k` additionally permits choosing any of that donor's existing legal routes. Both use labels and are diagnostic bounds, not inference methods. Their difference separates within-expert routing loss from expert selection loss. Each bound must be monotone over nested budgets, and voting cannot exceed the actual-routed bound.

The previously measured 97.90% bound used the full large peer set. This experiment does not assume a small budget retains that coverage.

## Outputs and interpretation

The output is `/kaggle/working/denice_peer_voting_diagnostics.zip`, containing:

- `summary.csv`: pooled accuracy, client-mean accuracy, pooled macro-F1, paired client-bootstrap CI versus self, and logical expert query counts.
- `per_client_metrics.csv`, receiver prediction CSVs, and expert caches with sample IDs.
- `random_summary.csv` and `alpha_vs_random.csv`: fixed-seed comparisons and paired CIs against the within-client random mean.
- `selection_manifest.json`, `peer_edges.csv`, `runtime.csv` and the notebook's protocol metadata.
- `accuracy_oracle_vs_budget.png` and `.pdf`.
- `voting_completion.json`: must contain `completed: true` before treating the archive as a complete result. Failed runs still produce a partial archive.

Normal ensemble policies query k+1 expert models; single-peer inference queries one. These are **logical expert queries**, not literal neural-module forward counts. Reported amortized sequential timings include an extra router-confidence pass, exclude model loading and communication, and are not deployment latency measurements. Oracle route enumeration adds substantial benchmark work. One GPU is used; a second T4 is not automatically utilized. Donor-major execution caches predictions across policies and budgets.

Report every declared budget and policy. Do not optimize a selector or choose a final deployment budget using this test panel. If a gate is needed, fit and select it using separate train/validation data.

## Verification

Local unit checks cover deterministic rankings, nested random candidates, votes, ties, invalid weights and sample-order invariance. Python and notebook syntax are checked locally. The full checkpoint benchmark must still run on Kaggle; no new accuracy is claimed before its artifacts are available.
