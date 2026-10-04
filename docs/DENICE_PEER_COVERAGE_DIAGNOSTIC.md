# Recorded-graph peer coverage diagnostic

Run `eval_denice_peer_coverage_kaggle.ipynb` on Kaggle with the original
100-clients dataset and Internet. The existing Drive links supply results (4)
and prior TIP diagnostics. No new file upload is required beyond this notebook.

## Fixed protocol

- Original 03b9b53 checkpoint, task 5 round 19, same 98 receivers and 50,000
  disjoint stratified sample IDs. Multiclass predictions must reproduce exactly.
- Peers are receiver-directed entries in the checkpoint's `cluster.groups`,
  confirmed against `alpha_debug.group_ids` with finite positive `alphas`.
  Self is always an allowed expert. Missing graph evidence fails explicitly.
- No graph changes, new profiles, backbone training, historical raw-train reads,
  weight averaging or continuous feature/basis transfer.

The checkpoint's group size mean is 84.90 including self. This experiment uses
that recorded dense topology; it does not evaluate a proposed sparse graph.

## Separate questions and outputs

1. **Coverage only:** true class/task availability locally and in legitimate peer
   support unions. These are label-assisted coverage counts, not accuracy.
2. **Union mask, local model:** keep the receiver's original route bank and
   adapters. Replace each task's class mask with the union of classes supported
   by recorded peers for that task. Report normal multiclass route, matched
   oracle (true task only if locally routable) and best allowed local route.
   No task profile or adapter is invented for an unavailable task. A larger mask
   can reduce accuracy because it introduces competing logits.
3. **PeerExpertBestAllowed:** allow selecting self or a permitted donor model and
   any route in that donor's existing binary bank, using the donor's own mask
   and adapter. Count samples with any correct expert/route prediction. Each
   model interprets raw inputs in its own feature space, so no encoder alignment
   assumption is required. This is an offline, label-assisted upper bound;
   neither peer query communication nor a deployable expert selector is built.

The local BestAllowedRoute must reproduce 45.3660% mean-client accuracy within
0.1 pp. Peer expert enumeration retains self, hence cannot lower this bound.
The union-mask branch is a different action family and need not be monotonic.
Do not combine these two branches into one unexplained "peer gain".

## Execution and artifacts

One device is used. Evaluation loads one donor at a time and batches the inputs
of receivers linked to it. This avoids loading all models on GPU but performs
many more forwards than earlier diagnostics (roughly peers × samples × routes).
It may take appreciably longer; no runtime promise is made.

`denice_peer_coverage_diagnostics.zip` contains:

- `peer_edges.csv`, `graph_support.json`: exact permitted peers and support sets.
- `peer_summary.csv`: pooled and mean-client fractions for each metric.
- `peer_per_client.csv`, `peer_per_task.csv`, `peer_predictions.csv`: detailed
  correctness/coverage flags and a first successful donor witness per sample.
- `peer_definition.json`: emitted with `completed=true` only after both branches
  finish. A ZIP written after failure can contain partial outputs.
- `protocol.json`: checkpoint, code version, sample partition and source paths.

The first successful donor is traversal-dependent and label-selected; it is
not a recommended deployment peer. A high expert ceiling demonstrates existence
of correct predictions, not that a learnable selector can identify them.

This implementation has not yet been run through local tests or Kaggle inference.
Interpret final results only after the notebook completes on the full panel.
