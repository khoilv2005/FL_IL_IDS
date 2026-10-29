# DENICE-ACE: technical design and implementation boundary

## Implemented: lossless masked uplink

`denice_v2/compact_transport.py` serializes a LocalContribution, including its
learning receipt and common-reference/topology/plan bindings. For each tensor,
the compact representation stores a bit-packed contribution mask and all values
under that mask, in flattened C order. It keeps dtype and shape. Zero-valued
contributors remain in the denominator. Frozen coordinates are reconstructed as
zero delta. Both encoding and decoding use the existing permission validation.

The sender chooses the shorter **whole serialized message**, including codec
metadata, from compact and dense alternatives. This bounds size by the dense
envelope; tiny messages can be slightly larger than the old unwrapped message.
There is no quantization or top-k approximation, and no residual state to resume.
Encoding both candidates costs CPU/memory: this is not a training-speed claim.

`HierarchicalRunner` exercises this transport before aggregation and accounts
for actual encoded client-to-leader bytes, excluding the leader's own local
contribution. Other traffic retains existing accounting. Temporary-server
election, class-wise aggregation, age/mask protection, BN collection and
inter-server common-reference reduction remain unchanged.

Opt-in: `RunConfig(compact_uploads=True, server_exchange=True, ...)` or:

```text
python tools/train_denice_v2.py --data-dir <DATA> --output <EMPTY_OUTPUT> --device cuda:0 --worker-devices cuda:0,cuda:1 --prediction-mode routerless_v1 --server-exchange --server-only-storage --defer-evaluation --compact-uploads
```

Run from the repository root with the existing dependencies. This runs the full
existing six-task protocol; it is not a quick benchmark. The field defaults to
false. Existing embedded Kaggle bundles/notebooks have not been regenerated;
use this repository source. Exact-source resume checks remain in force: use a
fresh experiment for comparison, not an old checkpoint with changed source.

## Proposed: coverage-constrained local work

Reserve private TRAIN/CAL roles once, before the experiment. The current V2 main
uses all current TRAIN rows, so a research runner must explicitly version this
protocol change and give the baseline the same split. Never repurpose TEST.

At task start, all new clients get one training opportunity. Subsequently send
only small local summaries: current class counts, prior held-out macro loss,
uncertainty, last participation, local duration and byte estimate. These metadata
are part of the FL privacy contract, not raw-row exchange.

Proposed selection score (normalize components within cluster):

```text
benefit_i = alpha * local_macro_loss_i
          + beta  * routing_uncertainty_i
          + gamma * marginal_class_coverage_i
          + delta * participation_debt_i
score_i = benefit_i / max(epsilon, predicted_seconds_i + lambda * upload_bytes_i)
```

First satisfy mandatory new-client and overdue-client slots. Greedily fill
class-coverage gaps, then maximize score under the round budget. Target at least
two distinct contributors per current class when available, otherwise one. If
coverage cannot fit, explicitly relax the budget or defer consolidation. Keep
exploration slots for low-scoring clients; do not claim this heuristic is an
unbiased FedAvg estimator. Log selection and class support each round.

Clients without successful optimizer steps produce no learning event and cannot
gain ownership or maturity. Resume stores participation debt, RNG, successful
update counts, validation history and decisions. Deadlines cannot silently
turn missing clients into zero-valued contributing updates.

## Proposed: age is evidence, expansion is a transaction

Separate exposure count from biological-style neuron age. Increment exposure
only on successful selected learning. Promote learners only at accepted
consolidation, never merely because a wall-clock round elapsed.

Expansion gate: local macro validation fails to improve by epsilon for P checks,
new-class recall remains poor, and usable learner capacity is below rho. First
try an existing compatible block; then allocate a small block/adapter within a
fixed per-task parameter budget. Initial candidates P=3, rho=10%, block=5% of
layer width are tuning points, not established optimums. Diagnose routing error
before allocating classifier capacity. Separate mature, routing and learner
dependency closures; changing an ancestor or BN can invalidate frozen features.

The temporary server proposes a new topology version; members acknowledge it;
only then train and aggregate. Enforce old/new validation acceptance on private
held-out data if available, plus exact protected-parameter invariants. Roll back
an unacceptable candidate. A replay-free run without historical validation can
prove invariants but cannot prove absence of functional forgetting.

## Proposed: stable clusters and bounded routing

Keep membership during short windows. Reassign only if a compatible cluster's
local validation utility exceeds the current cluster by a margin over several
observations; require minimum dwell time. Compute comparison locally, exchange
scores. Charge probing, downloads, leader migration and handoffs to communication.
Incompatible histories transfer separate experts, not arbitrary averaged weights.

Maintain a versioned routing feature branch with protected upstream/BN behavior.
Build balanced task sketches from local training data; do not pretend sketches
from different feature spaces are interchangeable. Fit/calibrate once per
accepted encoder version, or reuse a cache only when the full dependency hash
is unchanged. If sketches cannot support a new routing space without historical
rows, preserve that expert's old space instead of silently relabeling features.

Use top-1 on confident examples. On low calibrated margin, evaluate top-2
compatible task heads and combine normalized class probabilities using routing
weights, never uncalibrated logits from different heads. Choose thresholds on
CAL. Report top-1 route macro-F1, top-2 task recall and classification macro-F1
separately. No ground-truth task/class is an inference input.

Start with the existing synchronous hierarchy. Event-driven inter-server exchange
is a later experiment: independent server histories invalidate the current common
reference. It requires anchor/version management or separate experts, and cannot
be implemented safely by just skipping broadcasts every few rounds.

## Validation

Tests cover exact aggregation equivalence including zero updates, stale bindings,
permission violations, corrupt masks, encoded savings on a mostly protected
tensor, two-task native training, no TEST access during training, class coverage,
and compact checkpoint resume. See the accompanying test result file for actual
execution status in [validation results](DENICE_ACE_VALIDATION_20261028.md).
No full CICIoT training, accuracy gain or route-F1 gain is claimed.

Source audit: `training/decentralized_denice_il.py` round loop and router refresh;
`denice_v2/runner.py` locked budgets; `denice_v2/hierarchical.py` temporary servers;
`denice_v2/aggregation.py` masks and contributor-specific denominators.
