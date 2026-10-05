# Peer coverage diagnostic results — 2026-10-05

Input: `C:\Users\khoak\Downloads\denice_peer_coverage_diagnostics.zip`.
Completion marker is true. All 98 clients and 50,000 unique test rows are present.
No new inference or training was performed in this artifact analysis.

## Results (pooled percentages)

| Measurement | Result |
| --- | ---: |
| Multiclass baseline | 27.772 |
| Local BestAllowedRoute | 45.372 |
| Peer-union class mask, normal local route | 18.046 |
| Peer-union class mask, matched local route | 36.612 |
| Peer-union class mask, best allowed local route | 36.612 |
| Local class coverage | 63.914 |
| Peer class coverage | 100.000 |
| Local task availability | 85.980 |
| Peer task availability | 100.000 |
| PeerExpertBestAllowed | 97.896 |

The original directed graph has 50–98 permitted group members including self,
mean 84.898. No sparse-neighborhood experiment is represented here. The normal
baseline and local routing upper bound reproduce the previous runs.

## Expanding a local mask is harmful on this panel

Normal routing loses 6,655 previously correct samples while recovering 1,792:
net -4,863 samples, or -9.726 pp. The best-local-route mask comparison recovers
6,669 but loses 11,049: net -4,380, or -8.760 pp.

On the 31,957 samples with locally supported classes, matched/best local accuracy
falls from 70.989% to 36.415% with union masks. On 18,043 unsupported samples it
rises from 0 to 36.962%, insufficient to compensate. This demonstrates that
allowing more labels on the receiver classifier is not equivalent to giving it
peer classifier expertise. Feature separability and logit competition remain
relevant even with the task supplied.

Do not deploy peer-union masking as an accuracy fix on this evidence.

## Peer expert oracle is high, but is not deployable accuracy

There is at least one correct allowed expert/route on 48,948 samples; 1,052 are
unsolved. It rescues 26,262 samples relative to the local best-route bound, with
no losses because self remains an option. Every stored first-success donor is
self or a recorded positive-weight incoming peer; none violates edge membership.

The expert bound enumerates many models and task actions and chooses using the
true label. It demonstrates existence of a correct prediction, not that a
label-free gate can identify one, that expert logits are calibrated, or that
communication costs are acceptable. It must remain in diagnostic analysis.

Among stored first-success witnesses, 1,369 use a one-class task mask. Such an
action can return its only allowed label without discriminating within a task.
Larger peer/action sets and narrow masks can therefore make this union-of-hits
bound optimistic. Witnesses depend on traversal order; these counts do not
measure the marginal contribution of single-class experts.

Peer expert coverage of previously locally uncovered samples is 97.234%; of
locally covered samples 98.270%. It is 98.959% for samples whose task was absent
from the receiver bank. These are oracle diagnostics, not conditional selectors.

Unsolved samples by task: T0=119, T1=15, T2=0, T3=1, T4=723, T5=194. Class 28
accounts for 648/1,052 unsolved samples. This merits later class-level analysis,
without selecting new methods using these same test errors.

## Next experiment

Maintain frozen weights, multiclass routing and each expert's own class mask.
Evaluate a **label-free peer inference** protocol before changing training:

1. Select peers from recorded incoming edges by fixed training-time alpha,
   using client ID to break ties. Predeclare budgets (e.g. 1/4/8/16 peers plus
   self). This changes inference query budget, not the training graph.
2. Each selected expert runs its own multiclass task router and local classifier;
   no true task or class is passed to the selector or inference.
3. Include fixed baselines: self only, a deterministic single peer and uniform
   voting over expert class predictions. Treat confidence-weighted selection as
   separate work requiring training/validation calibration; raw max confidence
   is not comparable across differently sized masks.
4. Report accuracy/F1, per-task results, latency, number of expert queries,
   peer budget and paired client intervals. Do not automatically pick a budget
   from this already examined test panel and call it validated generalization.
5. If learning a gate, use legitimate training/validation access and lock it
   before evaluating a fresh held-out panel/seed. A failed simple vote does not
   prove every selector fails; the 97.896% bound does not promise 50% either.

Training/aggregation should remain separate until this deployability gap is
measured. No router/default, graph or model weights are changed by this audit.
