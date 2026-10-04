# Matched Oracle results — 2026-10-04

Source: `C:\Users\khoak\Downloads\denice_matched_oracle_diagnostics.zip`.
Checkpoint `03b9b53`, evaluator `faeec3d`. Complete 98-client panel: 50,000
samples per policy, 200,000 policy/sample rows with no duplicate policy/sample
identities. This audit reads saved artifacts; it does not run inference/training.

## Fixed-action routing limits

Pooled accuracy, percentage:

| Policy | Observed | OracleMatched | BestAllowedRoute | Remaining routing gap, pp |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 22.790 | 45.372 | 45.372 | 22.582 |
| Multiclass | 27.772 | 45.372 | 45.372 | 17.600 |
| Mahalanobis | 28.886 | 36.960 | 36.960 | 8.074 |
| TIP | 23.588 | 36.960 | 36.960 | 13.372 |

Mean-client bounds are 45.3660% (binary task bank) and 36.9594% (continuous
task bank). Exact matched hit counts equal enumerated best-allowed hit counts:
22,686 and 18,480 respectively. No policy correctly classifies a sample marked
task-unavailable or class-unsupported in this panel.

Therefore 50% cannot be attained by changing only task selection among these
existing actions on this frozen checkpoint and sampled panel. This statement
does not cover expanded class masks, added task actions, soft/mixed classifiers,
peer inference, changed weights or the full test distribution.

The previous OracleLocal 50.94% allowed true-task actions outside the router's
bank and invoked the existing empty-mask fallback. It is not the matched-policy
routing limit; use 45.37% for the existing binary-bank action set here.

## Disjoint prediction/error buckets

Each column partitions its 50,000 samples. Unavailable-task errors precede
unavailable-class errors; flags may overlap but counts below do not.

| Bucket | Multiclass | Mahalanobis |
| --- | ---: | ---: |
| Correct | 13,886 (27.772%) | 14,443 (28.886%) |
| True task profile unavailable | 7,010 (14.020%) | 13,583 (27.166%) |
| Class unavailable, task available | 11,033 (22.066%) | 11,001 (22.002%) |
| Wrong task, task/class available | 14,104 (28.208%) | 6,959 (13.918%) |
| Correct task/class, classifier wrong | 3,967 (7.934%) | 4,014 (8.028%) |

These are accounting categories, not amounts of accuracy independently
recoverable by fixes. For example multiclass has 14,104 wrong-task errors but
only 8,800 additional correct predictions under matched routing.

## Profile lifecycle evidence

Of 588 client/task pairs, 514 have local class support and binary memory, while
442 have continuous profiles. There are **72 locally supported pairs without a
continuous profile**. The lifecycle export labels 146 absent-profile pairs for
provenance investigation; 74 of those do not have local class support either.
This does not establish that 72 previously trained profiles were lost.

The continuous bank's fixed-action bound is 4,206 correct samples (8.412 pp)
below the binary bank's bound. This quantifies the opportunity from legitimate
additional task actions, not a promised gain from fitting new profiles. The
earlier diagnostic found 6,541 samples whose class is locally supported but whose
continuous task profile is unavailable.

## Next implementation decision

1. Integrate the exact saved-memory multiclass logistic configuration into
   normal routing, opt-in first. Reproduce 27.7681% mean-client accuracy within
   0.1 pp before changing the default. Preserve masks, adapter policy and training.
2. Audit the 72 pairs using historical participation, capture, persistence and
   restore evidence. Do not synthesize access to unseen historical task data.
3. Investigate legitimate peer-supported class/task coverage as a separate
   diagnostic. A larger class mask alone does not show the local model can
   classify those newly permitted labels.
4. To reach 50% on this panel, some part of the fixed-action system must change:
   coverage/action set, classifier or training. Raising the measured bound is
   necessary, not sufficient. Separate any training change from routing changes.

TIP remains an ablation. No inference/training implementation was changed by
this audit.
