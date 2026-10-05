# Label-blind inference and comparison fairness audit

## Conclusion

The frozen Class Meta pipeline does not require the test sample's true class or true task to select its output. An isolated process reproduced all 50,000 decisions without reading test targets or reference predictions. Meta accuracy remains **58.358%**, with zero mismatches.

This establishes label independence of cached expert-to-final inference. It does **not** establish equivalence to single-model CGoFed/FedNICE protocols, strict replay-free learning, an untouched backbone holdout, or privacy-preserving distributed deployment.

## Executed label-blind check

Tool: `tools/audit_denice_label_blind.py`.

The parent starts a separate Python process. That worker only reads:

1. Fresh expert NPZs with an exact field whitelist: predicted class/task, router confidence, classifier confidence/margin/entropy, mask count, predicted task support and predicted class support.
2. Frozen Gate V2 and Class Meta payloads.
3. Recorded graph edges.

The worker never reads prediction CSVs, panel manifest, target NPZs, role provenance or test labels. It saves its predictions and exits **before** the parent reads reference predictions/labels for scoring. Fit methods are blocked; worker fit calls are zero.

| Policy | Decisions | Mismatches vs completed run | Accuracy |
|---|---:|---:|---:|
| Self | 50,000 | 0 | 32.146% |
| Majority | 50,000 | 0 | 50.326% |
| Gate V2 | 50,000 | 0 | 54.184% |
| Class Meta | 50,000 | 0 | 58.358% |

This run used local scikit-learn 1.9.0 for replay of 1.6.1 serialized artifacts, suppressing persistence warnings. The Kaggle notebook remains pinned to 1.6.1. All outputs still matched exactly.

## Raw-input path inspection

`expert_features()` in `tools/eval_denice_competence_gate.py` accepts model, detector, inputs, seen-class vocabulary, device and batch size. It calls `_denice_routed_logits_with_episodes(..., inference_policy='pred_hard')`; no oracle episode argument or label tensor is passed.

In `fed_learning/training/denice_eval.py`, `pred_hard` uses predicted episodes for both adapters and masks. Oracle policies exist separately for diagnostics and are not selected by expert feature extraction.

`feature_matrix()` reads predicted task/class fields and calibration-derived frozen priors. `class_features()` constructs evidence for every class in the fixed vocabulary, using experts' predicted classes. `masked_class_decision()` chooses among classes actually predicted by those experts. None accepts the sample's true label.

The frozen panel evaluator computes predictions before obtaining `truth` for metrics and candidate-oracle flags. Test labels are used for class-stratified panel construction and reporting, but not for sample inference. A labeled benchmark sampler is distinct from giving a true class to the predictor.

No local raw test dataset is available, so this audit did not rerun raw-input GPU inference with labels physically removed. The runtime check covers the cached expert-to-final path; the raw-input path has code-level evidence, supplemented by the previously completed 850,000-reference GPU reproduction check.

## Gate/Meta training data

Provenance and code show the following allocation:

| Role | Unique content hashes | Usage |
|---|---:|---|
| Calibration | 10,342 | Frozen donor and donor-task correctness priors |
| Fit | 39,491 | Gate correctness targets and Class Meta supervised targets |
| Validation | 19,247 | Gate policy / meta hyperparameter selection and diagnostics |

These are unique content counts, not receiver-role row counts or final usable meta fitting counts. Content may appear for multiple receivers within the same role. Some fitting targets are excluded from Class Meta fitting when no queried expert predicts the true class; validation and test retain all samples.

Independent checks reproduced deterministic hash-role assignments, global role disjointness and these counts. New confirmation-panel content has **zero overlap** with any Gate V2 role. Frozen model checksums match the protocol.

Crucially, rows come from historical `client_<id>_train.npz` files, restricted to origins with recorded local participation evidence. The saved audit explicitly reports:

- `original_backbone_holdout=false`;
- `historical_revisit=true`;
- `centralized_gate_fit=true`;
- `privacy_deployment_claim=false`.

Thus validation is held out from Gate/Meta fitting, but is not proven held out from original backbone training. This is retrospective supervised stacking on historical training data, not cross-fitting. It does not automatically imply test-label leakage; it must be disclosed because it changes data-access, retention and fitting assumptions.

The primary gate has 56 input features and 2,369 learned MLP parameters. The primary meta has 578 input features and 19,107 learned LR parameters, plus scaler state and frozen competence priors. Its fitted class vocabulary has **33 classes, missing class 28**; the decision helper aligns that vocabulary to the declared 34-class action space. The saved fit coverage shows 1,470 class-28 fitting rows, but zero reachable targets at each budget, so they were excluded from meta fitting. Class 28 is also absent from the fresh panel, so this run cannot validate behavior on that class. Include that limitation when planning full-scope replication.

The original 50k panel informed successive designs and remains development data. The fresh 28-class panel is a frozen confirmation; further tuning on it would turn it into another development panel.

## Fair comparison protocol required

Do not publish a direct claim of superiority to CGoFed/FedNICE based on this 58.358% alone. A defensible comparison needs:

- The same untouched test rows, class coverage, receiver assignment and training seeds for all methods. This panel misses classes 0, 3, 28, 30, 31 and 33.
- An explicit total labeled-data budget and data-access policy. Before new training, either reserve calibration/meta-fit/validation data before backbone fitting or declare reuse of backbone training data and permit corresponding access for baselines. Out-of-fold expert predictions are another option, with their training cost reported.
- Separate single-model and peer-ensemble results; include matched candidate-budget majority and comparable ensemble baselines where feasible.
- Report k, inference latency, model memory/caching, transferred model bytes and any communication. The primary inference uses 17 logical expert evaluations per sample, plus Gate and Meta. No raw-sample network queries occurred in the diagnostic runtime, but that is not proof of a decentralized deployment protocol.
- Preserve validation-only selection and freeze before each new test evaluation. Replicate across independent original training seeds using the fixed design.

Name the current result **DeNICE + frozen peer Class Meta (self + 16 experts)**. The claim supported now is improvement over same-panel majority/Gate V2 in a retrospective frozen ensemble experiment, not standalone DeNICE at 58.358% across 34 classes.
