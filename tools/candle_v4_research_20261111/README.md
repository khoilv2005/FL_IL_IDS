# Separate checkpoint component smoke

No V1 or production runtime changes. Run from repo root:

```powershell
.venv-denice-gpu/Scripts/python.exe tools/candle_v4_research_20261111/test_components.py
.venv-denice-gpu/Scripts/python.exe tools/candle_v4_research_20261111/run_smoke.py
```

All100clients, same204800-row repeated TEST development panel as prior routing
audits. All fits finish before TEST opens, from saved receiver TRAIN activations.
Each probe selects one of the original two task candidates; task-specific adapter,
class mask and experts remain scoped. This tests routing components, not replacement
of the full CANDLE class predictor. Baseline confusion and frozen-state hashes
are checked by the existing evaluator. Unique audit output prevents overwrite.

Six methods are **inspired component probes**, not full paper reproductions:

| Component | Fixed implementation | Omitted assumptions/components |
|---|---|---|
|FeTrIL translation|32 pseudo-features/class from nearest latest-task class residuals, LinearSVC C1|Original frozen-backbone training, full current data, binary SVM original details; here multiclass SVC|
|PASS augmentation|Shared isotropic noise around class means,32/class, linear CE head|Self-supervised backbone training, paper-specific radius/schedule|
|IL2A diagonal semantic loss|Class means + diagonal covariance correction to CE logits, gamma1|classAug, end-to-end representation learning, full covariance|
|SLCA alignment|LedoitWolf Gaussian32/class, linear CE head|Slow learner/pretrained ViT, original classifier alignment settings|
|ACIL recursive|Fixed Gaussian projection256+ReLU, task-batched RLS lambda1|Original expansion/training/data budgets; only saved features, no full stream sufficient stats|
|DS-AL residual|Same main analytic head + tanh residual ridge branch|Original DAC null-space construction; cannot claim full DS-AL reproduction|

Fixed seed42, heads100 full-batch Adam steps LR.01; no sweep. Scalars are not
calibrated task posteriors. Class score max gives task score. No donor-space means,
receiver TEST target features or raw historical inputs used for fitting. Singleton
covariance uses shared isotropic variance, jitter1e-6 ensures sampling stability.

SSRE, EASE, SupSup masks and FedETF require learned branches/masks/classifier
training absent from this checkpoint; do not report their accuracy from an
untrained dummy branch. No claim of faithful ten-method benchmark or independent
confirmation. Extra network bytes0 here; added fit/inference cost remains.
