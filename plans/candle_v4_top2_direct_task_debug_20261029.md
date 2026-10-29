# Bounded direct-task Top2 diagnostic

User requested further debugging after mass/prototype reranking failed. This is
an additional development experiment, not an independent confirmation.

- Frozen results20/21 final checkpoint; original V1 and training remain unchanged.
- Original router supplies two candidate tasks. Exactly one task determines
  adapter, class mask, and expert scope. No cross-task union.
- Fit a separate task classifier from each receiver's existing saved TRAIN
  activation memory, using its existing scaler. Labels come from saved
  class-to-task metadata, never evaluation labels.
- Fixed LogisticRegression C=1, balanced task weights, lbfgs, max_iter=1000,
  random_state=0; these mirror the native classifier settings. No tuning.
- Choose higher task probability within the original two candidates; retain
  first on ties or absent runner-up. Missing tasks cannot be introduced.
- Same 204,800-row development panel and seed as preceding diagnostics.
  Evaluation labels are used only to compute metrics after prediction.
- Report route rescue/break and final class rescue/break. Exact baseline CM
  parity and native/expert state hashes are required.
- This fits an auxiliary router, so it is not a zero-fit scoring rule. It does
  not train the backbone or replay historical raw inputs. Historical feature
  drift and absent-class coverage are not solved by this experiment.
- Do not change production/notebook defaults based on an unconfirmed gain.

## One bounded follow-up

After the direct-task LR yielded 47.2378% against Top1 48.0718%, check a nonlinear
task boundary using SVC RBF, C=1, gamma=scale, balanced task weights, one-vs-rest
decision scores. Binary scores are [-margin,+margin]. No probability calibration,
no hyperparameter search, same saved TRAIN features and evaluation panel. This
follow-up is development motivated by the preceding failure, not preregistered
independent confirmation. The scalar decision values are not task probabilities.
