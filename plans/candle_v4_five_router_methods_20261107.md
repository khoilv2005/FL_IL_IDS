# Five-method fixed checkpoint development comparison

User requests literature review and at least five alternative methods. Scope:
frozen results20/21, all100 receivers, identical204800-row prior development
panel, original Top2 candidates. Exactly one chosen task controls native adapter,
class mask and experts. Original V1/native training remains unchanged.

All five scorers fit from saved receiver TRAIN activations before opening TEST.
Use existing StandardScaler, missing task score minus infinity, class-to-task
maximum, tie keeps production Top1. No sweep or tuning on TEST.

1. NCM Euclidean: nearest standardized class centroid, inspired by NCM classification
   in iCaRL; no raw exemplar rehearsal, herding or distillation from full iCaRL.
2. SLDA-inspired: shared residual LedoitWolf covariance, squared Mahalanobis
   class distance, equal class prior. Batch fit saved features, not streaming SLDA.
3. FeCAM-inspired: per-class LedoitWolf covariance and Mahalanobis distance;
   singleton uses shared covariance. Omits paper's Tukey transform/correlation
   normalization; do not label a faithful FeCAM reproduction.
4. RanPAC-inspired: fixed Gaussian random projection256, seed42, ReLU, ridge
   class one-hot regression, lambda1. No pretrained ViT or first-session adaptation.
5. kNN5: standardized feature distances, uniform vote among min(5,n) nearest
   saved samples. Related to neighbor-based feature confidence, not full DkNN
   multilayer conformal calibration.

Sources:
- https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html
- https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf
- https://arxiv.org/abs/2309.14062
- https://arxiv.org/abs/2307.02251
- https://arxiv.org/abs/1803.04765

Report Accuracy, fixed34 MacroF1, route accuracy, rescue/break, switches and
per-client accuracy. Require exact baseline confusion match and frozen service/
native hashes. New network bytes0 because existing local saved memory is used;
extra fitting, scoring and model memory are not free. Combined wall time reported,
not falsely attributed as isolated latency per method.

Best means best among these fixed variants on repeated TEST development only;
not proof of global optimum or independent benchmark gain. Do not automatically
promote a winner to production or promise61% oracle. Historical representation
drift, absent-task candidates and missing-class coverage remain unresolved.
