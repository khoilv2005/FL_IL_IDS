# Smoke sáu thành phần từ literature trên checkpoint hiện tại

Code riêng:`tools/candle_v4_research_20261111/`; không trùng/ghi đè các script
pilot trước. Dùng shared evaluator hiện có qua interface có sẵn, không sửa file
V1/native hoặc defaults. Đây là thử các thành phần lấy cảm hứng từ paper trong
bài toán rerankTop2, không phải tái lập đầy đủ6 phương pháp hoặc training6tasks.

## Phạm vi và controls

-Checkpoint finalTask5/R19 results20/21,100clients.
-Saved receiver TRAIN routing activation memory; không đọc raw train cũ.
-Fit tất cả classifier trước mởTEST. Không fit/tune từTEST hoặc tạo receiver
  target prototypes từTEST. Fixedseed42; không hyperparameter search.
-Cùng2048 rows/client từ prior official disjoint shards, tổng204800,
  seed20261028+clientID. Panel TEST đã dùng development nhiều lần.
-Mỗi phương pháp chọn đúng1task trong2 ứng viên gốc; adapter, class mask,
  expert scope của task đó. Không cộng raw logits/experts xuyêntask.
-ProductionTop1 CM khớp prior receipt; native/service hashes giữ nguyên.
-Headline là pooled CM accuracy và fixed34 MacroF1 trên panel, không phải
  finalfullTEST. Không suy xếp hạng với thuật toán khác từ con số panel.

## Sáu component probes

| Tên trong receipt | Thành phần thực hiện | Phần chưa tái lập |
|---|---|---|
|fetril_translation|Dịch residual của class latest-task gần nhất tới oldmean,32pseudo-features/class,LinearSVC C1|Frozen-backbone fullstream, original SVM details; ở đây multiclassSVC|
|pass_isotropic|Noise đẳng hướng, variance chung,32/class,linearCE head|SSL và original schedule/augmentation radius|
|il2a_diagonal_loss|Diagonal covariance semantic correction trongCEloss, gamma1|ClassAug,backbone learning,fullcovariance|
|slca_gaussian_alignment|Gaussian LedoitWolf32/class,linearCE head|Slow learner,PTM,original alignment schedule|
|acil_recursive|Gaussian projection256+ReLU,task-batchedRLS lambda1|Original backbone/expansion,budget và stream stats; ở đây chỉ featurememory|
|dsal_residual|Analyticmain+tanhridge residual compensation|DAC null-space construction; không được coi fullDS-AL|

CácheadCE học100full-batchAdamsteps,LR.01. Noiserandom fixedseed42; singleton
covariance fallbacksharedvariance+jitter1e-6. Mean/covariance đúng receiver-space
nhưng memory cũ có thể không đại diện đủ realdistribution. Mọi score so giữa
tasks chưa calibrated; không tự bảo đảm negative-transfer safety.

SSRE/EASE/SupSup/FedETF cần trained branches/masks/classifier geometry absent
ở checkpoint này. Không đo bằng nhánh random rồi gọi là accuracy của paper.

## Verification và chi phí

-2unit testsPASS:recursive ridge khớpjoint ridge ở numericaltolerance;
 6scorers hữu hạn, singleton handling, absenttask giữ-inf,memorybất biến.
-8protected V1/native filehashes khớp manifest.
-Extra peerbytes0 trong smoke vì không có featuretransfer mới. Pseudo-features,
 headfit và extra scoring có compute/storage overhead; không gọi là miễn phí.
-Combined evaluation seconds trongreceipt chia sẻ backbone forward giữa6method,
 không là latency độc lập. Fit/loading không nằm trong timer đó.
-Pairedbootstrapclient là uncertainty mô tả development, không thay được nhiều
 trainingseeds hoặc confirmation độc lập, không điều chỉnh winnerselectionbias.

## Kết quả

Receipt:`artifacts/candle_v4_six_component_smoke_20261111.json`.

| Thành phần thử | Accuracy | Macro-F1 | Rescue | Break | Net |
|---|---:|---:|---:|---:|---:|
|ProductionTop1|48,0718%|27,5621%|0|0|0|
|FeTrIL-inspired translation|46,1387%|26,0702%|5.308|9.267|-3.959|
|PASS-inspired isotropicaugmentation|46,9346%|26,3857%|7.628|9.957|-2.329|
|IL2A-inspired semanticloss|47,2134%|26,5370%|5.877|7.635|-1.758|
|SLCA-inspired Gaussianalignment|47,7637%|27,0782%|5.517|6.148|-631|
|ACIL-inspired recursiveclassifier|48,0781%|27,3719%|3.012|2.999|+13|
|DS-AL-inspired residualbranch|48,0327%|27,2844%|3.101|3.181|-80|

ACILgain+0,00635điểmaccuracy, MacroF1 giảm0,1902điểm; không phải improvement
đáng tin hoặc qualityPASS. Pairedbootstrapclient95%gain interval
[-0,3662;+0,2861]điểm, chứa0. DS-AL residual cũng chưa vượtbaseline.
Không chọn thông số khác từTEST để cứu các bản thử thất bại.

Trong tất cả các pilot cùng204800panel gần đây, SLDA-inspired từlượttrước
vẫn cao hơn:48,1724%/27,6606%; gain nhỏ và uncertainty vẫn chứa0.
Không có bằng chứng đạt50% hoặc khai thác được oracle61,30%.
Kết quả chính thức fullTEST chưa đổi:48,08%/27,61%.

## Quyết định

Giữ inferenceTop1. Không cần fulltraining lại với sáu component variants này.
Chưa kết luận các paper gốc thất bại vì backbone/feature representation/budget
và application target khác. Smoke hiện tại chỉ đo tác dụng của các classifier
components trong candidate-restricted routing của checkpointCANDLE.

Code/receipts tách riêng, có thể chạy tái lập theoREADME. Một next-step có
ý nghĩa hơn là bổ sung missing-class evidence đúng receiver-space cho cohort
chọn bằng metadata, hoặc kiểm chứng redesign cần freshtraining nhưFedETF/EASE
ở quy mô nhỏ. Chưa implement/train các redesign đó trong lượt này.
