# DENICE incremental upgrade — rationale, implementation and evidence

## Dữ liệu hiện có và giả thuyết

Người dùng báo accuracy task 0–5 lần lượt **99%, 98%, 87%, 70%, 70%, 70%**.
Chưa có file log để xác minh đó là cumulative accuracy hay accuracy task hiện
tại; chưa có macro-F1, class recall hoặc route accuracy tương ứng. Không kết
luận nguyên nhân của đường cong chỉ từ sáu con số này.

Code cho thấy hai điểm có thể làm replay không tăng metric chính:

1. Mature classifier rows bị khóa. Loss giữ logits trên các đường truyền hoàn
   toàn đóng băng có thể gần như không thay đổi quyết định.
2. Hard routing loại mọi lớp ngoài task dự đoán. Cải thiện cạnh tranh giữa task
   trong logits không giúp được mẫu có true class đã bị router loại bỏ.

Bản nâng cấp lần này giữ `algorithm="denice"`, vòng huấn luyện neurogenesis,
CANC, adapter và giao thức P2P; thêm readout có khả năng học lại ranh giới mọi
lớp và một cơ chế chọn mẫu replay theo đặc trưng. Không sửa nhãn/test split hoặc
dùng oracle context để tạo metric chính.

## Căn cứ nghiên cứu và phần thích nghi riêng

| Nguồn | Ý tưởng được dùng | Khác biệt của triển khai DENICE |
|---|---|---|
| [iCaRL — Rebuffi et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) | Exemplar selection theo trung bình đặc trưng; prototype classification | Herding trên tập candidate hữu hạn, re-encode bằng backbone DENICE, không tái lập toàn bộ iCaRL |
| [Deep SLDA — Hayes & Kanan, CVPRW 2020](https://arxiv.org/abs/1909.01520) | Deep features + discriminant classifier học tăng dần | Refit shrinkage LDA từ replay/current samples, covariance cân bằng theo lớp; không phải streaming covariance update của SLDA |
| [Decoupling Representation and Classifier — Kang et al., ICLR 2020](https://openreview.net/pdf?id=r1gRTCVFvB) | Tách học representation và classifier; cân bằng classifier cho dữ liệu long-tail | Không retrain neural fc2: fit readout đóng dạng riêng để giữ gradient freeze và aggregation cũ |
| [Large Scale Incremental Learning / BiC — Wu et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf) | Bias ở classifier mới/cũ và hiệu chỉnh bằng validation | Chọn probability mixture trên lưới cố định, không phải affine bias correction hai tham số của BiC; holdout chỉ tách khỏi fit readout, không hoàn toàn khỏi học backbone |

Đây là một tổ hợp thích nghi có cơ sở, **chưa là phương pháp được peer-review**.
Các bài trên không chứng minh tổ hợp này cải thiện CICIoT2023, môi trường IDS,
decentralized learning hay cấu hình 100 clients của người dùng. Không dùng số
phần trăm cải thiện của các bài để dự báo cải thiện cho DENICE.

## 1. Feature-representative replay

Giữ quota lớp của bộ nhớ hiện có; mặc định tổng B=1024 exemplar/client. Với
mỗi lớp, lấy ngẫu nhiên tối đa max(quota,512) candidate mới và ghép exemplar cũ.
Tính normalized penultimate features ở eval mode, không active adapter, rồi
greedy chọn exemplar sao cho trung bình tập đã chọn gần trung bình candidate.
Không chọn trùng index. Các exemplar cũ được giữ nguyên stored logits và mask
các lớp đã biết; không refresh teacher targets theo student.

Đây là approximate herding: candidate subsampling có thể bỏ sót mode hiếm;
trung bình đơn không mô tả tốt mọi phân phối đa mode. Config `priority` vẫn
giữ random-priority reservoir cũ để ablation. Không đổi selection khi resume:
chạy experiment mới từ task 0 để so sánh công bằng.

## 2. Readout cân bằng tại mỗi client

Không sửa neural fc2 hoặc mở khóa mature neurons. Mỗi client fit một affine
readout trong `model.local_classifier` (algorithm state, không phải parameter
hay buffer của `model.state_dict`). Sau aggregation mỗi round và sau
consolidation cuối task, lấy tối đa 128 train examples mỗi lớp mới và exemplar
các lớp cũ. Lớp tái xuất hiện dùng current training examples; lớp vắng mặt dùng
memory. Re-encode tất cả mẫu được chọn bằng backbone hiện tại, tránh dùng
prototype cũ trong feature space đã thay đổi.

Với z = normalize(penultimate_features(x)), tính mu_c và covariance S_c trong
từng lớp. Covariance chung là trung bình **đều theo lớp** có >=2 mẫu; singleton
vẫn có mean nhưng không đóng góp covariance chưa xác định.

```text
S = mean_c(S_c)
S_reg = (1-rho) * S + rho * max(trace(S)/d, 1e-6) * I
w_c = solve(S_reg, mu_c)
b_c = -0.5 * mu_c^T * w_c
score_c(x) = (z(x)^T * w_c + b_c) / temperature
```

rho=0.1, temperature=1.0 là cấu hình khởi đầu, chưa tune trên CICIoT2023.
Prior bằng nhau để tránh số lượng mẫu mới áp đảo lớp cũ. Solve CPU float64,
lưu weight/bias float32; không lưu inverse/covariance hoặc thêm raw bank.
Head chỉ biết các lớp có dữ liệu cục bộ/memory; không tự có tri thức lớp chỉ
có tại peer. Label support được mask theo seen classes, kể cả nhãn không liền nhau.

Feature path cho head tắt adapter để có cùng representation khi fit và predict.
Adapter vẫn phục vụ backbone learning và nhánh hard routing; khi trọng số head
bằng 1, quyết định cuối không dùng adapter. Đây là thay đổi có chủ đích ở
readout, không tuyên bố giữ nguyên mọi chi tiết inference của CANDLE.

## 3. Chọn probability mixture bằng validation cục bộ

```text
p_final = alpha * p_LDA + (1-alpha) * p_hard
alpha in {0, 0.25, 0.5, 0.75, 1}
```

Chọn alpha có balanced accuracy cao nhất trên tập hiệu chỉnh riêng tại client;
nếu hòa, ưu tiên alpha nhỏ hơn (đường cũ). Lớp mới dùng `X_validation` từ split
hiện có. Lớp cũ giữ riêng khoảng 20% exemplar (tối đa 32/lớp), loại khỏi dữ liệu
fit LDA; số còn lại mới dùng để fit. Lớp chỉ có một exemplar không tạo holdout.
Không có holdout nào thì alpha=0. Không dùng test accuracy để chọn alpha.

**Giới hạn về độc lập:** exemplar cũ có thể đã tham gia replay training của
backbone. Chúng chỉ độc lập với bước fit LDA đang xét, không phải validation
hoàn toàn chưa từng dùng của toàn hệ thống. Vì vậy score chọn alpha có thể lạc
quan; kiểm thử tách input chỉ xác nhận disjointness với fit readout. Tập current
validation tách khỏi current training. Không có bảo đảm test metric sẽ tăng
hay không giảm. Balanced accuracy hướng đến recall cân bằng/macro metric, có
thể đánh đổi natural-frequency accuracy hoặc benign false-positive rate.

## 4. Decentralization và lưu trạng thái

- Không thêm central server, global teacher, trao đổi raw replay hoặc pooled
  classifier. Capsule và masked peer aggregation vẫn như trước.
- Head fit từ current training/validation và memory của đúng client đó.
  Không đưa head vào `model.state_dict` hoặc capsule; bootstrap client mới
  bỏ head của donor. Rejoin refit trên dữ liệu riêng.
- Full algorithm checkpoint lưu head, alpha và selection audit. Các tensor
  head giữ float32 cả ở compact metadata để tránh fp16 overflow.
- Model thay đổi sẽ vô hiệu hóa head cũ. Refit sau round để round checkpoint
  có head tương ứng; refit sau consolidation để task checkpoint có head đúng.
- Classifier fitting dùng RNG riêng để không đổi chuỗi ngẫu nhiên training.
- Readout memory khoảng O(d*C) ngoài replay; covariance tạm O(C*d^2) trong
  triển khai, solve O(d^3), với d=256. Refit và validation tăng compute cục bộ;
  không tăng payload P2P nhưng tăng kích thước checkpoint cục bộ.

## Cách chạy và ablation

`train_incremental_kaggle.py` đã bật trực tiếp, vẫn `algorithm="denice"` và
`mode="decentralized"`. Giữ lựa chọn người dùng: phase 1 (task 0–1) và B=1024.
Chạy các phase tiếp theo với continuation của **cùng cấu hình mới**, hoặc đặt
phase 5 để chạy mới toàn bộ task 0–5. Không resume từ thí nghiệm selection/head
khác rồi coi là cùng experiment. Source ZIP chứa script và fed_learning cạnh
nhau; chạy script trong thư mục giải nén để tránh clone source GitHub cũ.

```python
"denice_replay_selection": "herding",
"denice_replay_candidate_limit": 512,
"denice_classifier_enabled": True,
"denice_classifier_per_class": 128,
"denice_classifier_batch_size": 256,
"denice_classifier_shrinkage": 0.1,
"denice_classifier_temperature": 1.0,
"denice_classifier_validation_select": True,
"denice_classifier_validation_per_class": 32,
"denice_eval_route_mode": "local_lda",
```

| Ablation | selection | classifier_enabled | validation_select | route_mode |
|---|---|---|---|---|
| DENICE replay trước lần nâng cấp này | priority | false | false | hard |
| Chỉ herding | herding | false | false | hard |
| Chỉ readout mới, memory cũ | priority | true | true | local_lda |
| Readout thuần, không mixture | herding | true | false | local_lda |
| Bản đầy đủ | herding | true | true | local_lda |

Mỗi nhánh train mới cùng split/seed/round budget. `--route-modes hard,local_lda`
trên một checkpoint chỉ đo tác động **readout trên cùng backbone**, không đo
toàn bộ đóng góp herding hoặc replay. Evaluator mặc định `auto` lấy route mode
từ checkpoint; có thể override `hard` để ablation. Notebook eval_selective đã
đọc route mode từ config, không cần sửa notebook.

Log chính bổ sung hard_accuracy, hard_f1_macro, gain_vs_hard_accuracy,
gain_vs_hard_f1_macro và classifier_weight. `history.local_classifier_fits`
và `round_metrics.local_classifier` lưu fit counts, bytes, time, selection grid,
alpha và validation class counts. Route accuracy vẫn đo router cũ độc lập;
không thay bằng accuracy của head để làm metric trông đẹp hơn.

## Kiểm chứng đã thực hiện

- 217 regression tests đạt (253.64 giây) trước bổ sung validation selection.
- Sau validation selection và integration test nâng cấp: 31 tests classifier/
  replay đạt (25.85 giây), bao gồm CUDA FP32/AMP, CPU, model/BN/mode/RNG
  preservation, singular covariance, label support, private head, herding,
  old-target retention và resume hai client/ba task với dropout–rejoin.
- Test kiểm tra mẫu dùng chọn mixture không nằm trong tập fit LDA.
- Sau thêm kiểm tra bootstrap không copy head của peer và fallback khi không
  có holdout: chạy lại 16 tests classifier, tất cả đạt trong 5.43 giây. Các số
  test ở đây là các đợt có chồng lấp, không cộng thành tổng tests độc lập.
- Evaluator độc lập load task checkpoint và tái tạo paired metrics khớp.
- Syntax compilation và git diff --check được kiểm tra riêng.

## Thí nghiệm synthetic — KHÔNG PHẢI IDS BENCHMARK

`tools/benchmark_denice_incremental.py`: 3 seeds (23,42,73), 2 clients, 3 tasks,
2 classes/task, Gaussian sequence clusters 16x1; train/test noise độc lập,
40 train samples/class/client, 80 test samples/class, 2 rounds/task, 1 epoch,
B=48. Validation chọn mixture; test chỉ đo kết quả. Mean ± sample std (%):

| Task | Hard routing accuracy | Local readout + mixture accuracy |
|---|---:|---:|
| 0 | 50.00 ± 0.00 | 100.00 ± 0.00 |
| 1 | 63.80 ± 11.96 | 98.70 ± 1.13 |
| 2 | 71.46 ± 14.46 | 98.06 ± 2.10 |

Task 2 macro-F1: hard 64.28 ± 17.59, readout 98.05 ± 2.10.
Kết quả nằm ở `output/denice_incremental_probe/summary.json`, từng seed có log,
config, source hash và checkpoint. `checkpoint_eval.json` là kiểm tra độc lập
seed 23, task 2; cùng checkpoint, cùng 480 test samples.

Dataset synthetic này dễ phân cụm; backbone train rất ngắn khiến neural head
yếu ngay task 0 (50%), khác hẳn baseline task 0=99% của người dùng. Do đó mức
tăng lớn trên đây chỉ xác nhận readout có tác dụng trong một tình huống cụ thể.
**Không suy ra DENICE thực tế tăng từ 70% lên 98%, không coi đây là benchmark
end-to-end thắng phiên bản trước.** Chưa có kết quả CICIoT2023 mới.

## Tiêu chí chấp nhận trên dữ liệu thật

Chạy 5 seeds, cùng split, cohort client và metric protocol. Báo cáo cumulative
accuracy/macro-F1, recall lớp cũ/mới, per-task accuracy, false-positive rate của
benign, routing accuracy, memory, train time và communication. So sánh với cả
ngân sách số round bằng nhau và wall-time tương đương. Chỉ xem là cải tiến khi
tăng ổn định ở task 2–5 mà không đánh đổi quá mức task 0–1 hoặc benign FPR.
Nếu classifier_weight luôn 0, validation không ủng hộ head mới; nếu weight>0
nhưng test kém, xem domain mismatch/holdout bias trước khi tăng capacity/loss.

Hạn chế còn nguyên: classifier không sửa backbone thiếu biểu diễn, CANC hết
reserve, recycling xóa mất feature, hoặc P2P grouping thiếu cộng tác. Herding
mean có thể bỏ mode hiếm. Không tự tuyên bố đã giải quyết mọi dạng forgetting.
