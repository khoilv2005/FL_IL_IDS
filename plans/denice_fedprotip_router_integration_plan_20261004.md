# Kế hoạch thay router DeNICE bằng subspace task inference

Ngày: 2026-10-04. Trạng thái: đã triển khai evaluator frozen P0/P1/P2; chưa chạy thí nghiệm Kaggle và chưa tích hợp streaming/training.

Entrypoint: `eval_denice_tip_router_kaggle.ipynb`. Hướng dẫn và các khác biệt protocol cụ thể: `docs/DENICE_TIP_FROZEN_EVALUATION.md`. Các hạng mục kiểm chứng trong mục 7 vẫn là kế hoạch; chưa có kết quả kiểm chứng từ bộ evaluator mới.

## 1. Quyết định và phạm vi

Triển khai router lấy cảm hứng từ FedProTIP theo hai giai đoạn: thí nghiệm hồi cứu trên checkpoint `03b9b53`, sau đó triển khai streaming nếu đạt tiêu chí. Tên kỹ thuật đề xuất `tip_subspace`; đây là bản thích nghi cho DeNICE, không phải tái hiện nguyên bản FedProTIP.

- Checkpoint chính: `03b9b534f4c3f12b1072e8eafa3802d50490f25f`, task 5 round 19 từ `results (4).zip`.
- Training đối chứng tiếp theo: DeNICE + local CGoFed, giữ AMP fix, `denice_cgofed_peer_projection=False`.
- Thay router riêng biệt với optimizer, aggregation, capacity, adapter và classifier. Không bật lại peer CGoFed cùng thí nghiệm router.
- Thử frozen checkpoint trước để tránh chi phí một lần full training chưa biết router có hiệu quả hay không.
- Inference chính phải chọn task cho từng sample; nhãn và true task ID chỉ dùng tính metric hoặc chạy Oracle riêng.

## 2. Căn cứ và điều chỉnh đối với route.txt

Nguồn tham khảo:

1. `C:\Users\khoak\Downloads\route.txt`.
2. [FedProTIP repository](https://github.com/seohyeon-cha/FedProTIP), bản đã đọc `54193fa2d44f6203f39299a0ac3845097559a440`.
3. [Paper v3](https://arxiv.org/abs/2509.21606v3), đặc biệt phần task identity prediction.
4. Audit nội bộ: `docs/DENICE_ROUTER_MECHANISM_AUDIT_20261004.md`.

FedProTIP dùng subspace của activation và mức liên quan với subspace để suy ra task. Đây là ý tưởng phù hợp để kiểm tra trên DeNICE. Tuy nhiên phải điều chỉnh các giả định sau.

### 2.1 Khác biệt protocol rất quan trọng

Trong [server_fedprotip.py](https://github.com/seohyeon-cha/FedProTIP/blob/54193fa2d44f6203f39299a0ac3845097559a440/server/server_fedprotip.py), `eval_task` duyệt từng test dataset; `_eval_cnn` tính norm trên activation của cả batch, chọn một `pred_task` và mask toàn bộ batch theo task đó. Bộ đếm task accuracy tăng theo batch, đối chiếu `targets[0]`. Trường hợp hai task còn dùng ngưỡng riêng, thay vì cosine reference thông thường.

Do đó không dùng con số 95–99% trong route.txt làm cam kết cho per-sample routing của IDS. Khi báo cáo cần phân biệt rõ sample-level và batch-level; batch chứa nhiều flow của cùng task có lợi thế thông tin khác sample đơn lẻ.

Trong [client_fedprotip.py](https://github.com/seohyeon-cha/FedProTIP/blob/54193fa2d44f6203f39299a0ac3845097559a440/client/client_fedprotip.py), `compute_references` dùng `space_mats` được lưu để tính lại reference. Không thể bê nguyên cơ chế này rồi tuyên bố chỉ lưu basis mà không giữ activation lịch sử.

### 2.2 Router hiện tại và bằng chứng

Router đang chạy là `binary_cosine`, không phải logistic regression. Activation conv/GRU được nhị phân hóa, mask, rồi so cosine với prototype trung bình từng episode.

| Quan sát baseline | Giá trị / ý nghĩa |
| --- | --- |
| Feature gốc / feature giữ lại | 548 / 17–102 mỗi client |
| Cosine giữa prototype khác task | Trung bình 0.982793 |
| Route lại chính memory đã lưu | 50.05%; representation/prototype phân biệt task kém |
| Test samples có local class mapping | 31,957 / 50,000 |
| Route accuracy pooled trên phần có mapping | 47.05% |
| Route accuracy mean-client hiện báo | 47.42%; không phải global task accuracy |

Đây là bằng chứng cần thử continuous features và cách biểu diễn task giàu hơn. Chưa đủ để quy mọi lỗi cho feature drift.

| Frozen checkpoint diagnostic | Cũ 03b9b53 | Mới fb86983 |
| --- | ---: | ---: |
| Routed classification, mean-client | 22.79% | 22.54% |
| Oracle hiện có | 46.87% | 41.99% |
| AllClasses | 12.51% | 11.16% |

Oracle hiện dùng global task-class mask còn router thường dùng local mask. Chênh lệch có cả class coverage; 46.87% không phải trần thuần túy đã được kiểm soát chỉ theo routing. Cần Oracle matched-mask trước khi kết luận khả năng phục hồi.

## 3. Kiến trúc router đề xuất

Pipeline: input → encoder router không phụ thuộc task → continuous feature → relevance vector → task prediction → adapter/mask/classifier hiện có.

### 3.1 Feature extractor

V1 dùng feature trước fc2 (`penultimate_features`, dự kiến 256 chiều; đọc dimension thực tế từ model). Bắt buộc vô hiệu adapter theo task khi trích feature router để tránh cần biết task trước khi route. Giữ lại và khôi phục trạng thái adapter, mode train/eval sau khi capture; BN không cập nhật, dropout tắt.

Trong thí nghiệm checkpoint, encoder cuối được freeze trong suốt profile fitting và evaluation. Hash gồm weights, masks, BN buffers, preprocessing, feature layer, normalization và adapter policy. `topology_signature` một mình không đủ xác nhận feature tương thích.

Không dùng trực tiếp bank local CGoFed: bank đó có thể được capture với adapter task và encoder ở thời điểm khác; mục tiêu là bảo vệ update fc2, không bảo đảm tương thích cho route hiện tại.

### 3.2 Basis và relevance theo từng sample

Với feature hàng `z` và basis trực chuẩn `U_t` kích thước d×k:

```
r_t(z) = ||z @ U_t||_2
r(z) = [r_0(z), ..., r_T(z)]
score_t(z) = cosine(r(z), reference_t)
prediction(z) = argmax_t score_t(z)
```

Không tạo ma trận d×d khi inference; giữ trục batch và norm theo trục basis. Một sample phải có prediction giống nhau khi chạy đơn lẻ, đổi batch size hoặc ghép với sample thuộc task khác.

Hai cách tạo bank được kiểm tra trên validation:

- `independent`: SVD từng task trên cùng frozen feature space; baseline triển khai đầu tiên, tránh task mới bị thiếu residual dimension.
- `residual`: loại projection lên union basis cũ trước SVD; gần ý tưởng không gian mới của FedProTIP hơn. Ghi residual energy, total rank và basis saturation. Tổng rank trực giao không vượt d.

Chốt rank bằng squared singular-value energy, mặc định energy 0.95 và max rank 32; validation thử rank 16/32/64 có giới hạn. Không mặc định 99% tốt hơn: có thể giữ thêm noise và khiến subspace các task giống nhau.

Các trường hợp biên: một task trả task đó; zero vector/empty bank dùng fallback đã cấu hình; rank 0 không chia zero; không trả NaN. Tie-break xác định theo task ID. Ghi tỷ lệ fallback.

### 3.3 Reference có thể cập nhật mà không giữ mẫu cũ

V1 hồi cứu có thể tính reference từ từng sample training với toàn bộ bank đã fit. Đặt tên `mean_relevance`; đây là diagnostic được truy cập dữ liệu lịch sử.

V2 streaming chọn thống kê second moment làm mặc định:

```
M_t = mean(z.T @ z)     # triển khai bằng sum outer products / count
reference_t[s] = sqrt(max(trace(U_s.T @ M_t @ U_s), 0))
```

Đây là RMS projection, khác mean projection norm; tên `tip_rms`, phải đối chứng riêng. Lưu `M_t` cho phép tính thêm cột reference khi task mới xuất hiện mà không đọc lại dữ liệu cũ. Không zero-pad reference cũ theo giả định trực giao chưa được chứng minh.

Với d=256: moment float32 ~256 KiB/task, 6 task ~1.5 MiB/client. Basis rank64 ~64 KiB/task. Ước lượng moment cho 100 client đủ 6 task ~150 MiB, chưa tính encoder; phải đo footprint thực và peak RAM. Xử lý từng client, không tải toàn bộ model lên GPU cùng lúc.

## 4. Thí nghiệm checkpoint trước khi training

### P0 — Khóa protocol và đo baseline công bằng

1. Tạo notebook riêng `eval_denice_tip_router_kaggle.ipynb`; giữ notebook Oracle hiện tại làm đối chứng.
2. Drive mặc định là ZIP cũ `1BEjP4iGJbPT0uFcHZ7WXX4oM_1vyvx0M`. Dataset dùng path cũ kèm auto-discovery hiện có.
3. Restore checkpoint bằng helper hiện có, cùng client order, 50,000 test samples, sampling seed và shard assignment như baseline. Lưu manifest sample ID/client ID để ghép kết quả.
4. Source clone main mới nhất như yêu cầu; ghi source commit, checkpoint training commit và SHA256. Không ép checkout commit cũ; báo thiếu schema/API trước khi chạy lâu.
5. Chạy router legacy, Oracle local-mask, Oracle global-mask, AllClasses. Oracle local-mask dùng đúng local class policy của router; unsupported label được đếm rõ.
6. Kiểm tra khớp baseline restored khoảng 22.79%; sai quá 0.1 điểm phần trăm phải tìm khác biệt protocol trước khi so thuật toán.

### P1 — Fit router trên frozen checkpoint

1. Dùng training data của từng client và các task client thực sự có quyền truy cập. Lập manifest participation từ history/state; thiếu bằng chứng phải đánh dấu thay vì mặc nhiên cấp đủ 6 task cho mọi client.
2. Split profile-fit/validation xác định bằng sample ID, có phân tầng theo task/class. Validation không tham gia SVD/moment/calibration. Với checkpoint đã train trên dữ liệu này, ghi rõ đây là router holdout, không phải holdout của backbone.
3. Lấy tối đa 512 profile samples/client/task, có quota lớp và seed cố định; ghi số thực lấy. Không đọc test để fit basis hay chọn hyperparameter.
4. So các baseline bắt buộc: legacy binary cosine; multiclass trên binary sketch có sẵn; nearest centroid trên continuous feature; Mahalanobis task prototype; continuous subspace router. Multiclass sketch chỉ là đối chứng representation/rule, không giả định tăng accuracy test từ kết quả holdout memory.
5. Thử độc lập `mean_relevance` và `tip_rms`; independent basis trước, residual basis sau. Chọn cấu hình trên validation rồi khóa cấu hình trước final test.
6. Router đổi task prediction; adapter và local class mask giữ cùng policy ở các nhánh so sánh chính.

**Giới hạn công bố:** fit lại bằng historical local training data là retrospective router refit. Không gọi đây là kết quả streaming replay-free. Checkpoint cũ không có continuous statistics tương thích thì không thể sinh bank chính xác từ binary sketches.

### Hai baseline continuous bắt buộc — bổ sung theo quyết định triển khai

Mục đích: tách lợi ích của continuous representation khỏi lợi ích riêng của subspace routing. Centroid, Mahalanobis và TIP dùng cùng adapter-free frozen features, cùng sample IDs để fit, cùng validation split, cùng danh sách task khả dụng và cùng local class-mask/adapter policy. Không cấp thêm dữ liệu hoặc class coverage cho riêng một phương pháp.

- **Nearest centroid:** mỗi task lưu `mu_t = mean(z)` trên profile-fit samples; chọn task có squared Euclidean distance `||z - mu_t||²` nhỏ nhất. Không dùng task prior lấy từ test. Đây là baseline mặc định; cosine-centroid chỉ là ablation thêm nếu được chọn trên validation trước test.
- **Mahalanobis task prototype:** cùng các centroid trên, dùng covariance within-task gộp để ổn định khi mỗi task ít mẫu. Ước lượng `S = sum_t sum_i (z_i-mu_t)(z_i-mu_t)^T / max(N-K,1)`, với N là tổng profile samples và K là số task có mẫu. Dùng `Sigma = (1-lambda)S + lambda*trace(S)/d*I + epsilon*I`, mặc định lambda=0.1, epsilon theo scale với floor 1e-6. Chọn task có `(z-mu_t)^T Sigma^-1 (z-mu_t)` nhỏ nhất. Dùng Cholesky solve, không tính inverse trực tiếp. Nếu covariance suy biến, tăng jitter có giới hạn và log; không âm thầm đổi estimator.
- Dùng covariance gộp ở bản đầu để tránh covariance riêng từng task quá nhiễu. Covariance riêng hoặc Gaussian likelihood có log determinant là ablation khác, không gộp tên với baseline này.
- Feature transform mặc định là identity trên continuous activation. Nếu thử normalization/standardization, fit chỉ trên profile-fit và áp dụng cùng transform cho cả ba phương pháp; ghi transform vào protocol/hash. Không chuẩn hóa bằng thống kê của batch inference.
- Giới hạn tuning: centroid không có hyperparameter; Mahalanobis chỉ thử lambda trong {0.01, 0.1, 0.5} trên validation. TIP dùng budget đã nêu. Ghi đầy đủ lựa chọn trước khi mở final test.
- Lưu scores và predicted task cho cả hai baseline; kiểm tra sample prediction không đổi theo batch composition như TIP. Dùng cùng quy tắc tie-break và xử lý zero/invalid feature.

Nếu centroid hoặc Mahalanobis ngang/better TIP, chọn phương pháp đơn giản đạt gate cho hướng triển khai tiếp. Không kết luận subspace là nguyên nhân cải thiện chỉ từ việc TIP thắng binary cosine. Gate tăng ≥2pp so legacy vẫn giữ nguyên; tuyên bố TIP vượt các baseline continuous cần paired comparison riêng và khoảng tin cậy tương ứng.

### P2 — Chốt kết quả frozen evaluation

- Chạy cả 98 client và cùng shards. Report pooled accuracy, mean-client accuracy, macro F1, per-task recall, confusion task, confusion class.
- Route metrics: global task accuracy (mọi sample), task-available accuracy, local-class-covered accuracy và coverage fraction; luôn kèm numerator/denominator.
- Report phân rã: đúng task nhưng thiếu class; sai task; đúng task có class nhưng classifier sai.
- Lưu per-sample predictions để so paired; bootstrap theo client cho khoảng tin cậy 95%, không coi 50,000 mẫu là độc lập giữa client.
- Ghi latency/sample, profile fitting time, memory, fallback, rank từng task, cosine giữa reference, encoder hash.
- Bảng chính: Legacy / Multiclass / Continuous centroid / Mahalanobis / TIP / Oracle local / Oracle global / AllClasses. Bảng phụ đổi local/global mask phải tách khỏi router-only ablation.

Gate đề xuất, chốt trước final test: TIP tăng classification ≥2 điểm phần trăm so legacy trên cùng protocol, macro F1 không giảm quá 0.5 điểm, paired interval cải thiện accuracy không chứa 0, không có lỗi batch-dependent. Đây là tiêu chí quyết định kỹ thuật, không phải dự báo kết quả. Nếu hụt gate: xem confusion, coverage và separability trước khi chạy full.

## 5. Streaming replay-free sau khi P2 đạt gate

### 5.1 Giải quyết encoder drift

Không capture task mới bằng encoder đã đổi trong khi giữ statistics task cũ mà không đổi version. Giải pháp V2: router encoder độc lập, snapshot tại cuối task đầu tiên client tham gia, freeze weights, masks, BN và preprocessing. Classifier tiếp tục học bình thường. Snapshot không dùng future task data.

Client mới snapshot khi tham gia, profile task hiện có; local bank của các client có thể khác hệ tọa độ. Không ép dùng chung basis. Đo khả năng encoder đầu task biểu diễn task muộn: đây là rủi ro chính, chưa được giải quyết bởi kết quả posthoc encoder cuối.

Nếu encoder frozen quá yếu: dừng mở rộng peer, thử feature layer khác hoặc một routing encoder dùng initialization/pretraining hợp lệ trước stream. Đây là ablation mới, không tự thay đổi backbone chính trong cùng run.

### 5.2 Lifecycle bank

Ở task hiện tại, lấy mẫu chỉ từ current local train data. Cập nhật đủ thống kê theo sample membership, không cộng lặp cùng mẫu ở mỗi round làm sai weighting. Khuyến nghị finalize một lần tại task end; nếu cần route giữa task thì dùng provisional bank có version riêng.

Khi task kết thúc: finalize basis/moment → cập nhật mọi reference từ moment → kiểm tra finite/orthogonality → atomic save → evaluate. Resume phải idempotent, không append trùng task.

Không cần replay buffer dữ liệu cũ. Statistics vẫn có thể mang thông tin dữ liệu, nên không tuyên bố đảm bảo privacy/DP chỉ vì không lưu raw inputs.

### 5.3 Peer router là giai đoạn riêng

Mặc định `denice_tip_peer_enabled=False`. Chỉ chia sẻ/ghép basis khi encoder hash, preprocessing, normalization, schema, task IDs đều tương thích. Cùng d hoặc topology không đủ. Nhận bank không tương thích phải reject và log.

Nếu muốn router chung, cần thiết kế một frozen shared routing encoder với provenance hợp lệ trước khi gom statistics. Không lấy trung bình trực tiếp các basis khác hệ tọa độ; QR/SVD và moment phải có quy tắc weight, sample counts, deduplicate peer message và budget.

Peer score voting chỉ dùng khi từng peer có thể chấm cùng query trong feature space của mình, kèm phân tích communication/data exposure. Hoãn nhánh này đến sau local streaming gate; không trộn nó với peer CGoFed optimizer.

## 6. Các file cần triển khai

| File | Công việc |
| --- | --- |
| `fed_learning/strategies/incremental/denice_tip_router.py` (mới) | Bank, SVD, moments, references, sample scores, version và fallback |
| `fed_learning/strategies/incremental/denice_router_baselines.py` (mới) | Continuous centroid và pooled shrinkage Mahalanobis, cùng interface fit/predict/scores |
| `fed_learning/models/denice_model.py` | API router features không cần task; quản lý frozen encoder và phục hồi adapter/mode |
| `fed_learning/training/denice_eval.py` | Dispatch continuous router, tách global/local routing metrics, Oracle matched-mask |
| `fed_learning/servers/nice_server.py` | Giữ compatibility detector cũ; dispatch rõ mode, không đưa continuous feature qua binarize |
| `fed_learning/training/decentralized_denice_il.py` | Capture/finalize task profile đúng lifecycle; logs và participation manifest |
| `fed_learning/training/checkpoint_state.py` | Snapshot/restore bank, encoder, version; default legacy khi checkpoint cũ thiếu trường |
| `fed_learning/training/denice_delta_checkpoint.py` | Giữ bases/moments float32, không tự cast fp16; hỗ trợ delta/continuation |
| `eval_checkpoint.py` | Restore đầy đủ router state, kiểm tra hash và schema |
| `tools/eval_denice_tip_router.py` (mới) | Reusable CLI fit/eval, manifests, summaries và predictions |
| `eval_denice_tip_router_kaggle.ipynb` (mới) | Clone/download/restore/profile/eval, output ZIP; không launch training |
| `configs/denice_cgofed_tip.json` (mới) | Opt-in config streaming, tách baseline |
| `tools/build_denice_launchers.py` | Sinh launcher riêng `train_denice_cgofed_tip_kaggle.py` sau khi gate đạt |
| `fed_learning/strategies/decentralized/denice_capsule.py` | Chỉ mở rộng khi bắt đầu giai đoạn peer, có schema/hash |

Config đề xuất: `denice_router_type=binary_cosine|tip_subspace`, `denice_tip_feature_layer=fc1`, `denice_tip_basis_mode=independent`, `denice_tip_reference_mode=rms`, `denice_tip_energy=0.95`, `denice_tip_max_rank=32`, `denice_tip_max_samples=512`, `denice_tip_peer_enabled=False`. Các tên này là API dự kiến, chưa có hiệu lực trong code.

## 7. Kiểm chứng dự kiến khi triển khai

Các kiểm chứng dưới đây là hạng mục tương lai, chưa được chạy trong bước lập kế hoạch:

1. Feature extraction không làm đổi weights/BN/adapter; không đọc label khi predict.
2. Basis trực chuẩn, rank cap đúng, zero/NaN/empty/task thiếu xử lý rõ.
3. RMS reference từ moment khớp RMS tính trực tiếp trên toy features.
4. Prediction invariant với batch size 1/32/512, hoán vị và batch trộn task.
5. Baseline mode unchanged; checkpoint cũ load được; full/delta/resume giữ scores trong tolerance float32.
6. Task0/task1 và task IDs không liên tiếp; thêm task cập nhật reference không cần old data; resume không double count.
7. Encoder hash mismatch reject; stats peer trùng không tăng count lần hai.
8. Smoke 2 task, nhóm client có overlap, lấy ít current-task samples, checkpoint/resume và evaluation đủ nhánh.
9. Full 6 task chỉ sau smoke. Checkpoint mỗi 5 round theo yêu cầu hiện tại. Eval nặng ở cuối task cuối; muốn learning/forgetting matrix thì đánh giá offline checkpoint từng task trên panel cố định và báo protocol riêng.

Không thể lấy model cuối, gắn bank từng task rồi gọi đó là accuracy lịch sử. Learning matrix cần weights và bank đúng thời điểm tương ứng.

## 8. Thứ tự commit và tiêu chí bàn giao

1. `feat: add coverage-aware routing diagnostics` — matched Oracle, global task metrics, paired manifests.
2. `feat: add per-sample subspace router for frozen checkpoints` — module, extractor, notebook và exports.
3. `feat: persist streaming router moments and frozen encoder` — lifecycle, schema, resume, memory budget.
4. `feat: add opt-in DeNICE CGoFed TIP training launcher` — config và launcher riêng; baseline vẫn chọn được.
5. Peer integration chỉ mở thành commit/experiment riêng nếu có encoder alignment và bằng chứng local TIP tốt.

Artifact mỗi run: `protocol.json`, `profile_manifest.json`, `summary.csv`, `per_client_metrics.csv`, `per_task_metrics.csv`, `route_confusion.csv`, `predictions.csv`, `router_diagnostics.json`, `router_profiles.pt`. Protocol ghi mode retrospective/streaming, data access, mask policy, adapter policy, source/checkpoint hash, seeds, split IDs và metric denominator.

## 9. Việc thực hiện đầu tiên

Bắt đầu P0 và P1: notebook đánh giá router mới trên checkpoint cũ, giữ nguyên classifier. Ưu tiên xác định mức tăng thực tế từ per-sample subspace routing với mask công bằng. Chưa đầu tư một full training hoặc peer voting trước khi có bằng chứng này.

Phạm vi triển khai đã chốt: **P0 → P1 → P2**, chưa retrain, chưa bật peer TIP, chưa sửa graph. Các mục streaming và peer trong tài liệu là giai đoạn sau, chỉ xem xét khi frozen evaluation đạt gate. Không thay kiến trúc lớn trước kết quả P1.

Mục tiêu là tăng accuracy phân loại đo được; chưa có cơ sở cam kết routing 95–99% hoặc main accuracy 50%. Frozen retrospective thành công chỉ chứng minh representation cuối hỗ trợ router tốt hơn; streaming còn phải vượt kiểm tra encoder drift và task coverage.
