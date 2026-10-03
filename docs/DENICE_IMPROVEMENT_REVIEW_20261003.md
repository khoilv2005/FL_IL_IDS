# DeNICE: điểm yếu và hướng cải thiện từ code hiện tại

Ngày: 03/10/2026. Source: `fdc0587b484d5615853f9335fe6f85e703497dec`.

Người dùng xác nhận: **accuracy cuối của bản kết hợp chỉ cao hơn DeNICE legacy khoảng 1 điểm phần trăm**. Chưa có config/checkpoint của chính phép so sánh này, chưa biết mức tăng thuộc DER++, EWC hay cả hai. Vì vậy báo cáo phân biệt hành vi đã xác nhận trong code, bằng chứng từ run lịch sử, và giả thuyết cần đo trên run mới. Không coi kết quả lịch sử là kết quả của commit hiện tại; không kết luận +1 điểm có ý nghĩa thống kê nếu chưa có training seeds độc lập.

Khuyến nghị ưu tiên: **DeNICE + replay cân bằng theo lớp kiểu FedCBDR + học lại classifier cân bằng kiểu stage 2 của DER mở rộng representation**, cùng sửa hợp đồng router/encoder và kiểm soát aggregation. Không ưu tiên cộng thêm EWC/LwF vào DER++ trước khi xác định nút thắt. Đây là đề xuất triển khai/thí nghiệm, chưa phải một preset đã có hay một mức tăng accuracy được đo.

## 1. Thuật toán đang thực sự chạy

Luồng chính: `train_incremental_kaggle.py` → `training/task_loop.py:run_incremental_training` → `training/decentralized_denice_il.py:run_decentralized_denice_il`.

Mỗi client giữ CNN–GRU/NICE, neuron ages, connection masks, adapter, router và trạng thái continual learning riêng. Theo từng task: xác định lớp có dữ liệu cục bộ; cấp phát neuron; train theo round; tạo capsule; lập đồ thị cộng tác; trộn local deltas có mask; cập nhật router; cuối task consolidate, replay/Fisher và đánh giá.

Các điểm khác mô tả tổng quát trong tài liệu cũ:

- Default hiện tại dùng `fixed_per_class`, `canc_mode=paper`, `clustering_mode=paper`, `aggregation_update_mode=local_delta`, router `binary_cosine` và inference `hard`.
- `select_learner_units()` trả về ngay khi `fixed_task_allocation=True`: việc chọn neuron tau-greedy của NICE không hoạt động trong cấu hình này.
- `paper_context_cluster()` trả `K_t=1` và labels toàn 0, sau đó giới hạn cộng tác bằng adjacency. Không nên đọc telemetry này như Dynamic-K/AP của nhánh legacy.
- Legacy hiện tại trong launcher chung có local replay và router replay; từ “legacy” không mặc nhiên đồng nghĩa replay-free.
- DER/EWC presets thay đổi nhiều thứ đồng thời: supervised forward, train mature neurons, loại replay, router refresh, classifier/transfer options. So sánh preset không phải ablation chỉ thay một loss.

**Có hai thuật toán khác nhau cùng viết tắt DER trong repo:**

| Thành phần | Ý nghĩa | Mã nguồn |
|---|---|---|
| `denice_cl_method=der/derpp` | Dark Experience Replay, NeurIPS 2020 | `strategies/incremental/denice_der.py` |
| `algorithm=der` | Dynamically Expandable Representation, CVPR 2021 | `models/der_model.py`, `strategies/incremental/der.py`, `clients/der_client.py` |

Stage 2 classifier cân bằng được đề xuất ở đây lấy ý tưởng từ DER **CVPR 2021**, chưa nằm trong DeNICE + Dark DER++ đang có.

## 2. Những điểm yếu quan trọng

### A. Hợp đồng router/encoder bị phá khi mở khóa mature neurons

**Xác nhận từ code, chưa đo mức ảnh hưởng trên run mới.**

`denice_variants.py:58` bật `denice_cl_train_mature=True` cho DER/EWC nhưng đặt `denice_router_replay_enabled=False` và giữ `denice_eval_route_mode=hard`.

`clients/denice_client.py:208` cho phép gradient qua neuron đã cấp phát, kể cả mature. Trong khi đó runner lưu binary sketches cũ và sau mỗi round chỉ cập nhật sketches của task hiện tại (`local_task_loop.py:944`). Việc gọi `mark_router_fresh` không chứng minh các sketches lịch sử được mã hóa lại trong feature space mới.

Structural connection masks giữ topology, **không giữ nguyên hàm khi chính mature weights được cập nhật**. Các thresholds cũ và binary sketches cũ vì thế có thể không còn khớp với encoder. Feature mask được lấy từ task đầu (`decentralized_denice_il.py:2539`); nếu không có router replay thay thế `routing_feature_mask`, router cũng không dùng các chiều mới học ở task sau.

Trong task classes rời nhau, với hard mask hợp lệ:

`P(class đúng) = P(route đúng) × P(class đúng | route đúng)`.

Cải thiện classification loss không cứu được mẫu bị router loại mất true class. Đây là một cơ chế hợp lý khiến DER/EWC chỉ tăng ít, nhưng cần đo paired `nomask_accuracy`, hard accuracy và oracle diagnostic trên checkpoint mới để quy kết.

Hướng xử lý:

- DER++: ablation bật router replay cục bộ, mã hóa lại mọi episode bằng encoder hiện tại. Cấu hình này **đã được code chấp nhận**.
- Theo dõi số lần `refresh_replay_router` trả `updated=False`; client mới thiếu replay của các episode kế thừa sẽ không refresh được toàn bank.
- EWC replay-free: cần giữ một encoder router thực sự frozen, hoặc chuyển sang classifier toàn bộ lớp đã biết có calibration. Không bật raw replay rồi vẫn gọi đó là EWC-only.
- Nếu dùng classifier toàn lớp, phải học/calibrate classifier đó; bỏ hard mask một mình có thể làm kết quả tệ hơn.

### B. Client mới nhận model nhưng không có cơ chế bảo vệ đầy đủ tri thức kế thừa

**Xác nhận luồng state; tác động accuracy cần tách cohort.**

`_bootstrap_denice_model()` (`decentralized_denice_il.py:399`) copy model/masks nhưng xóa `ewc_state` và local classifier; replay không được copy từ donor. Việc giữ private state tại client là có chủ đích.

Tuy nhiên DER/EWC lại cho client mới cập nhật mature weights. Ở task đầu mà client đó tham gia, EWC chưa có Fisher/anchor từ donor; DER không có exemplar lịch sử của donor. Do đó trọng số kế thừa có thể bị thay đổi chỉ dựa trên dữ liệu mới. Tuổi neuron không chứng minh client có dữ liệu/loss bảo vệ những lớp mà neuron đó đã học.

Cần báo cáo riêng: clients tham gia từ đầu, clients mới, clients quay lại; đồng thời giữ một cohort cố định khi đo forgetting.

Hướng kết hợp có mục tiêu: snapshot teacher cục bộ từ model bootstrap để distill kiểu LwF, hoặc proximal anchor trên phần model kế thừa kiểu FedProx; tạm giữ router encoder frozen. Teacher phải clone cả topology/masks/adapters. Distillation trên dữ liệu task mới vẫn thiếu support lớp cũ, nên không xem đây là thay thế hoàn toàn replay.

### C. Aggregation có thể làm suy yếu chính local update

**Đã tái hiện bằng hàm aggregation hiện tại.**

Trong `strategies/decentralized/denice_aggregation.py:211`, mỗi peer có mask theo tuổi và label support. Nhưng tại dòng 271, alpha đã được chuẩn hóa trên cả nhóm trước khi nhân mask; alpha không được chuẩn hóa lại theo tọa độ còn hợp lệ.

Với local-delta mode, điểm xuất phát là **model trước local training**. Một phần local update vì thế có thể bị mất nếu peer không hỗ trợ tọa độ đó mà vẫn chiếm mẫu số.

Probe: receiver biết lớp `[0,1]`, peer biết `[1,2]`, alpha `[0.1,0.9]`; local update cho lớp 0 bằng `1.0`. Sau aggregation, update lớp 0 chỉ còn **0.1**, lớp 1 vẫn là **1.0**. Tất cả peers đều hợp lệ ở cấp nhóm vì có lớp 1 chung.

Đây là cơ chế trong công thức hiện tại, chưa phải bằng chứng nó giải thích bao nhiêu accuracy giảm. Cần log:

`eligible_mass_i[p] = sum_j(alpha_ij * M_ij[p])`, đặc biệt theo từng hàng classifier và từng lớp hiếm.

Các ablation:

- Có sẵn: `denice_selective_fc2_peer_rows=True` chuẩn hóa theo support cho các hàng learner fc2. Nó là một phần xử lý, chưa chuẩn hóa mọi mask ở backbone.
- Thiết kế mới: chuẩn hóa theo tọa độ hợp lệ, có denominator floor; hoặc giữ đầy đủ post-local model và chỉ nhận phần cải thiện của peer được local validation chấp nhận. Đây là thay đổi công thức DeNICE, cần đặt tên và kiểm tra riêng.
- Giữ nguyên mature rows, connection masks, local BN và cấu trúc adapters; không dùng median/trimmed mean như một cách sửa mặc định cho vấn đề normalization.

Paper graph còn có thể tạo edge vì neuron ages giống nhau: beta=threshold=0.5 cho similarity **0.5** ngay cả khi hai client không có shared class prototype. Probe xác nhận edge vẫn tồn tại. Pair mask ngăn cập nhật không phù hợp nhưng những peer đó vẫn có thể làm loãng alpha. Tuổi bằng nhau cũng không chứng minh neuron index tương ứng có cùng ý nghĩa sau các lịch sử học khác nhau.

### D. Mất cân bằng lớp và thiếu class support không được DER/EWC giải quyết tự động

**Cơ chế đã xác nhận; có bằng chứng lịch sử về data support.**

Dark DER dùng uniform stream reservoir (`denice_der.py:28`), không có quota theo lớp. `denice_replay_selection=herding` không thay đổi `DERReplay.observe()`; tham số đó thuộc `LocalReplay` legacy. Không thể bật herding cho DER chỉ bằng config và cho rằng đã thay thuật toán lấy mẫu.

EWC lấy ngẫu nhiên tối đa 128 train examples để tính Fisher (`denice_ewc.py:14`). Với dữ liệu lệch lớp, cả replay và Fisher có thể thiếu lớp hiếm. Ví dụ minh họa với tần suất lớp `p=0.001`: xác suất không gặp lớp đó trong 128 mẫu độc lập xấp xỉ 88%; đây là tính toán minh họa, không phải histogram của run mới.

Một lớp không có trong local train thì tăng memory tại client đó không tạo ra positive examples. Paper preparation chỉ đăng ký lớp thực sự local (`_prepare_client_task`); pair aggregation fc2 cũng giới hạn theo giao của label support. Cần xác định mục tiêu là toàn bộ label space trên mỗi client hay local support, và giữ protocol toàn lớp khi so benchmark nếu đó là bài toán chính. Supported-only metrics chỉ là diagnostic bổ sung.

Chú ý: replay batch 32 so với current batch 2048 **không** đồng nghĩa replay loss tự bị nhân hệ số `32/2048`: mỗi loss được lấy mean riêng rồi nhân alpha/beta. Batch nhỏ chủ yếu làm giảm coverage trong từng step và tăng phương sai; phải đo loss/gradient trước khi tăng trọng số.

Hướng xử lý ưu tiên: class-balanced memory và sampling, cân bằng classifier sau representation learning. Các mảnh đã có trong FedCBDR, `LocalReplay`, DER stage 2 và GLFC.

### E. CANC có thể không kích hoạt đúng năng lực mà tên “Expand” gợi ý

**Đã tái hiện bằng controller hiện tại.**

`candle_prototype_drift` (`denice_capacity.py:130`) chỉ so prototype trên lớp chung. Với class-incremental tasks rời nhau, drift trả `defined=False`, giá trị trung tính 0. Không được đổi thành “drift cao” tùy ý vì đó là đại lượng không xác định.

`candle_capacity_plan` chỉ thêm adapter khi `shift=True`. Probe ở utilization 87.5%, task disjoint: action **Expand**, nhưng `adapters_to_add=[]`, chỉ `reserve_to_promote={'fc1':1}`. Cùng capacity, nếu có shared-class drift lớn thì adapter mới được thêm.

Default `fixed_per_class` còn dành capacity theo số lớp local, và tắt tau-greedy selection. Với đủ 6 lớp task 0, số units cấp phát là conv1=12, conv2=24, conv3=48, GRU=18, fc1=48. Với ít lớp local hơn, budget còn nhỏ hơn. Đây không phải số lượng tham số trainable và không chứng minh underfitting; cần so task-0 fitting/validation với backbone đầy đủ.

Probe chỉ allocation, đủ 34 lớp, không CANC extra: task cuối chỉ còn conv1=4, conv2=8, conv3=16, GRU=10, fc1=16. Đây là hệ quả làm tròn `ceil(width/34)` và tiêu budget qua task; không kết luận mọi client thực tế hết capacity, vì local support và CANC khác nhau.

Hướng sửa thiết kế: tách tín hiệu **task mới cần representation** khỏi **domain drift trên cùng lớp**. Quyết định adapter/extra capacity theo training/validation loss hoặc novelty đo trong một feature space chung. Thử `class_blocks` để kiểm tra allocation, hoặc adaptive budget riêng. Không bật đồng thời nhiều cơ chế trước khi đo.

### F. Không thể bật adaptive routing hiện tại như một giải pháp đơn giản

**Đã kiểm tra bằng router thật trên dữ liệu cơ sở trực giao.**

`nice_server.py:518` dùng cosine giữa binary vector và mean prototype, sau đó softmax với temperature 1. Vì các vector không âm, score nằm trong `[0,1]`. Với K episode bank không rỗng:

`max confidence <= exp(1) / (exp(1) + K - 1)`.

| Số episode | Confidence tối đa | Nhánh adaptive ở trường hợp phân tách tốt nhất |
|---|---:|---|
| 2 | 0.7311 | top-k |
| 3 | 0.5761 | top-k |
| 4 | 0.4754 | top-k |
| 5 | 0.4046 | nomask |
| 6 | 0.3522 | nomask |

Các ngưỡng trong `denice_eval.py` là 0.75/0.45. Từ 5 episode đầy đủ trở lên, adaptive sẽ đi nhánh nomask ngay cả với phân tách cosine tốt nhất. Điều này **không giải thích trực tiếp default hard thấp**, nhưng loại bỏ lời khuyên “chuyển adaptive là xong”. Cần calibrate temperature/margin bằng validation; không diễn giải softmax cosine là confidence đã hiệu chỉnh. Mean prototype theo episode còn gom nhiều attack classes đa mode vào một vector; per-class prototypes hoặc discriminative router là ablation khác.

## 3. Bằng chứng có sẵn: điều gì đã được đo và điều gì chưa

Nguồn lịch sử `audit_denice/results3/README.md`: run 100-client split, seed 42, tasks 0–3. Logged task-3 accuracy **33.0014%**, route accuracy **57.4769%**; chỉ 47/80 active clients được chọn eval, trên 50,000 sampled examples. Ensemble **52.372%** là inference protocol khác, không được dùng thay mean personalized accuracy để báo tăng.

Fixed-cohort probe 15 clients, 768 mẫu cân bằng:

- Task-0 oracle accuracy khoảng **23.16% → 24.83%** từ checkpoint task 0 đến task 3.
- Task-0 backbone/nomask accuracy **23.16% → 0.07%** trên cùng cohort và task-0 panel.
- Số này cho thấy old-task discrimination với task đã biết có thể còn, trong khi cạnh tranh giữa các task sụp đổ. Task-0 đã yếu từ đầu; không thể đổ mọi lỗi cho catastrophic forgetting.

Nguồn `audit_denice/results3/router_budget/README.md`: tăng router references từ 20 lên 100/class chỉ tăng classification **0.694 và 1.102 điểm** trên hai reference-sampling seeds, backbone cố định. Trung bình vẫn thiếu **6.933/24 lớp** trong bank local. Đây không phải hai training seeds và không phải hai bản DER/EWC người dùng đang báo.

Các probe lịch sử dùng checkpoint FP16 delta, có khác sklearn version và policy so với hiện tại. Đặc biệt run router cũ có logistic classifier, còn launcher mới mặc định binary-cosine. Chỉ dùng chúng để định hướng giả thuyết.

`all_results_summary.md` ghi FedCBDR có kết quả lịch sử đáng tham khảo, nhưng chưa phải so sánh kiểm soát với DeNICE hiện tại: seed provenance, client cohort, evaluation distribution và model reconstruction khác nhau. Không dùng bảng đó để hứa một mức tăng cụ thể.

## 4. Kết hợp gì từ code hiện có?

| Hướng | Phần có thể tái sử dụng | Điểm yếu nhắm tới | Ưu tiên / hạn chế |
|---|---|---|---|
| DeNICE + replay cân bằng kiểu FedCBDR | `fedcbdr.py:ReplayBuffer`, `fedcbdr_client.py`, `denice_replay.py:LocalReplay` | Thiếu rehearsal cho lớp hiếm, lệch lớp cũ/mới | Cao; giữ memory private, không bê global replay/server vào P2P |
| DeNICE + classifier stage 2 của DER CVPR 2021 | `der_client.py:_create_balanced_batches`, `der_model.py:get_classifier_params` | Logits giữa các task không so sánh được | Cao; viết riêng head DeNICE, không cần nhân toàn bộ extractor |
| DeNICE + calibrated local LDA | `denice_classifier.py`, eval route `local_lda` | Kiểm tra representation còn thông tin nhưng neural classifier yếu | Có sẵn cho legacy; hiện bị validator chặn trong DER/EWC; thiếu local classes vẫn là hạn chế |
| DeNICE + GLFC local compensation | `glfc.py:efficient_old_class_weight`, local distillation | Gradient cũ/mới mất cân bằng | Sau baseline balanced replay; không dùng nguyên proxy server/inversion của GLFC |
| DeNICE + LwF cho bootstrap | `incremental/lwf.py` | Client mới làm hỏng model kế thừa vì không có old memory/Fisher | Có mục tiêu; teacher theo client và clone topology đầy đủ; current-only KD có hạn chế support |
| DeNICE + FedProx trên shared parameters | `federated/fedprox.py` | Local drift gây peer updates lệch nhau | Chỉ khi pre/post aggregation diagnostics xác nhận; runner hiện truyền `global_params=None` nên đổi mu đơn thuần không thêm penalty |
| DeNICE + CGoFed projection | `fed_incremental/cgofed.py:pre_step` và SVD spaces | Drift representation trên phần shared | Chi phí cao; NICEClient không gọi `trainer.pre_step`, cần nối gradient hook; không bảo vệ GRU tự động chỉ bằng đổi trainer |
| DeNICE + ReFed sample selection | `refed_client.py:_select_and_cache`/PIM | Chọn exemplar hữu ích trong memory hạn chế | Sau baseline cân bằng; per-sample gradients tốn compute, dễ chọn outlier nếu thiếu quota |
| DeNICE + EWC + DER++ cùng lúc | Các loss hiện có | Thêm parameter anchoring lên replay | Chưa ưu tiên; có thể cộng ràng buộc dư trong khi router/classifier vẫn là nút thắt; chưa có preset hợp lệ |

Nếu cần giữ replay-free, lựa chọn hẹp hơn: frozen router encoder + classifier/calibration học trên support hợp lệ + EWC hoặc LwF/gradient projection. Không kỳ vọng regularization tự cung cấp thông tin positive của lớp chưa từng có tại client.

## 5. Thiết kế được đề xuất trước tiên

Tên mô tả: **DeNICE với replay cân bằng và classifier học hai giai đoạn**. Cần công khai đây là một biến thể thích nghi, chưa phải tái lập nguyên bản FedCBDR/DER.

1. **Stage representation:** giữ age/mask/peer protocol; train current data và memory private có quota lớp. Khởi đầu bằng balanced CE; thêm KD nhỏ chỉ khi baseline cho thấy quên representation. Đây là bước kiểm tra xem dark-logit loss có đang giữ cả dự đoán yếu của mô hình cũ hay không.
2. **Stage classifier:** sau representation/aggregation, freeze backbone và BN; fit một head trên dữ liệu current + memory được cân bằng theo lớp. Dùng tất cả lớp client có positive support; mask unseen labels đúng theo ID. Head cần được lưu trong algorithm checkpoint và phục hồi đúng khi resume.
3. **Inference:** dự đoán toàn bộ lớp đã biết từ head đã học cân bằng. So với hard routing trên cùng panel; không chọn policy bằng test. Router có thể là nhánh hỗ trợ được chọn bằng validation. Các classes client thiếu phải được giải quyết qua giao thức học/transfer hoặc báo coverage, không âm thầm bỏ khỏi metric chính.
4. **Aggregation:** log pre/post score và eligible mass; giữ local head nếu peer không có support tương ứng. Thử selective fc2 normalization trước khi mở rộng normalized masked aggregation trên backbone.
5. **Cold start:** giữ phần model/router kế thừa ổn định cho tới khi client có cơ chế bảo vệ đủ; có thể thử local teacher LwF hoặc anchor FedProx cho đúng cohort này.

Weight alignment có sẵn trong DERModel là một ablation bổ sung cho head, không phải copy trực tiếp: code đó giả định old/new classes liên tiếp ở cuối classifier; DeNICE dùng nhãn toàn cục, local support có thể không liên tiếp. Cần dùng explicit class IDs, xử lý bias, và không scale hàng mature trong phiên bản tuyên bố hard-freeze tuyệt đối.

## 6. Thí nghiệm tối thiểu để tránh thêm loss một cách mù quáng

**Trước hết dùng checkpoint mới nhất, giữ nguyên trọng số:** so hard, nomask và oracle-hard diagnostic trên cùng test indices và fixed client cohort; báo per-class recall, confusion matrix, support và route accuracy. Oracle chỉ định vị lỗi, không là metric deploy. Refit/calibrate trên train/validation private; test chỉ scoring.

| Kết quả | Hướng ưu tiên |
|---|---|
| Nomask của DER++ tốt lên rõ nhưng hard gần như không tăng | Router/calibration đang che lợi ích continual learning |
| Oracle cũng yếu ngay task 0 | Data/preprocessing, local label imbalance, capacity và optimization |
| Oracle cũ ổn, nomask cũ giảm mạnh | Cạnh tranh logits cũ/mới; balanced classifier stage 2 |
| Score local tốt rồi giảm ngay sau aggregation | Eligible mass, neuron-coordinate alignment, negative transfer |
| Chủ yếu client mới giảm | Bootstrap teacher/anchor và thiếu replay/Fisher kế thừa |

Sau đó chạy các ablation **tách biệt**, cùng train steps, memory budget, data split và source commit:

| ID | Thay đổi so với parent | Có chạy bằng code hiện tại? |
|---|---|---|
| A0 | Tái lập chính xác legacy và bản kết hợp người dùng đang so | Có; cần saved configs thực tế |
| A1 | DER++ + router replay; parent DER++ | Có; theo dõi refresh coverage/skip reasons |
| A2 | Class-balanced current batches; parent DER++ | Có: `denice_batch_sampling=class_balanced`, class weights `none` |
| A3 | Selective fc2 peer rows; parent DER++ | Có: `denice_selective_fc2_peer_rows=True` |
| A4 | Balanced local head stage 2 | Neural head cần triển khai; legacy LDA có sẵn làm diagnostic |
| A5 | Class-balanced memory/loss kiểu FedCBDR | Cần nối vào DeNICE variant riêng; setting herding không sửa DERReplay |
| A6 | Bootstrap LwF/anchor | Cần triển khai và đo riêng new-client cohort |

Ví dụ override A1, áp lên launcher DER++ trong **run mới**:

```json
{
  "denice_router_replay_enabled": true,
  "denice_router_replay_per_class": 64
}
```

Ví dụ A2 (riêng với A1 trong vòng kiểm tra đầu):

```json
{
  "denice_batch_sampling": "class_balanced",
  "denice_class_weight_mode": "none"
}
```

Không ghép classifier/transfer với DER/EWC chỉ bằng JSON: `variant_config` hiện từ chối cấu hình đó. Mở rộng cần tên biến thể, checkpoint state, compatibility checks và kiểm tra train/eval nhất quán. A1 không hợp lệ cho EWC-only.

Dùng pilot ít client nhưng đủ dữ liệu và tối thiểu tới task 2, vì nút thắt phải xuất hiện khi đổi task. Không dùng 300 mẫu/client với batch 2048 rồi xem một optimizer step là benchmark. Chỉ các hướng có lợi trên validation mới được xác nhận trên full 100-client protocol. Nếu đổi số clients, ghi rõ graph/participation cũng đổi nên pilot không ước lượng trực tiếp full-run gain.

Đánh giá cuối: accuracy trên phân phối test chính, macro-F1, per-class recall, old/new accuracy và forgetting trên cùng client-task pairs. Thêm panel cân bằng như diagnostic, không thay distribution rồi gọi đó là tăng accuracy. Giữ cohort eval cố định giữa variants; `_select_eval_clients` có thể thay tập client tùy full class coverage. Ensemble là metric riêng. Dùng ít nhất vài training seeds ghép cặp nếu compute cho phép, báo chênh lệch từng seed và độ bất định; không nhầm reference/evaluation seeds với training seeds.

## 7. Probe đã thực hiện trong lần review này

Script: `audit_denice/review_20261003/mechanism_probe.py`.

Kết quả: `audit_denice/review_20261003/mechanism_evidence.json`.

```powershell
.\.venv-audit\Scripts\python.exe audit_denice/review_20261003/mechanism_probe.py
```

Đã chạy trên CPU với torch 2.14.0; xác nhận aggregation dilution, CANC disjoint-class behavior, paper graph edge, confidence bound của binary-cosine, allocation budget và preset constraints. Probe không train IDS, không đọc/sửa checkpoint thực và không chứng minh accuracy của phương án mới. Không sửa production algorithm/config hay các file tools chưa được Git theo dõi của người dùng.

## 8. Cơ sở nghiên cứu đối chiếu

- [NICE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gurbuz_NICE_Neurogenesis_Inspired_Contextual_Encoding_for_Replay-free_Class_Incremental_Learning_CVPR_2024_paper.html): kiến trúc theo maturation và context inference. Việc thích nghi sang decentralized clients có feature spaces khác nhau cần kiểm chứng riêng.
- [Dark Experience Replay, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html): replay logits từ trajectory; không tự giải quyết protocol router của DeNICE.
- [DER, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.pdf): học representation mở rộng theo hai giai đoạn; là thuật toán khác với Dark DER.
- [GLFC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.pdf): local class-aware compensation/distillation và một proxy server. Đề xuất ở đây chỉ tái sử dụng phần local nếu cần giữ P2P.
- [Weight Aligning, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Maintaining_Discrimination_and_Fairness_in_Class_Incremental_Learning_CVPR_2020_paper.pdf): phân biệt giữ discrimination trong lớp cũ với hiệu chỉnh bias cũ/mới; gợi ý ablation classifier thay vì chỉ tăng KD.

Các kết quả bài báo không là bằng chứng tổ hợp đề xuất sẽ tăng accuracy trên CICIoT2023 của repo này.
