# Kế hoạch tích hợp DeNICE + CGoFed

Ngày: 2026-10-03. Cơ sở đọc code: HEAD `fdc0587b484d5615853f9335fe6f85e703497dec`.
Trạng thái: thiết kế đề xuất; chưa triển khai hoặc chạy thực nghiệm hybrid.

## 1. Mục tiêu và giả thuyết

**Dùng CGoFed để xử lý sự đánh đổi giữa khả năng học tiếp và giữ kiến thức của DeNICE.** DeNICE đóng băng neuron mature nên khó sửa representation/classifier đã học; mở mature như các biến thể hiện tại lại có thể làm quên kiến thức và lệch router. Hybrid sẽ mở có chọn lọc trọng số cũ, đồng thời hạn chế cập nhật trên không gian biểu diễn của các task cũ.

Thông tin người dùng cung cấp: biến thể hiện tại chỉ tăng khoảng **1 điểm phần trăm accuracy cuối** so với legacy. Chưa có bộ log tương ứng để xác nhận nguyên nhân hoặc độ ổn định giữa các seed. Không dùng con số lịch sử khác làm kết quả của commit hiện tại.

Tên thực nghiệm: `DeNICE-CGoFed-head` cho bản đầu, `DeNICE-CGoFed` khi đã mở rộng phạm vi được bảo vệ. Đây là tích hợp các thành phần CGoFed vào kiến trúc P2P của DeNICE; không tuyên bố tái hiện nguyên bản toàn bộ CGoFed.

| Điểm yếu của DeNICE | Cơ chế hybrid đề xuất | Bằng chứng cần đo |
|---|---|---|
| Mature classifier bị khóa, khó thích nghi task mới | Cho cập nhật mature `fc2` dưới ràng buộc subspace cũ | Accuracy task mới, accuracy task cũ, chuẩn cập nhật mature |
| Mở mature tự do gây interference/forgetting | Projection theo basis activation lịch sử của từng client | Forgetting, old-logit drift, thành phần update trong subspace cũ |
| Không gian plastic dần ít đi | Sau bản head, mở `fc1` có kiểm soát để tái sử dụng capacity | Oracle accuracy theo task, utilization và rank còn tự do |
| Cập nhật peer có thể phá kết quả local | Bảo vệ phần correction do aggregation bằng subspace của receiver | Drift trước/sau aggregation, hiệu quả projection local so với local+peer |
| Router cũ lệch khi backbone thay đổi | Bản đầu giữ mature encoder/BN và đường feature router ổn định | Router accuracy và drift sketch trên tập chẩn đoán cố định |
| Client mới không có bộ nhớ bảo vệ lịch sử | Bootstrap bảo thủ, không tự mở mature được kế thừa khi chưa có basis phù hợp | Accuracy riêng nhóm join/return và coverage của basis |

CGoFed không tự bổ sung mẫu của lớp chưa xuất hiện ở client, không tự sửa class imbalance, lỗi normalization aggregation hoặc routing sai. Các vấn đề này cần theo dõi riêng để không gán mọi thay đổi accuracy cho projection.

## 2. Các ràng buộc đã thấy trong code

- `fed_learning/strategies/fed_incremental/cgofed.py`: có thu activation, tạo basis, importance, hợp nhất subspace và `pre_step`. State/cache hiện tổ chức theo trainer/task/device; hybrid cần thêm ranh giới client.
- `fed_learning/clients/nice_client.py:183`: tạo Adam trực tiếp. Đường training này không tự gọi `CGoFedTrainer.pre_step`; đổi tên strategy sẽ không kích hoạt projection đúng cách.
- `fed_learning/models/nice_model.py:282,319`: gọi `F.conv1d`/`F.linear` với masked weights. Hook trên module Conv/Linear của collector CGoFed hiện tại không bắt được các lần gọi này.
- Collector/projection CGoFed hiện chọn Conv/Linear, chưa xử lý GRU. Không được mở mature GRU rồi coi là đã được bảo vệ.
- DeNICE có mask theo neuron, connection, age, adapter và class support. Projection thông thường có thể tạo lại gradient trên connection bị cấm.
- File basis của CGoFed chứa task/layer trong tên; dùng chung thư mục cho nhiều client có thể ghi đè. Resume dựa vào đường dẫn tạm không đủ bền vững.
- `fed_learning/training/decentralized_denice_il.py`: bootstrap, local training, aggregation, router và continuation state đều cần được tích hợp. Không thay runner bằng `CGoFedServer` vì sẽ đổi protocol.
- Router lấy feature từ conv/GRU; mở `fc2` trước cho phép đánh giá projection mà không chủ động mở các trọng số mature dùng tạo router feature.

## 3. Thiết kế được chọn

### 3.1 Bản đầu: projection classifier, giữ ổn định encoder

Giữ `algorithm=denice`, `mode=decentralized`; thêm variant `cgofed`. Giữ topology, age, structural protection, CANC và router của DeNICE theo cấu hình baseline. Dùng supervised logits của các lớp đã thấy; không dùng `LetLearner` để triệt gradient của các mature class đang cần thích nghi.

Phạm vi mở mature đầu tiên là **`fc2.weight`** có basis và support hợp lệ. Giữ mature bias đóng băng trong bản đầu để tránh đường cập nhật không được projection bảo vệ. Learner bias vẫn học theo quy tắc DeNICE. Mature conv/GRU/fc1, BN và adapter lịch sử giữ quy tắc đóng băng. Các neuron mới vẫn học theo mask DeNICE.

Không bật DER/EWC trong preset này. Bản pure dùng replay off; mọi đối chứng dùng cùng policy replay. Nếu cần balanced replay ở giai đoạn sau, đặt tên rõ `DeNICE-CGoFed+replay` và so sánh với đối chứng replay không projection.

Lợi ích kỳ vọng của bản đầu là sửa classifier mà hạn chế phá phản hồi trên feature cũ. Bản đầu **chưa giải quyết đầy đủ chất lượng encoder thấp**. Nếu oracle accuracy vẫn thấp và classifier-only không cải thiện, bước kế tiếp là `fc1+fc2`, không tăng tùy tiện độ mạnh projection.

### 3.2 State riêng cho từng client

Tạo `CGoFedProjectionState` và engine dùng chung về logic, không dùng chung mutable bank:

```text
client_id -> schema_version, completed_local_tasks, bank_version
             layer/task -> basis_fp32, importance_fp32, input_dimension
                           sample_count, class_support, topology_signature
                           adapter_context, model_revision
             projection_schedule, diagnostics
```

Basis chứa tensor, không chứa đường dẫn bắt buộc tới temp directory. Cache khóa theo `(client_id, bank_version, layer, support_signature, device, dtype)`. Invalidate khi đổi task, mask, dimension, adapter topology hoặc restore checkpoint. Client inactive giữ state; client khác không được kế thừa cache ngầm.

Không gửi basis hay dữ liệu thô qua capsule ở bản đầu. Điều này tránh phải giả định hai client có cùng hệ tọa độ neuron sau quá trình học riêng.

### 3.3 Thu activation đúng đường forward

Thêm API capture rõ ràng vào masked forward, mặc định tắt và không thay kết quả forward. Capture **input thực tế của `fc2`**, sau các phép masking/activation liên quan, kèm class, context và topology. Không dựa vào module forward hooks hiện có.

Cuối mỗi task local, sau aggregation cuối, calibration và consolidation đã được chấp nhận:

1. Chọn mẫu từ train partition của client, ưu tiên cân bằng lớp; giới hạn chính xác số mẫu, không lấy test.
2. Chạy chế độ eval/no-grad, tắt dropout và không cập nhật BN; khôi phục train/eval flags và RNG sau khi thu.
3. Thu theo forward path/context đã khai báo; các adapter context khác nhau phải có metadata riêng. Không gọi basis của một context là bảo vệ mọi context.
4. Tạo Gram/SVD có kiểm tra hữu hạn; chọn rank theo năng lượng và giới hạn bộ nhớ.
5. Commit bank sau khi hoàn tất task. Không dùng basis của task hiện tại để tự khóa chính task đang học.

Với activation `X` có shape `n × d`, phân rã `XᵀX`, giữ basis `U ∈ R^(d×r)` và importance. Khi hợp nhất các task, tái trực chuẩn hóa và chuẩn hóa trọng số như ý tưởng implementation CGoFed hiện tại; không cộng projector không giới hạn làm eigenvalue vượt 1.

Lưu low-rank `U,w` thay vì luôn lưu ma trận `d × d`; tính `G P = ((G U) ⊙ w) Uᵀ`. Ghi rank thực tế, phần năng lượng bị bỏ và số byte. Rank cap làm giảm bảo vệ nên phải được báo cáo.

### 3.4 Projection phải tôn trọng mask

Với ma trận cập nhật `D`, định nghĩa:

`T(D) = D - μ_t D U diag(w) Uᵀ`, trong đó `0 ≤ μ_t ≤ 1`, `0 ≤ w ≤ 1`.

Đây là ràng buộc mềm. Chỉ khi hệ số và trọng số phù hợp mới có projection trực giao hoàn toàn; không gọi mọi cấu hình là “không quên”.

Mỗi output row có tập input connection được phép `J`. Lấy basis trên `J`, tái phân rã để xây projector hợp lệ trong không gian này; nhóm các row có cùng support để cache. Không chỉ project toàn bộ rồi zero mask và tuyên bố còn giữ tính trực giao. Kiểm tra mask lần cuối trước commit parameter; parameter bị đóng băng không được đổi.

Bản đầu không project bias. Giai đoạn sau có thể ghép hằng số 1 vào activation để bảo vệ đồng thời weight/bias; đây là thay đổi riêng, cần đối chứng.

### 3.5 Optimizer: tách bản tham chiếu và bản dùng Adam

**Bản tham chiếu toán học:** SGD momentum=0, weight_decay=0, áp projection vào gradient sau AMP unscale và mask, trước clipping/step. Chạy cặp `μ=0` và `μ>0` cùng SGD để đo đóng góp riêng của projection.

**Bản tương thích pipeline hiện tại:** giữ Adam, nhưng project **parameter delta do Adam đề xuất** sau một optimizer step thành công: `W_final = W_before + T(W_candidate - W_before)`. Mask/frozen restore nằm trong bước commit này. Gọi rõ đây là adaptation theo optimizer; không đồng thời project gradient lần nữa với cùng hệ số.

Lý do: Adam preconditioning và momentum khiến project gradient không bảo đảm actual update còn tuân theo ràng buộc. Với adaptation trên, chỉ delta được commit chịu ràng buộc; momentum vẫn tạo proposal cho bước sau. Clear optimizer state ở connection bị cấm khi mask đổi, kiểm tra delta ở mỗi bước.

AMP skip không được làm tăng counter thành công hoặc gọi nhầm callback commit/replay. Compose callback hiện có thay vì ghi đè. Giữ behavior legacy/DER/EWC khi không chọn hybrid.

### 3.6 Aggregation và router

Giữ cơ chế peer selection/age-aware của DeNICE. Bảo toàn các mature head update local được cho phép; không để aggregation vô tình phục hồi giá trị trước local training.

Biến thể bảo vệ peer correction lưu `W_local`, dựng candidate bằng aggregation hiện tại rồi lấy `C_peer = W_candidate - W_local`. Chỉ trên phần được phép nhận peer, áp `T_receiver(C_peer)` và commit `W_local + T_receiver(C_peer)`. Không project lại toàn bộ local delta lần hai. Log riêng local update và peer correction.

Vấn đề weight dilution do normalize trước mask cần đối chứng sửa riêng, dùng cùng cách sửa cho cả baseline và hybrid nếu bật. Projection không thay thế normalization theo peer thực sự hỗ trợ từng coordinate.

Router vẫn hard theo baseline để so sánh công bằng. Đo routing accuracy và oracle-task classification cùng lúc. Nếu router là nút thắt chính, báo cáo rõ; không dùng oracle task để làm inference thật hoặc tăng metric chính.

### 3.7 Relaxation, client mới và mở rộng

Schedule đầu tiên cố định theo số task **client đã hoàn thành**, bắt đầu thử `μ0 ∈ {0.25,0.5,0.75,1.0}`, decay `{1.0,0.8}`. Không dùng accuracy test để reset hệ số. Adaptive forgetting chỉ bật khi có validation lịch sử tách biệt và policy lưu trữ rõ ràng.

Client mới nhận model nhưng chưa có bank: giữ mature được kế thừa frozen; chỉ cho mở phần có bảo vệ hợp lệ sau khi đã xây basis local. Basis mới chỉ bảo vệ dữ liệu client thực sự quan sát, không chứng minh bảo vệ toàn bộ tri thức donor. Log donor classes chưa có local protection.

Giai đoạn hai mở `fc1+fc2`; giữ mature conv/GRU để hạn chế router drift. Việc đổi `fc1` làm đầu vào `fc2` thay đổi, nên projection trên activation lịch sử không bảo đảm old logits bất biến. Đo old-feature/logit drift; nếu cần refresh bằng balanced replay thì ghi thành biến thể có replay.

Chỉ mở conv khi đã có collector unfold đúng kernel/stride/padding và đánh giá router riêng. GRU cần thiết kế projection recurrent riêng; không nằm trong MVP.

Regularization theo historical model và personalization kiểu CGoFed là giai đoạn tùy chọn sau projection. Ban đầu `lambda_cross_task=0`; không copy full-model blending của server vào DeNICE. Nếu thêm anchor loss, dùng tensor theo layer/mask tương thích và nêu rõ policy similarity; không so prototype từ hai feature spaces khác nhau một cách mặc định.

## 4. Kế hoạch thay đổi theo file

Các đường dẫn dưới đây là file có sẵn, trừ các mục ghi “mới”.

| File | Thay đổi dự kiến |
|---|---|
| `fed_learning/strategies/incremental/denice_variants.py` | Đăng ký `cgofed`, validate preset và tránh kích hoạt DER/EWC |
| `fed_learning/strategies/incremental/denice_cgofed.py` — mới | Per-client state, low-rank bank, support-aware projector, schedule và serialization |
| `fed_learning/strategies/fed_incremental/cgofed.py` | Tách helper số học có thể tái sử dụng; giữ nguyên hành vi baseline CGoFed |
| `fed_learning/models/nice_model.py`, `denice_model.py` | Explicit activation capture; API trainable support; selective mature head masks |
| `fed_learning/clients/nice_client.py` | Optimizer factory/callback trước-sau step; AMP success; giữ đường mặc định cũ |
| `fed_learning/clients/denice_client.py` | Truyền bank đúng client, supervised scope, selective thaw, projection callback |
| `fed_learning/training/decentralized_denice_il.py` | Lifecycle bank, bootstrap/return, peer correction, task-final capture và diagnostics |
| `fed_learning/training/checkpoint_state.py` | Serialize tensor bank và schema; phân biệt eval checkpoint với full continuation |
| `train_incremental_kaggle.py` | Parse/validate cấu hình mới và ghi resolved config |
| `tools/build_denice_launchers.py` | Thêm launcher cgofed, tạo parent directory trước khi ghi config |
| `tests/` — các test mới khi triển khai | Projection, isolation, masks, AMP, resume và integration |

Checkpoint lưu bank FP32, schedule, task mapping, topology signature và config ảnh hưởng thuật toán. Không ép basis qua đường nén model FP16. Resume hybrid thiếu bank phải fail rõ hoặc được người dùng chọn warm-start có tên riêng; không âm thầm coi là continuation tương đương. Checkpoint cũ vẫn đọc được khi chạy variant cũ.

## 5. Trình tự triển khai và điều kiện qua từng bước

### P0 — Khóa baseline và cách đo

- Lưu commit, resolved config, task/class mapping, data split hash, seed, danh sách client/participation, budgets.
- Thu log legacy, DER++, EWC hiện tại nếu muốn so sánh kết quả người dùng; chưa có thì đánh dấu thiếu.
- Chọn fixed evaluation cohort từ đầu; báo cáo thêm client join/return riêng, không chọn cohort sau khi nhìn accuracy.
- Chốt pure/no-replay baseline so với head-thaw `μ=0` và head-thaw có projection. Không thay router/aggregation giữa một cặp đối chứng.

### P1 — Engine và collector

- Viết state/collector/projector độc lập; validate empty bank, zero-rank, rank cap, dimension mismatch.
- Điều kiện qua: capture không rỗng đúng layer; client state tách biệt; projection không tạo update ngoài support; tensor hữu hạn.

### P2 — Local integration `fc2`

- Thêm variant, supervised scope và selective thaw; nối SGD-reference trước, Adam delta sau.
- Điều kiện qua: có actual projected update từ task thứ hai; `μ=0` trùng control cùng code path/optimizer trong tolerance; frozen encoder không đổi do local step.
- Log loss, norm proposal/accepted, old-subspace component, projection coverage, skipped steps và rank.

### P3 — P2P lifecycle và checkpoint

- Nối final task bank, peer-correction option, bootstrap/return, continuation schema và launchers.
- Điều kiện qua: mature update hợp lệ không mất sau aggregation; không trộn bank client; resume giữ task/bank/schedule và tái lập bước tiếp theo trong tolerance đã khai báo.

### P4 — Pilot rồi full benchmark

- Pilot ít client, ít task để phát hiện projection vô hiệu hoặc bóp chết learning. Pilot không đủ để kết luận final accuracy.
- Chạy toàn bộ task schedule trên cùng dữ liệu/participation sau khi pilot đạt; ưu tiên 3 training seed ghép cặp, tăng seed nếu chênh lệch nhỏ/nhiễu.
- Chỉ mở `fc1` sau khi bản head hoạt động đúng và số liệu chỉ ra hạn chế representation/capacity.

### P5 — Mở rộng có bằng chứng

- So `fc2` với `fc1+fc2`; local-only với local+peer protection; schedule cố định với decay.
- Balanced replay/anchor regularization chỉ thêm từng thành phần, có ablation riêng. Không bật mọi thành phần cùng lúc rồi quy kết cải thiện cho CGoFed.

## 6. Ma trận thí nghiệm tối thiểu

| ID | Cấu hình | Câu hỏi |
|---|---|---|
| A | Legacy nguyên trạng | Mốc thực tế của người dùng |
| B | Hybrid head scope, Adam, `μ=0` | Chỉ mở head và thay supervised scope có giúp không? |
| C | Giống B + Adam delta projection local | CGoFed có đóng góp vượt phần mở head không? |
| D | Giống C + receiver peer correction | Aggregation có phá bảo vệ local không? |
| E0/E1 | Cùng head scope, SGD không momentum; `μ=0`/`μ>0` | Kiểm chứng gradient projection với optimizer đơn giản |
| F | Cấu hình thắng + `fc1+fc2` | Có cần cải thiện representation không? |
| G0/G1 | Cùng balanced replay budget; projection off/on | CGoFed còn giúp khi thiếu dữ liệu cũ đã được giảm? |

Giữ data split, seed, task order, participation, training steps, batch size, eval cohort và tuning budget giống nhau trong mỗi cặp. Chênh lệch A–C là hiệu quả của cả hybrid; B–C mới tách được contribution projection. So DER++/EWC ở cùng memory budget hoặc công khai khác biệt storage/compute.

Grid nhỏ ban đầu: energy `{0.90,0.95}`, `μ0` như trên; giảm số tổ hợp bằng pilot và validation. Giá trị hiện tại `num_samples_rep=100` của CGoFed không mặc nhiên đủ cho client nhiều lớp; thử ngân sách theo lớp, báo cáo tổng mẫu thực dùng. Chọn cấu hình bằng validation, không chọn seed hoặc hyperparameter tốt nhất theo test.

## 7. Metrics và tiêu chí thành công

Metric chính: **mean-client final accuracy trên all-seen classes, fixed cohort**, cùng định nghĩa của baseline. Ghi gain bằng điểm phần trăm. Ensemble accuracy là metric phụ riêng.

Metric bắt buộc: macro-F1, balanced accuracy, per-class recall, task-accuracy matrix, forgetting/BWT với định nghĩa thống nhất; accuracy task mới; router accuracy; oracle-task accuracy chỉ để chẩn đoán; nhóm client cũ/join/return; coverage lớp local và coverage bank.

Metric cơ chế: rank/energy, số chiều còn tự do, projection coverage, update residual trong subspace, old-logit drift trước/sau local và aggregation, router-feature drift, frozen-coordinate max delta, RAM/VRAM, wall time, checkpoint bytes và communication bytes.

Thành công cần đồng thời: C tốt hơn B theo các seed ghép cặp; hybrid vượt legacy/current best về final accuracy với mức biến thiên công khai; forgetting giảm mà task mới không sụt nghiêm trọng; chi phí chấp nhận được. Có thể đặt mục tiêu thực dụng **+3 điểm phần trăm so với legacy** để vượt mức +1 hiện tại, nhưng đó là mục tiêu nghiên cứu, không phải dự báo hoặc cam kết.

Nếu gain chỉ quanh 1 điểm phần trăm và biến thiên seed lớn hơn gain, kết luận chưa đủ bằng chứng. Nếu oracle tăng nhưng hard-route không tăng, ưu tiên router. Nếu oracle không tăng và rank gần đầy, nới projection hoặc mở rộng representation; nếu B thắng C thì projection đang quá mạnh hoặc basis không phù hợp. Nếu chỉ client lâu năm được lợi, xem lại bootstrap và local support.

## 8. Kiểm tra cần thực hiện khi triển khai

Đây là danh sách kiểm tra tương lai; kế hoạch này chưa thêm hoặc chạy test implementation.

1. Empty bank/first task là no-op; `μ=0` là no-op; `μ=1,w=1` loại thành phần trong basis trong tolerance số học.
2. Low-rank so dense cho cùng kết quả; union projector hữu hạn, spectrum trong `[0,1]`.
3. Hai client chung task/device không dùng nhầm state/cache hoặc ghi đè file.
4. Functional capture đúng input thực sự; giới hạn sample và class balance; flags/RNG được khôi phục kể cả exception.
5. Mask nhiều kiểu, learner/mature rows, bias, adapter và thay topology; không có forbidden-coordinate drift.
6. SGD-reference và Adam accepted-delta có ràng buộc mong muốn; AMP overflow không commit callback sai.
7. Aggregation không vô tình project local step lần hai; mature local delta được giữ theo policy.
8. Client late join, return, class support rỗng và thiếu basis có hành vi xác định.
9. Full continuation round-trip có bank tensor độc lập temp files; không làm hỏng checkpoint variant cũ.
10. Chạy nhỏ end-to-end ít nhất hai task để chứng minh projection thực sự hoạt động, rồi benchmark đầy đủ.

## 9. Nguồn và giới hạn

- Mã nguồn hiện tại: các file ở mục 2 và 4; đây là căn cứ cho các điểm tích hợp cụ thể.
- Audit nội bộ: `docs/DENICE_IMPROVEMENT_REVIEW_20261003.md`.
- Repository CGoFed gốc: https://github.com/fengjiyuan/cgofed — mô tả phương pháp constrained gradient optimization cho federated class-incremental learning. Kế hoạch này không suy ra hiệu quả trên CICIoT2023 từ kết quả của repository đó.

**Thứ tự ưu tiên cuối cùng: mở head có bảo vệ → bảo vệ correction từ peer → mở rộng representation khi số liệu yêu cầu.** Mỗi bước phải chứng minh đang xử lý một hạn chế cụ thể của DeNICE, thay vì chỉ thêm tên CGoFed vào preset.
