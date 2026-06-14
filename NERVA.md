Cơ chế mới: NERVA — Neurogenesis-gated Efficient Recurrent Vector Adapters
NERVA kết hợp logic “nhẹ, không cần task-id” của NICE với logic “truyền đặc trưng tuần tự, chống feature diffusion” của RNE, nhưng tránh hai điểm yếu chính: NICE hết neuron và RNE phình to do thêm expert theo task.
Ý tưởng trung tâm:
Không thêm full task expert như RNE, cũng không chỉ tiêu thụ neuron cố định như NICE. Thay vào đó, NERVA dùng một backbone chia theo “tuổi neuron” như NICE, nhưng khi capacity bắt đầu cạn, nó không mở rộng bằng network mới; nó thêm hoặc kích hoạt micro recurrent adapters cực nhỏ tại các layer quan trọng, được nối tuần tự bằng mapping module kiểu RNE. NICE chọn neuron tối thiểu theo activation coverage và đóng băng/prune neuron già; RNE dùng shared mapping module để truyền feature từ expert trước sang expert sau, đồng thời dùng compressed expert, decoupled classifier và pseudo-feature bias correction. NERVA giữ lại những phần đó nhưng thay “task expert lớn” bằng age-bank + low-rank adapter. NICE vốn chọn tập neuron age-1 nhỏ nhất đủ bao phủ một tỷ lệ activation τ, rồi mature/freeze chúng; nhưng chính paper cũng chỉ ra NICE cuối cùng sẽ cạn neuron và đề xuất hướng giảm tuổi/tái cấp phát capacity. RNE thì giải quyết feature diffusion bằng shared mapping module giữa các expert và RNE-compress giảm chiều layer xuống 1/4, khoảng 6% tham số so với network gốc cho expert bổ sung.
________________________________________
1. Trực giác thiết kế
NICE mạnh ở ba điểm: một là kiến trúc cố định, nhẹ; hai là neuron có tuổi nên có cơ chế stability-plasticity rõ ràng; ba là context detector tự suy ra episode/class context lúc inference, không cần task-id. NICE freeze incoming connections vào neuron già hơn age-1 và prune connection từ neuron trẻ sang neuron già, nhờ đó neuron mature không bị update ngược bởi task mới. NICE cũng dùng binary activation/context detector để mask output neuron theo context lúc test, chi phí logistic regression nhỏ hơn nhiều so với replay.
RNE mạnh ở chỗ nó nhận ra vấn đề của network expansion thông thường: expert mới có thể tái biểu diễn sai old classes, gây feature diffusion. RNE khắc phục bằng cách cho feature maps truyền tuần tự qua shared mapping module, để expert mới “điều chỉnh từ feature cũ” thay vì re-extract mọi thứ từ đầu. RNE còn dùng decoupled classifier và pseudo-feature generation để giảm feature confusion/classifier bias.
NERVA lấy kết luận sau:
Neuron mature của NICE chính là “old expert” ở dạng rải trong backbone. Micro-adapter của NERVA chính là “new expert” ở dạng cực nhỏ. Recurrent mapping của RNE nối old mature feature sang new plastic capacity, giúp học task mới mà tiêu ít neuron hơn.
________________________________________
2. Kiến trúc tổng thể của NERVA
NERVA gồm 5 khối:
2.1. Shared Age-Backbone
Backbone (B) được chia theo channel/neuron age như NICE:
[
N_l = N_l^{0} \cup N_l^{1} \cup N_l^{2+}
]
Trong đó:
•	(N_l^0): neuron dự trữ, chưa dùng.
•	(N_l^1): neuron plastic cho episode hiện tại.
•	(N_l^{2+}): neuron mature, đóng vai trò memory.
•	Early layers được ưu tiên dùng chung vì NICE quan sát rằng early layers nhanh cạn nhưng thường học low-level features có tính tổng quát cao.
2.2. Recurrent Micro-Adapter Bank
Thay vì thêm một task expert hoàn chỉnh như RNE, NERVA chỉ thêm micro-adapter tại các layer quan trọng, đặc biệt là layer đổi kích thước feature map hoặc đổi số channel. RNE cũng chỉ xây connection ở key layers vì dense connection quá tốn, có thể giảm plasticity và đưa nhiễu vào feature mới.
Một adapter tại layer (l), episode (t):
[
A_{l,t}(h) = U_{l,t}\sigma(V_{l,t}h)
]
với bottleneck rank (r_l \ll d_l). Đây là low-rank adapter, không phải full conv block. Ví dụ thực dụng:
[
r_l = \max(4, d_l/16)
]
Nếu cần nhẹ hơn, dùng depthwise-separable adapter hoặc LoRA-style (1 \times 1) adapter.
2.3. Shared Recurrent Mapping Module
Tại key layer (l), NERVA dùng mapping module (M_l) kiểu RNE nhưng chia sẻ qua episode:
[
\tilde{h}{l,t} = B_l^{plastic}(h{l-1,t})
]
[
h_{l,t}^{new} = \tilde{h}{l,t} + \alpha{l,t} M_l(h_{l}^{mature}) + A_{l,t}(h_{l-1,t})
]
Trong đó:
•	(h_l^{mature}): feature từ neuron mature/old age-bank.
•	(M_l): mapping module chia sẻ, rất nhỏ.
•	(\alpha_{l,t}): gate học được hoặc do context router sinh ra.
•	(A_{l,t}): micro-adapter chỉ thêm khi capacity pressure cao hoặc novelty cao.
Khác với RNE, NERVA không forward qua tất cả expert. Nó chỉ forward một backbone và một số adapter được router bật. Vì vậy compute gần NICE hơn là RNE.
2.4. Context Router thay cho chuỗi logistic regression
NICE dùng thresholded binary activations và chuỗi logistic regression để suy context. Paper cũng nêu hạn chế: chaining logistic regressions khá đơn giản và quan sát mỗi episode có chiều khác nhau, có thể cần cơ chế sequential/attention tốt hơn.
NERVA thay bằng Tiny Recurrent Context Router:
[
q(e|x) = R_{\psi}(b_1(x), b_2(x), ..., b_L(x))
]
Trong đó:
[
b_l(x) = \mathbb{1}[A_l(x) > \mu_l + \kappa\sigma_l]
]
Router có thể là GRU nhỏ hoặc attention pooling rất nhỏ trên binary activation sketches. Output:
•	Xác suất episode/context (q(e|x)).
•	Age mask (m_e).
•	Adapter mask (g_t).
•	Logit mask cho classifier.
2.5. Causal Age-Decoupled Classifier
NERVA dùng classifier kiểu RNE nhưng gắn với age/context thay vì task expert lớn:
[
H = {h_1, h_2, ..., h_t}
]
•	Subclassifier cũ update rất chậm hoặc freeze.
•	Subclassifier mới update mạnh hơn.
•	New classifier được phép nhìn feature sequence/age feature tổng hợp.
•	Old classifier không bị new feature làm lệch quá mức.
RNE dùng decoupled classifier để giảm feature confusion và cho new subclassifier sử dụng feature sequence trong khi old subclassifier ít bị ảnh hưởng bởi new features.
________________________________________
3. Cơ chế chống “hết neuron” của NICE
NERVA không đợi đến khi neuron cạn mới xử lý. Nó dùng Capacity Controller ở mỗi layer.
3.1. Đo áp lực capacity
Tại layer (l):
[
P_l = 1 - \frac{|N_l^0|}{|N_l|}
]
•	(P_l) thấp: còn nhiều neuron dự trữ, học như NICE.
•	(P_l) trung bình: giảm số neuron age-1 được giữ lại, tăng recurrent transfer.
•	(P_l) cao: không cấp thêm neuron trừ khi novelty rất lớn; bật micro-adapter hoặc nén/recycle neuron mature.
3.2. Chọn neuron bằng activation + gradient + novelty
NICE chọn neuron theo activation coverage. NERVA mở rộng score:
[
I_i^l = \lambda_A \bar{A}_i^l + \lambda_G \overline{|\nabla_i L|} + \lambda_C C_i^l - \lambda_R R_i^l
]
Trong đó:
•	(\bar{A}_i^l): activation trung bình.
•	(\overline{|\nabla_i L|}): độ quan trọng theo gradient.
•	(C_i^l): context uniqueness, neuron có giúp phân biệt context mới không.
•	(R_i^l): redundancy/correlation với neuron mature khác.
Chọn tập nhỏ nhất:
[
S_l = \arg\min |S_l|
]
sao cho:
[
\sum_{i \in S_l} I_i^l \geq \tau_l \sum_{i \in N_l^1} I_i^l
]
Với (\tau_l) không cố định 0.95 như NICE, mà adaptive:
[
\tau_l = clip(\tau_0 - \rho P_l + \delta Novelty_t,\tau_{min},\tau_{max})
]
Nghĩa là:
•	Capacity càng cạn, giữ ít neuron hơn.
•	Novelty/domain shift càng cao, cho phép giữ nhiều neuron hơn.
•	Nếu old mature features đã đủ tốt, ưu tiên adapter thay vì neuron mới.
3.3. Neuron leasing/recycling
Neuron mature không bị xoá bừa. NERVA dùng 3 trạng thái mature:
•	Mature-locked: neuron rất quan trọng, không recycle.
•	Mature-shared: neuron dùng chung nhiều context, được giữ làm shared feature.
•	Mature-recyclable: neuron ít dùng, trùng lặp cao, có thể nén rồi giải phóng.
Trước khi recycle, NERVA distill contribution của nhóm neuron đó vào:
1.	micro-adapter,
2.	feature moments,
3.	classifier prototypes,
4.	binary activation sketch memory.
Điều kiện recycle:
[
Usage_i < \epsilon_u,\quad Redundancy_i > \epsilon_r,\quad \Delta Acc_{proxy} < \epsilon_a
]
Trong đó (\Delta Acc_{proxy}) đo bằng pseudo-feature validation, không cần raw old images.
________________________________________
4. Cơ chế chống “phình model” của RNE
RNE thêm expert theo task, dù RNE-compress đã giảm xuống khoảng 6% tham số của network gốc cho expert bổ sung. NERVA đi xa hơn bằng cách không thêm expert hoàn chỉnh.
Ba mức tăng tham số:
Mức	Khi nào dùng	Tham số thêm
Level 0: NICE-only	Capacity còn nhiều	Không thêm
Level 1: Recurrent mapping	Task mới giống domain cũ	Chỉ dùng shared (M_l), gần như constant
Level 2: Micro-adapter	Novelty cao hoặc layer cạn neuron	Low-rank adapter tại key layers
Level 3: Emergency expansion	Domain shift lớn	Thêm tiny block rất nhỏ, không thêm full network
Vì vậy tăng trưởng tham số:
[
P_{NERVA} \approx P_{backbone} + P_{router} + \sum_l P(M_l) + \sum_{t,l \in K} P(A_{l,t})
]
Trong khi RNE/full NE có xu hướng:
[
P_{RNE} \approx P_{backbone} + tP_{expert}
]
NERVA chỉ tăng theo số adapter thực sự cần, không tăng tuyến tính theo task bằng full expert.
________________________________________
5. End-to-end methodology
Giai đoạn 0: Khởi tạo
Khởi tạo backbone (B), chia neuron/channel theo layer.
•	Tất cả hidden neurons ban đầu ở age-0.
•	Input layer luôn shared.
•	Output head mở theo class mới.
•	Khởi tạo shared mapping modules (M_l) tại key layers.
•	Khởi tạo context router (R_\psi).
•	Khởi tạo age-decoupled classifier (H).
Memory lưu:
[
\mathcal{M} = {b_c, \mu_c, \sigma_c, p_c, age_c}
]
Trong đó:
•	(b_c): binary activation sketch.
•	(\mu_c, \sigma_c): feature mean/variance theo class.
•	(p_c): prototype/logit prototype.
•	Không lưu raw images mặc định.
________________________________________
Giai đoạn 1: Nhận episode mới (D_t)
Với episode (t), model nhận dữ liệu mới:
[
D_t = {(x_i, y_i)}
]
Không cần task-id lúc inference, nhưng lúc training biết episode hiện tại để train router.
Tính novelty:
[
Novelty_t = 1 - \max_e sim(\mu(D_t), \mu_e)
]
Nếu novelty thấp, dùng lại mature features nhiều hơn. Nếu novelty cao, cấp neuron/adapters nhiều hơn.
________________________________________
Giai đoạn 2: Tạm kích hoạt neuron age-0 thành age-1
Giống NICE:
[
N_l^0 \rightarrow N_l^1
]
Nhưng chỉ kích hoạt một phần theo capacity budget:
[
|N_l^{1,active}| = Budget(P_l, Novelty_t)
]
Nếu layer đã gần cạn, không mở toàn bộ age-0 mà mở top-k candidates hoặc bật adapter.
________________________________________
Giai đoạn 3: Forward recurrent neurogenesis
Với input (x), tại layer (l):
[
h_l^{mature} = B_l^{N^{2+}}(h_{l-1})
]
[
h_l^{plastic} = B_l^{N^1}(h_{l-1})
]
[
h_l = h_l^{mature} + h_l^{plastic} + \alpha_l M_l(h_l^{mature}) + g_l A_{l,t}(h_{l-1})
]
Trong đó:
•	(M_l(h_l^{mature})) truyền knowledge cũ sang vùng plastic.
•	(A_{l,t}) chỉ bật khi cần.
•	(g_l \in {0,1}) hoặc soft gate.
•	Không có đường từ neuron trẻ làm thay đổi neuron già.
________________________________________
Giai đoạn 4: Loss training chính
Tổng loss:
[
L = L_{CE}^{new} + \lambda_{ctx}L_{ctx} + \lambda_{sp}L_{sparse} + \lambda_{align}L_{recurrent} + \lambda_{cal}L_{calib}
]
Trong đó:
4.1. New class CE
[
L_{CE}^{new} = CE(H(f(x)), y)
]
4.2. Context loss
[
L_{ctx} = CE(R_\psi(b(x)), e_t)
]
Giúp router nhận ra episode/context từ binary activations.
4.3. Sparsity loss
[
L_{sparse} = \sum_l ||g_l||1 + \sum_l ||A{l,t}||_1
]
Ép adapter và gate nhỏ nhất có thể.
4.4. Recurrent alignment loss
[
L_{recurrent} = ||h_l^{plastic} - stopgrad(M_l(h_l^{mature}))||_2^2
]
Mục tiêu: neuron/adapters mới học bằng cách “điều chỉnh từ old mature feature”, không re-extract hỗn loạn. Đây là tinh thần của RNE: expert mới nhận feature từ expert trước qua mapping module thay vì tái trích xuất redundant.
4.5. Calibration loss
Dùng pseudo-features của old classes để classifier không bias về class mới.
________________________________________
Giai đoạn 5: Chọn neuron sống sót
Sau mỗi (p) epoch:
1.	Tính score (I_i^l).
2.	Sort giảm dần.
3.	Chọn tập nhỏ nhất (S_l) đạt adaptive coverage (\tau_l).
4.	Neuron trong (S_l) giữ age-1.
5.	Neuron còn lại quay về age-0.
6.	Prune connection từ neuron trẻ sang neuron già.
7.	Freeze incoming connection của neuron mature.
Đây là NICE nhưng có thêm capacity-aware score và adapter fallback.
________________________________________
Giai đoạn 6: Maturation cuối episode
Cuối episode:
[
N_l^1 \rightarrow N_l^2
]
[
N_l^a \rightarrow N_l^{a+1}, a \geq 2
]
Các neuron mature được phân loại:
•	locked,
•	shared,
•	recyclable.
Adapter (A_{l,t}) cũng được mature:
•	Nếu được dùng nhiều: giữ.
•	Nếu ít dùng: merge vào shared adapter (M_l) hoặc prune.
•	Nếu trùng với old adapter: low-rank merge.
________________________________________
Giai đoạn 7: Update memory không replay ảnh
Cập nhật memory bằng feature statistics và binary sketches:
[
\mu_c = \frac{1}{n_c}\sum_i z_i
]
[
\sigma_c^2 = \frac{1}{n_c}\sum_i(z_i - \mu_c)^2
]
[
b_c = Quantize(\mathbb{1}[A(x) > \theta])
]
Memory này phục vụ:
•	context router,
•	pseudo-feature generation,
•	classifier calibration,
•	neuron recycling validation.
NICE đã dùng binary activations cho context inference; RNE dùng mean/variance feature để generate pseudo-features và retrain classifier.
________________________________________
Giai đoạn 8: Pseudo-feature bias correction
Sinh pseudo-features cho old classes:
[
\hat{z}_c = \mu_c + \sigma_c \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
]
Hoặc dùng cách RNE-style: chọn một new class xa old classes nhất làm generator rồi transform theo mean/variance old class.
Tạo balanced feature set:
[
\hat{D}{bal} = {z{new}} \cup {\hat{z}_{old}}
]
Freeze backbone, chỉ retrain classifier/router nhỏ:
[
\min_H CE(H(\hat{z}), \hat{y})
]
Mục tiêu: giảm classifier bias mà không cần raw replay. RNE paper mô tả pseudo-feature generation để tái tạo old category features từ new task samples, tạo balanced feature set rồi retrain classifier.
________________________________________
6. Inference end-to-end
Với test sample (x):
1.	Forward một lần qua shared backbone.
2.	Tạo binary activation sketch:
[
b(x) = {b_1(x),...,b_L(x)}
]
3.	Context router dự đoán:
[
q(e|x)=R_\psi(b(x))
]
4.	Chọn top-1 hoặc top-k context:
[
E^* = topk(q(e|x))
]
5.	Bật age mask + adapter mask tương ứng:
[
m = Mask(E^*)
]
6.	Mask logits không liên quan hoặc reweight theo context posterior:
[
logits = \sum_{e \in E^*} q(e|x) H_e(f_m(x))
]
7.	Predict class:
[
\hat{y} = \arg\max softmax(logits)
]
Khác biệt quan trọng với RNE: không cần chạy tất cả task expert. Khác biệt với NICE: nếu neuron cũ cạn, vẫn còn adapter nhỏ và recycling/compaction để tiếp tục học.
________________________________________
7. Pseudocode cấp thuật toán
Algorithm NERVA: Neurogenesis-gated Efficient Recurrent Vector Adapters

Input:
  Episodes D1...DT
  Backbone B with age-labeled neurons
  Shared mapping modules M_l at key layers
  Context router R
  Decoupled classifier H
  Memory M = {}

For episode t = 1...T:

  1. Estimate novelty and layer capacity pressure:
       Novelty_t = novelty(D_t, M)
       P_l = 1 - |N_l^0| / |N_l|

  2. Allocate plastic capacity:
       Activate subset of age-0 neurons as age-1
       If P_l high or Novelty_t high:
           instantiate or activate micro-adapter A_l,t

  3. Train for p epochs:
       For each batch (x, y) in D_t:
           Forward through mature + plastic neurons
           Inject recurrent mapped mature feature:
               h_l = h_l^mature + h_l^plastic + alpha_l M_l(h_l^mature) + g_l A_l,t(h)
           Predict logits through causal age-decoupled classifier
           Compute:
               L = CE + context_loss + sparsity_loss + recurrent_alignment
           Update only:
               age-1 neurons
               current adapters
               mapping modules with low LR
               new classifier head
               context router

  4. Every p epochs:
       Score age-1 neurons
       Keep minimal subset satisfying adaptive coverage tau_l
       Return unimportant age-1 neurons to age-0
       Prune younger-to-older connections

  5. End episode:
       Mature surviving age-1 neurons to age-2
       Freeze mature neurons
       Compress/merge weak adapters
       Identify recyclable mature neurons

  6. Update memory:
       Store binary activation sketches
       Store feature moments mu_c, sigma_c
       Store class prototypes

  7. Bias correction:
       Generate pseudo-features for old classes
       Build balanced feature set
       Freeze backbone
       Retrain only classifier/router for few epochs

Return:
  Lightweight incremental model with age-gated backbone, recurrent adapters,
  context router, and calibrated classifier
________________________________________
8. Vì sao NERVA nhẹ hơn nhưng vẫn giữ hiệu năng
Vấn đề	NICE	RNE	NERVA
Catastrophic forgetting	Tốt nhờ freeze/prune neuron mature	Tốt nhờ old experts frozen	Tốt nhờ neuron mature + recurrent transfer
Hết neuron	Có, đặc biệt early layers	Không hết neuron nhưng phình expert	Giảm tiêu neuron bằng recurrent reuse + adapter + recycling
Model size	Nhẹ	Tăng theo expert/task	Gần NICE, chỉ thêm micro-adapter khi cần
Compute inference	Gần 1 backbone	Có thể phải dùng nhiều expert/feature sequence	1 backbone + top-k adapter/context
Không cần task-id	Có	Có CIL inference nhưng expert sequence lớn	Có, dùng context router
Feature diffusion	NICE không xử lý theo expert sequence	RNE xử lý tốt bằng recurrent connection	Dùng recurrent mapping từ mature feature sang plastic feature
Classifier bias	NICE chủ yếu context mask	RNE có decoupled classifier + pseudo-feature	Giữ RNE-style calibration nhưng không replay ảnh
________________________________________
9. Công thức độ phức tạp
9.1. Tham số
Với:
•	(P_B): backbone.
•	(P_M): shared mapping modules.
•	(P_A): tổng adapter đang giữ.
•	(P_R): router.
•	(P_H): classifier.
[
P_{NERVA} = P_B + P_M + P_A + P_R + P_H
]
Trong đó:
[
P_A = \sum_{t,l} 2d_lr_l
]
và (r_l \ll d_l). Nếu adapter rank (r_l=d_l/16), adapter thấp hơn nhiều so với full block.
9.2. FLOPs
[
F_{NERVA} \approx F_B + F_{topk-adapter} + F_R
]
Vì router chỉ bật top-k adapter, inference không scale mạnh theo số episode.
9.3. Memory
[
Mem = Mem(binary\ sketches) + Mem(\mu,\sigma) + Mem(prototypes)
]
Không cần lưu raw images mặc định. NICE cũng nhấn mạnh activation memory rất nhỏ so với image memory trong thiết lập của họ.
________________________________________
10. Training protocol đề xuất để kiểm chứng
Nên đánh giá NERVA trên cùng benchmark của NICE/RNE:
•	Split CIFAR-100: 5, 10, 25 episodes.
•	ImageNet-100.
•	TinyImageNet hoặc ImageNet-1K nếu có compute.
•	Metrics:
o	Average Accuracy (A_{avg})
o	Final Accuracy (A_T)
o	Forgetting (F)
o	Params
o	FLOPs
o	Active params at inference
o	Memory budget
o	Context accuracy
o	Neuron depletion rate per layer
Ablation bắt buộc:
Ablation	Mục đích
NERVA without adapters	Kiểm tra còn bị hết neuron không
NERVA without recurrent mapping	Kiểm tra feature diffusion
NERVA with logistic context detector	So với NICE detector
NERVA without pseudo-feature correction	Đo classifier bias
NERVA without recycling	Đo lợi ích capacity controller
Full adapter vs key-layer adapter	Đo trade-off FLOPs/accuracy
Top-1 context vs Top-k context	Đo routing robustness
________________________________________
11. Kết luận thiết kế
NERVA là cơ chế lai hợp lý nhất giữa NICE và RNE nếu mục tiêu là:
1.	Nhẹ như NICE: backbone chính vẫn cố định, neuron được chọn sparse, memory là binary sketches + feature moments.
2.	Không cạn nhanh như NICE: dùng recurrent transfer để tái sử dụng mature features, adaptive neuron budget, micro-adapter, neuron recycling.
3.	Không phình như RNE: không thêm full task expert; chỉ thêm low-rank adapter ở key layers khi cần.
4.	Giữ hiệu năng kiểu RNE: chống feature diffusion bằng shared recurrent mapping, giảm classifier bias bằng decoupled classifier + pseudo-feature correction.
5.	CIL đúng nghĩa: inference không cần task-id, router tự chọn context/age path.
Điểm cần kiểm chứng thực nghiệm: NERVA nhiều khả năng sẽ nằm giữa NICE và RNE-compress về accuracy, nhưng tốt hơn cả hai về active FLOPs/parameter efficiency. Nó sẽ không còn “zero forgetting cứng” tuyệt đối như NICE nếu dùng recycling mạnh, nhưng có thể đạt trade-off tốt hơn cho lifelong CIL dài hạn, nơi NICE cạn neuron còn RNE tăng tham số theo episode.

Cơ chế mới: NERVA — Neurogenesis-gated Efficient Recurrent Vector Adapters
NERVA kết hợp logic “nhẹ, không cần task-id” của NICE với logic “truyền đặc trưng tuần tự, chống feature diffusion” của RNE, nhưng tránh hai điểm yếu chính: NICE hết neuron và RNE phình to do thêm expert theo task.
Ý tưởng trung tâm:
Không thêm full task expert như RNE, cũng không chỉ tiêu thụ neuron cố định như NICE. Thay vào đó, NERVA dùng một backbone chia theo “tuổi neuron” như NICE, nhưng khi capacity bắt đầu cạn, nó không mở rộng bằng network mới; nó thêm hoặc kích hoạt micro recurrent adapters cực nhỏ tại các layer quan trọng, được nối tuần tự bằng mapping module kiểu RNE. NICE chọn neuron tối thiểu theo activation coverage và đóng băng/prune neuron già; RNE dùng shared mapping module để truyền feature từ expert trước sang expert sau, đồng thời dùng compressed expert, decoupled classifier và pseudo-feature bias correction. NERVA giữ lại những phần đó nhưng thay “task expert lớn” bằng age-bank + low-rank adapter. NICE vốn chọn tập neuron age-1 nhỏ nhất đủ bao phủ một tỷ lệ activation τ, rồi mature/freeze chúng; nhưng chính paper cũng chỉ ra NICE cuối cùng sẽ cạn neuron và đề xuất hướng giảm tuổi/tái cấp phát capacity. RNE thì giải quyết feature diffusion bằng shared mapping module giữa các expert và RNE-compress giảm chiều layer xuống 1/4, khoảng 6% tham số so với network gốc cho expert bổ sung.
________________________________________
1. Trực giác thiết kế
NICE mạnh ở ba điểm: một là kiến trúc cố định, nhẹ; hai là neuron có tuổi nên có cơ chế stability-plasticity rõ ràng; ba là context detector tự suy ra episode/class context lúc inference, không cần task-id. NICE freeze incoming connections vào neuron già hơn age-1 và prune connection từ neuron trẻ sang neuron già, nhờ đó neuron mature không bị update ngược bởi task mới. NICE cũng dùng binary activation/context detector để mask output neuron theo context lúc test, chi phí logistic regression nhỏ hơn nhiều so với replay.
RNE mạnh ở chỗ nó nhận ra vấn đề của network expansion thông thường: expert mới có thể tái biểu diễn sai old classes, gây feature diffusion. RNE khắc phục bằng cách cho feature maps truyền tuần tự qua shared mapping module, để expert mới “điều chỉnh từ feature cũ” thay vì re-extract mọi thứ từ đầu. RNE còn dùng decoupled classifier và pseudo-feature generation để giảm feature confusion/classifier bias.
NERVA lấy kết luận sau:
Neuron mature của NICE chính là “old expert” ở dạng rải trong backbone. Micro-adapter của NERVA chính là “new expert” ở dạng cực nhỏ. Recurrent mapping của RNE nối old mature feature sang new plastic capacity, giúp học task mới mà tiêu ít neuron hơn.
________________________________________
2. Kiến trúc tổng thể của NERVA
NERVA gồm 5 khối:
2.1. Shared Age-Backbone
Backbone (B) được chia theo channel/neuron age như NICE:
[
N_l = N_l^{0} \cup N_l^{1} \cup N_l^{2+}
]
Trong đó:
•	(N_l^0): neuron dự trữ, chưa dùng.
•	(N_l^1): neuron plastic cho episode hiện tại.
•	(N_l^{2+}): neuron mature, đóng vai trò memory.
•	Early layers được ưu tiên dùng chung vì NICE quan sát rằng early layers nhanh cạn nhưng thường học low-level features có tính tổng quát cao.
2.2. Recurrent Micro-Adapter Bank
Thay vì thêm một task expert hoàn chỉnh như RNE, NERVA chỉ thêm micro-adapter tại các layer quan trọng, đặc biệt là layer đổi kích thước feature map hoặc đổi số channel. RNE cũng chỉ xây connection ở key layers vì dense connection quá tốn, có thể giảm plasticity và đưa nhiễu vào feature mới.
Một adapter tại layer (l), episode (t):
[
A_{l,t}(h) = U_{l,t}\sigma(V_{l,t}h)
]
với bottleneck rank (r_l \ll d_l). Đây là low-rank adapter, không phải full conv block. Ví dụ thực dụng:
[
r_l = \max(4, d_l/16)
]
Nếu cần nhẹ hơn, dùng depthwise-separable adapter hoặc LoRA-style (1 \times 1) adapter.
2.3. Shared Recurrent Mapping Module
Tại key layer (l), NERVA dùng mapping module (M_l) kiểu RNE nhưng chia sẻ qua episode:
[
\tilde{h}{l,t} = B_l^{plastic}(h{l-1,t})
]
[
h_{l,t}^{new} = \tilde{h}{l,t} + \alpha{l,t} M_l(h_{l}^{mature}) + A_{l,t}(h_{l-1,t})
]
Trong đó:
•	(h_l^{mature}): feature từ neuron mature/old age-bank.
•	(M_l): mapping module chia sẻ, rất nhỏ.
•	(\alpha_{l,t}): gate học được hoặc do context router sinh ra.
•	(A_{l,t}): micro-adapter chỉ thêm khi capacity pressure cao hoặc novelty cao.
Khác với RNE, NERVA không forward qua tất cả expert. Nó chỉ forward một backbone và một số adapter được router bật. Vì vậy compute gần NICE hơn là RNE.
2.4. Context Router thay cho chuỗi logistic regression
NICE dùng thresholded binary activations và chuỗi logistic regression để suy context. Paper cũng nêu hạn chế: chaining logistic regressions khá đơn giản và quan sát mỗi episode có chiều khác nhau, có thể cần cơ chế sequential/attention tốt hơn.
NERVA thay bằng Tiny Recurrent Context Router:
[
q(e|x) = R_{\psi}(b_1(x), b_2(x), ..., b_L(x))
]
Trong đó:
[
b_l(x) = \mathbb{1}[A_l(x) > \mu_l + \kappa\sigma_l]
]
Router có thể là GRU nhỏ hoặc attention pooling rất nhỏ trên binary activation sketches. Output:
•	Xác suất episode/context (q(e|x)).
•	Age mask (m_e).
•	Adapter mask (g_t).
•	Logit mask cho classifier.
2.5. Causal Age-Decoupled Classifier
NERVA dùng classifier kiểu RNE nhưng gắn với age/context thay vì task expert lớn:
[
H = {h_1, h_2, ..., h_t}
]
•	Subclassifier cũ update rất chậm hoặc freeze.
•	Subclassifier mới update mạnh hơn.
•	New classifier được phép nhìn feature sequence/age feature tổng hợp.
•	Old classifier không bị new feature làm lệch quá mức.
RNE dùng decoupled classifier để giảm feature confusion và cho new subclassifier sử dụng feature sequence trong khi old subclassifier ít bị ảnh hưởng bởi new features.
________________________________________
3. Cơ chế chống “hết neuron” của NICE
NERVA không đợi đến khi neuron cạn mới xử lý. Nó dùng Capacity Controller ở mỗi layer.
3.1. Đo áp lực capacity
Tại layer (l):
[
P_l = 1 - \frac{|N_l^0|}{|N_l|}
]
•	(P_l) thấp: còn nhiều neuron dự trữ, học như NICE.
•	(P_l) trung bình: giảm số neuron age-1 được giữ lại, tăng recurrent transfer.
•	(P_l) cao: không cấp thêm neuron trừ khi novelty rất lớn; bật micro-adapter hoặc nén/recycle neuron mature.
3.2. Chọn neuron bằng activation + gradient + novelty
NICE chọn neuron theo activation coverage. NERVA mở rộng score:
[
I_i^l = \lambda_A \bar{A}_i^l + \lambda_G \overline{|\nabla_i L|} + \lambda_C C_i^l - \lambda_R R_i^l
]
Trong đó:
•	(\bar{A}_i^l): activation trung bình.
•	(\overline{|\nabla_i L|}): độ quan trọng theo gradient.
•	(C_i^l): context uniqueness, neuron có giúp phân biệt context mới không.
•	(R_i^l): redundancy/correlation với neuron mature khác.
Chọn tập nhỏ nhất:
[
S_l = \arg\min |S_l|
]
sao cho:
[
\sum_{i \in S_l} I_i^l \geq \tau_l \sum_{i \in N_l^1} I_i^l
]
Với (\tau_l) không cố định 0.95 như NICE, mà adaptive:
[
\tau_l = clip(\tau_0 - \rho P_l + \delta Novelty_t,\tau_{min},\tau_{max})
]
Nghĩa là:
•	Capacity càng cạn, giữ ít neuron hơn.
•	Novelty/domain shift càng cao, cho phép giữ nhiều neuron hơn.
•	Nếu old mature features đã đủ tốt, ưu tiên adapter thay vì neuron mới.
3.3. Neuron leasing/recycling
Neuron mature không bị xoá bừa. NERVA dùng 3 trạng thái mature:
•	Mature-locked: neuron rất quan trọng, không recycle.
•	Mature-shared: neuron dùng chung nhiều context, được giữ làm shared feature.
•	Mature-recyclable: neuron ít dùng, trùng lặp cao, có thể nén rồi giải phóng.
Trước khi recycle, NERVA distill contribution của nhóm neuron đó vào:
1.	micro-adapter,
2.	feature moments,
3.	classifier prototypes,
4.	binary activation sketch memory.
Điều kiện recycle:
[
Usage_i < \epsilon_u,\quad Redundancy_i > \epsilon_r,\quad \Delta Acc_{proxy} < \epsilon_a
]
Trong đó (\Delta Acc_{proxy}) đo bằng pseudo-feature validation, không cần raw old images.
________________________________________
4. Cơ chế chống “phình model” của RNE
RNE thêm expert theo task, dù RNE-compress đã giảm xuống khoảng 6% tham số của network gốc cho expert bổ sung. NERVA đi xa hơn bằng cách không thêm expert hoàn chỉnh.
Ba mức tăng tham số:
Mức	Khi nào dùng	Tham số thêm
Level 0: NICE-only	Capacity còn nhiều	Không thêm
Level 1: Recurrent mapping	Task mới giống domain cũ	Chỉ dùng shared (M_l), gần như constant
Level 2: Micro-adapter	Novelty cao hoặc layer cạn neuron	Low-rank adapter tại key layers
Level 3: Emergency expansion	Domain shift lớn	Thêm tiny block rất nhỏ, không thêm full network
Vì vậy tăng trưởng tham số:
[
P_{NERVA} \approx P_{backbone} + P_{router} + \sum_l P(M_l) + \sum_{t,l \in K} P(A_{l,t})
]
Trong khi RNE/full NE có xu hướng:
[
P_{RNE} \approx P_{backbone} + tP_{expert}
]
NERVA chỉ tăng theo số adapter thực sự cần, không tăng tuyến tính theo task bằng full expert.
________________________________________
5. End-to-end methodology
Giai đoạn 0: Khởi tạo
Khởi tạo backbone (B), chia neuron/channel theo layer.
•	Tất cả hidden neurons ban đầu ở age-0.
•	Input layer luôn shared.
•	Output head mở theo class mới.
•	Khởi tạo shared mapping modules (M_l) tại key layers.
•	Khởi tạo context router (R_\psi).
•	Khởi tạo age-decoupled classifier (H).
Memory lưu:
[
\mathcal{M} = {b_c, \mu_c, \sigma_c, p_c, age_c}
]
Trong đó:
•	(b_c): binary activation sketch.
•	(\mu_c, \sigma_c): feature mean/variance theo class.
•	(p_c): prototype/logit prototype.
•	Không lưu raw images mặc định.
________________________________________
Giai đoạn 1: Nhận episode mới (D_t)
Với episode (t), model nhận dữ liệu mới:
[
D_t = {(x_i, y_i)}
]
Không cần task-id lúc inference, nhưng lúc training biết episode hiện tại để train router.
Tính novelty:
[
Novelty_t = 1 - \max_e sim(\mu(D_t), \mu_e)
]
Nếu novelty thấp, dùng lại mature features nhiều hơn. Nếu novelty cao, cấp neuron/adapters nhiều hơn.
________________________________________
Giai đoạn 2: Tạm kích hoạt neuron age-0 thành age-1
Giống NICE:
[
N_l^0 \rightarrow N_l^1
]
Nhưng chỉ kích hoạt một phần theo capacity budget:
[
|N_l^{1,active}| = Budget(P_l, Novelty_t)
]
Nếu layer đã gần cạn, không mở toàn bộ age-0 mà mở top-k candidates hoặc bật adapter.
________________________________________
Giai đoạn 3: Forward recurrent neurogenesis
Với input (x), tại layer (l):
[
h_l^{mature} = B_l^{N^{2+}}(h_{l-1})
]
[
h_l^{plastic} = B_l^{N^1}(h_{l-1})
]
[
h_l = h_l^{mature} + h_l^{plastic} + \alpha_l M_l(h_l^{mature}) + g_l A_{l,t}(h_{l-1})
]
Trong đó:
•	(M_l(h_l^{mature})) truyền knowledge cũ sang vùng plastic.
•	(A_{l,t}) chỉ bật khi cần.
•	(g_l \in {0,1}) hoặc soft gate.
•	Không có đường từ neuron trẻ làm thay đổi neuron già.
________________________________________
Giai đoạn 4: Loss training chính
Tổng loss:
[
L = L_{CE}^{new} + \lambda_{ctx}L_{ctx} + \lambda_{sp}L_{sparse} + \lambda_{align}L_{recurrent} + \lambda_{cal}L_{calib}
]
Trong đó:
4.1. New class CE
[
L_{CE}^{new} = CE(H(f(x)), y)
]
4.2. Context loss
[
L_{ctx} = CE(R_\psi(b(x)), e_t)
]
Giúp router nhận ra episode/context từ binary activations.
4.3. Sparsity loss
[
L_{sparse} = \sum_l ||g_l||1 + \sum_l ||A{l,t}||_1
]
Ép adapter và gate nhỏ nhất có thể.
4.4. Recurrent alignment loss
[
L_{recurrent} = ||h_l^{plastic} - stopgrad(M_l(h_l^{mature}))||_2^2
]
Mục tiêu: neuron/adapters mới học bằng cách “điều chỉnh từ old mature feature”, không re-extract hỗn loạn. Đây là tinh thần của RNE: expert mới nhận feature từ expert trước qua mapping module thay vì tái trích xuất redundant.
4.5. Calibration loss
Dùng pseudo-features của old classes để classifier không bias về class mới.
________________________________________
Giai đoạn 5: Chọn neuron sống sót
Sau mỗi (p) epoch:
1.	Tính score (I_i^l).
2.	Sort giảm dần.
3.	Chọn tập nhỏ nhất (S_l) đạt adaptive coverage (\tau_l).
4.	Neuron trong (S_l) giữ age-1.
5.	Neuron còn lại quay về age-0.
6.	Prune connection từ neuron trẻ sang neuron già.
7.	Freeze incoming connection của neuron mature.
Đây là NICE nhưng có thêm capacity-aware score và adapter fallback.
________________________________________
Giai đoạn 6: Maturation cuối episode
Cuối episode:
[
N_l^1 \rightarrow N_l^2
]
[
N_l^a \rightarrow N_l^{a+1}, a \geq 2
]
Các neuron mature được phân loại:
•	locked,
•	shared,
•	recyclable.
Adapter (A_{l,t}) cũng được mature:
•	Nếu được dùng nhiều: giữ.
•	Nếu ít dùng: merge vào shared adapter (M_l) hoặc prune.
•	Nếu trùng với old adapter: low-rank merge.
________________________________________
Giai đoạn 7: Update memory không replay ảnh
Cập nhật memory bằng feature statistics và binary sketches:
[
\mu_c = \frac{1}{n_c}\sum_i z_i
]
[
\sigma_c^2 = \frac{1}{n_c}\sum_i(z_i - \mu_c)^2
]
[
b_c = Quantize(\mathbb{1}[A(x) > \theta])
]
Memory này phục vụ:
•	context router,
•	pseudo-feature generation,
•	classifier calibration,
•	neuron recycling validation.
NICE đã dùng binary activations cho context inference; RNE dùng mean/variance feature để generate pseudo-features và retrain classifier.
________________________________________
Giai đoạn 8: Pseudo-feature bias correction
Sinh pseudo-features cho old classes:
[
\hat{z}_c = \mu_c + \sigma_c \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
]
Hoặc dùng cách RNE-style: chọn một new class xa old classes nhất làm generator rồi transform theo mean/variance old class.
Tạo balanced feature set:
[
\hat{D}{bal} = {z{new}} \cup {\hat{z}_{old}}
]
Freeze backbone, chỉ retrain classifier/router nhỏ:
[
\min_H CE(H(\hat{z}), \hat{y})
]
Mục tiêu: giảm classifier bias mà không cần raw replay. RNE paper mô tả pseudo-feature generation để tái tạo old category features từ new task samples, tạo balanced feature set rồi retrain classifier.
________________________________________
6. Inference end-to-end
Với test sample (x):
1.	Forward một lần qua shared backbone.
2.	Tạo binary activation sketch:
[
b(x) = {b_1(x),...,b_L(x)}
]
3.	Context router dự đoán:
[
q(e|x)=R_\psi(b(x))
]
4.	Chọn top-1 hoặc top-k context:
[
E^* = topk(q(e|x))
]
5.	Bật age mask + adapter mask tương ứng:
[
m = Mask(E^*)
]
6.	Mask logits không liên quan hoặc reweight theo context posterior:
[
logits = \sum_{e \in E^*} q(e|x) H_e(f_m(x))
]
7.	Predict class:
[
\hat{y} = \arg\max softmax(logits)
]
Khác biệt quan trọng với RNE: không cần chạy tất cả task expert. Khác biệt với NICE: nếu neuron cũ cạn, vẫn còn adapter nhỏ và recycling/compaction để tiếp tục học.
________________________________________
7. Pseudocode cấp thuật toán
Algorithm NERVA: Neurogenesis-gated Efficient Recurrent Vector Adapters

Input:
  Episodes D1...DT
  Backbone B with age-labeled neurons
  Shared mapping modules M_l at key layers
  Context router R
  Decoupled classifier H
  Memory M = {}

For episode t = 1...T:

  1. Estimate novelty and layer capacity pressure:
       Novelty_t = novelty(D_t, M)
       P_l = 1 - |N_l^0| / |N_l|

  2. Allocate plastic capacity:
       Activate subset of age-0 neurons as age-1
       If P_l high or Novelty_t high:
           instantiate or activate micro-adapter A_l,t

  3. Train for p epochs:
       For each batch (x, y) in D_t:
           Forward through mature + plastic neurons
           Inject recurrent mapped mature feature:
               h_l = h_l^mature + h_l^plastic + alpha_l M_l(h_l^mature) + g_l A_l,t(h)
           Predict logits through causal age-decoupled classifier
           Compute:
               L = CE + context_loss + sparsity_loss + recurrent_alignment
           Update only:
               age-1 neurons
               current adapters
               mapping modules with low LR
               new classifier head
               context router

  4. Every p epochs:
       Score age-1 neurons
       Keep minimal subset satisfying adaptive coverage tau_l
       Return unimportant age-1 neurons to age-0
       Prune younger-to-older connections

  5. End episode:
       Mature surviving age-1 neurons to age-2
       Freeze mature neurons
       Compress/merge weak adapters
       Identify recyclable mature neurons

  6. Update memory:
       Store binary activation sketches
       Store feature moments mu_c, sigma_c
       Store class prototypes

  7. Bias correction:
       Generate pseudo-features for old classes
       Build balanced feature set
       Freeze backbone
       Retrain only classifier/router for few epochs

Return:
  Lightweight incremental model with age-gated backbone, recurrent adapters,
  context router, and calibrated classifier
________________________________________
8. Vì sao NERVA nhẹ hơn nhưng vẫn giữ hiệu năng
Vấn đề	NICE	RNE	NERVA
Catastrophic forgetting	Tốt nhờ freeze/prune neuron mature	Tốt nhờ old experts frozen	Tốt nhờ neuron mature + recurrent transfer
Hết neuron	Có, đặc biệt early layers	Không hết neuron nhưng phình expert	Giảm tiêu neuron bằng recurrent reuse + adapter + recycling
Model size	Nhẹ	Tăng theo expert/task	Gần NICE, chỉ thêm micro-adapter khi cần
Compute inference	Gần 1 backbone	Có thể phải dùng nhiều expert/feature sequence	1 backbone + top-k adapter/context
Không cần task-id	Có	Có CIL inference nhưng expert sequence lớn	Có, dùng context router
Feature diffusion	NICE không xử lý theo expert sequence	RNE xử lý tốt bằng recurrent connection	Dùng recurrent mapping từ mature feature sang plastic feature
Classifier bias	NICE chủ yếu context mask	RNE có decoupled classifier + pseudo-feature	Giữ RNE-style calibration nhưng không replay ảnh
________________________________________
9. Công thức độ phức tạp
9.1. Tham số
Với:
•	(P_B): backbone.
•	(P_M): shared mapping modules.
•	(P_A): tổng adapter đang giữ.
•	(P_R): router.
•	(P_H): classifier.
[
P_{NERVA} = P_B + P_M + P_A + P_R + P_H
]
Trong đó:
[
P_A = \sum_{t,l} 2d_lr_l
]
và (r_l \ll d_l). Nếu adapter rank (r_l=d_l/16), adapter thấp hơn nhiều so với full block.
9.2. FLOPs
[
F_{NERVA} \approx F_B + F_{topk-adapter} + F_R
]
Vì router chỉ bật top-k adapter, inference không scale mạnh theo số episode.
9.3. Memory
[
Mem = Mem(binary\ sketches) + Mem(\mu,\sigma) + Mem(prototypes)
]
Không cần lưu raw images mặc định. NICE cũng nhấn mạnh activation memory rất nhỏ so với image memory trong thiết lập của họ.
________________________________________
10. Training protocol đề xuất để kiểm chứng
Nên đánh giá NERVA trên cùng benchmark của NICE/RNE:
•	Split CIFAR-100: 5, 10, 25 episodes.
•	ImageNet-100.
•	TinyImageNet hoặc ImageNet-1K nếu có compute.
•	Metrics:
o	Average Accuracy (A_{avg})
o	Final Accuracy (A_T)
o	Forgetting (F)
o	Params
o	FLOPs
o	Active params at inference
o	Memory budget
o	Context accuracy
o	Neuron depletion rate per layer
Ablation bắt buộc:
Ablation	Mục đích
NERVA without adapters	Kiểm tra còn bị hết neuron không
NERVA without recurrent mapping	Kiểm tra feature diffusion
NERVA with logistic context detector	So với NICE detector
NERVA without pseudo-feature correction	Đo classifier bias
NERVA without recycling	Đo lợi ích capacity controller
Full adapter vs key-layer adapter	Đo trade-off FLOPs/accuracy
Top-1 context vs Top-k context	Đo routing robustness
________________________________________
11. Kết luận thiết kế
NERVA là cơ chế lai hợp lý nhất giữa NICE và RNE nếu mục tiêu là:
1.	Nhẹ như NICE: backbone chính vẫn cố định, neuron được chọn sparse, memory là binary sketches + feature moments.
2.	Không cạn nhanh như NICE: dùng recurrent transfer để tái sử dụng mature features, adaptive neuron budget, micro-adapter, neuron recycling.
3.	Không phình như RNE: không thêm full task expert; chỉ thêm low-rank adapter ở key layers khi cần.
4.	Giữ hiệu năng kiểu RNE: chống feature diffusion bằng shared recurrent mapping, giảm classifier bias bằng decoupled classifier + pseudo-feature correction.
5.	CIL đúng nghĩa: inference không cần task-id, router tự chọn context/age path.
Điểm cần kiểm chứng thực nghiệm: NERVA nhiều khả năng sẽ nằm giữa NICE và RNE-compress về accuracy, nhưng tốt hơn cả hai về active FLOPs/parameter efficiency. Nó sẽ không còn “zero forgetting cứng” tuyệt đối như NICE nếu dùng recycling mạnh, nhưng có thể đạt trade-off tốt hơn cho lifelong CIL dài hạn, nơi NICE cạn neuron còn RNE tăng tham số theo episode.

