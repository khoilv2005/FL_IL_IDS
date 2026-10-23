# CANDLE V4 — ý tưởng Round-wise Complementary Expert Learning

## 0. Trạng thái và nguồn

Bản thiết kế nghiên cứu hình thành từ `C:/Users/khoak/Downloads/candle v4.txt`.
Chưa implement thuật toán, chưa train expert hoặc chạy benchmark. Core tại
`candle_v4_runtime/` vẫn là 367 source files V1 nguyên bản từ
`a9714ac96b3ecec3910985d6fda8224b1584b570`; không thay clone manifest.

Tên chính giữ **CANDLE V4**. “Round-wise Complementary Masked Gossip / RCMG” là
tên mô tả cơ chế dự kiến, chưa phải tên một phương pháp đã công bố hoặc có kết quả.

## 1. Giả thuyết chính

Mỗi receiver học một expert nhận diện class còn thiếu trong input space chung.
Donor cung cấp gradient dương và âm từ current BASE của mình; receiver cung cấp
gradient âm từ current BASE của mình. Tất cả tính trên đúng cùng expert reference
của receiver, rồi receiver tự aggregate. Việc này diễn ra trong từng native round.

Kiến thức mới nằm trong expert; chưa được tích hợp vào CNN-GRU/FC2 của V1.
Một input đi qua base predictor và các expert ACTIVE ở local. Không cần donor model
lúc inference, không có central aggregation server, không có global expert bắt buộc.
Đây là **một local predictor nhiều nhánh**, không còn là chỉ một CNN-GRU + FC2 gốc.

Hai giả thuyết phải đo riêng:

1. Input-space expert có nhận diện được target và bảo vệ negatives mà receiver không
   có target positives local hay không?
2. Học trong từng round có tạo giá trị sớm hoặc tận dụng peers thay đổi tốt hơn so
   với học cuối task trên cùng compute/data/communication budget hay không?

Nếu peers, dữ liệu và thứ tự gradient cố định, cùng initialization và cùng tổng
optimizer steps có thể tạo cùng expert cuối dù thực hiện các bước ở cuối task.
Vì vậy không quy mọi gain cuối task cho “round-wise”. Cần learning curves và
time-to-activation bên cạnh final accuracy.

## 2. Kế thừa và thay đổi

| Thành phần | Quyết định ban đầu |
| --- | --- |
| Native local training, NICE age/masks | Giữ V1 |
| Native peer graph và aggregation | Giữ V1, paper threshold ξ=0,8 |
| Native router | Giữ cho base predictions |
| Module bổ sung | MLP expert trên cùng 39 features đã preprocess |
| Chủ expert | Receiver, định danh `(receiver, class, generation)` |
| Trao đổi | Reference/gradient nhỏ của expert, không cần receiver CNN-GRU snapshot cho expert |
| Training expert | Current BASE tại chủ dữ liệu, mỗi round |
| Selection/calibration | FIT hợp lệ, tách khỏi optimization và acceptance |
| Acceptance | CAL hợp lệ, gate/version bị khóa trước |
| Dữ liệu task cũ | Không raw replay, không truy cập lại historical CAL trong training |
| Inference imported class | Expert gate không phụ thuộc task do native router chọn |
| Nhiều expert | Phải kiểm tra arbitration và joint risk riêng |

Native V1 paper neighborhoods có thể chồng lấn; không coi sentinel K=1 hoặc labels
là partition cluster thực. Pilot đầu chỉ lấy donor trong actual native neighbors.
Không âm thầm thêm cross-neighborhood search. Nếu neighborhood không có positives,
ghi NO_DONOR, không cam kết mọi client sẽ phủ đủ 34 classes.

## 3. Input contract và expert

### 3.1. Input contract

Mỗi packet bind feature order, shape, dtype, class mapping và preprocessing fingerprint.
Input V1 hiện có dạng `(N,39,1)`; expert flatten thành `(N,39)`.

Loader V1 đọc dữ liệu đã preprocess; nó không chứng minh mọi dataset mount dùng cùng
scaler. Source `prepare_data/step2_federated_splits.py` có fit StandardScaler trên TRAIN
rồi transform train/test trước khi tạo benchmark. Pilot giữ đúng dữ liệu đã transform;
không fit thêm một scaler riêng tại mỗi client hoặc trên TEST. Cần bind được scaler/
feature-order provenance; thiếu authority thì ghi UNKNOWN, không suy từ shape giống nhau.
Quy trình benchmark preprocessing tập trung là giả định có sẵn; không gọi đó là một
quy trình triển khai online hoàn toàn không cần coordination. Training vẫn P2P.

### 3.2. Expert nhỏ đầu tiên

`Linear(39,64) → ReLU → Linear(64,1)`, không BN hoặc dropout trong pilot đầu.
Tham số: `39*64+64 + 64+1 = 2.625`; FP32 weights = 10.500 bytes.
Sigmoid output là **score nhị phân**, chưa tự động là posterior được calibration tốt,
đặc biệt khi objective dùng positive/negative class balancing.

Mọi donor dùng đúng `phi_i,c^(r,k)` của receiver tại inner step k; không lấy trung bình
các MLP tự train độc lập. Cùng architecture hoặc cùng init seed không bảo đảm models
còn cùng reference sau nhiều rounds. Snapshot/version hash là bắt buộc.

## 4. Metadata và chọn peer

Giữ riêng các trạng thái:

- `current_data_mask`: class có current BASE hợp lệ, là nguồn tạo gradient dương mới.
- `local_history_mask`: class từng học local; không cấp quyền đọc lại dữ liệu cũ.
- `native_availability_mask`: class được native inference mở.
- `expert_active_mask`: expert đã được cấp phép inference, không biến thành ownership.
- `request_mask`: class có gap theo FIT hợp lệ; UNKNOWN không tự thành request có thể cài.

Trước khi học, so base/native với các no-learning controls hợp lệ. Nếu registration
repair đã giải quyết được thì không gọi gain của expert là knowledge-transfer gain.
V4 có thể giải quyết cả lỗi representation và lỗi routing, nhưng phải phân rã đóng góp.

Receiver quét metadata tất cả actual neighbors. Donor phải có current positives và
đủ train/FIT/CAL authority theo protocol. Không yêu cầu FC2 donor đã rank=2: expert
không trích FC2 và có thể bắt đầu học khi backbone native còn đang học task.

Đề xuất điểm xếp hạng sau hard eligibility:

`S(i,j) = a*Similarity(i,j) + b*CoverageGain(i,j)`

Chuẩn hóa CoverageGain theo số target requests để cùng thang với Similarity.
Coverage dùng current ownership của donor, không dùng availability do import làm
bằng chứng rằng donor có positives. Chọn top-L theo rule deterministic; L và weights
được khóa bằng development trước khi chạy acceptance. Không thử donor vô hạn.

## 5. Training và aggregation có định nghĩa rõ

Receiver là owner optimizer. Donor không gửi raw samples và không aggregate Adam states.
Cho mỗi class c, peer j gửi mean gradient và contribution mass theo hai nhóm:

`g_j,c^+ = mean gradient BCE trên y=c`

`g_j,c^- = mean gradient BCE trên y!=c`

Với collaboration weights không âm a_ij, pooled gradients là:

`g_i,c^+ = sum_j(a_ij*n_j^+*g_j,c^+) / sum_j(a_ij*n_j^+)`

`g_i,c^- = sum_j(a_ij*n_j^-*g_j,c^-) / sum_j(a_ij*n_j^-)`

Receiver tham gia nhóm âm dù không có positives. Không coi nhóm rỗng là gradient 0
hợp lệ cho transfer: không có positive mass thì không thể học class mới trong round đó.
Nếu cần negative class balancing, phải định nghĩa thêm aggregation theo từng negative
class rồi average classes; không thay rule giữa run hoặc để lớp phổ biến lấn át lớp hiếm.

Có thể gửi hai gradient tensors riêng để debug, hoặc tiết kiệm payload bằng cách
trao counts trước. Receiver công bố D+ và D- (hai denominators), lambda và a_ij;
mỗi peer trả một tensor contribution đã gồm đúng hai hệ số nhóm. Receiver **cộng**
contributions, không FedAvg lần nữa. Cách một tensor chỉ đúng khi metadata nhóm và
reference khớp; phải đo các message counts này. Self-weight của receiver cũng phải
được định nghĩa trước, không mặc nhiên bằng reserve-neuron weight trong native CANDLE.

Update dự kiến:

`phi_next = phi - eta*M_expert*(lambda_plus*g_plus + lambda_minus*g_minus + mu*(phi-phi_anchor))`

M_expert chỉ tác động params expert được phép học. Không ghi vào mature CNN-GRU/FC2.
lambda_plus/lambda_minus được định nghĩa trước, không chỉnh theo test/CAL failures.
Core tuổi neuron V1 và tuổi expert là hai state riêng; không để native freeze masks
tự vô hiệu hóa gradient của một expert mới.

Anchor phải được cố định và versioned. Nếu vừa đặt anchor=phi ở đầu mỗi single-step
rồi tính `mu*(phi-anchor)`, regularization bằng 0. Anchor từ initialization chỉ là
initialization/proximal prior, chưa phải kiến thức được chứng nhận. Không mô tả nó
như bằng chứng bảo vệ mọi class cũ.

Mỗi native round có S inner expert updates đồng bộ. Phải khóa S, learning rate,
sampling budget và tổng optimizer steps trước đánh giá. Không mặc định 20 rounds
tương đương một quá trình MLP đã hội tụ. Mỗi inner step mới cần đúng shared reference;
các gradient stale, sai layout/mask/version hoặc nonfinite bị từ chối.

## 6. Draft và deployment phải tách

Tối thiểu hai namespace khi một expert vừa ACTIVE vừa muốn tiếp tục nghiên cứu update:

`draft expert`: được học từ current BASE mỗi round.

`active expert`: immutable snapshot + calibration + gate + evidence scope.

Nếu tiếp tục học từ một active snapshot, tạo draft generation mới với optimizer/masks
riêng; không unfreeze hoặc ghi đè trực tiếp active generation. Draft copy cũng làm
tăng memory và phải được checkpoint/accounting. Thay active generation cần acceptance mới.

State dự kiến: ABSENT → DRAFT → FIT_READY → CAL_ACCEPTED → ACTIVE → MATURE.
UNKNOWN/REJECTED giữ base predictor; SUSPENDED lưu weights/state nhưng không inference.
Optimizer checkpoint gồm draft, active generation, reference hashes, counters, masks,
anchor, RNG và communication ledger; resume không học lặp hoặc dùng gate của version khác.

Pilot đầu chỉ một target expert và một lần xét activation tại round được khóa trước
(ví dụ round thứ 10, index 9). FIT dùng theo dõi trajectory/calibrate theo rule trước
CAL; CAL chỉ dùng acceptance cho candidate version đã chọn trước. Nếu ACTIVE, đóng
băng snapshot đó; draft có thể tiếp tục tới round 20 để phân tích bằng FIT nhưng không
tự thay active generation. Không chuyển sang thử lại ở round 11 nếu CAL round 10 FAIL.

Mở gate nhiều lần trong tương lai cần các CAL partitions mới hoặc protocol kiểm định
tuần tự phù hợp. Không coi 20 lần dùng cùng CAL là 20 bộ evidence độc lập.
Nếu recall gate thực nghiệm ≥95%, FAR ≤0,1%, break=0 và minimum evidence được giữ,
32 negative samples không tự chứng nhận FAR thật ≤0,1%; báo rõ empirical scope.

## 7. Inference và giới hạn bảo vệ

Luồng đề xuất:

`x → native CANDLE base prediction`

`x → ACTIVE raw-input expert scores`

`gate/arbitration → expert class hoặc fallback base prediction`

Expert imported class được đánh giá độc lập task do native router chọn. Vì vậy
native router không còn là điều kiện cần để imported class xuất hiện ở output.
Không dùng true task hoặc true class khi inference. Không so trực tiếp sigmoid score
của expert với raw FC2 logits hoặc native softmax như những đại lượng đã calibration chung.

Pilot một expert: score vượt threshold đã khóa thì chọn target, còn lại dùng base.
Ngay cả khi không sửa một weight native nào, override vẫn có thể biến dự đoán đúng
thành sai. Gate phải kiểm tra receiver negatives và donor negatives hợp lệ, kể cả
hard negatives hiện có; không được bỏ pool khó sau khi nhìn kết quả.

Nhiều expert: cần rule khi nhiều gate cùng vượt, reject/abstain khi không phân xử được,
và đánh giá **bank cuối cùng sau kết hợp**. FAR riêng lẻ không bảo đảm FAR toàn bank.
Không gọi các sigmoid score là mutually exclusive multiclass probabilities.

Sau task transition, expert cũ không có quyền đọc lại positives cũ. Freeze weights và
input transform giữ hàm expert ổn định, nhưng không bảo đảm không kích hoạt trên class
mới. Chỉ negatives task mới có thể không đủ chứng nhận recall cũ. Phải đánh giá current
negative risk và áp dụng ACTIVE/SUSPENDED theo policy empirical/scope đã khóa; không
gọi đó là strict old-class safety hoặc tự mở scope sang mọi input.
Policy nghiên cứu đề xuất là empirical risk-controlled, công bố evidence coverage và
unknown risk; chưa đủ căn cứ cho strict protection trên toàn bộ cumulative classes.

## 8. Pipeline mỗi round

1. Native V1 local training, age/mask protection.
2. Native capsule exchange, paper graph và masked aggregation.
3. Refresh native state theo lịch V1; lấy actual current neighbor/version snapshot.
4. Discovery metadata, gap probe và evidence feasibility; chọn budget expert sessions.
5. Receiver gửi expert reference; peers tính positive/negative gradients trên current BASE.
6. Receiver aggregate, kiểm tra finite/compatibility/protection, cập nhật draft expert.
7. FIT diagnostics theo lịch; chỉ activation/calibration theo protocol đã đăng ký.
8. Lưu round checkpoint toàn bộ base + expert + gate + optimizer + ledger.

Base model hiện có được freeze dưới vai trò reference chỉ trong các probe cần so sánh;
native training các rounds tiếp theo vẫn chạy. Gate/certificate không được phụ thuộc
vào một base confidence score đã drift mà không version/check lại dependency đó.

## 9. Communication và capacity accounting

Cho một receiver–donor–class và S=1:

- Hai tensors g+ và g- riêng: reference + 2 gradients = 31.500 bytes/round,
  20 rounds = 630.000 bytes trước overhead khi donor có cả hai nhóm.
- Một combined contribution sau khi chốt global group denominators: reference +
  gradient = 21.000 bytes/round, 20 rounds = 420.000 bytes, cộng count protocol.

Với S inner steps, L donors, Q receiver–class sessions: khoảng
`(2 hoặc 3)*R*S*L*Q*10.500` bytes, tùy codec gradient, cộng metadata/gates/verification.
Caching chỉ miễn phí payload khi recipient thật sự có đúng hash; version id không
thay thế bytes cần để tạo model reference mới. Chỉ gửi tensor cache chưa có hoặc delta
thật sự đủ để reconstruct cùng reference.

Không cần CNN-GRU snapshot cho expert-only gradient/recall validation. Native CANDLE
communication vẫn được cộng đầy đủ. Verification expert có thể chạy tại donor bằng
expert parameters và gate nhỏ; arbitration với base prediction trên donor positives
có thể vẫn cần receiver base function để đo end-to-end rescue/break chính xác.
Phải tính chi phí đó nếu protocol sử dụng, không tự coi mọi verification chỉ 10,5 KB.
Đặc biệt, functional-gap probe trên positives tại donor vẫn cần truy cập đúng receiver
base function. Native model cache chỉ được tái dùng khi hash/version thật sự khớp.
Không loại chi phí snapshot khỏi ledger chỉ vì expert optimization không dùng encoder.

34 experts tối đa: 89.250 parameters, 357.000 FP32 bytes, xấp xỉ 17,41% của backbone
512.654 parameters. Đây chỉ là illustration cho bank đầy đủ, chưa tính optimizer/draft
copies/activations. Mỗi prediction phải chạy các active experts; đo inference latency,
peak memory và số expert/client. Không gọi chi phí này bằng 0 hoặc luôn không đáng kể.

## 10. Pilot có điểm dừng

### P0 — contract

Input/version authority, receiver-owned reference, positive/negative aggregation,
draft/active state, graph eligibility và counters. Chưa sửa core trước khi contract rõ.

### P1 — standalone feasibility

Một receiver, một donor có current positives, một class, cùng 39D preprocessing.
Chọn pair bằng metadata trước evaluation. BASE để train, FIT để quan sát và calibration
theo rule, CAL để acceptance một lần. Không gọi data đã xem qua nhiều audit là holdout
mới độc lập. Không mở TEST để chọn expert size, thresholds hoặc learning rate.
Recall trên holdout của donor chỉ chứng minh trên phân bố donor đang được kiểm tra;
receiver không có target positives thì chưa xác nhận target recall trong miền receiver.

Chạy fixed 20 round-equivalent expert schedule, ghi actual optimizer steps/convergence,
target recall, per-owner/per-negative-class FP/break, timing và serialized bytes.
Đây chưa phải bằng chứng native continual survival.

### P2 — native round integration

Một bounded current-task run có local training và native aggregation thật. Học expert
qua từng round, activation tại thời điểm khóa trước; kiểm tra checkpoint/resume và
predictions khớp. Không cài vào core V1 hoặc chuyển qua 100 clients ngay.

### P3 — task transition

Current-task-only data access, expert retention, new-class false activation, suspension,
multi-expert arbitration tối thiểu hai targets và communication. Chỉ mở full một seed
sau khi có gain dữ liệu thật và accounting/protection/restore đúng trong phạm vi đã nêu.

NO-GO nếu một pilot đã khóa không cải thiện net rescue/recall vượt controls ở ngân sách
risk phù hợp. Không cứu bằng cách hạ gate, tăng donor vô hạn hoặc chọn panel thuận lợi.

## 11. Đối chứng để diễn giải đúng gain

| Control/variant | Câu hỏi |
| --- | --- |
| Native CANDLE V1 | Baseline cuối và learning curve |
| Availability/router/mask-only hợp lệ | Đã có knowledge bị che hay cần expert học thật? |
| Cùng MLP, receiver-only | Donor positives có đóng góp thực không? |
| Cùng MLP, donor-only | Receiver negatives có giảm FP/break không? |
| Round-wise positive+negative | Cơ chế chính |
| Cùng expert, train cuối task với data/steps/bytes khớp | Giá trị do timing hay chỉ do thêm optimizer work? |
| NoReg và random compatible peer | Vai trò regularization và complementary selection |
| Capacity-matched baseline | Gain đến từ transfer hay chỉ thêm parameters/compute? |

Đo Mean/Pooled Accuracy, Macro-F1 theo cùng protocol, target metrics, forgetting,
time-to-first-activation, area under round learning curve, full-system communication,
inference cost. Direct FC2/receiver-head cũ là references bổ sung, không trộn protocol.

## 12. Liên hệ literature và claim

[FedSaC, ICML 2024](https://proceedings.mlr.press/v235/yan24a.html) nghiên cứu cân bằng
similarity/complementarity. Vì vậy riêng điểm chọn peer có coverage bonus không đủ novelty.

[FedCPD, Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253524002598)
nghiên cứu label distribution skew với class proxy decoupling/proxy regularization và
global model; bài được dẫn này không phải chứng minh cho DFCL no-replay của V4.
Không nhầm với các công trình khác cũng dùng acronym FedCPD.

[FedProK, CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/FedVision-2024/html/Gao_FedProK_Trustworthy_Federated_Class-Incremental_Learning_via_Prototypical_Feature_Knowledge_Transfer_CVPRW_2024_paper.html)
dùng prototypical knowledge transfer, gồm client feature translation và server-side
prototypical fusion. Không phải cùng cơ chế receiver-owned raw-input expert của đề xuất này.

Claim dự kiến cần chứng minh: decentralized complementary acquisition trên input-space
expert reference chung, joint positive/negative gradient aggregation, early utility
trong continual rounds và empirical risk/version control. Chưa tuyên bố novel toàn
bộ literature, privacy tuyệt đối, strict safety, hay khả năng cải thiện benchmark.

## 13. Kết luận thiết kế

Hướng V4 đáng thử ở quy mô giới hạn vì bỏ được dependence vào receiver FC1 và task
gating cho imported expert. Đánh đổi là expert inference bổ sung và calibration/risk
coverage vẫn phải giải quyết. Bước kế tiếp hợp lý là khóa contract rồi triển khai
standalone expert pilot, chưa viết notebook full training hoặc đổi core V1.
