# CANDLE V2 — Training-Time Prototype Integration

Ngày: 22/10/2026. Implementation/engineering configuration mới, chưa có training
hoặc smoke receipt. Không gọi đây là phương pháp đã cải thiện Accuracy/F1.

## Chạy notebook

Upload `train_candle_v2_kaggle.ipynb` lên Kaggle, bật Internet và GPU, attach
dataset `/kaggle/input/datasets/khoilv2005/100-clients`, rồi Run All.
Notebook clone đúng commit trên GitHub; không nhúng code/data/checkpoint ZIP.

- Fresh Task0→5, 20 rounds/task, local epoch1, seed42, ξ=0,8.
- Batch mặc định2048, sửa `BATCH_SIZE` ngay trong notebook nếu cần.
- Một GPU, một personalized model/client, native DeNICE peer graph.
- Checkpoint mỗi round; seal ZIP mỗi task; output budget12GiB.
- Source và roles trong `/kaggle/temp`, results trong `/kaggle/working`.
- Chỉ eval sau terminal Task5/Round19/consolidation.

Đây là **CANDLE V2 trên DeNICE legacy**, khác nhánh DeNICE V2 cluster-model trong
`denice_v2/`. Không có central aggregation server hoặc global prototype center.
V1 config/notebook vẫn pin source cũ; behavior mặc định không bật V2 hooks.

## Training protocol

Giữ data budget V1: global unique-content role split BASE80%, CAL2%, FIT8%,
validation10%, split seed20261006; runner giữ current BASE local holdout10% như
V1. Tức training không phải100% original TRAIN. Reuse mounted roles bằng cách
đặt `ROLES_DIR` tới folder có role_lock.json, hoặc notebook tạo lại cùng split.

Local optimizer và prototype construction chỉ dùng current `client.X_train`.
Prototype không lấy CAL/FIT/global validation hoặc TEST; V2 capsule cũng dùng
current training reference thay vì private held-out validation. Khi sang task
mới, không train lại task cũ, không đọc historical raw data và xóa feature bank
task cũ. Knowledge weights, availability và provenance vẫn được giữ.

## Thay đổi thuật toán

```
Round r:
  Local current-data training với bank từ r−1
    + cosine real CE
    + shared-class alignment / prototype contrastive loss
    + missing-class virtual prototype CE
    + frozen-teacher preservation trên current inputs
  Native graph masked aggregation
  Giữ imported rows receiver đã train khỏi peer averaging không phù hợp
  Tạo class means từ post-aggregation model + current local TRAIN
  Trao đổi với peer có positive alpha trong graph hiện tại
  Receiver aggregate prototypes theo alpha × sampled class support
  Checkpoint toàn bộ model/algorithm state
  Round r+1
```

Không copy Dense2 weights của donor. New output capacity có thể được khởi tạo
từ peer prototype rồi học bằng virtual CE; mature output rows không bị unfreeze.
Không nâng local ownership chỉ vì đã import class. Metadata thiếu local data
là optimization candidate, không được gọi là functional gap đã chứng nhận.
Không tự mở thêm inter-neighborhood edges để ép coverage34 class.
Prototype không có support trên allocated FC1 coordinates được bỏ qua và log
trong `unsupported_prototype_classes`, không mở class đó chỉ để tăng coverage.

Prototypes dùng như stop-gradient targets. Shared-class loss kết hợp cosine
attraction và CE trên các class means của peer. Virtual prototypes đi thẳng vào
FC2, không ngụ ý backbone đã nhìn thấy positive samples của class thiếu.
Current real CE có imported classes làm negatives nên có gradient hạn chế false
activation; việc này chưa bảo đảm generalization sang mọi class cũ.

## Inference thay đổi có chủ đích

Main policy: **TaskFreeCosineSelf**.

```
x → local CNN/GRU/FC1 → normalized cosine FC2
  → mask class chưa có successful local/virtual training
  → argmax
```

Scale10, không dùng FC2 bias. Chỉ dùng allocated FC1 coordinates cho cả real và
virtual inputs. Không nhận true task/class, không imported router gate, không
donor-model inference. Context-specific adapters được tắt để training và
task-free inference dùng cùng feature function. Native binary router metadata
vẫn phục vụ đường DeNICE/capsule hiện hữu, không quyết định prediction chính.

Training softmax gồm cumulative available classes và toàn bộ label IDs của
**current** task đã được protocol công bố. Những row current chưa train vẫn
bị frozen và không được mở ở inference. Điều này tránh CE=0 ở client chỉ có
một local class mà không cho dùng future task labels hoặc test information.

Đây là nhiều thay đổi cùng lúc để tạo một pipeline có thể chạy. Chưa được quy
mọi gain cho prototype transfer: benchmark sau đó cần cosine/task-free-only,
no-alignment, no-virtual-CE và no-preservation controls cùng budget.

## Cấu hình đầu tiên, chưa tune

| Thành phần | Giá trị |
|---|---:|
| Alignment weight | 0,1 |
| Virtual CE weight | 1,0 |
| Preservation weight | 0,5 |
| Cosine logit scale | 10 |
| Prototype samples/class/round | ≤128 |
| Prototype forward batch | 256 |
| AMP initial scale | 1024 |

Teacher là clone frozen của receiver trước local update, chạy trên current
inputs và chỉ distill các old-class logits. Không lưu raw old examples. Một
teacher forward bổ sung có chi phí; không hứa training chỉ tăng vài phần trăm.
KL trên current data và NICE maturity protection không chứng minh old-class
safety trên vùng input không được quan sát.

## Checkpoint, resume, logging

`candle_v2_state` nằm trong snapshot/restore algorithm state: availability,
class birth tasks, ownership, import provenance, bank round và prototype bank.
Bank chỉ được dùng cho đúng task và round r−1. Native bootstrap giữ inherited
knowledge nhưng reset ownership của client mới. Resume không cho chuyển V1
thành V2 hoặc đổi các loss/representation controls.

Muốn resume: mount continuation/task ZIP của chính CANDLE V2, đặt `RESUME_STATE`
trong notebook, giữ batch/config và dùng một output directory mới.

- `candle_v2_training.jsonl`: real CE, total objective, alignment, virtual CE,
  preservation, optimizer steps, local/imported/available classes.
- `candle_v2_communication.jsonl`: số prototype packets và bytes JSON thực sự
  encode trong harness, cùng native dense-model payload estimate.
- `checkpoint_task_*_all_rounds.zip`: model/masks/state/continuation.
- `candle_v2_roles.zip`: exact role manifest/lock và indices để reuse khi resume.
- `candle_v2_self_task_5/completion.json`: final pooled Accuracy/Macro-F1.
- `candle_v2_completion.json`: entry point tổng kết sau khi train/eval hoàn tất.

Communication log không phải full wire accounting: còn thiếu native capsule
bytes và transport headers. Không tuyên bố chỉ truyền prototype nên tổng bytes
bằng0 hoặc chắc chắn rẻ hơn V1.

## Evaluation

Giữ sample allocation của CANDLE V1: full34-class TEST chia thành receiver
shards deterministic, disjoint; tổng mỗi sample được dự đoán đúng một lần bởi
một model local. Headline là pooled Accuracy/Macro-F1 trên mọi class, kể cả
class receiver chưa học. Có confusion matrix, per-class/per-client metrics và
label coverage. Đây không phải mean cluster accuracy hoặc ensemble.

Route accuracy=null vì main policy không dự đoán task. Không dùng oracle mask,
không chọn config từ TEST, không eval trung gian; vì vậy run này cũng chưa tạo
được đường forgetting theo từng task. Cần campaign riêng nếu báo cáo forgetting.

## Tình trạng và giới hạn

Đã viết service/losses, runner hooks, snapshot/restore, terminal evaluator,
launcher/config và notebook builder. Chưa chạy unit tests, native smoke hoặc
training của implementation mới trong phiên này. Chưa có bằng chứng V2 vượt V1.
Fresh full run đầu tiên là engineering baseline; nếu TEST dùng để cải tiến
thiết kế, cần confirmation độc lập sau khi khóa cấu hình.

Prototype regularization trong training có tiền lệ ở
[FedProto](https://ojs.aaai.org/index.php/AAAI/article/view/20819); feature alignment
và classifier collaboration được nghiên cứu trong
[FedPAC](https://arxiv.org/abs/2306.11867). Các kết quả đó không tự chứng minh
variant decentralized continual này hoạt động, hoặc virtual missing-class
injection là đóng góp mới. Không khẳng định hai lớp neo xác định đầy đủ không
gian256D, centering giải quyết collapse, hoặc alignment loss tự bảo đảm shared
coordinates và chống forgetting.
