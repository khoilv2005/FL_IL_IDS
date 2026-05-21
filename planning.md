# 15/05/2026 - Chốt NICE Và Hướng Protocol Mới

## Mục Tiêu

Mục tiêu không phải chỉ là chạy NICE trong FL có server, mà là tạo một protocol mới:

```text
Decentralized NICE
```

Ý tưởng chính:

```text
NICE hiện tại:
clients -> server aggregate -> global model

Protocol mới:
clients tự trao đổi / tự aggregate / không có server trung tâm
nhưng vẫn giữ cơ chế NICE: neuron age, learner/mature/young, freeze mask, context detector
```

Sau khi đọc DFCA paper, hướng tốt nhất là:

```text
DFCA + NICE
```

Plexus có thể dùng phụ để chọn sample/neighbor nếu cần scale, nhưng core protocol nên dựa trên DFCA vì DFCA đã có decentralized clustering đúng bản chất.

## NICE Hiện Tại Gửi Gì Lên Để Tổng Hợp

Trong repo hiện tại, mỗi NICE client gửi lên:

```text
1. params       = weights model sau local training
2. num_samples  = số sample của client
3. loss         = local training loss
4. neuron_ages = tuổi neuron/unit ranks sau train
```

`neuron_ages` có các trạng thái:

```text
0  = young
1  = learner
2+ = mature
```

Tất cả client có cùng số neuron và cùng architecture. Khác nhau nằm ở:

```text
- neuron nào là young
- neuron nào là learner
- neuron nào là mature
- neuron nào được dùng để học task hiện tại
- weight update của từng neuron/layer
```

## Age Có Đang Được Dùng Không

Có, nhưng chưa tận dụng hết.

Hiện tại:

```text
server merge neuron_ages bằng max
aggregator dùng freeze mask từ age để bảo vệ mature neurons
```

Nghĩa là `age` đang được dùng để bảo vệ tri thức cũ, vì mature neurons không bị average đè lung tung.

Nhưng `age` chưa được dùng để:

```text
- cluster client
- tính similarity giữa client
- tính trọng số aggregation
- chọn neighbor
- chọn client đại diện
- dynamic split/merge cluster
- quyết định client nào nên đóng góp nhiều hơn
```

Kết luận:

```text
NICE hiện tại dùng age để bảo vệ weights.
Protocol mới nên dùng age + context + weight update để cluster và aggregate thông minh hơn.
```

## Cluster Là Gì

Cluster là nhóm client giống nhau theo một tiêu chí nào đó.

Ví dụ:

```text
Cluster 0: client có data/class giống nhau
Cluster 1: client có neuron age pattern giống nhau
Cluster 2: client có weight update giống nhau
```

Trong FL non-IID, nếu aggregate tất cả client chung một lần:

```text
100 clients -> FedAvg -> global model
```

thì client có data khác nhau có thể kéo model theo hướng khác nhau.

Nếu dùng cluster:

```text
clients -> cluster theo similarity -> aggregate trong từng cluster -> merge cluster
```

sẽ giảm việc trộn bừa các client quá khác nhau.

## Thông Tin Trong Quá Trình Incremental Có Thể Dùng Để Cluster

Protocol mới nên tận dụng nhiều thông tin hơn NICE hiện tại:

```text
1. weight params
   Model weights sau train.

2. weight delta
   delta_i = params_i - params_global
   Cho biết client đang đẩy model theo hướng nào.

3. num_samples
   Số sample của client.

4. loss
   Chất lượng local training.

5. neuron_ages
   Pattern young / learner / mature.

6. freeze_masks
   Neuron nào cần bảo vệ.

7. learner unit pattern
   Client đang dùng nhóm neuron nào để học task mới.

8. class histogram
   Client có class nào nhiều/ít.

9. context activation summary
   Đại diện context/task/episode của client.

10. context confidence
   Context detector route có chắc không.
```

Những thông tin này có thể tạo `client_repr`:

```text
client_repr_i = [
  weight_delta_summary,
  neuron_age_histogram,
  learner_mask,
  class_histogram,
  context_summary,
  loss,
  num_samples
]
```

Sau đó cluster client dựa trên `client_repr_i`.

## DFCA Cơ Chế Chính

DFCA là Decentralized Federated Clustering Algorithm.

Mỗi client giữ `k` model cluster:

```text
theta_i,1
theta_i,2
...
theta_i,k
```

Mỗi round có 3 bước:

```text
Step 1: AssignCluster
client chọn cluster có loss nhỏ nhất trên local data

Step 2: LocalUpdate
client chỉ train model của cluster đã chọn

Step 3: Decentralized Aggregation
client gửi model cluster vừa train cho neighbors
neighbors update bằng running average
```

Công thức assignment:

```text
c(i) = argmin_j F_client(theta_i,j, D_i)
```

Công thức running average:

```text
theta_i,j <- r/(r+1) * theta_i,j + 1/(r+1) * theta_m,j
```

DFCA hợp với bài toán này vì:

```text
- không cần server
- clustering là core protocol
- hợp non-IID
- mỗi client có local cluster bank
- cluster assignment thay đổi theo round
```

## Plexus Hay DFCA Cho Decentralized NICE

Plexus:

```text
Ưu điểm:
- nhẹ hơn
- sample/aggregator rõ
- dễ scale

Nhược điểm:
- cluster không phải core
- vẫn giống mini-server mỗi round
- NICE metadata chưa có vai trò mạnh
```

DFCA:

```text
Ưu điểm:
- cluster là core
- fully decentralized
- mỗi client tự assign cluster
- hợp với non-IID và incremental
- dễ gắn neuron age/context vào assignment và aggregation

Nhược điểm:
- nặng hơn vì mỗi client giữ k model
- assign cluster phải eval k model
```

Lựa chọn tốt nhất:

```text
Core protocol: DFCA + NICE
Optional communication scaling: Plexus sampling
```

Không nên lấy Plexus-NICE hiện tại làm final protocol. Plexus-NICE hiện tại chủ yếu là Plexus communication + NICE train, chưa tận dụng đầy đủ cluster/age/context.

## Protocol Đề Xuất: DeNICE

Tên tạm:

```text
DeNICE = Decentralized Clustered NICE
```

Mỗi client giữ `k` NICE cluster states:

```text
cluster_state_i,j = {
  theta_i,j,
  neuron_ages_i,j,
  freeze_masks_i,j,
  context_proto_i,j,
  class_hist_i,j
}
```

## DeNICE Round

### Step 1: NICE-Aware AssignCluster

Thay vì chỉ chọn cluster bằng loss như DFCA gốc, DeNICE chọn cluster bằng score:

```text
score_j =
  CE_loss(theta_i,j, local_data)
  + lambda_age * age_distance(age_i,j, local_age_pattern)
  + lambda_ctx * context_distance(context_i,j, local_context)
  + lambda_class * class_mismatch(class_hist_i,j, local_class_hist)
```

Client chọn:

```text
c(i) = argmin_j score_j
```

### Step 2: NICE LocalUpdate

Client train cluster đã chọn bằng NICE:

```text
theta_i,c(i)
neuron_ages_i,c(i)
freeze_masks_i,c(i)
context_proto_i,c(i)
```

NICE vẫn giữ:

```text
- tau-greedy learner unit selection
- young / learner / mature neurons
- freeze mature neurons
- context detector / context summary
- replay-free incremental learning
```

### Step 3: Message Passing

Client gửi cho neighbor:

```text
{
  client_id,
  cluster_id,
  params_or_delta,
  num_samples,
  loss,
  neuron_ages,
  freeze_masks,
  context_summary,
  class_histogram,
  assignment_margin
}
```

### Step 4: NICE-Aware DFCA Aggregation

Neighbor nhận message và update cluster tương ứng:

```text
theta_i,j <- weighted_running_average(theta_i,j, theta_m,j)
age_i,j <- merge_age(age_i,j, age_m,j)
context_i,j <- average_context(context_i,j, context_m,j)
freeze_i,j <- merge_freeze_mask(freeze_i,j, freeze_m,j)
```

Trọng số aggregation có thể là:

```text
w_m =
  sample_weight
  * quality_weight
  * context_confidence
  * age_compatibility
  * cluster_assignment_confidence
```

Trong đó:

```text
sample_weight                 = dựa trên num_samples
quality_weight                = loss thấp hơn -> weight cao hơn
context_confidence            = route context chắc hơn -> weight cao hơn
age_compatibility             = update ít phá mature neurons -> weight cao hơn
cluster_assignment_confidence = assignment margin lớn hơn -> weight cao hơn
```

## Dynamic Cluster

Cluster không nên fix cứng.

Khi class/task/client tăng, cluster nên có khả năng thay đổi:

```text
- thêm cluster mới nếu client_repr quá khác các cluster cũ
- split cluster nếu variance cao
- merge cluster nếu hai cluster quá giống
- remove cluster nếu rỗng lâu
```

Rule đơn giản:

```text
K_t = min(max_clusters, base_K + alpha * num_seen_tasks + beta * num_seen_classes)
```

Rule tốt hơn:

```text
if distance_to_nearest_cluster > threshold:
    create new cluster

if cluster_variance > split_threshold:
    split cluster

if distance(cluster_a, cluster_b) < merge_threshold:
    merge clusters

if cluster_empty_rounds > patience:
    remove cluster
```

## Sự Khác Biệt Chính So Với NICE Gốc

NICE gốc:

```text
- có server
- aggregate chung
- age dùng để protect neurons
- context dùng chủ yếu cho eval/inference
```

DeNICE:

```text
- không server
- mỗi client có cluster bank
- cluster assignment dùng loss + age + context + class histogram
- aggregation dùng weights + NICE metadata
- age/context không chỉ để protect/eval mà còn để cluster và aggregate
```

## Sự Khác Biệt Chính So Với DFCA Gốc

DFCA gốc:

```text
- mỗi client giữ k normal models
- assign cluster bằng local loss
- aggregate bằng running average
```

DeNICE:

```text
- mỗi client giữ k NICE models/states
- assign cluster bằng NICE-aware score
- aggregate bằng age/context-aware running average
- hỗ trợ class-incremental learning
- tận dụng neuron age và context detector
```

## Sự Khác Biệt Chính So Với Plexus-NICE

Plexus-NICE:

```text
- chọn sample/aggregator theo Plexus
- aggregator tạm thời average update
- cluster không phải core
```

DeNICE:

```text
- cluster là core state
- mỗi client tự giữ cluster bank
- không cần aggregator trung tâm mỗi round
- neighbor update bằng running average
- NICE metadata là thành phần của protocol
```

## Kết Luận

Plan tốt nhất:

```text
Dùng DFCA làm skeleton decentralized clustering.
Dùng NICE làm local incremental learner.
Mở rộng DFCA assignment và aggregation bằng NICE metadata.
Dùng Plexus chỉ như optional communication/sample layer nếu cần scale.
```

Novelty chính:

```text
Decentralized clustered continual learning bằng NICE state-aware cluster assignment và aggregation.
```
