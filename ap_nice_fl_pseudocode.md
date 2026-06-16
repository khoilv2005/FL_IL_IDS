# Pseudo Code: Dynamic-K AP Clustering cho Decentralized FL với NICE

## 1. Tổng quan

Tài liệu này trình bày hai pseudo code cho bài toán phân cụm client trong decentralized Federated Learning:

1. **Algorithm 1**: Dynamic-K AP-Based Client Clustering dựa trên NICE Context Capsule.
2. **Algorithm 2**: Một round decentralized FL với NICE, dynamic clustering và age-aware neighbor aggregation.

Ý tưởng chính là không cố định số cụm `K` trước. Mỗi round, số cụm `K_t` được hình thành động từ thông tin context của client. Affinity Propagation phù hợp với mục tiêu này vì AP tự chọn exemplars và tự sinh số cụm từ similarity matrix.

Thông tin dùng để clustering không chỉ gồm model update và neuron age, mà còn tận dụng:

- **Activation prototype**: biểu diễn context/task mà client đang học.
- **Neuron age / selected-neuron mask**: cho biết vùng plastic và vùng mature.
- **Neuron importance score**: mức quan trọng của neuron dựa trên activation.
- **Label overlap**: mức trùng ngữ nghĩa giữa các class/task đã học.
- **Capacity histogram**: tình trạng còn capacity của từng layer.
- **Local reliability**: độ tin cậy local như validation accuracy, F1 hoặc loss.
- **Model update distance**: độ lệch giữa hướng update của các client.

Nếu clustering có silhouette thấp hoặc chỉ sinh một cụm không đủ ý nghĩa, hệ thống fallback về neighbor averaging thông thường.

Tài liệu tham khảo:

- **Affinity Propagation**
  - Paper: https://doi.org/10.1126/science.1136800
  - Wikipedia: https://en.wikipedia.org/wiki/Affinity_propagation
- **Silhouette Score**
  - Paper: https://doi.org/10.1016/0377-0427(87)90125-7
  - Wikipedia: https://en.wikipedia.org/wiki/Silhouette_(clustering)
- **NICE / Neuron Age**
  - Paper: https://doi.org/10.1109/CVPR52733.2024.02233

---

## 2. Ký hiệu chính

| Ký hiệu | Ý nghĩa |
|---|---|
| `M_i^t` | Model hiện tại của client `i` tại round `t` |
| `M_i,ref^t` | Model tham chiếu local của client `i`, là bản sao `M_i^t` trước local training |
| `W_i^{t+1}` | Local model của client `i` sau local training |
| `Delta_i^t` | Model update của client `i`, tính từ `reference_model_i` |
| `Age_i^t` | Neuron age map đầy đủ của client `i` |
| `B_i^t` hoặc `M_i^t` | Age mask hoặc selected-neuron mask, cho biết neuron  trẻ và trưởng thành |
| `P_i^t` hoặc `P_{i,c}^t` | Activation prototype theo class/task/context, lấy trung bình binary activation vector `b_i(x)` |
| `A_i^t` | Neuron importance score dựa trên activation |
| `Y_i^t` | Tập nhãn hoặc task semantic mà client đã học |
| `H_i^t` | Capacity histogram theo layer |
| `R_i^t` | Local reliability, ví dụ F1, validation accuracy hoặc calibration score |
| `Q_i^t` | Context-detector parameters hoặc context probability summary |
| `n_i^t` | Số mẫu local theo class/task |
| `Capsule_i^t` | NICE Context Capsule của client `i` |
| `N_i^t` | Tập neighbor của client `i` tại round `t` |
| `G_i^t` | Cụm cộng tác động của client `i` sau clustering |
| `S[i,j]` | Similarity giữa client `i` và client `j` |
| `E_context[i,j]` | Cạnh cộng tác cục bộ, bật khi `s_ij > delta_sim` |
| `C[i]` | Nhãn cụm của client `i` |
| `E` | Tập exemplars / cluster heads |
| `K_t` | Số cụm động tại round `t` |
| `S_avg` | Average silhouette score |
| `CompatibleMask_{ij}^t` | Mask kiểm soát phần parameter client `i` được phép nhận từ client `j` |

---

## 3. NICE Context Capsule

Sau local training, mỗi client tạo một gói metadata gọi là **NICE Context Capsule**. Capsule này được trao đổi với neighbors và dùng làm tín hiệu clustering.

```text
Capsule_i^t = {
    P_i^t,        # activation prototype theo class/task, tương ứng P_{i,c}^t
    B_i^t,        # age mask M_i^t: neuron trẻ, mature
    A_i^t,        # neuron importance score dựa trên activation
    H_i^t,        # capacity histogram theo layer
    Y_i^t,        # label/task semantic set đã học
    R_i^t,        # local reliability: accuracy, loss, F1, calibration
    Q_i^t,        # context-detector parameters hoặc context probability summary
    n_i^t,        # số mẫu local theo class/task
    Delta_i^t     # local model update, dùng cho update distance/anomaly score
}
```

Trong đó, `P_i^t` cho biết client đang học ngữ cảnh hoặc loại tấn công nào, còn `B_i^t` cho biết vùng neuron nào đang plastic và vùng nào đã mature. `A_i^t` bổ sung neuron importance score mà NICE dùng khi chọn neuron quan trọng theo activation. `H_i^t` không chỉ là capacity chung, mà là histogram/tỷ lệ trạng thái neuron theo layer. `Q_i^t` có thể là tham số context-detector hoặc summary xác suất context. `Delta_i^t` không nằm trong capsule lõi ở mục 5 của đề xuất, nhưng được giữ trong pseudocode vì mục 6 dùng `D(Delta_i, Delta_j)` để tính update distance hoặc anomaly score.

Ý nghĩa của capsule là biến thông tin activation, neuron age và context thành tín hiệu điều phối cộng tác trong decentralized FL. So với clustering chỉ dựa trên gradient/update, capsule ổn định hơn vì một phần activation signature đến từ mature neurons đã bị đóng băng.

---

# Algorithm 1: Dynamic-K AP-Based Client Clustering with NICE Context Capsule

## Mục tiêu

Thuật toán này phân cụm client trong phạm vi neighbor/collaboration graph mà không cần nhập trước số cụm `K`. Số cụm `K_t` được xác định động từ số exemplars mà Affinity Propagation tìm được.

## Pseudo code

```text
Algorithm 1: Dynamic-K AP-Based Client Clustering with NICE Context Capsule

Input:
    Local reference client models {M_i,ref^t = M_i^t} for i = 1..N
    Local client models {W_i^{t+1}} for i = 1..N
    NICE Context Capsules {Capsule_i^t}
    Neighbor graph or candidate clients V_t
    Similarity weights:
        lambda_proto, lambda_age, lambda_label,
        lambda_importance, lambda_capacity,
        lambda_reliability, lambda_update
    beta >= 0              # preference weight
    damping in [0.5,1)     # AP damping factor
    T_max                  # maximum AP iterations
    conv_iter              # stable iterations for convergence
    theta_s                # silhouette threshold
    delta_sim              # minimum context similarity for collaboration
    epsilon > 0            # numerical stability constant

Output:
    Client cluster labels C[i]
    Exemplars E
    Dynamic number of clusters K_t
    Average silhouette score S_avg
    Similarity matrix S
    Local context collaboration graph E_context

Function SAFE_L2_NORMALIZE(x):
    if ||x||_2 < epsilon:
        return zero_vector_like(x)
    else:
        return x / ||x||_2

Function LABEL_OVERLAP(Y_i, Y_j):
    if |Y_i union Y_j| = 0:
        return 0
    else:
        return |Y_i intersection Y_j| / |Y_i union Y_j|

Function CAPACITY_COMPATIBILITY(H_i, H_j):
    return 1 - normalized_distance(H_i, H_j)

Function UPDATE_DISTANCE(Delta_i, Delta_j):
    return ||SAFE_L2_NORMALIZE(Delta_i) - SAFE_L2_NORMALIZE(Delta_j)||_2

Function UPDATE_DISTANCE_OR_ANOMALY(Delta_i, Delta_j):
    return UPDATE_DISTANCE(Delta_i, Delta_j)

Step 0: Handle small client set
    N <- |V_t|

    if N < 2:
        C[1] <- 1
        E <- {1}
        K_t <- 1
        S_avg <- invalid
        E_context <- zero matrix N x N
        return C, E, K_t, S_avg, S, E_context

Step 1: Build capsule-aware client representation
    for each client i in V_t:
        Delta_i <- flatten(W_i^{t+1} - M_i,ref^t)
        Delta_i <- compress(Delta_i)
        Delta_i_norm <- SAFE_L2_NORMALIZE(Delta_i)

        Age_i_vec <- vectorize(Age_i^t)
        Age_i_norm <- SAFE_L2_NORMALIZE(Age_i_vec)

        P_i_norm <- SAFE_L2_NORMALIZE(vectorize(P_i^t))
        A_i_norm <- SAFE_L2_NORMALIZE(vectorize(A_i^t))
        H_i_norm <- SAFE_L2_NORMALIZE(vectorize(H_i^t))

        f_i <- concat(Delta_i_norm,
                      Age_i_norm,
                      P_i_norm,
                      A_i_norm,
                      H_i_norm)

Step 2: Compute context-aware similarity matrix S
    for each pair i != j:
        proto_sim <- cosine(P_i^t, P_j^t)
        age_sim <- Jaccard(B_i^t, B_j^t)
        importance_sim <- cosine(A_i^t, A_j^t)
        label_sim <- LABEL_OVERLAP(Y_i^t, Y_j^t)
        capacity_sim <- CAPACITY_COMPATIBILITY(H_i^t, H_j^t)
        reliability_sim <- R_j^t
        update_dist <- UPDATE_DISTANCE_OR_ANOMALY(Delta_i^t, Delta_j^t)

        s_ij <- lambda_proto * proto_sim
                 + lambda_age * age_sim
                 + lambda_importance * importance_sim
                 + lambda_label * label_sim
                 + lambda_capacity * capacity_sim
                 + lambda_reliability * reliability_sim
                 - lambda_update * update_dist

        if s_ij > delta_sim:
            E_context[i,j] <- 1
            S[i,j] <- s_ij
        else:
            E_context[i,j] <- 0
            S[i,j] <- large_negative_value

Step 3: Set AP preferences
    valid_similarities <- {S[i,j] | i != j and S[i,j] > large_negative_value}

    if valid_similarities is empty:
        p0 <- 0
    else:
        p0 <- median(valid_similarities)

    for each client i:
        q_i <- normalize(total_count(n_i^t) * R_i^t)
        S[i,i] <- p0 + beta * q_i
            # high-reliability and data-rich clients are more likely
            # to become exemplars / cluster heads

Step 4: Initialize AP messages
    R_msg <- zero matrix N x N
    A_msg <- zero matrix N x N
    stable_count <- 0
    previous_assignment <- None

Step 5: Affinity Propagation message passing
    for iter = 1 to T_max:

        # 5.1 Update responsibility
        for each i,k:
            R_raw[i,k] <- S[i,k] -
                          max_{k' != k} { A_msg[i,k'] + S[i,k'] }

        R_msg <- damping * R_msg + (1 - damping) * R_raw

        # 5.2 Update availability
        for each i,k where i != k:
            A_raw[i,k] <- min(0,
                              R_msg[k,k] +
                              sum_{i' not in {i,k}}
                                  max(0, R_msg[i',k]))

        for each k:
            A_raw[k,k] <- sum_{i' != k} max(0, R_msg[i',k])

        A_msg <- damping * A_msg + (1 - damping) * A_raw

        # 5.3 Compute current assignments
        for each client i:
            C_iter[i] <- argmax_k { A_msg[i,k] + R_msg[i,k] }

        if C_iter == previous_assignment:
            stable_count <- stable_count + 1
        else:
            stable_count <- 0

        previous_assignment <- C_iter

        if stable_count >= conv_iter:
            break

Step 6: Extract dynamic clusters
    for each client i:
        C[i] <- argmax_k { A_msg[i,k] + R_msg[i,k] }

    E <- { k | C[k] = k }
    K_t <- number_of_unique_clusters(C)

Step 7: Silhouette evaluation
    if K_t < 2:
        S_avg <- invalid
    else:
        for each client i:
            same_cluster <- {j | C[j] = C[i], j != i}

            if same_cluster is empty:
                a_i <- 0
            else:
                a_i <- mean_{j in same_cluster} distance(f_i, f_j)

            b_i <- min over clusters c != C[i]
                   mean_{j:C[j]=c} distance(f_i, f_j)

            if max(a_i, b_i) < epsilon:
                s_i <- 0
            else:
                s_i <- (b_i - a_i) / max(a_i, b_i)

        S_avg <- mean_i(s_i)

Step 8: Validate clustering quality
    if K_t < 2 or S_avg is invalid or S_avg < theta_s:
        mark clustering as low_quality
    else:
        mark clustering as valid

Return:
    C, E, K_t, S_avg, S, E_context
```

## Giải thích từng bước

### Step 1: Tạo biểu diễn client từ capsule

Mỗi client không còn chỉ được biểu diễn bằng `Delta_i` và `Age_i`. Thuật toán bổ sung `P_i^t`, `A_i^t` và `H_i^t` để phản ánh context, neuron importance và capacity. `A_i^t` giúp silhouette/distance không đánh đồng hai client chỉ vì age mask giống nhau trong khi neuron quan trọng thực tế khác nhau.

### Step 2: Tính context-aware similarity

Similarity giữa hai client được tính bằng nhiều tín hiệu:

```text
s_ij =
    lambda_proto * cos(P_i, P_j)
  + lambda_age * Jaccard(B_i, B_j)
  + lambda_importance * cos(A_i, A_j)
  + lambda_label * Overlap(Y_i, Y_j)
  + lambda_capacity * C(H_i, H_j)
  + lambda_reliability * R_j
  - lambda_update * D_or_anomaly(Delta_i, Delta_j)

E_context[i,j] = 1 if s_ij > delta_sim
E_context[i,j] = 0 otherwise
S[i,j] = s_ij if E_context[i,j] = 1
S[i,j] = large_negative_value otherwise
```

Ý nghĩa:

- Prototype giống nhau: hai client có context/task gần nhau.
- Age mask giống nhau: hai client đang dùng vùng neuron tương thích.
- Neuron importance giống nhau: hai client dựa vào các neuron quan trọng tương tự.
- Label overlap cao: hai client có semantic class gần nhau.
- Capacity histogram tương thích: hai client có khả năng chia sẻ update an toàn hơn.
- Reliability cao: neighbor đáng tin hơn.
- Update distance lớn: giảm similarity để tránh aggregate sai hướng.

Mục 6 trong đề xuất nêu các lựa chọn clustering phi tập trung như label propagation, gossip-based community detection hoặc Louvain cục bộ. Pseudocode này chọn **Affinity Propagation** như một hiện thực cụ thể vì AP không cần nhập trước số cụm `K`. AP vẫn dùng đúng NICE similarity `s_ij` và chỉ cho phép cạnh hợp lệ qua `E_context`; vì vậy kết quả cuối vẫn là cụm cộng tác động giữa các client có context tương thích.

### Step 3: Preference và exemplar

Preference `S[i,i]` quyết định client nào dễ trở thành exemplar. `n_i^t` được lưu theo class/task để phục vụ aggregation có điều kiện; khi tính preference, dùng tổng số mẫu hoặc số mẫu hữu hiệu của client:

```text
q_i = normalize(total_count(n_i^t) * R_i^t)
```

### Step 6: Dynamic K

Số cụm không phải input. Sau khi AP hội tụ:

```text
K_t = number_of_unique_clusters(C)
```

Do đó `K_t` có thể thay đổi theo round, theo context, theo neighbor graph và theo trạng thái NICE của các client.

### Step 7-8: Silhouette và fallback

Silhouette score được dùng để kiểm tra cụm có thật sự tách biệt hay không. Nếu cụm kém chất lượng, round đó không dùng cluster-based aggregation mà fallback về neighbor averaging.

---

# Algorithm 2: One FL Round with NICE, Dynamic-K Clustering, and Age-Aware Neighbor Aggregation

## Mục tiêu

Thuật toán này mô tả một round decentralized FL. Mỗi client học local bằng NICE, tạo NICE Context Capsule, trao đổi capsule/update với neighbors, hình thành cụm động bằng Algorithm 1, sau đó aggregate với neighbors trong cụm bằng age-aware mask.

## Pseudo code

```text
Algorithm 2: One FL Round with NICE, Dynamic-K AP Clustering,
             and Age-Aware Neighbor Aggregation

Input:
    Current client models {M_i^t}
    Participating clients P_t
    Neighbor sets {N_i^t} for each client i
    Client datasets {D_i}
    Client-side neuron ages {Age_i^t}
    Client-local frozen weights {FrozenWeights_i}
    Context memory {P_i^old, Y_i^old}
    Local epochs E_local
    Learning rate eta
    Aggregation step size eta_agg
    Validation fraction v
    NICE parameters: tau, pruning_interval
    Dynamic AP parameters:
        lambda_proto, lambda_age, lambda_label,
        lambda_importance, lambda_capacity,
        lambda_reliability, lambda_update,
        beta, damping, T_max, conv_iter, theta_s, delta_sim
    epsilon > 0

Output:
    Updated client models {M_i^{t+1}}
    Dynamic cluster labels C
    Exemplars E
    Dynamic number of clusters K_t
    Silhouette score S_avg
    Context collaboration graph E_context
    Updated neuron ages {Age_i^{t+1}}
    Updated NICE Context Capsules {Capsule_i^t}

Phase 1: Initialize local reference models
    for each client i in P_t:
        M_i,ref^t <- copy(M_i^t)
            # no full neighbor-model exchange; neighbors later exchange
            # capsules, updates, and trainable masks only

Phase 2: Local training with NICE
    parallel for each client i in P_t:

        age_i <- copy(Age_i^t)
        model_i <- copy(M_i,ref^t)

        # Restore client-specific frozen neurons
        for each neuron n where age_i[n] >= 2:
            model_i[n] <- FrozenWeights_i[n]

        reference_model_i <- copy(model_i)

        D_i_train, D_i_val <- split(D_i, validation_fraction=v)

        # Temporarily activate reserve neurons
        for each neuron n where age_i[n] = 0:
            age_i[n] <- 1

        for epoch = 1 to E_local:
            for each mini-batch (x,y) in D_i_train:
                output <- forward_NICE(model_i, x, age_i)
                loss <- IDS_loss(output, y)

                update model_i using SGD with learning rate eta
                    only update parameters allowed by NICE mask

            if epoch mod pruning_interval == 0:
                activation_score <- compute_neuron_activation(
                                        model_i,
                                        sample(D_i_train))

                selected_neurons <- greedy_select_neurons(
                                        activation_score,
                                        threshold=tau)

                for each age-1 neuron n not in selected_neurons:
                    age_i[n] <- 0

                prune_young_to_old_connections(model_i, age_i)

        # Compute update and mask before increasing age
        Delta_i^t <- model_i - reference_model_i

        TrainableMask_i^t <- trainable_parameter_mask(age_i)
            # 1 for parameters trained in this local round

        local_metrics_i <- evaluate(model_i, D_i_val)
        R_i^t <- compute_reliability(local_metrics_i)
        n_i^t <- count_samples_by_class_or_task(D_i_train)

Phase 3: Build NICE Context Capsule
    for each client i in P_t:
        P_i^t <- compute_activation_prototype(
                    model_i,
                    D_i_train,
                    selected_neurons)

        B_i^t <- build_age_or_selected_neuron_mask(age_i, selected_neurons)
            # compact mask M_i^t used for Jaccard similarity

        A_i^t <- compute_neuron_importance_score(
                    model_i,
                    D_i_train,
                    selected_neurons)
            # activation-based importance score used by NICE

        Y_i^t <- extract_label_or_task_set(D_i_train)

        H_i^t <- compute_capacity_histogram(age_i)
            # e.g., ratio of reserve, young, and mature neurons per layer

        Q_i^t <- summarize_context_detector(model_i)
            # context-detector parameters or context probability summary

        Capsule_i^t <- {
            P_i^t,
            B_i^t,
            A_i^t,
            Age_i^t = age_i,
            H_i^t,
            Y_i^t,
            R_i^t,
            Q_i^t,
            n_i^t,
            Delta_i^t
        }

        send to neighbors or cluster exemplar:
            Capsule_i^t,
            Delta_i^t,
            TrainableMask_i^t

Phase 4: Dynamic-K AP-based client clustering
    for each client i in P_t:
        W_i^{t+1} <- reference_model_i + Delta_i^t

    C, E, K_t, S_avg, S, E_context <- DYNAMIC_AP_CLUSTER(
                                          reference_models={reference_model_i},
                                          local_models={W_i^{t+1}},
                                          capsules={Capsule_i^t},
                                          candidate_clients=P_t,
                                          lambda_proto=lambda_proto,
                                          lambda_age=lambda_age,
                                          lambda_label=lambda_label,
                                          lambda_importance=lambda_importance,
                                          lambda_capacity=lambda_capacity,
                                          lambda_reliability=lambda_reliability,
                                          lambda_update=lambda_update,
                                          beta=beta,
                                          damping=damping,
                                          T_max=T_max,
                                          conv_iter=conv_iter,
                                          theta_s=theta_s,
                                          delta_sim=delta_sim,
                                          epsilon=epsilon)

    if K_t < 2 or S_avg is invalid or S_avg < theta_s:
        C <- one_neighbor_group_for_all_clients_in_P_t
        E <- None
        K_t <- 1
        log "Low-quality clustering, fallback to neighbor averaging"

Phase 5: Build dynamic collaboration groups
    for each client i in P_t:
        G_i^t <- {i} union
                 {j in P_t | C[j] = C[i] and j in N_i^t}

Phase 6: Age-aware neighbor aggregation
    for each client i in P_t:

        for each neighbor j in G_i^t:
            s_plus_ij <- max(S[i,j], 0)
            n_eff_j <- effective_count(n_j^t,
                                       target_labels=Y_i^t,
                                       neighbor_labels=Y_j^t)
                # use matched class/task counts when available

            alpha_ij <- s_plus_ij * n_eff_j * R_j^t

        normalize alpha_ij over all j in G_i^t:
            alpha_ij <- alpha_ij /
                        (sum_{k in G_i^t} max(S[i,k], 0)
                                             * effective_count(n_k^t,
                                                               Y_i^t,
                                                               Y_k^t)
                                             * R_k^t
                         + epsilon)

        if sum_{j in G_i^t} alpha_ij < epsilon:
            set alpha_ii <- 1
            set alpha_ij <- 0 for all j != i

        AggregatedUpdate_i <- zero_like(M_i^t)

        for each neighbor j in G_i^t:
            CompatibleMask_ij^t <- build_compatible_mask(
                                      target_age=Age_i^t,
                                      neighbor_age=Age_j^t,
                                      target_labels=Y_i^t,
                                      neighbor_labels=Y_j^t,
                                      target_prototype=P_i^t,
                                      neighbor_prototype=P_j^t,
                                      cluster_centroid=centroid(P_k^t
                                                               for k in G_i^t),
                                      trainable_mask=TrainableMask_j^t)

            AggregatedUpdate_i <- AggregatedUpdate_i
                                  + alpha_ij *
                                    (CompatibleMask_ij^t odot Delta_j^t)

        M_i^{t+1} <- reference_model_i + eta_agg * AggregatedUpdate_i

Phase 7: Update local ages, frozen weights, and context memory
    for each client i in P_t:

        for each neuron n where age_i[n] >= 1:
            age_i[n] <- age_i[n] + 1

        for each neuron n where age_i[n] >= 2:
            freeze neuron n
            FrozenWeights_i[n] <- model_i[n]

        Age_i^{t+1} <- age_i
        update context memory with P_i^t and Y_i^t

Phase 8: Logging
    log:
        round index
        participating clients P_t
        neighbor sets {N_i^t}
        dynamic number of clusters K_t
        exemplars E
        silhouette score S_avg
        cluster sizes
        valid context edges E_context
        local IDS metrics
        reliability scores {R_i^t}

Return:
    {M_i^{t+1}}, C, E, K_t, S_avg, E_context, Age, {Capsule_i^t}
```

## Giải thích từng phase

### Phase 1: Khởi tạo local reference model

Mỗi client tạo `M_i,ref^t` bằng cách copy model hiện tại của chính nó: `M_i,ref^t <- copy(M_i^t)`. Phase này không nhận full model từ neighbors, vì trong đề xuất neighbor chỉ trao đổi capsule, update và mask phục vụ clustering/aggregation.

### Phase 2: Local training với NICE

Client học local trên task/dữ liệu mới từ `M_i,ref^t`, tức model local trước training. Trước khi học, frozen weights được restore để mature neurons của chính client không bị thay đổi sai. `Delta_i^t` và `TrainableMask_i^t` được tính trước khi tăng age để mask phản ánh đúng vùng vừa được train.

### Phase 3: Tạo NICE Context Capsule

Sau local training, client tạo capsule gồm activation prototype, age mask, neuron importance, reliability, context-detector summary, capacity histogram, số mẫu theo class/task, label/task set và update. `B_i^t` là mask gọn dùng để trao đổi và tính Jaccard, còn `Age_i^t` là full age map dùng cho compatible mask trong aggregation. Capsule biến activation/age/context từ thông tin phục vụ inference cục bộ thành tín hiệu điều phối clustering và aggregation trong decentralized FL.

### Phase 4: Dynamic-K AP clustering

Clustering dùng capsule từ neighbors để tính `s_ij`, lọc cạnh cộng tác bằng `delta_sim`, rồi AP dùng các cạnh hợp lệ trong `E_context` để tự tìm exemplars. Số cụm:

```text
K_t = number_of_unique_clusters(C)
```

Do đó `K_t` thay đổi theo round và theo trạng thái context của clients. Nếu silhouette thấp, clustering bị xem là không đáng tin và fallback về neighbor averaging. Cách này là một hiện thực cụ thể của mục 6 trong đề xuất: thay vì label propagation, gossip-based community detection hoặc Louvain cục bộ, pseudocode dùng AP để giữ tính dynamic-K.

### Phase 5: Dynamic collaboration groups

Sau khi có nhãn cụm, mỗi client chỉ cộng tác với neighbors cùng cụm:

```text
G_i^t = {i} union {j | C[j] = C[i] and j in N_i^t}
```

Điều này giữ đúng tinh thần decentralized FL: client không aggregate với toàn mạng, mà chỉ aggregate với neighbor có context tương thích.

### Phase 6: Age-aware neighbor aggregation

Trọng số neighbor được tính theo similarity, số mẫu và reliability:

```text
alpha_ij =
    max(S[i,j], 0) * n_eff_j * R_j^t
    / sum_{k in G_i^t} max(S[i,k], 0) * n_eff_k * R_k^t
```

Trong đó `n_eff_j` được lấy từ `n_j^t` theo class/task tương thích với `Y_i^t`. Nếu không có thống kê chi tiết hoặc không có label overlap hợp lệ, dùng tổng số mẫu như fallback. Nếu tổng trọng số bằng 0, client không nhận update từ neighbor trong round đó và giữ update của chính nó.

Sau đó update được mask bởi `CompatibleMask_ij^t`:

```text
M_i^{t+1} =
    reference_model_i
    + eta_agg * sum_j alpha_ij *
      (CompatibleMask_ij^t odot Delta_j^t)
```

Mask này đảm bảo không aggregate toàn bộ model. Nó chỉ cho phép chia sẻ parameter thuộc vùng plastic hoặc vùng cùng context, đồng thời tránh sửa mature neurons của client khác.

### Phase 7: Update NICE memory

Sau aggregation, client tăng tuổi neuron, đóng băng mature neurons và cập nhật context memory. Mature neurons tiếp tục được bảo vệ đúng tinh thần continual learning.

---

## 4. Compatible Mask trong age-aware aggregation

`CompatibleMask_ij^t` nên bật parameter chỉ khi các điều kiện sau thỏa mãn:

1. Parameter thuộc vùng plastic của client nhận hoặc vùng cùng context.
2. Neuron age giữa client `i` và neighbor `j` tương thích.
3. Output label tương ứng cùng semantic class.
4. Activation prototype của neighbor không lệch xa cluster centroid.
5. Update của neighbor không cố gắng sửa mature neurons của client nhận.

Lựa chọn an toàn cho bài báo:

- Không aggregate weight của mature neurons.
- Chỉ dùng mature neurons để tạo context capsule/prototype.
- Chỉ aggregate mature weights nếu có cùng class/task, cùng activation signature và cùng age alignment.




