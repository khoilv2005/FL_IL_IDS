# So Sánh Các Phương Pháp Regularization trong Federated Class Incremental Learning

## Tổng Quan

| Thuật Toán | Tên Đầy Đủ | Cơ Chế Chính | Paper Gốc |
|------------|------------|--------------|-----------|
| **EWC** | Elastic Weight Consolidation | Fisher Information + Parameter Regularization | Kirkpatrick et al., PNAS 2017 |
| **LwF** | Learning without Forgetting | Knowledge Distillation | Li & Hoiem, ECCV 2016 |
| **CGoFed** | Constrained Gradient Optimization | Gradient Projection + SVD | Feng et al., IEEE TKDE 2025 |

---

## 1. EWC (Elastic Weight Consolidation)

### Công Thức Loss Function
```
L_EWC = L_CE + (λ/2) * Σ_i F_i * (θ_i - θ_i*)²
```

- **L_CE**: Cross-entropy loss trên task hiện tại
- **F_i**: Fisher Information (độ quan trọng của parameter i)
- **θ_i***: Giá trị tối ưu của parameter sau task trước
- **λ**: Hệ số regularization (thường ~1000)

### Điểm Mạnh ✅

| STT | Điểm Mạnh | Giải Thích |
|-----|-----------|------------|
| 1 | **Có lý thuyết vững chắc** | Dựa trên Fisher Information Matrix từ thống kê Bayesian |
| 2 | **Bảo vệ parameter cụ thể** | Xác định chính xác parameter nào quan trọng cho task cũ |
| 3 | **Linh hoạt với parameter không quan trọng** | Cho phép thay đổi thoải mái các parameter có F_i nhỏ |
| 4 | **Online EWC variant** | Có thể dùng running average thay vì lưu tất cả task |
| 5 | **Dễ tích hợp** | Mixin pattern - có thể thêm vào bất kỳ base trainer nào |

### Điểm Yếu ❌

| STT | Điểm Yếu | Hệ Quả |
|-----|----------|--------|
| 1 | **Tính toán Fisher expensive** | Cần backward pass cho từng sample (~200 samples) |
| 2 | **Over-constraint** | Giữ quá chặt tất cả parameters quan trọng, khó học task mới |
| 3 | **Lưu trữ lớn** | Phải lưu F_i và θ_i* cho mỗi task (2x model size per task) |
| 4 | **Không adaptive** | λ cố định, không điều chỉnh theo mức độ forgetting |
| 5 | **Diagonal approximation** | Bỏ qua tương quan giữa các parameters (off-diagonal Fisher) |

### Vấn Đề Đã Giải Quyết ✅
- [x] Xác định được parameters quan trọng cho task cũ
- [x] Ngăn chặn catastrophic forgetting cơ bản
- [x] Có thể kết hợp với các phương pháp FL khác (FedAvg, FedProx)

### Vấn Đề Chưa Giải Quyết ❌
- [ ] Không linh hoạt trong việc điều chỉnh mức độ bảo vệ
- [ ] Chi phí tính toán và lưu trữ cao khi nhiều task
- [ ] Không tận dụng được thông tin giữa các task liên quan

---

## 2. LwF (Learning without Forgetting)

### Công Thức Loss Function
```
L_LwF = L_CE + α * T² * KL(σ(z_old/T) || σ(z_new/T))
```

- **L_CE**: Cross-entropy loss trên task hiện tại
- **z_old**: Logits từ old model (frozen teacher)
- **z_new**: Logits từ current model (student)
- **T**: Temperature scaling (thường = 2.0)
- **α**: Distillation weight (thường = 1.0)

### Điểm Mạnh ✅

| STT | Điểm Mạnh | Giải Thích |
|-----|-----------|------------|
| 1 | **Không cần lưu data cũ** | Chỉ cần old model snapshot, không cần rehearsal |
| 2 | **Đơn giản, dễ hiểu** | Dựa trên knowledge distillation quen thuộc |
| 3 | **Soft targets giàu thông tin** | Temperature scaling giữ được quan hệ giữa các classes |
| 4 | **Linh hoạt** | Có thể distill trên old classes only hoặc all classes |
| 5 | **Lưu trữ nhỏ** | Chỉ cần lưu model snapshot (1x model size per task) |

### Điểm Yếu ❌

| STT | Điểm Yếu | Hệ Quả |
|-----|----------|--------|
| 1 | **Phụ thuộc old model** | Nếu old model kém chất lượng, distillation truyền lỗi |
| 2 | **Không bảo vệ trực tiếp weights** | Chỉ gián tiếp qua output matching, weights vẫn có thể drift |
| 3 | **Task interference** | Task mới có thể "ghi đè" hoàn toàn kiến thức cũ |
| 4 | **Temperature sensitive** | Hiệu quả phụ thuộc nhiều vào chọn T |
| 5 | **Không xử lý class imbalance** | Phân phối non-IID trong FL làm giảm hiệu quả |

### Vấn Đề Đã Giải Quyết ✅
- [x] Tránh được việc lưu trữ data cũ (privacy-friendly)
- [x] Transfer knowledge qua soft targets
- [x] Dễ triển khai trong federated setting

### Vấn Đề Chưa Giải Quyết ❌
- [ ] Không có cơ chế hard constraint lên gradient/weights
- [ ] Dễ bị ảnh hưởng bởi chất lượng model cũ
- [ ] Không tận dụng được cấu trúc không gian representation

---

## 3. CGoFed (Constrained Gradient Optimization)

### Công Thức

#### Representation Space (Eq. 2-4):
```
R^t = F(Θ^t, X^t) = [z_1, ..., z_n]  # Activations từ forward pass
SVD: R^t = U^t Σ^t (V^t)^T
M^t = [u_1, ..., u_κ]  # κ basis vectors từ U
```

#### Gradient Projection (Eq. 9):
```
g' = g - μ_t * (g @ M^t @ M^t^T)
```

#### Adaptive Relaxation (Eq. 7-8):
```
μ_t = μ_init * α^(t - t_τ)  if AF >= θ (reset)
```

### Điểm Mạnh ✅

| STT | Điểm Mạnh | Giải Thích |
|-----|-----------|------------|
| 1 | **Gradient projection trực tiếp** | Loại bỏ thành phần gradient trong old task space |
| 2 | **Adaptive relaxation** | Điều chỉnh μ_t dựa trên Average Forgetting (AF) |
| 3 | **Activation-based SVD** | Dựng không gian representation từ activations (hiệu quả) |
| 4 | **Cross-task regularization** | Chọn top-K task tương tự để weighted aggregation |
| 5 | **Per-layer projection** | Linh hoạt projection theo từng layer |
| 6 | **Importance weighting** | Sigmoid(Σ) để weight các basis vectors |

### Điểm Yếu ❌

| STT | Điểm Yếu | Giải Thích |
|-----|----------|------------|
| 1 | **Phức tạp hơn EWC/LwF** | Nhiều hyperparameters (α, θ, energy_threshold) |
| 2 | **Chi phí SVD** | Tính SVD cho mỗi layer mỗi task |
| 3 | **Dimension matching** | Phải xử lý Conv vs Linear layers khác nhau (unfold) |
| 4 | **Storage per-layer** | Lưu basis matrices cho mỗi layer, mỗi task |
| 5 | **Yêu cầu samples cho SVD** | Cần ~100 samples để build representation space |

### Vấn Đề Đã Giải Quyết ✅
- [x] Hard constraint lên gradient (mạnh hơn soft regularization)
- [x] Adaptive điều chỉnh theo mức độ forgetting
- [x] Tận dụng cấu trúc representation space qua SVD
- [x] Cross-task knowledge transfer qua similarity matching

---

## 4. CGoFed Giải Quyết Điểm Yếu Của EWC và LwF Như Thế Nào?

### 4.1 Giải Quyết Vấn Đề Của EWC

| Vấn Đề EWC | Cách CGoFed Giải Quyết | Chi Tiết |
|------------|------------------------|----------|
| **Over-constraint** | **Gradient Projection thay vì Parameter Regularization** | EWC penalize mọi thay đổi của params quan trọng. CGoFed chỉ loại bỏ thành phần gradient "có hại" (trong old space), vẫn cho phép thay đổi theo hướng orthogonal. |
| **Không adaptive** | **AF-based Relaxation Coefficient** | EWC dùng λ cố định. CGoFed điều chỉnh μ_t = μ_init * α^(t-t_τ) dựa trên AF. Khi forgetting cao → reset μ_t = 1.0 để bảo vệ mạnh hơn. |
| **Diagonal approximation** | **SVD-based Representation Space** | EWC bỏ qua tương quan giữa params (Fisher diagonal). CGoFed dùng SVD để capture đầy đủ cấu trúc representation space. |
| **Không cross-task** | **Top-K Similarity Aggregation** | EWC xử lý từng task độc lập. CGoFed chọn top-K task tương tự nhất để weighted aggregation: θ_final = (1-λ)*θ_current + λ*Σ(w_i*θ_hist_i). |

**Ví dụ cụ thể:**
```python
# EWC: Penalize tất cả thay đổi của params quan trọng
loss = ce_loss + (lambda/2) * sum(F[i] * (param[i] - param_old[i])^2)

# CGoFed: Chỉ loại bỏ thành phần gradient trong old space
grad_projected = grad - mu * (grad @ M @ M^T)  # Chỉ remove "bad" component
```

### 4.2 Giải Quyết Vấn Đề Của LwF

| Vấn Đề LwF | Cách CGoFed Giải Quyết | Chi Tiết |
|------------|------------------------|----------|
| **Không bảo vệ trực tiếp weights** | **Hard Gradient Constraint** | LwF chỉ match outputs (soft). CGoFed constraint gradient ở mức weight space, đảm bảo weights không drift vào old task space. |
| **Phụ thuộc old model chất lượng** | **Representation Space từ nhiều samples** | LwF dùng 1 old model snapshot. CGoFed dựng representation space từ ~100 samples, robust hơn. |
| **Task interference** | **Orthogonal Gradient Update** | LwF vẫn có thể ghi đè kiến thức cũ. CGoFed đảm bảo gradient update orthogonal với old space, không interference. |
| **Không handle non-IID** | **Per-client + Cross-task Aggregation** | LwF không xử lý đặc thù FL. CGoFed có cơ chế aggregation riêng cho federated setting. |

**Ví dụ cụ thể:**
```python
# LwF: Match outputs (gián tiếp, weights vẫn có thể drift)
loss = ce_loss + alpha * KL(softmax(old_logits/T), softmax(new_logits/T))

# CGoFed: Hard constraint lên gradient (trực tiếp)
# Chỉ update theo hướng orthogonal với old task space
grad = grad - mu * projection_matrix @ grad
```

---

## 5. Bảng Tổng Hợp So Sánh

| Tiêu Chí | EWC | LwF | CGoFed |
|----------|-----|-----|--------|
| **Cơ chế chính** | Parameter regularization | Knowledge distillation | Gradient projection |
| **Hard/Soft constraint** | Soft | Soft | **Hard** |
| **Adaptive** | ❌ Cố định λ | ❌ Cố định α | **✅ AF-based μ_t** |
| **Cross-task transfer** | ❌ Không | ❌ Không | **✅ Top-K similarity** |
| **Chi phí tính toán** | Cao (Fisher) | Thấp | Trung bình (SVD) |
| **Chi phí lưu trữ** | Cao (2x/task) | Trung bình (1x/task) | Trung bình (basis matrices) |
| **Non-IID handling** | Yếu | Yếu | **Tốt** |
| **Convergence speed** | Chậm | Nhanh | Trung bình |
| **Memory efficiency** | Thấp | Cao | Trung bình |

---

## 6. Khi Nào Dùng Phương Pháp Nào?

### Chọn EWC khi:
- ✅ Cần lý thuyết vững chắc, dễ giải thích
- ✅ Số task ít (memory không quan trọng)
- ✅ Muốn tích hợp nhanh vào hệ thống có sẵn (mixin pattern)
- ❌ Không dùng khi có rất nhiều task (quá nhiều Fisher matrices)

### Chọn LwF khi:
- ✅ Cần đơn giản, dễ implement
- ✅ Tài nguyên hạn chế (chỉ cần lưu model snapshots)
- ✅ Old model có chất lượng tốt
- ❌ Không dùng khi old model kém hoặc có nhiều task liên quan

### Chọn CGoFed khi:
- ✅ Cần hiệu quả cao nhất trong FCIL
- ✅ Có nhiều task liên quan (tận dụng cross-task)
- ✅ Non-IID data distribution (phổ biến trong FL)
- ✅ Cần adaptive regularization theo mức độ forgetting
- ❌ Không dùng khi tài nguyên rất hạn chế (cần tính SVD)

---

## 7. Kết Luận

**CGoFed là sự kết hợp và cải tiến của cả EWC và LwF:**

1. **Từ EWC**: Học được ý tưởng "bảo vệ kiến thức cũ", nhưng thay vì soft regularization, dùng hard gradient constraint.

2. **Từ LwF**: Học được ý tưởng "không cần data cũ", nhưng thay vì match outputs, dùng SVD để capture representation structure.

3. **Cải tiến mới**: 
   - Adaptive relaxation dựa trên AF
   - Cross-task regularization qua similarity matching
   - Per-layer gradient projection

**Thứ tự hiệu quả (trong FCIL):**
```
CGoFed > FedProx+EWC > FedAvg+EWC > FedLwF > FedProx > FedAvg
```
