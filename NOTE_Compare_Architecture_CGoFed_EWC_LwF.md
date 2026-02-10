# So Sánh Kiến Trúc Model: CGoFed vs EWC vs LwF
## Task 1 đến Task N - Neural Network Architecture Analysis

---

## 1. TỔNG QUAN KIẾN TRÚC

### 1.1 Cả 3 thuật toán đều dùng: **FIXED ARCHITECTURE**

| Thuật toán | Task 1 | Task N | Output Layer | Có Expand? |
|------------|---------|---------|--------------|------------|
| **CGoFed** | 34 neurons | 34 neurons | Cố định từ đầu | ❌ KHÔNG |
| **EWC** | 34 neurons | 34 neurons | Cố định từ đầu | ❌ KHÔNG |
| **LwF** | 34 neurons | 34 neurons | Cố định từ đầu | ❌ KHÔNG |

### 1.2 Khác biệt với Dynamic Expansion

| Phương pháp | Task 1 | Task N | Cách làm |
|-------------|---------|---------|----------|
| **iCaRL** | 10 neurons | 34 neurons | **EXPAND** output layer |
| **DER** | 10 neurons | 34 neurons | **EXPAND** cả feature extractor |
| **CGoFed/EWC/LwF** | 34 neurons | 34 neurons | **FIXED**, dùng regularization |

---

## 2. CHI TIẾT TỪNG THUẬT TOÁN

### 🔵 CGoFed (Constrained Gradient Optimization)

**Kiến trúc:**
```
Task 0: CNN-GRU → FC(256→34) 
        └─> Train classes 0-9, neurons 10-33 giữ random
        
Task 1: CNN-GRU → FC(256→34)
        └─> Train classes 10-15
        └─> Bảo vệ neurons 0-9 bằng Gradient Projection (SVD)
        
Task 2: CNN-GRU → FC(256→34)
        └─> Train classes 16-21
        └─> Bảo vệ neurons 0-15 bằng Gradient Projection (SVD)
        
Task N: Tương tự, dần dần train hết 34 classes
```

**Cơ chế chống forgetting:**
- **Projection**: ∇L ← ∇L - μ·(∇L @ M @ M^T) (Eq. 9)
- **Cross-task**: Blend với historical models (Eq. 11)
- **Relaxation**: μ decay theo task

**Cập nhật:** ✓ **Liên tục mỗi task**, không chờ task cuối

---

### 🟢 EWC (Elastic Weight Consolidation)

**Kiến trúc:**
```
Task 0: CNN-GRU → FC(256→34)
        └─> Train classes 0-9
        └─> Tính Fisher Information F_i (độ quan trọng của từng weight)
        
Task 1: CNN-GRU → FC(256→34)
        └─> Train classes 10-15
        └─> Loss += λ/2 * Σ F_i * (θ_i - θ*_i)^2
        └─> Cố gắng giữ weights quan trọng của Task 0
        
Task 2: CNN-GRU → FC(256→34)
        └─> Train classes 16-21
        └─> Loss += λ/2 * Σ F_i * (θ_i - θ*_i)^2
        └─> Bảo vệ weights quan trọng của Task 0+1
```

**Cơ chế chống forgetting:**
- **Fisher Information**: Xác định weights nào quan trọng cho task cũ
- **Penalty term**: Phạt khi thay đổi weights quan trọng
- **Regularization**: Lagrangian constraint

**Cập nhật:** ✓ **Liên tục mỗi task**, tính Fisher sau mỗi task

---

### 🟡 LwF (Learning without Forgetting)

**Kiến trúc:**
```
Task 0: CNN-GRU → FC(256→34)
        └─> Train classes 0-9 (standard)
        
Task 1: CNN-GRU → FC(256→34)
        └─> Train classes 10-15
        └─> Save snapshot model Task 0 (θ_old)
        └─> Distillation loss: Giữ outputs giống Task 0 cho classes cũ
        
Task 2: CNN-GRU → FC(256→34)
        └─> Train classes 16-21
        └─> Save snapshot model Task 1
        └─> Distillation loss: Giữ outputs giống Task 1 cho classes cũ
```

**Cơ chế chống forgetting:**
- **Knowledge Distillation**: Giữ "kiến thức" từ model cũ
- **Temperature scaling**: Softmax với T > 1 để giữ information
- **Multi-task loss**: L_CE (new) + α·L_distill (old)

**Cập nhật:** ✓ **Liên tục mỗi task**, save snapshot sau mỗi task

---

## 3. SO SÁNH CHI TIẾT

### 3.1 Kiến trúc Layer

| Layer | CGoFed | EWC | LwF | Note |
|-------|--------|-----|-----|------|
| **CNN** | Conv1d(3 layers) | Conv1d(3 layers) | Conv1d(3 layers) | Giống nhau |
| **GRU** | 2 layers, 100 units | 2 layers, 100 units | 2 layers, 100 units | Giống nhau |
| **FC1** | Linear(concat→256) | Linear(concat→256) | Linear(concat→256) | Giống nhau |
| **FC2** | **Linear(256→34)** | **Linear(256→34)** | **Linear(256→34)** | **34 neurons cố định** |
| **Output** | 34 classes | 34 classes | 34 classes | Cố định |

### 3.2 Cách xử lý Task mới

| Thuật toán | Task mới | Cách học | Cách bảo vệ cũ |
|------------|----------|----------|----------------|
| **CGoFed** | Train classes mới | Gradient descent | **Gradient Projection** (chặn gradient directions) |
| **EWC** | Train classes mới | Gradient descent + **Fisher penalty** (phạt đổi weights quan trọng) |
| **LwF** | Train classes mới | Gradient descent + **Distillation** (bắt outputs giống model cũ) |

### 3.3 Thứ tự cập nhật

```
CẢ 3 THUẬT TOÁN ĐỀU:
========================

Task 0:
  ├─ Train classes 0-9
  ├─ Lưu trạng thái (Fisher/Snapshot/Representation)
  └─ Hoàn thành Task 0

Task 1:
  ├─ Train classes 10-15  ← CẬP NHẬT NGAY
  ├─ Bảo vệ Task 0 (Projection/Fisher/Distill)
  ├─ Lưu trạng thái
  └─ Hoàn thành Task 1

Task 2:
  ├─ Train classes 16-21  ← CẬP NHẬT NGAY
  ├─ Bảo vệ Task 0+1
  ├─ Lưu trạng thái
  └─ Hoàn thành Task 2

...

Task N:
  ├─ Train classes cuối  ← CẬP NHẬT NGAY
  ├─ Bảo vệ tất cả tasks trước
  └─ Hoàn thành

❌ KHÔNG PHẢI: Chỉ train ở task cuối
✓ CẬP NHẬT: Mỗi task đều train ngay
```

---

## 4. CÓ CHUNG Ý NGHĨA KHÔNG?

### 4.1 Cả 3 đều: **REGULARIZATION-BASED**

```
┌─────────────────────────────────────────────────────────────┐
│           FIXED ARCHITECTURE + REGULARIZATION               │
├─────────────────────────────────────────────────────────────┤
│  CGoFed:  Gradient Projection (constrain directions)        │
│  EWC:     Fisher Penalty (constraint important weights)     │
│  LwF:     Distillation (constraint output behavior)         │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Khác với: **DYNAMIC EXPANSION**

```
┌─────────────────────────────────────────────────────────────┐
│              DYNAMIC ARCHITECTURE (iCaRL/DER)               │
├─────────────────────────────────────────────────────────────┤
│  Task 0:  10 neurons                                        │
│  Task 1:  16 neurons (expand +6)                            │
│  Task 2:  22 neurons (expand +6)                            │
│  ...                                                        │
│  Task N:  34 neurons (expand dần)                           │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 So sánh tư duy

| Đặc điểm | Regularization (CGoFed/EWC/LwF) | Dynamic Expansion (iCaRL) |
|----------|----------------------------------|---------------------------|
| **Tư duy** | "Bảo vệ kiến thức cũ" | "Thêm không gian cho mới" |
| **Arch** | Fixed 34 neurons | Expand từ 10 → 34 |
| **Memory** | O(1) - cố định | O(N) - tăng dần |
| **Implementation** | Phức tạp (projection/fisher) | Đơn giản hơn |

---

## 5. TÓM TẮT TRẢ LỜI CÂU HỎI

### ❓ Câu 1: Kiến trúc Task 1 vs Task N?

**Trả lờI:**
- **Task 1**: CNN-GRU-FC(34 neurons), chỉ train classes 0-9, 24 neurons khác random
- **Task N**: CNN-GRU-FC(34 neurons), đã train hết 34 classes, tất cả neurons đều có giá trị

**Khác biệt duy nhất:** Weights của output layer (34 neurons dần được train qua các task)

### ❓ Câu 2: Cập nhật liên tục hay task cuối?

**Trả lờI:**
- ✓ **Cập nhật liên tục** mỗi task (không chờ task cuối)
- CGoFed: Projection được tính và áp dụng mỗi task
- EWC: Fisher được tính sau mỗi task
- LwF: Snapshot được lưu sau mỗi task

### ❓ Câu 3: Có chung ý nghĩa với Dynamic không?

**Trả lờI:**
- **KHÔNG chung ý nghĩa**
- CGoFed/EWC/LwF: **Regularization-based** (bảo vệ kiến thức cũ trong cùng không gian)
- iCaRL/DER: **Architecture-based** (mở rộng không gian cho kiến thức mới)

---

## 6. BẢNG SO SÁNH TỔNG HỢP

| Feature | CGoFed | EWC | LwF | iCaRL (Dynamic) |
|---------|--------|-----|-----|-----------------|
| **Output neurons Task 1** | 34 | 34 | 34 | 10 |
| **Output neurons Task N** | 34 | 34 | 34 | 34 |
| **Có expand?** | ❌ | ❌ | ❌ | ✅ |
| **Chống forgetting** | Gradient Projection | Fisher Penalty | Distillation | Nearest Class Mean |
| **Cập nhật** | Mỗi task | Mỗi task | Mỗi task | Mỗi task |
| **Memory/Sample** | Không cần | Không cần | Snapshot model | Exemplar set |
| **Multi-GPU** | ✅ Support | ✅ Support | ✅ Support | ⚠️ Complex |

---

## 7. KẾT LUẬN

### CGoFed, EWC, LwF đều:
1. ✅ Giữ **34 neurons cố định** từ Task 0
2. ✅ **Cập nhật liên tục** qua từng task
3. ✅ Dùng **regularization** để chống forgetting
4. ❌ **KHÔNG expand** architecture

### Khác biệt chính:
- **CGoFed**: Chặn gradient directions (SVD basis)
- **EWC**: Phạt thay đổi weights quan trọng (Fisher)
- **LwF**: Bắt outputs giống model cũ (Distillation)

### Dynamic (iCaRL) thì:
- Expand neurons dần dần
- Khác hoàn toàn approach
