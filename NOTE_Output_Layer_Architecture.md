# CGoFed Neural Network Architecture Note
## Output Layer Design: Fixed 34 Neurons vs Incremental Expansion

---

## 1. KIẾN TRÚC HIỆN TẠI

### 1.1 Model Architecture (cnn_gru.py)
```python
class CNN_GRU_Model(nn.Module):
    def __init__(self, input_shape, num_classes: int = 34):
        # ... CNN layers ...
        # ... GRU layers ...
        self.fc2 = nn.Linear(256, num_classes)  # ← 34 neurons từ đầu!
```

**Output layer**: Luôn có **34 neurons** (0-33) ngay từ Task 0, không expand dần.

### 1.2 Config (train_incremental_kaggle.py)
```python
CONFIG = {
    "total_classes": 34,  # Tổng số classes trong dataset
    "num_classes": 34,    # Model có 34 output neurons ngay từ đầu
}
```

---

## 2. CƠ CHẾ HOẠT ĐỘNG QUA CÁC TASK

### Task 0 (Classes 0-9)

**Trạng thái weights:**
```
Neurons 0-9:   Được train, weights cập nhật
Neurons 10-33: Giữ giá trị khởi tạo random (PyTorch default: ~N(0, 0.01))
```

**Training:**
- Loss: `CrossEntropyLoss` chỉ tính trên classes 0-9
- Gradient: Chỉ flow qua neurons 0-9
- Neurons 10-33: **Không nhận gradient**, giữ nguyên giá trị khởi tạo

**Evaluation:**
```python
# IncrementalServer.evaluate_global()
seen_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Chỉ 10 classes
mask = [y in seen_classes for y in y_test]     # Mask test set
# → Chỉ evaluate trên classes đã thấy, ignore neurons 10-33
```

### Task 1 (Classes 10-15)

**Trạng thái weights:**
```
Neurons 0-9:   Đã train Task 0, được bảo vệ bởi CGoFed projection
Neurons 10-15: Được train lần đầu, weights cập nhật
Neurons 16-33: Vẫn random
```

**CGoFed Protection:**
```python
# cgofed.py:pre_step()
# Projection chỉ cho phép gradient update hướng không bị constrain
grad_new = grad - μ_t * (grad @ Uf)  # Bảo vệ weights classes cũ (0-9)
```

### Task 2-4 (Classes 16-33)

Tương tự, dần dần train thêm các classes mới trong khi bảo vệ classes cũ.

---

## 3. TẠI SAO KHÔNG EXPAND OUTPUT LAYER?

### ❌ Cách 1: Expand dần (KHÔNG dùng)
```python
# Task 0: 10 neurons
# Task 1: 10 + 6 = 16 neurons  
# Task 2: 16 + 6 = 22 neurons
# ...
```

**Vấn đề:**
1. **Complexity**: Phải remap class indices, resize weights mỗi task
2. **CGoFed projection**: Khó áp dụng khi dimension thay đổi liên tục
3. **Historical models**: Cross-task regularization phức tạp với shapes khác nhau

### ✅ Cách 2: Fixed 34 neurons (ĐANG dùng)

**Ưu điểm:**
1. **Đơn giản**: Shape cố định, dễ implement
2. **CGoFed hoạt động tốt**: Projection matrix Uf cố định dimension
3. **Cross-task**: Dễ dàng blend historical models (cùng shape)
4. **Memory**: Không cần reallocate memory mỗi task

---

## 4. CÁC NEURONS CHƯA TRAIN CÓ SAO KHÔNG?

### 4.1 Khởi tạo PyTorch mặc định
```python
# nn.Linear(256, 34) khởi tạo:
self.weight ~ Uniform(-sqrt(k), +sqrt(k)) với k = 1/256
→ Giá trị ~N(0, 0.06), rất nhỏ
```

### 4.2 Softmax tự điều chỉnh
```python
logits = model(x)  # neurons chưa train có giá trị ~0
probs = softmax(logits)
# → Classes chưa train có probability ~1/34 ≈ 3%, không ảnh hưởng lớn
```

### 4.3 Evaluation mask
```python
# Chỉ evaluate trên classes đã thấy
seen_set = {0, 1, ..., current_max_class}
mask = [y in seen_set for y in y_test]
```

### 4.4 Gradients
```python
# Task 0: labels chỉ trong [0-9]
loss = CrossEntropyLoss(logits[:, 0:10], labels)  # Chỉ tính loss trên 10 classes đầu
# → Gradients = 0 cho neurons 10-33
```

---

## 5. VISUALIZATION

### Task 0 Weight Norms
```
Class 0:  0.57 ✓ (đã train)
Class 1:  0.70 ✓ (đã train)
...
Class 9:  0.82 ✓ (đã train)
Class 10: 0.05  (random - chưa train)
Class 11: 0.03  (random - chưa train)
...
Class 33: 0.04  (random - chưa train)
```

### Task 2 Weight Norms (sau khi train classes 16-21)
```
Class 0:  0.57 ✓ (bảo vệ bởi CGoFed)
Class 1:  0.68 ✓ (bảo vệ bởi CGoFed)
...
Class 9:  0.81 ✓ (bảo vệ bởi CGoFed)
Class 10: 0.61 ✓ (bảo vệ bởi CGoFed - train Task 1)
...
Class 15: 0.78 ✓ (bảo vệ bởi CGoFed - train Task 1)
Class 16: 0.58 ✓ (mới train Task 2)
...
Class 21: 0.88 ✓ (mới train Task 2) ← CAO NHẤT, dẫn đến explosion!
Class 22: 0.02  (random - chưa train)
...
Class 33: 0.03  (random - chưa train)
```

---

## 6. VẤN ĐỀ THỰC TẾ GẶP PHẢI

### Vấn đề: Weight Explosion ở Class 21

**Nguyên nhân:**
1. Class 21 có **2.2 triệu samples** (siêu imbalanced)
2. Là class **cuối cùng** trong Task 2
3. Khi μ reset, gradient mạnh → weight explode từ 0.88 → 2.95 → 3.23

**Giải pháp:**
- Giảm μ_cgofed từ 1.5 → 0.8 (đã làm)
- Tăng SVD rank (energy_threshold 0.85 → 0.7) (đã làm)

---

## 7. SO SÁNH VỚI CÁC PHƯƠNG PHÁP KHÁC

| Feature | CGoFed (Fixed) | iCaRL (Expand) | LwF (Fixed) |
|---------|---------------|----------------|-------------|
| **Output neurons** | 34 cố định | Expand dần | 34 cố định |
| **Projection** | Có (SVD) | Không | Không |
| **Cross-task** | Blend models | N/A | Distillation |
| **Complexity** | Trung bình | Cao | Thấp |

---

## 8. KẾT LUẬN

✅ **Khởi tạo 34 neurons ngay từ đầu là CÁCH ĐÚNG**

- Không có vấn đề gì với neurons chưa train (random ~0)
- Evaluation có mask nên không bị nhiễu
- Phù hợp với CGoFed projection (cố định dimension)
- Đơn giản, hiệu quả, không cần resize network

⚠️ **Vấn đề thực sự là:**
- μ quá cao khi reset → weight explosion ở imbalanced classes
- SVD rank thấp → constrain không đủ mạnh
→ **Đã fix bằng cách giảm μ_cgofed và energy_threshold**

---

**Note này giải thích:**
1. Tại sao output layer có 34 neurons từ Task 0
2. Các neurons chưa train không gây vấn đề
3. Cơ chế bảo vệ classes cũ qua CGoFed projection
4. Vấn đề weight explosion và cách fix
