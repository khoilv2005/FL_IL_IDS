Rất tốt 👍
Dưới đây là **mã giả Algorithm 1 được viết lại hoàn toàn bằng tiếng Việt**, **không dùng ký hiệu (Eq.x)** nữa mà **viết thẳng công thức**, kèm **chú thích ngắn gọn ngay tại bước đó** để bạn đọc *không cần lật lại bài báo*.

---

# THUẬT TOÁN: Chiến lược Tối ưu Gradient Có Ràng buộc cho Federated Class Incremental Learning (CGoFed)

---

## **ĐẦU VÀO**

* (K): số lượng client
* (T): số lượng task

## **ĐẦU RA**

* (\Theta^{T,g}): mô hình toàn cục (hoặc cá nhân hoá) sau khi học xong task cuối

---

## **KHỞI TẠO**

1. Với mỗi client (k):

   * Chuẩn bị chuỗi dữ liệu theo task:
     $$
     {D_k^1, D_k^2, \dots, D_k^T}
     $$
2. Khởi tạo tham số mô hình ban đầu cho mỗi client:
   $$
   \Theta_k^{init}
   $$

---

## **VÒNG LẶP THEO TASK**

3. **Cho** (t = 1) **đến** (T) **làm**:

---

## 🔹 PHẦN A — THỰC HIỆN TRÊN CLIENT (CHẠY SONG SONG)

4. **Cho mỗi client** (k = 1 \dots K) **chạy song song**:

---

### **TRƯỜNG HỢP 1: TASK ĐẦU TIÊN ((t = 1))**

5. **Huấn luyện mô hình của client k bằng loss phân loại (Cross-Entropy):**
   $$
   L_k(\Theta_k^1)
   =
   \frac{1}{n_k^1}
   \sum_{i=1}^{n_k^1}
   \ell\big(f(x_{k,i}^1;\Theta_k^1),; y_{k,i}^1\big)
   $$

6. **Tính gradient của loss:**
   $$
   g = \nabla_{\Theta_k^1} L_k(\Theta_k^1)
   $$

7. **Cập nhật gradient theo không gian trực giao (nếu có memory):**
   $$
   g \leftarrow g - g M^{0}(M^{0})^\top
   $$
   *(với task đầu, (M^{0}) gần như rỗng)*

8. **Cập nhật tham số mô hình:**
   $$
   \Theta_k^1 \leftarrow \Theta_k^1 - \eta g
   $$

---

### **TRƯỜNG HỢP 2: TASK THỨ (t > 1)**

9. **Huấn luyện mô hình với hàm loss tổng (loss phân loại + regularization):**
   $$
   \min_{\Theta_k^t}
   \Bigg(
   \frac{1}{n_k^t}
   \sum_{i=1}^{n_k^t}
   \ell\big(f(x_{k,i}^t;\Theta_k^t),; y_{k,i}^t\big)
   ;+;
   A(\Theta_k^t,\Theta^{old})
   \Bigg)
   $$

Trong đó regularization:
$$
A(\Theta_k^t,\Theta^{old})
==========================
\sum_{j < t}
\sum_{i \in \pi}
w_i^j
\left|
\Theta_k^t - \Theta_i^j
\right|_2^2
$$
---

10. **Tính hệ số siết ràng buộc (\mu_t):**

* Hàm decay:
  $$
  f(\alpha,t) = \alpha^t
  $$

* Công thức xác định:
  $$
  \mu_t =
  \begin{cases}
  \mu_{init},\alpha^t, & \text{nếu } AF < \tau \
  \mu_{init},\alpha^{t - t_\tau}, & \text{nếu } AF \ge \tau
  \end{cases}
  $$

---

11. **Tính gradient của loss tổng:**
    $$
    g = \nabla_{\Theta_k^t} L_k(\Theta_k^t)
    $$

---

12. **Chỉnh gradient để tránh phá task cũ (gradient constraint):**
    $$
    g \leftarrow g - \mu_t , g M^{t-1}(M^{t-1})^\top
    $$

> 👉 Bước này đảm bảo:
>
> * gradient vẫn giảm loss task mới
> * nhưng bị “bẻ hướng” để ít làm hỏng task cũ

---

13. **Cập nhật tham số mô hình:**
    $$
    \Theta_k^t \leftarrow \Theta_k^t - \eta g
    $$

---

## 🔹 SAU KHI TRAIN XONG TASK (t) (CLIENT LƯU KÝ ỨC)

14. **Tạo ma trận biểu diễn (representation) cho task t:**
    $$
    R_k^t = F(\Theta_k^t, X^t)
    $$

---

15. **Phân rã SVD để tìm các hướng quan trọng của task:**
    $$
    R_k^t = U_k^t \Sigma_k^t (V_k^t)^\top
    $$

---

16. **Tính trọng số cho mỗi vector cơ sở (độ quan trọng):**
    $$
    \Lambda_k^t = \frac{1}{1 + e^{-\Sigma_k^t}}
    $$

---

17. **Tạo và lưu ma trận nhớ (memory subspace):**
    $$
    M_k^t =
    \big[
    \lambda_1^t u_1^t,;
    \lambda_2^t u_2^t,;
    \dots
    \big]
    $$

---

18. **Gửi lên server:**

* Tham số mô hình (\Theta_k^t)
* Ma trận biểu diễn (R_k^t)

---

## 🔹 PHẦN B — THỰC HIỆN TRÊN SERVER

19. **Nếu là task đầu tiên ((t = 1)) thì gộp mô hình đơn giản:**
    $$
    \Theta_k^{1,g} = \Theta_k^1
    $$

---

20. **Tính độ tương đồng giữa các task của các client:**
    $$
    \phi_i^t
    =
    \left|
    R_k^t - R_i^{t-1}
    \right|_2
    $$

---

21. **Gộp mô hình theo độ tương đồng (personalized aggregation):**
    $$
    \Theta_k^{t,g}
    =
    \Theta_k^t

*

\sum_{i \neq k}
\frac{\phi_i^t}{\sum_{j \neq k} \phi_j^t}
\Theta_i^t
$$

---

22. **Kết thúc task t**

---

## ✅ **TÓM TẮT BẰNG 1 CÂU**

> **CGoFed học bằng cách:**
> dùng **loss để học task mới**,
> **bẻ hướng gradient để không phá task cũ**,
> và **thêm regularization + aggregation thông minh trên server** để các client hỗ trợ nhau đúng task.

---

Nếu bạn muốn, bước tiếp theo mình có thể:

* viết **phiên bản “giải thích bằng lời ngay dưới từng dòng mã giả”**, hoặc
* làm **ví dụ số 2D** để bạn thấy gradient bị chiếu như thế nào.
