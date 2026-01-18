Dưới đây là bản tóm tắt chi tiết nội dung của bài báo khoa học "CGoFed: Constrained Gradient Optimization Strategy for Federated Class Incremental Learning" được chuyển đổi sang định dạng Markdown.

---

# CGoFed: Chiến lược Tối ưu hóa Gradient Ràng buộc cho Học Tăng cường Lớp Liên kết (Federated Class Incremental Learning)

**Tác giả:** Jiyuan Feng, Xu Yang, Liwen Liang, Weihong Han, Binxing Fang, và Qing Liao 
**Tạp chí:** IEEE Transactions on Knowledge and Data Engineering, Vol. 37, No. 5, May 2025 

---

## 1. Tóm tắt (Abstract)

Bài báo giới thiệu **CGoFed**, một phương pháp mới cho Federated Class Incremental Learning (FCIL). FCIL là mô hình nơi các client liên tục tạo ra dữ liệu với các nhãn lớp mới chưa từng thấy và không chia sẻ dữ liệu cục bộ do quyền riêng tư. Các phương pháp hiện tại gặp hai thách thức lớn:

1. Thiếu sự cân bằng giữa việc duy trì kiến thức cũ (độ ổn định - stability) và thích nghi với nhiệm vụ mới (tính dẻo - plasticity).


2. Bỏ qua việc dữ liệu tăng cường có phân phối nhãn không đồng nhất (non-identical label distribution).



CGoFed đề xuất các module cập nhật gradient ràng buộc nới lỏng (relax-constrained) và điều chuẩn gradient chéo nhiệm vụ (cross-task gradient regularization). Kết quả cho thấy CGoFed cải thiện hiệu suất mô hình từ 8% - 23% so với các phương pháp SOTA.

---

## 2. Giới thiệu (Introduction)

* 
**Bối cảnh:** Federated Learning (FL) thường giả định phân phối nhãn tĩnh, nhưng trong thực tế, phân phối nhãn thay đổi động theo thời gian, dẫn đến bài toán FCIL.


* 
**Vấn đề:** Do thiết bị client có dung lượng hạn chế, việc chỉ lưu trữ và huấn luyện trên dữ liệu mới dẫn đến "lãng quên thảm khốc" (catastrophic forgetting) các lớp cũ.


* 
**Hạn chế của phương pháp cũ:** Các phương pháp dựa trên chưng cất (distillation) như CFeD phụ thuộc vào nhãn giả (pseudo-labels), gây sai lệch khi số lượng tác vụ tăng lên. Các phương pháp dựa trên bộ nhớ đệm (memory buffer) như GLFC giả định phân phối dữ liệu mới là giống nhau giữa các client, điều này không thực tế.



---

## 3. Các công trình liên quan (Related Work)

Bài báo phân loại các phương pháp FCIL hiện có thành 3 nhóm:

* **Dựa trên mở rộng (Expansion-Based):** Mở rộng cấu trúc mô hình để tăng dung lượng (ví dụ: FedWeIT). Tuy nhiên, mô hình dung lượng cố định thực tế hơn cho các thiết bị hạn chế tài nguyên.


* **Dựa trên điều chuẩn (Regularization-Based):** Phạt các thay đổi tham số lớn liên quan đến nhiệm vụ cũ (ví dụ: EWC, FedCurv, LwF, CFeD). Phương pháp này thường gặp khó khăn khi số lượng tác vụ tăng hoặc mối quan hệ ánh xạ bị suy giảm.


* **Dựa trên bộ nhớ (Memory-Based):** Lưu trữ một tập hợp con dữ liệu cũ (ví dụ: GLFC, TARGET). Các phương pháp này hiệu quả nhưng chưa giải quyết tốt vấn đề phân phối nhãn không đồng nhất trong dữ liệu tăng cường.



---

## 4. Định nghĩa bài toán & Thảo luận (Discussion on FCIL)

### 4.1. Federated Class Incremental Learning (FCIL)

* Mục tiêu là tối ưu hóa hàm mất mát (loss) trên tất cả các tác vụ của tất cả client:






* 
**Lãng quên thảm khốc:** Xảy ra khi việc học tác vụ mới làm thay đổi trọng số, khiến hướng cập nhật gradient đi vào không gian không liên quan đến tác vụ cũ.



### 4.2. Kịch bản dữ liệu tăng cường (Incremental-Data Scenarios)

Bài báo định nghĩa hai kịch bản:

1. 
**Identical New-Class (Lớp mới đồng nhất):** Nhãn của tác vụ mới tại client  chưa từng xuất hiện trong lịch sử của bất kỳ client nào khác.


2. **Non-Identical New-Class (Lớp mới không đồng nhất):** Nhãn của tác vụ mới tại client  có thể đã xuất hiện trong các tác vụ lịch sử của client khác. Đây là kịch bản thực tế và khó hơn.



---

## 5. Phương pháp đề xuất: CGoFed

CGoFed giải quyết vấn đề bằng hai module chính:

### 5.1. Cập nhật Gradient Ràng buộc Nới lỏng (Relax-Constrained Gradient Update)

Để cân bằng giữa tính ổn định (cho task cũ) và tính dẻo (cho task mới):

* **Cơ chế cũ (Strong-Constraint):** Chiếu gradient của tác vụ mới lên không gian trực giao với tác vụ cũ. Điều này hạn chế khả năng học cái mới.


* 
**Cải tiến của CGoFed:** Sử dụng hệ số cường độ ràng buộc có thể điều chỉnh () để cho phép hướng gradient lệch nhẹ khỏi phương trực giao, mở rộng không gian cập nhật.
Công thức cập nhật:







* Hệ số  giảm dần theo hàm mũ () để tăng tính dẻo, nhưng sẽ được reset nếu mức độ quên (Average Forgetting - AF) vượt quá ngưỡng .



### 5.2. Điều chuẩn Gradient Chéo nhiệm vụ (Cross-Task Gradient Regularization)

Để giải quyết vấn đề phân phối nhãn không đồng nhất:

* Mỗi client tính toán ma trận đại diện  và gửi về server (bảo vệ quyền riêng tư cấp mẫu).


* Server tính ma trận tương đồng  giữa tác vụ hiện tại của client  và các tác vụ lịch sử của client khác.


* Server thực hiện tổng hợp mô hình cá nhân hóa cho client  bằng cách chọn các mô hình lịch sử có liên quan nhất từ các client khác để hỗ trợ.



---

## 6. Thực nghiệm (Experiments)

### 6.1. Thiết lập

* **Datasets:** CIFAR-100, Tiny-ImageNet-200 (Ảnh); CoraFull, Reddit (Đồ thị).


* 
**Baselines:** FedAvg, FedEWC, FedLwF, CFeD, TARGET, GLFC, FedWeIT.


* 
**Metrics:** Average Accuracy (Độ chính xác trung bình - ) và Average Forgetting (Độ quên trung bình - ).



### 6.2. Kết quả chính

1. **Hiệu suất chung:** CGoFed đạt hiệu suất tốt nhất trên cả hai kịch bản dữ liệu (Identical và Non-Identical).
* Trên Tiny-ImageNet-200 (Non-Identical), CGoFed đạt độ chính xác **43.85%** so với 22.5% của FedWeIT (cải thiện ~21%).


* Độ quên () cực thấp, ví dụ 0.0041 trên CIFAR-100.




2. 
**Trên dữ liệu đồ thị:** CGoFed cũng vượt trội hơn FedEWC và FedLwF trên CoraFull và Reddit.


3. 
**Khả năng thích nghi (Plasticity):** CGoFed thích nghi nhanh với tác vụ mới ngay từ các vòng giao tiếp đầu tiên, cân bằng tốt hơn so với CFeD (học nhanh nhưng quên nhanh) và FedWeIT (học chậm hơn).


4. 
**Kịch bản Non-IID:** Trong các thiết lập Non-IID khắc nghiệt (), CGoFed vẫn duy trì khả năng chống quên tốt nhất và độ chính xác cao hơn 21.89% so với phương pháp tốt thứ nhì.



### 6.3. Hiệu quả tính toán

* Thời gian huấn luyện của CGoFed thấp hơn đáng kể so với GLFC và FedWeIT trên cả hai tập dữ liệu, đảm bảo tính hiệu quả cho ứng dụng thực tế.



---

## 7. Kết luận (Conclusion)

CGoFed giải quyết hiệu quả vấn đề đánh đổi giữa tính ổn định và tính dẻo trong FCIL thông qua việc cập nhật gradient ràng buộc nới lỏng. Đồng thời, nó sử dụng cơ chế hợp tác chéo nhiệm vụ để xử lý dữ liệu không đồng nhất (Non-IID), mang lại hiệu suất vượt trội so với các phương pháp hiện hành mà không cần lưu trữ dữ liệu cũ hay dataset phụ trợ.

Dưới đây là tổng hợp chi tiết tất cả các phương pháp (method) và công thức (formula) được đề xuất trong bài báo **CGoFed**, được chia thành các module chính:

### 1. Module Cập nhật Gradient Ràng buộc Nới lỏng (Relax-Constrained Gradient Update)

Module này nhằm cân bằng giữa tính ổn định (ghi nhớ nhiệm vụ cũ) và tính dẻo (học nhiệm vụ mới) bằng cách kiểm soát hướng cập nhật gradient.

**Bước 1: Xây dựng không gian đại diện (Representation Space)**
Trước tiên, mô hình tính toán ma trận đại diện cho không gian gradient của nhiệm vụ hiện tại  thông qua lan truyền xuôi (forward propagation) trên một tập mẫu con.




* Trong đó  là tập hợp  mẫu ngẫu nhiên từ dữ liệu .



**Bước 2: Phân rã giá trị kỳ dị (SVD) để tìm cơ sở trực giao**
Áp dụng thuật toán SVD để phân rã ma trận đại diện:




* 
 là ma trận kỳ dị trái chứa các vector cơ sở trực giao.


* Chọn  giá trị kỳ dị lớn nhất dựa trên ngưỡng năng lượng  để giảm chiều dữ liệu: .



**Bước 3: Tính hệ số quan trọng (Scaling Basis Vector)**
Tính toán trọng số quan trọng  cho các vector cơ sở dựa trên giá trị kỳ dị bằng hàm sigmoid:



*(Lưu ý: Văn bản gốc bị lỗi hiển thị công thức (5), nhưng mô tả xác định đây là hàm sigmoid)*.

**Bước 4: Hệ số nới lỏng ràng buộc (Relaxation Strategy)**
Để tăng tính dẻo, phương pháp sử dụng hệ số cường độ ràng buộc , giảm dần theo thời gian (theo hàm mũ) và được đặt lại nếu mức độ quên (Average Forgetting - AF) vượt quá ngưỡng .
Hàm suy giảm theo cấp số nhân:




Công thức xác định :




* 
: Tốc độ suy giảm (decay rate).


* 
: Chỉ số nhiệm vụ gần nhất mà tại đó AF vượt ngưỡng .



**Bước 5: Cập nhật Gradient (Gradient Update Rule)**
Gradient của nhiệm vụ mới  được cập nhật sao cho hướng của nó bị hạn chế bởi không gian gradient cũ nhưng được nới lỏng bởi hệ số :




* 
: Ma trận chứa các vector cơ sở của không gian gradient cũ.



---

### 2. Module Điều chuẩn Gradient Chéo Nhiệm vụ (Cross-Task Gradient Regularization)

Module này giải quyết vấn đề phân phối nhãn không đồng nhất (Non-Identical Label Distribution) bằng cách tận dụng các mô hình lịch sử từ các client khác.

**Bước 1: Tính ma trận tương đồng (Similarity Matrix)**
Server tính độ tương đồng giữa nhiệm vụ hiện tại  của client  và các nhiệm vụ của client  khác dựa trên khoảng cách L2-norm của ma trận đại diện:




**Bước 2: Hàm mất mát hợp tác chéo nhiệm vụ (Cross-task Cooperation Loss)**
Chọn các mô hình lịch sử có độ tương đồng cao nhất để tính toán loss điều chuẩn (regularization loss):




* 
: Số lượng client được chọn (ví dụ: 2 client) có giá trị  thấp nhất (tương đồng nhất).



**Bước 3: Tổng hợp mô hình cá nhân hóa (Personalized Aggregation)**
Server tổng hợp mô hình toàn cục cho client  dựa trên độ tương đồng:




---

### 3. Mục tiêu Tối ưu hóa Tổng thể (Optimization Objective)

Hàm mục tiêu cuối cùng kết hợp giữa hàm mất mát cục bộ (Local Cross-Entropy Loss) và hàm điều chuẩn gradient chéo nhiệm vụ:



Trong đó hàm mất mát cục bộ  được định nghĩa là:




---

### 4. Các chỉ số đánh giá (Evaluation Metrics)

**Độ chính xác trung bình (Average Accuracy - ):**
Được tính sau khi huấn luyện xong  nhiệm vụ:


*(Trong đó  là độ chính xác của mô hình trên nhiệm vụ  sau khi đã huấn luyện xong nhiệm vụ )*.

**Độ quên trung bình (Average Forgetting - ):**
Đo lường sự sụt giảm hiệu suất trên các nhiệm vụ cũ:



*(Lưu ý:  là độ chính xác cao nhất đạt được trên nhiệm vụ  trong quá khứ, so sánh với độ chính xác hiện tại )*.