Dưới đây là bản viết lại có cấu trúc hơn để bạn dùng báo cáo.

---

# Báo cáo cơ chế NICE

## 1. Bài toán NICE giải quyết

NICE, viết tắt của **Neurogenesis Inspired Contextual Encoding**, là một phương pháp dùng cho **continual learning**, cụ thể là **class-incremental learning**.

Trong class-incremental learning, mô hình không học toàn bộ dữ liệu cùng lúc. Thay vào đó, dữ liệu được chia thành nhiều **episode**. Mỗi episode chứa một nhóm class mới, và mỗi episode có thể được train trong nhiều epoch.

Ví dụ:

```text
Episode 1: học class 0, 1
Episode 2: học class 2, 3
Episode 3: học class 4, 5
...
```

Vấn đề chính là khi mô hình học episode mới, nó dễ quên kiến thức từ episode cũ. Hiện tượng này gọi là **catastrophic forgetting**.

Các phương pháp replay thường giải quyết bằng cách lưu lại một số ảnh cũ rồi train lại cùng dữ liệu mới. Tuy nhiên, NICE không lưu ảnh cũ. NICE giải quyết bằng cách thay đổi cách tổ chức neuron trong mạng.

Ý tưởng chính của NICE là:

> Chia neuron thành nhiều nhóm tuổi khác nhau. Neuron trẻ học kiến thức mới, neuron đã trưởng thành thì bị freeze để giữ kiến thức cũ. Khi test, mô hình dùng context-detector để đoán input thuộc episode nào, rồi chỉ dùng subnet tương ứng để phân loại. 

---

## 2. Tuổi của neuron: cơ chế cốt lõi

Trong NICE, mỗi neuron được gán một giá trị gọi là **age**.

Neuron được chia thành ba nhóm chính:

```text
Age = 0   → neuron dự trữ, chưa được gán cho episode nào
Age = 1   → neuron đang học episode hiện tại
Age > 1   → neuron đã học xong episode cũ, bị freeze để lưu memory
```

Ví dụ, nếu mô hình đang ở episode hiện tại `E = 5`, thì:

```text
Neuron age 1: đang học episode 5
Neuron age 2: từng học episode 4
Neuron age 3: từng học episode 3
Neuron age 4: từng học episode 2
Neuron age 5: từng học episode 1
```

Công thức ánh xạ giữa episode và tuổi neuron là:

[
\text{age của neuron phụ trách episode } e = E - e + 1
]

Trong đó:

```text
E = episode hiện tại
e = episode mà neuron từng học
```

Ví dụ:

```text
E = 5, e = 1
age = 5 - 1 + 1 = 5
```

Nghĩa là neuron học episode 1 sẽ có age bằng 5 khi mô hình đang ở episode 5.

---

## 3. Neurogenesis: đưa neuron age-0 vào học

Ở đầu mỗi episode mới, NICE tạm thời chuyển toàn bộ neuron `age = 0` thành `age = 1`.

Ký hiệu:

[
N_{=0} \rightarrow N_{=1}
]

Các neuron này được xem như neuron mới sinh. Chúng bắt đầu tham gia học dữ liệu của episode hiện tại.

Ví dụ, trước episode mới:

```text
Neuron age-0: đang dự trữ
```

Khi episode mới bắt đầu:

```text
Neuron age-0 → age-1
```

Sau đó, các neuron age-1 này được train trên dữ liệu của episode hiện tại.

Tuy nhiên, NICE không giữ lại tất cả neuron mới. Sau một số epoch, nó sẽ chọn neuron nào thật sự quan trọng để giữ lại, còn neuron không quan trọng thì trả về age-0.

---

## 4. Chọn neuron age-1 nào được giữ lại

Sau mỗi `p` epoch, NICE đánh giá các neuron age-1 trong từng layer để xem neuron nào đang đóng góp nhiều cho việc học episode hiện tại.

Với mỗi layer `l`, NICE lấy một subset dữ liệu của episode hiện tại, ví dụ 1024 mẫu, cho đi qua mạng. Sau đó, nó tính tổng activation của các neuron age-1 trong layer đó.

Activation có thể hiểu là mức độ một neuron phản ứng với input. Neuron activation cao nghĩa là neuron đó đang hoạt động mạnh trên dữ liệu hiện tại.

NICE tính tổng activation của toàn bộ neuron age-1 trong layer:

[
A^l_{=1}
]

Sau đó, NICE tìm một tập nhỏ nhất các neuron age-1, ký hiệu là:

[
S^l_1
]

sao cho tổng activation của tập này chiếm ít nhất tỷ lệ `τ` tổng activation của toàn bộ neuron age-1 trong layer.

Trong paper, tác giả dùng:

[
\tau = 0.95
]

Nghĩa là NICE muốn giữ lại tập neuron nhỏ nhất nhưng vẫn giữ được ít nhất 95% tổng activation của layer.

Ví dụ một layer có 6 neuron age-1:

| Neuron | Tổng activation |
| ------ | --------------: |
| n1     |              40 |
| n2     |              30 |
| n3     |              15 |
| n4     |              10 |
| n5     |               3 |
| n6     |               2 |

Tổng activation là:

```text
40 + 30 + 15 + 10 + 3 + 2 = 100
```

95% của 100 là 95.

NICE sắp xếp neuron theo activation giảm dần rồi cộng dần:

```text
n1 = 40
n1 + n2 = 70
n1 + n2 + n3 = 85
n1 + n2 + n3 + n4 = 95
```

Vậy NICE giữ lại:

```text
n1, n2, n3, n4
```

Còn:

```text
n5, n6
```

sẽ quay về `age = 0`.

Ý nghĩa của bước này là:

> Neuron nào phản ứng mạnh với dữ liệu episode hiện tại thì được giữ lại để học và lưu kiến thức. Neuron nào gần như không đóng góp thì được trả về kho dự trữ để dùng cho episode sau.

---

## 5. Maturation: neuron trưởng thành sau mỗi episode

Khi một episode kết thúc, NICE tăng tuổi của tất cả neuron đã được dùng:

[
age \leftarrow age + 1
]

Các neuron age-1 được giữ lại sau quá trình chọn lọc sẽ trở thành age-2.

Ví dụ:

```text
Cuối episode 3:
neuron age-1 → age-2
```

Từ lúc này, các neuron đó được xem là neuron đã trưởng thành. Chúng lưu kiến thức của episode vừa học và sẽ bị freeze để tránh bị thay đổi khi học episode mới.

Tóm tắt quá trình:

```text
Đầu episode:
age-0 → age-1

Trong episode:
chọn neuron age-1 quan trọng

Cuối episode:
age-1 được giữ lại → age-2
age-2 trở lên → memory neuron, bị freeze
```

---

## 6. Tránh quên: freeze và pruning

NICE cần đảm bảo rằng neuron đã học episode cũ không bị thay đổi khi học episode mới.

Có hai cách mà kiến thức cũ có thể bị ảnh hưởng:

```text
1. Trọng số đi vào neuron cũ bị update trực tiếp
2. Input đi vào neuron cũ thay đổi do neuron trẻ phía trước bị update
```

NICE xử lý bằng hai cơ chế.

### 6.1. Freeze neuron đã trưởng thành

Neuron có `age > 1` sẽ bị freeze. Nghĩa là trọng số đi vào các neuron này không còn được cập nhật nữa.

```text
age = 1   → được train
age > 1   → bị freeze
```

Như vậy, kiến thức cũ được bảo vệ.

### 6.2. Prune connection từ neuron trẻ sang neuron già

Nếu neuron già nhận input từ neuron trẻ, thì dù neuron già đã freeze, output của nó vẫn có thể thay đổi vì input từ neuron trẻ thay đổi.

Vì vậy, NICE prune các connection có hướng:

```text
neuron trẻ → neuron già
```

Nói cách khác, NICE không cho neuron mới đang học làm ảnh hưởng neuron cũ đã lưu memory.

Các đường đi được cho phép:

```text
neuron già → neuron trẻ     // cho phép transfer knowledge từ cũ sang mới
neuron cùng tuổi → cùng tuổi // lưu kiến thức của cùng một episode
```

Đường đi bị cấm:

```text
neuron trẻ → neuron già
```

Nhờ vậy, kiến thức cũ được giữ ổn định trong quá trình học episode mới.

---

## 7. Training loss: chỉ xét output của class trong batch

Khi train episode hiện tại, batch chỉ chứa class của episode hiện tại.

Ví dụ:

```text
Episode 1: cat, dog
Episode 2: car, truck
```

Khi đang train episode 2, batch chỉ có ảnh `car` và `truck`.

Nếu dùng softmax trên toàn bộ output:

```text
cat, dog, car, truck
```

thì output của `cat`, `dog` vẫn tham gia vào loss. Nhưng trong NICE, output của `cat`, `dog` đã thuộc episode cũ và bị freeze. Nếu chúng có giá trị cao, optimizer không thể giảm chúng xuống vì chúng đã bị khóa.

Điều này có thể gây training instability.

Vì vậy, khi train episode 2, NICE chỉ xét các output class trong batch:

```text
car, truck
```

Nó không xét:

```text
cat, dog
```

Nghĩa là loss chỉ hỏi:

```text
Trong car và truck, đáp án đúng là gì?
```

chứ không bắt class mới phải cạnh tranh với toàn bộ class cũ đã bị freeze.

---

## 8. Binary activation memory

NICE không lưu ảnh cũ như replay methods. Thay vào đó, nó lưu **binary activation memory**.

Sau mỗi `p` epoch, NICE lấy `m` mẫu ngẫu nhiên từ episode hiện tại, cho đi qua mạng, rồi ghi lại neuron nào kích hoạt mạnh.

Với mỗi layer `l`, NICE đặt một threshold:

[
t_l = mean_l + std_l
]

Trong paper, threshold này được lấy từ thống kê activation sau episode đầu tiên để đơn giản hóa.

Sau đó, activation được nhị phân hóa:

```text
Activation > threshold  → 1
Activation ≤ threshold  → 0
```

Ví dụ một ảnh đi qua mạng tạo ra activation của 7 neuron:

```text
[4.2, 0.3, 5.1, 0.0, 0.5, 6.0, 4.7]
```

Giả sử threshold là 3.0, ta có binary vector:

```text
[1, 0, 1, 0, 0, 1, 1]
```

Vector này cho biết neuron nào đang kích hoạt mạnh.

Ví dụ memory có thể như sau:

```text
Episode 1 memory:
[1, 0, 1, 1, 0, 0, 1]
[1, 0, 1, 0, 0, 1, 1]

Episode 2 memory:
[0, 1, 0, 0, 1, 1, 0]
[0, 1, 0, 0, 1, 0, 1]

Episode 3 memory:
[0, 0, 1, 0, 1, 1, 1]
[0, 0, 1, 1, 1, 0, 1]
```

Mỗi vector là một “dấu vân tay activation” của dữ liệu trong một episode.

---

## 9. Context-detector: làm sao NICE biết ảnh test thuộc episode nào?

Khi test, NICE không được cho biết ảnh thuộc episode nào.

Ví dụ, sau khi học xong ba episode:

```text
Episode 1: cat, dog
Episode 2: car, truck
Episode 3: bird, horse
```

Khi đưa một ảnh test vào, NICE phải tự đoán ảnh này thuộc episode nào.

NICE làm điều đó bằng **context-detector**.

Quy trình như sau:

```text
Input test image
→ chạy qua network
→ lấy activation
→ threshold thành binary vector
→ đưa vào context-detector
→ context-detector dự đoán episode
```

Ví dụ ảnh test tạo ra binary activation vector:

```text
[1, 0, 1, 1, 0, 0, 1]
```

Context-detector so pattern này với memory đã lưu:

```text
Episode 1 memory:
[1, 0, 1, 1, 0, 0, 1]
[1, 0, 1, 0, 0, 1, 1]

Episode 2 memory:
[0, 1, 0, 0, 1, 1, 0]
[0, 1, 0, 0, 1, 0, 1]

Episode 3 memory:
[0, 0, 1, 0, 1, 1, 1]
[0, 0, 1, 1, 1, 0, 1]
```

Vector ảnh test giống episode 1 nhất. Vì vậy context-detector có thể trả ra:

```text
P(episode 1) = 0.91
P(episode 2) = 0.06
P(episode 3) = 0.03
```

Do đó NICE chọn episode 1.

Điểm quan trọng là:

> NICE không biết chắc ảnh test thuộc episode nào. Nó dự đoán dựa trên activation pattern của ảnh đó.

---

## 10. Logistic regression theo chuỗi

Để tính xác suất input thuộc từng episode, NICE dùng một chuỗi logistic regression.

Giả sử hiện tại đang ở episode `E`. NICE cần tính:

[
p_1, p_2, ..., p_E
]

Trong đó:

```text
p1 = xác suất input thuộc episode 1
p2 = xác suất input thuộc episode 2
...
pE = xác suất input thuộc episode E
```

Tuy nhiên, mỗi episode tương ứng với một nhóm neuron có age khác nhau. Vì vậy NICE không dùng một classifier duy nhất, mà dùng nhiều logistic regression theo chuỗi.

### Bước 1: xác suất thuộc episode 1

Dùng neuron già nhất, tức neuron liên quan đến episode 1:

[
P(E_1 \mid N_{=E})
]

### Bước 2: xác suất thuộc episode 2, với điều kiện không thuộc episode 1

[
P(E_2 \mid N_{\geq E-1}, \bar{E}_1)
]

### Bước 3: xác suất thuộc episode 3, với điều kiện không thuộc episode 1 và episode 2

[
P(E_3 \mid N_{\geq E-2}, \bar{E}_1, \bar{E}_2)
]

Tiếp tục như vậy cho các episode sau.

Công thức tổng quát cho episode `e` là:

[
p_e =
P(E_e \mid N_{\geq E-e+1}, \bar{E}*1, ..., \bar{E}*{e-1})
\prod_{i=1}^{e-1}
\left(
1 -
P(E_i \mid N_{\geq E-i+1}, \bar{E}*1, ..., \bar{E}*{i-1})
\right)
]

Ý nghĩa đơn giản là:

```text
Xác suất thuộc episode e
=
xác suất không thuộc các episode trước
×
xác suất thuộc episode e
```

Với episode cuối cùng:

[
p_E = 1 - \sum_{i=1}^{E-1} p_i
]

Sau đó NICE chọn episode có xác suất cao nhất.

Ví dụ:

```text
P(episode 1) = 0.91
P(episode 2) = 0.06
P(episode 3) = 0.03
```

NICE chọn episode 1.

---

## 11. Inference: quá trình dự đoán khi test

Khi test một input `x`, NICE thực hiện các bước sau:

```text
Input x
→ lấy activation
→ threshold thành binary vector
→ context-detector đoán episode
→ mask output không thuộc episode đó
→ classify trong subnet tương ứng
```

Ví dụ context-detector đoán input thuộc episode 1:

```text
Episode 1: cat, dog
Episode 2: car, truck
Episode 3: bird, horse
```

NICE sẽ chỉ mở output:

```text
cat, dog
```

và mask output:

```text
car, truck, bird, horse
```

Sau đó mô hình chỉ phân loại trong nhóm class của episode 1.

Nếu ảnh là cat, mô hình sẽ chọn giữa:

```text
cat hoặc dog
```

chứ không còn so với car, truck, bird, horse.

---

## 12. Toàn bộ thuật toán NICE

Có thể tóm tắt quá trình training như sau:

```text
For each episode e:
    1. Chuyển neuron age-0 thành age-1

    2. Train trên dữ liệu episode hiện tại

    3. Mỗi p epoch:
        a. Tính activation của neuron age-1
        b. Giữ neuron quan trọng nhất đủ 95% activation
        c. Trả neuron dư về age-0
        d. Prune connection trẻ → già
        e. Cập nhật binary activation memory
        f. Fit/update context-detector

    4. Cuối episode:
        a. Tăng tuổi neuron đã dùng
        b. Freeze neuron age > 1
```

Khi test:

```text
Input x
→ chạy qua network
→ lấy activation
→ threshold thành binary vector
→ context-detector đoán episode
→ mask output không thuộc episode đó
→ classify trong subnet tương ứng
```

---

## 13. Kết luận

NICE là một phương pháp continual learning không dùng replay ảnh cũ.

Cơ chế chính của NICE gồm:

```text
1. Chia neuron theo tuổi
2. Dùng neuron age-1 để học episode hiện tại
3. Chọn neuron age-1 quan trọng dựa trên activation
4. Trả neuron không quan trọng về age-0
5. Tăng tuổi neuron sau mỗi episode
6. Freeze neuron đã trưởng thành để giữ kiến thức cũ
7. Prune connection trẻ → già để tránh interference
8. Lưu binary activation memory thay vì lưu ảnh
9. Dùng context-detector để đoán input thuộc episode nào
10. Khi test, chỉ dùng subnet tương ứng với episode được dự đoán
```

Tóm lại:

> NICE đưa tính tuần tự của continual learning vào kiến trúc mạng. Neuron trẻ học kiến thức mới, neuron già lưu kiến thức cũ, và context-detector giúp mô hình chọn đúng nhóm neuron khi dự đoán.
