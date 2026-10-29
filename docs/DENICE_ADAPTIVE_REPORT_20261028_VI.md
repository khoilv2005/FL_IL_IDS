# DENICE thích nghi theo nhu cầu: học ít hơn ở nơi đã ổn định

Ngày 28/10/2026. Đây là đề xuất phương pháp kèm một phần triển khai đã kiểm thử;
chưa phải kết quả thực nghiệm toàn bộ thuật toán mới.

## 1. DENICE đang làm gì và vì sao chậm?

Mỗi **client** giữ dữ liệu riêng và huấn luyện mô hình tại chỗ. Các client được
gom thành **cluster** để cộng tác. Một client được bầu làm **temporary server**:
nhận cập nhật, tổng hợp rồi phân phối mô hình; vai trò này có thể thay đổi.
**Incremental learning** nghĩa là học thêm loại tấn công mới mà vẫn nhận ra loại cũ.
DENICE dùng tuổi neuron và mask — “khóa cho phép học” — để bảo vệ kiến thức cũ.

Kiểm tra code cho thấy ba nguồn chi phí cần xử lý:

- Nhánh cũ chạy vòng huấn luyện qua mọi client đang hoạt động; một số cấu hình
  xây lại thông tin router cả trước và sau tổng hợp.
- Nhánh V2 mặc định 20 round mỗi task; mọi client tham gia quét dữ liệu hiện tại
  mỗi round. Nhiều client hơn làm tổng công việc tăng. V2 đã hỗ trợ hai GPU,
  nên không thể kết luận mọi phiên bản đều chạy tuần tự.
- Dù chỉ một phần trọng số được học, đóng gói cập nhật hiện tại vẫn chứa tensor
  delta và mask đầy đủ. Khóa gradient không tự biến phép nhân ma trận thành phép
  tính sparse, nên giảm neuron được học chưa chắc giảm thời gian GPU.

Đây là nhận xét từ code, chưa phải phân rã thời gian của lần chạy thực tế của bạn.

## 2. Phương pháp đề xuất: DENICE-ACE

Tên làm việc: **Adaptive Coverage and Expansion** — chọn nơi cần học, giữ độ phủ
kiến thức, và chỉ mở rộng khi thiếu khả năng học. Không khẳng định đây là thuật
toán đầu tiên trong nghiên cứu dùng các thành phần này.

| Thành phần | Cơ chế đề xuất | Lợi ích kỳ vọng |
|---|---|---|
| Client | Ưu tiên client mới, lớp hiếm, lỗi còn cao; client ổn định nghỉ luân phiên | Giảm số lượt train nhưng tránh bỏ quên nhóm khó |
| Tuổi neuron | Tuổi tăng sau lần học/củng cố thành công, không tăng vì client nghỉ | Client tham gia muộn không bị “già hóa” giả |
| Mở rộng | Thử tái sử dụng trước; chỉ cấp block neuron/adapter nhỏ khi validation trì trệ và thiếu capacity | Tránh mở rộng theo lịch một cách máy móc |
| Cluster | Giữ cluster ổn định; chỉ chuyển khi lợi ích vượt ngưỡng trong nhiều round | Giảm clustering, chuyển trạng thái và dao động |
| Communication | Gửi delta theo mask; server cụm trao đổi có chọn lọc, có phiên bản | Giảm truyền phần không được cập nhật |
| Router | Nhánh đặc trưng ổn định, bộ nhớ cân bằng và dự đoán top-2 khi không chắc | Hạn chế chọn sai task rồi loại mất lớp đúng |

**Một round:** temporary server chọn nhóm client đủ độ phủ → client học trên dữ
liệu riêng → gửi cập nhật đã nén → server tổng hợp theo mask và số mẫu → các
client tự kiểm tra trên validation riêng → quyết định tiếp tục, nghỉ hoặc xin
thêm capacity. Server chỉ nhận số liệu tổng hợp, không nhận mẫu validation.

Client mới phải có lượt khởi động. Client lâu không được chọn phải có lượt quay
lại; client giữ lớp hiếm không bị loại chỉ vì máy chậm. Nếu ngân sách không đủ
để phủ các lớp, hệ thống phải ghi nhận thiếu phủ và tăng ngân sách hoặc hoãn
chốt task, không báo lớp đó là đã học.

## 3. Accuracy và route F1 sẽ được cải thiện bằng cách nào?

**Accuracy** là tỷ lệ phân loại đúng. **Route macro-F1** đánh giá việc chọn đúng
task/context, coi mỗi context quan trọng như nhau. Nó khác macro-F1 phân loại
các loại tấn công.

Router chọn sai có thể khiến classifier không còn quyền chọn lớp đúng. Đề xuất
giữ một nhánh đặc trưng ổn định cho router, dùng bộ nhớ tóm tắt cân bằng giữa
context cũ và mới. Khi hai context có điểm gần nhau, trộn xác suất của hai head
đã hiệu chỉnh thay vì khóa cứng vào một head. Giới hạn top-2 để kiểm soát chi phí.
Top-2 có thể cứu phân loại nhưng **không tự chứng minh route F1 top-1 tăng**;
phải đo riêng cả hai.

Để giữ kiến thức cũ, giữ nguyên bảo vệ mature neurons; nếu cho phép replay,
dùng bộ nhớ mẫu nhỏ cân bằng lớp **chỉ tại client sở hữu** và ràng buộc đầu ra
cũ. Nếu buộc replay-free, dùng sketches và cấu trúc bảo vệ, đồng thời chấp nhận
rằng không có mẫu cũ sẽ khó kiểm tra và sửa quên hơn. Mở rộng mạng chỉ giúp khi
thực sự thiếu capacity; nó không sửa được lỗi routing hoặc thiếu lớp trong cluster.

## 4. Giữ đúng Federated Learning và temporary server

Client giữ raw data, nhãn từng mẫu và bộ nhớ replay tại chỗ. Temporary server
nhận cập nhật mô hình, số đếm và thống kê được cho phép; không có kho dữ liệu
huấn luyện tập trung. Mask, tuổi neuron, ánh xạ lớp và phiên bản đặc trưng phải
được kiểm tra trước khi gộp. Không lấy trung bình hai mô hình có lịch sử neuron
khác nhau chỉ vì chúng cùng kích thước.

Đây là hướng semi-decentralized. Temporary server vẫn được bầu từ client.
Federated Learning không đồng nghĩa bảo đảm riêng tư tuyệt đối: cập nhật và
thống kê vẫn có thể tiết lộ thông tin; bản này không triển khai DP/secure aggregation.

## 5. Phần đã triển khai và giới hạn

Đã thêm **đóng gói delta theo mask không mất thông tin** vào HierarchicalRunner,
bật bằng `--compact-uploads`. Bộ gửi so sánh gói dense với gói compact và chọn
gói nhỏ hơn; bên nhận giải mã rồi kiểm tra quyền cập nhật và phiên bản round.
Giữ cả tọa độ có delta bằng 0 nhưng mask bằng 1 để không làm sai mẫu số tổng hợp.

Đây là bước giảm communication, **chưa triển khai toàn bộ DENICE-ACE**. Chọn
client thích nghi, mở rộng theo validation, cluster theo sự kiện và router top-2
ở trên là thiết kế nghiên cứu. Nhánh hierarchical hiện tại dùng routerless;
không được lấy kết quả nhánh này làm bằng chứng route F1 tăng.

Chi phí upload được đếm từ thông điệp thực sự mã hóa/giải mã trong mô phỏng;
chưa đo truyền mạng. Nén không giảm số batch train, có thêm chi phí CPU và chưa
nén broadcast/BN/trao đổi giữa server. Vì vậy chưa có cơ sở công bố train nhanh
hơn bao nhiêu lần hoặc accuracy tăng bao nhiêu trên dữ liệu thật.

Trên fixture tổng hợp nhỏ, CPU, 2 task × 2 round: upload mô hình giảm từ
61.554.618 xuống 10.513.014 byte (**82,9%**); tổng byte ứng dụng được mô phỏng
giảm từ 373.790.406 xuống 322.748.802 byte (**13,7%**). Trạng thái mô hình cuối
trùng baseline. Không ngoại suy các tỷ lệ này sang bộ dữ liệu thật.

## 6. Cách kiểm chứng công bằng

So sánh cùng lịch client đến, task, dữ liệu và thiết bị: baseline → thêm nén →
thêm chọn client → thêm mở rộng → thêm router. Mỗi bước đo thời gian đến cùng
mức chất lượng, tổng byte cả hai chiều, accuracy, macro-F1 phân loại, route
macro-F1, độ quên và kết quả lớp hiếm. Dùng ít nhất 3 seed trong runner nghiên
cứu; runner main hiện khóa seed 42 nên phải tạo protocol nghiên cứu riêng.

Chọn ngưỡng bằng validation đã tách khỏi TRAIN từ đầu; TEST chỉ dùng báo cáo.
Đánh giá trên tất cả lớp đã xuất hiện, kể cả lớp cluster chưa học. Chỉ chấp nhận
cơ chế tăng tốc nếu tiết kiệm thực tế và mức giảm chất lượng nằm trong dung sai
đã chốt trước. Không đổi số round rồi so hai kết quả ở ngân sách không tương đương.

## Tham khảo và tài liệu kỹ thuật

Chọn client theo ích lợi và tốc độ có cơ sở từ
[Oort, OSDI 2021](https://www.usenix.org/conference/osdi21/presentation/lai).
Replay kết hợp giữ đầu ra cũ tham khảo
[Dark Experience Replay, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf).
Các kết quả của những bài này không phải kết quả trên DENICE.

Xem [đặc tả, cách chạy và kiểm thử](DENICE_ACE_TECHNICAL_20261028.md).
