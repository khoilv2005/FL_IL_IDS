# P2-Shallow: protocol khóa trước pilot

Policy `conv1_only_v1`, ξ=0,8, binary-cosine native. Không sửa P0 partition,
P1 class-wise aggregation hoặc final TEST. Default router cũ giữ nguyên.

## Fixture và data roles

- Fresh legacy model, seed 42; cluster tự nhiên Task 0 gồm client 9 và 30.
- Giữ membership này qua Task 1 như fixture kiểm tra transition; không tuyên bố
  đã triển khai scheduler/reclustering full campaign.
- Mỗi task: một native round, Adam FP32, LR 0,001, batch 16, một epoch,
  tối đa 32 current BASE/class/client theo thứ tự role đã khóa.
- Client 9 không có BASE Task 1: zero-step receipt, không đóng góp BN/sketch.
- Memory router: tối đa 8 binary sketches/class/client từ current BASE.
- Threshold chỉ fit lần đầu trên current BASE; giữ nguyên qua Task 1.
- Fresh run khai báo BN1 counter làm dependency ban đầu, không thêm GRU witness.
  Với state đã có protection, luôn union dependency và giữ nguyên protection.
- Consolidate canonical neuron ages sau Task 0, bảo vệ mature parameters,
  rồi handoff; không dùng lại raw BASE/CAL/VALIDATION Task 0 ở Task 1.
- Source NPZ chứa nhiều task: tách file current BASE/VALIDATION theo task
  ở bước chuẩn bị dataset **trước khởi tạo model**, với row selection cố định
  ở trên. Online Task 1 chỉ đọc file Task 1; không gọi lại multi-task shard reader.
  File chuẩn bị dataset không phải replay bank của thuật toán.

## Gate kỹ thuật được đặt trước khi đọc kết quả

1. Capacity: actual protected backbone <90%, còn learner parameters;
   Conv2, Conv3, GRU có learner capacity khi allocate Task 1.
2. Optimizer thật thay đổi parameters được phép; protection, threshold và
   historical binary memory giữ byte exact.
3. Routing stability: trên cùng **current Task 1 input**, Conv1 features của
   checkpoint trước/sau round giống byte exact. Đây không phải re-encode raw T0.
4. Routing discrimination: balanced task accuracy ≥60%, mỗi task recall ≥25%,
   cả hai task có predicted rows. Đây là gate feasibility, không phải paper metric.
5. Model/router checkpoint restore cho prediction, task, score và bits giống hệt.

## Evidence routing

Trong lúc Task 0 còn current, đọc tối đa 64 VALIDATION/class/client, encode thành
binary sketches rồi giữ **chỉ bits và task labels**. Tại Task 1 chỉ đọc current
VALIDATION 6–11. Chấm task routing bằng old frozen validation bits + current
validation bits. Báo rõ: structural dependency stability bảo đảm old encoding
không đổi với cùng input/preprocessing; không có phép đo re-encoding T0 raw ở T1.
Không sử dụng TEST, không tune feature count/threshold sau pilot. Nếu discrimination
FAIL thì NO-GO policy shallow; không chạy full training để cứu kết quả.
