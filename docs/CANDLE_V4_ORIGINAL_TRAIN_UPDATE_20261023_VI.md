# CANDLE V4: chuyển full training sang TRAIN gốc

Đã thay đường dữ liệu full V4 bằng `IncrementalDataLoader` gốc. Launcher không
còn tham số `--roles`, không gọi prepare roles, không tạo role manifest hay
chia thêm BASE/FIT/CAL/VALIDATION. Chỉ lọc class của task hiện tại và giữ bước
local validation .1 vốn có của native V1.

Expert nhận đúng view `X_train/y_train` sau local validation mà native learner
nhận. Không cap4096/class như pilot. Sampling theo mini-batch vẫn giữ128positive
+128negative/update và44updates/round; không có nghĩa expert chạy full epoch
trên mọi mẫu. TRAIN của task trước không được đưa lại vào expert.

Mặc định chỉ có một backbone V1 và bank coverage-aware/client. Không chạy một
V1 shadow copy, không chạy hai expert bank đối chứng phụ. Các peer vẫn nhận
bản reference MLP nhỏ để tính gradient theo cùng tọa độ; đó không phải clone
CNN-GRU. Inference chính dùng native prediction cùng các expert đã ACTIVE.

Gate mới khóa threshold0.5 và dùng native local validation hiện có. Giữ recall
95%, observed FAR0.1%, break0 và yêu cầu count; thiếu evidence thì DEFERRED,
không tự ACTIVE. Loại validation content trùng TRAIN hiện tại khỏi gate,
không tạo split mới. Native validation từng được V1 dùng nội bộ nên kết quả
này không được gọi là CAL confirmation độc lập. Historical risk chưa đo vẫn
UNKNOWN. Không tune threshold bằng TRAIN hay TEST.

Giữ full TEST34class sau Task5/Round19/consolidation. Không dùng TEST trong
training/gate. Không còn cumulative historical VALIDATION; forgetting không
được đo với protocol này. Báo cáo rõ thay vì đưa ra số giả.

Notebook: `train_candle_v4_kaggle.ipynb`, source folder mới
`/kaggle/working/candle_v4_original_source`, output mặc định
`/kaggle/working/results_candle_v4_original_seed42`.

Dừng notebook cũ đang tạo roles và chạy fresh bằng notebook mới. Không resume
checkpoint role-based. Không cần xóa/ghép các role đã tạo: run mới không dùng.

V1 counterfactual trong final report dùng chính backbone được train theo TRAIN
gốc, không tạo predictor thứ hai. Muốn chạy riêng baseline cùng protocol, đặt
`VARIANT="native-only"` và OUTPUT_DIR khác. Không so trực tiếp accuracy với V1
lịch sử dùng80/2/8/10.

Regression dùng NPZ tổng hợp: lọc task gốc, full training view>4096/class,
gradient thật, một bank, native/RNG isolation, resume, gate validation có kiểm
soát, loại TRAIN overlap, thiếu evidence, task transition và từ chối resume cũ.
Không chạy full training hoặc mở TRAIN/CAL/TEST thật local trong lần cập nhật.
