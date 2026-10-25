# Train cluster-server trước, eval riêng sau

`train_incremental_kaggle.py` mặc định dùng profile `cluster_server` và train mới từ Task 0 đến Task 5 (base + 5 incremental tasks), 20 round/task. Không tải/đọc TEST, không eval cuối round/task. Không dùng tiếp checkpoint `.pt` per-client của row_overlay cũ.

## Chạy trên Kaggle

Upload các file trong `output/cluster_server_kaggle/`. Đặt launcher và `denice_cluster_server_source.zip` cùng thư mục; hoặc đặt `DENICE_SOURCE_ZIP` là đường dẫn ZIP đã mount. Không cần tải source cũ trên GitHub.

```python
import os
os.environ['DENICE_DATA_DIR'] = '/kaggle/input/datasets/khoilv2005/100-clients/100-clients'
os.environ['DENICE_SOURCE_ZIP'] = '/kaggle/input/YOUR_CODE_DATASET/denice_cluster_server_source.zip'
os.environ['DENICE_OUTPUT_DIR'] = '/kaggle/working/results_denice_cluster_server_seed_42'
# Chạy file train_incremental_kaggle.py đã cung cấp.
```

Nếu đã giải nén source: đặt `DENICE_CODE_DIR` tới thư mục chứa `fed_learning/` và `tools/`. Giữ các biến profile/config cũ không hoạt động: bỏ `DENICE_RUN_PROFILE`, `DENICE_VARIANT`, `DENICE_CONFIG_FILE`, `DENICE_CONFIG_OVERRIDES`, `DENICE_RESUME_STATE_PATH`, `DENICE_TRAIN_PHASE` của lần chạy row_overlay trước. Có thể đặt rõ `DENICE_RUN_PROFILE=cluster_server`.

Mặc định dùng 2 GPU nếu có, 1 GPU nếu chỉ có 1, CPU nếu không có CUDA; batch size 2048. Có thể đặt `DENICE_DEVICE`, `DENICE_WORKER_DEVICES` (rỗng để chạy tuần tự), `DENICE_BATCH_SIZE`, `DENICE_OUTPUT_BUDGET_GIB`.

## Model và communication

- Một state model/router/masks/BN/ages cho mỗi cluster. Mapping client -> cluster chỉ là tham chiếu, không phải model cá nhân.
- Client nhận reference chung, train bản tạm trên dữ liệu riêng, gửi packet cập nhật về coordinator. Bản tạm không được giữ giữa round hay lưu checkpoint.
- Tổng hợp class-wise và mask-aware ở cluster theo runner V2 đã có; bảo vệ mature parameters và router coordinates. Không có global model.
- Bỏ broadcast state cuối round. Round kế tiếp tải reference khi cần. Không chạy overlay full-mesh giữa các cluster.
- Communication ghi trong `communication.json`: model/reference downloads, uploads, BN moments, router sketches, handoff. Đây là ước lượng application payload, không đo mạng thật; chưa thể kết luận mức giảm so với run cũ trước khi chạy so sánh.
- Có các download phụ phục vụ BN/router; không được coi bỏ model cá nhân là loại bỏ toàn bộ download.

Đây là chuyển sang runner DENICE V2 có sẵn, không chỉ đổi cách lưu model V1: clustering tự động ở đầu task (`xi=0.8`), giữ membership trong task; handoff có kiểm tra tương thích; temporary server theo số class đã học local và tie-break ID, giữ leader trong task; router binary-cosine dùng Conv2 coordinate được bảo vệ. Dùng toàn bộ TRAIN hiện tại, không replay dữ liệu cũ. Vì vậy K/accuracy không được giả định bằng kết quả `adaptive_ap`/row_overlay cũ.

Các state dormant được giữ để client quay lại, không được tính thành temporary server đang hoạt động khi eval cuối task.

## Đầu ra train

- `checkpoint_task_T_all_rounds.zip`: state cluster, có checkpoint round và task-boundary; không có model riêng từng client.
- `temporary_servers_task_T.json`: cluster UID, leader ID, members, state hash.
- `partition_task_T.json`: partition/leader trong task.
- `completion.json`: `completed=true`, `tasks_completed=6`, `evaluation_pending=true` khi hoàn tất.
- `communication.json`, `protocol_lock.json`, `execution_provenance.json`.

Resume bằng `DENICE_CLUSTER_RESUME_ROOT` (output đã mount), `DENICE_CLUSTER_RESUME_TASK` (task có archive), tùy chọn `DENICE_CLUSTER_RESUME_ROUND`; bỏ ROUND để lấy task-boundary mới nhất. Source, dữ liệu và cấu hình phải giống lần train trước. Giữ output cũ để có đủ lịch sử archive.

## Resume từ ZIP Google Drive bằng launcher

Trong `train_incremental_kaggle.py`, điền `CLUSTER_RESUME_DRIVE_URL = "https://drive.google.com/file/d/.../view"` ở đầu file; hoặc đặt `DENICE_RESUME_DRIVE_URL`. Đây phải là link **file ZIP output cluster_server**, không phải link thư mục hay checkpoint `.pt` cũ. Để trống link khi train mới.

ZIP giữ nguyên các checkpoint ZIP bên trong, `execution_provenance.json` và `protocol_lock.json`; có thể chứa một thư mục bao ngoài. Không cần upload source ZIP riêng nếu Internet bật: launcher lấy commit source từ provenance và clone GitHub. Source/cấu hình/PyTorch vẫn được runner kiểm tra như trước. `current_data_cache` được bỏ qua khi giải nén vì có thể dựng lại từ dataset TRAIN.

Trên notebook Kaggle mới: thêm dataset 100-clients, bật Internet, chọn T4 x2 cho run V59, dùng PyTorch 2.10.0+cu128, rồi chạy toàn bộ launcher mới. Không đặt đồng thời link Drive và `DENICE_CLUSTER_RESUME_ROOT`. Bỏ các biến resume task/round từ phiên cũ để tự chọn checkpoint mới nhất; có thể đặt chúng khi chủ ý khôi phục một mốc cụ thể. V59 tự chọn T2 R0, runner tiếp tục T2 R1. Log round làm dở không được dùng làm checkpoint.

Giữ output các phiên trước để bảo toàn lịch sử. Nếu cần resume lần nữa, dùng ZIP output mới nhất. Không thay đổi source lõi để bỏ qua lỗi hash/runtime/protocol.

## Eval sau khi train xong

### Một file eval, tải output từ Google Drive

Dùng `eval_selective.py` mới: dán toàn bộ file vào một cell Kaggle, điền `EVAL_DRIVE_URL` ở đầu file bằng link FILE ZIP output train cuối cùng. Thêm dataset `100-clients` vào Input, bật Internet và GPU. Sửa `DATA_DIR` nếu đường dẫn mount khác. Không cần notebook `.ipynb` hoặc upload source riêng.

ZIP cần giữ nguyên `completion.json`, `execution_provenance.json`, `checkpoint_task_5_all_rounds.zip` trong cùng thư mục; có thể nén toàn bộ output, launcher chỉ giải nén các file cần cho eval. Không giải nén ZIP checkpoint bên trong. File tự kiểm tra train đủ Task 0–5, khớp PyTorch (tự cài bản 2.10.0+cu128 khi cần), clone đúng commit nguồn, đối chiếu hash rồi gọi evaluator trong tiến trình mới. TEST vẫn lấy từ Kaggle Input, không lấy từ Drive output. Đầu ra mặc định `/kaggle/working/eval_cluster_servers`.

CLI tương đương: `python eval_selective.py --drive-url LINK --data-dir DATA_DIR`. Vẫn hỗ trợ `--checkpoint-root` thay cho Drive, nhưng không đặt cả hai. Bỏ các biến `DENICE_CODE_DIR`/`DENICE_SOURCE_ZIP` cũ nếu muốn tự lấy source đúng commit.

Dùng `eval_selective.ipynb`: đặt `CHECKPOINT_ROOT` tới output có `completion.json` và ZIP Task 5, `DATA_DIR` chứa `global_test_data.npz`, `EVAL_SELECTIVE_SCRIPT` tới `eval_selective.py` đã cung cấp. Đặt cùng `DENICE_SOURCE_ZIP` như train. `EVAL_TASKS=5`, `EVAL_ROUNDS=None`, `EVAL_CLIENT_FRACTION=1.0`.

Hoặc chạy script:

```bash
python eval_selective.py --checkpoint-root /kaggle/input/YOUR_TRAIN_OUTPUT --data-dir /kaggle/input/YOUR_DATA --output /kaggle/working/eval_cluster_servers
```

Eval chỉ mở checkpoint cuối sau khi đủ Task 0–5. Mỗi temporary server dùng model/router cluster đã consolidate, chạy trên cùng toàn bộ TEST 34 class; không giới hạn 50.000 mẫu và không lọc class theo cluster. Nhãn TEST chỉ dùng tính metric. Có kiểm tra số mẫu từng server bằng tổng số mẫu TEST.

Đầu ra: `temporary_server_metrics.csv/json` và từng `evaluation_CLUSTER_UID.json`; gồm server ID, cluster UID, members, số mẫu, accuracy, macro-F1, confusion matrix và per-class metrics. Eval có thể tiếp tục từ receipt đã lưu. Train không tự gọi eval.
# Chẩn đoán router trên checkpoint đã train

Trong `eval_selective.py`, đặt `DIAGNOSTIC = True`, giữ link ZIP output train đã hoàn tất,
rồi paste toàn bộ file vào một cell Kaggle. `DIAGNOSTIC_PER_CLASS = 256` chọn tối đa 256
mẫu mỗi lớp, cố định seed 42 và dùng cùng mẫu cho tất cả server. Không cần train lại.
Kết quả ở `eval_cluster_servers/diagnostics/diagnostic_summary.json` và từng
`diagnostic_<cluster_uid>.json`. Chế độ này so sánh router thật và oracle biết task
trên cùng logits, cùng danh sách lớp đã đăng ký; task thiếu vẫn tính là dự đoán sai.
Oracle chỉ phục vụ chẩn đoán, không phải accuracy chính thức. Tập con cân bằng lớp
không so trực tiếp với accuracy full test. Không dùng TEST để chọn tham số sửa model.
Đặt `DIAGNOSTIC = False` để quay lại eval full test. Chế độ chẩn đoán không sửa
checkpoint, thuật toán train hoặc kết quả eval chính thức. File một cell đã nhúng
worker chẩn đoán, vẫn kiểm tra mã nguồn và phiên bản torch gốc.

