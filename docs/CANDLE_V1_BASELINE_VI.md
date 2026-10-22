# CANDLE V1 — cấu hình DeNICE V1 đã audit

## Phạm vi

CANDLE V1 là tên run baseline dùng lại engine `algorithm=denice`,
`denice_cl_method=legacy`. Không thêm thuật toán mới hoặc đổi tên registry/model
trong checkpoint. Không tuyên bố đây là một bản tái tạo paper độc lập.

`configs/candle_v1.json` lấy toàn bộ cấu hình thực tế từ
`results (11).zip / training_artifacts/config.json`. File provenance bên cạnh
lưu SHA-256 của config gốc, revision/source hash lịch sử và role manifest hash.
Những tham số của thuật toán khác trong config tổng hợp vẫn được giữ nhưng
không kích hoạt CGoFed/EWC/LwF/DER khi engine là DeNICE legacy.

Chỉ đổi đường dẫn dữ liệu/output/roles, bỏ ba giá trị provenance của run cũ để
runner tính lại, và đặt rõ hai cờ APPLIANCE về false cùng assertion phương pháp.
Supporting code hiện tại được pin bằng Git commit riêng: cùng cấu hình không
đồng nghĩa source bytes hoặc kết quả số học giống hệt run tháng 10 trước đó.

## Cấu hình

| Thành phần | CANDLE V1 |
|---|---|
| Fresh training | Seed 42, Task 0–5, 20 rounds/task |
| Dữ liệu local | Chỉ BASE của current task; không train lại raw task cũ |
| Split lịch sử | BASE 80%, calibration 2%, FIT 8%, validation 10%, theo unique content; split seed 20261006 |
| Training | Adam lr=0.001, batch=2048, một phase/epoch mỗi round |
| Số học | AMP bật như cấu hình V1 lịch sử |
| Collaboration | Paper threshold neighborhoods, ξ=0.8; K=1 là sentinel |
| Model/router | Một model + binary-cosine router cho mỗi client |
| Replay/APPLIANCE/CME | Tắt |
| Checkpoint | Mỗi round; compress rồi seal một ZIP/task; budget 12 GiB |
| TEST | Chỉ sau Task 5/Round 19 và terminal consolidation; đủ 34 classes |
| Metric chính | BinarySelf pooled Accuracy, Macro-F1 |
| Metric phụ | MulticlassSelf từ memory, freeze trước TEST |

**Đây là cấu hình V1 lịch sử, không phải cấu hình V2 full original TRAIN.**
Các holdout không đi vào optimizer. Việc chuẩn bị roles không đọc final TEST.
Ở paper mode, `denice_validation_fraction=0.1` trong config còn được giữ nguyên
cho đường capsule/CANC của implementation; không đổi sang preset V4/legacy capsule.

## Chạy Kaggle

Mở `train_candle_v1_kaggle.ipynb`, bật Internet và GPU. Không chạy launcher V4
trước đó trong cùng session. Notebook clone source từ GitHub, không nhúng source
bundle, checkpoints hoặc dữ liệu; không yêu cầu upload runtime ZIP.

Mặc định:

```text
DATA_DIR  = /kaggle/input/datasets/khoilv2005/100-clients
OUTPUT_DIR = /kaggle/working/results_candle_v1_xi_0p8_seed_42
ROLES_DIR = /kaggle/working/candle_v1_roles
```

DATA_DIR tự tìm dataset nested có metadata, client TRAIN và global TEST.
Để dùng exact historical role files, sửa ROLES_DIR tới thư mục mount chứa
`role_lock.json`, `role_manifest.json`, `indices/`. Nếu để mặc định trống, notebook
tạo roles lại từ TRAIN theo split protocol lịch sử; quá trình scan/hash có thể lâu.
Không khẳng định manifest hash tái tạo giống file cũ nếu đường dẫn/serialization khác.

Launcher dùng một GPU theo execution path V1; không tự áp dụng batch4086/two-GPU
của V2. Nó ép profile legacy và toàn bộ config baseline, tránh mặc định row-overlay
của launcher V4. Output có sẵn bị từ chối, không ghi đè.

## Kết quả

```text
results_candle_v1_xi_0p8_seed_42/
  candle_v1_protocol.json
  candle_v1_requested_config.json
  environment.txt
  config.json
  checkpoint_task_0_all_rounds.zip ... checkpoint_task_5_all_rounds.zip
  candle_v1_self_task_5/
    pipeline_lock.json
    completion.json
    confusion_matrices.npz
    client_metrics.json
    test_predictions.csv.gz
    frozen_routers.joblib
  candle_v1_self_task_5.zip
  candle_v1_completion.json
```

Evaluator cũ được dùng nguyên trạng, mỗi policy chạy toàn bộ original TEST bằng
disjoint receiver shards; mỗi row tính đúng một lần trong từng policy. Không chuyển
sang mỗi client FULL TEST như các protocol mới. Raw evaluator vẫn ghi method
DeNICE legacy và main_policy MulticlassSelf để bảo toàn format/provenance; báo cáo
ngoài `candle_v1_completion.json` ghi BinarySelf là baseline chính của CANDLE V1.

Không dùng số V1 cũ làm kết quả của run mới. Không có cam kết accuracy hoặc
Macro-F1 sẽ trùng run cũ; TEST nguồn này đã được phân tích trong development.

## Trạng thái chuẩn bị

Đã tạo config, launcher và notebook. Chưa chạy training, evaluation hoặc tests
local. Không sửa code V1/V2/V3/V4 hiện có và không tự thay AMP/loss/router để
tối ưu baseline trong bước đặt tên CANDLE V1 này.
