# CANDLE V4 — core gốc V1

Cập nhật: đã thêm [native DRAFT extension](CANDLE_V4_NATIVE_SHADOW_20261023_VI.md).
365/367 file gốc còn nguyên; hai file private thay đổi, thêm các module V4 shadow/bank/preflight.
Phần dưới mô tả **mốc clone ban đầu**, không phải trạng thái extension hiện tại.

## Phạm vi

Stage `v1_clone`: tạo `candle_v4_runtime/` từ commit V1 gốc
`a9714ac96b3ecec3910985d6fda8224b1584b570`, đúng `CODE_REF` của notebook V1.
Không lấy core V3 hiện tại hoặc working tree có sửa đổi làm nguồn.

367 file source, 3.804.166 byte khớp Git blobs của V1. Config trong runtime cũng
giữ nguyên V1. Các thay đổi thuật toán V4 chưa được triển khai.

| Thư mục/file trong runtime | Nội dung |
| --- | --- |
| `fed_learning/models/` | NICE/DeNICE CNN-GRU và masks |
| `fed_learning/clients/` | Local training |
| `fed_learning/training/` | Native orchestrator, aggregation hooks, checkpoint |
| `fed_learning/strategies/` | Peer selection, age-masked aggregation và continual strategies V1 |
| `fed_learning/servers/` | Native context detector/router và dependencies |
| `fed_learning/data/` | Dataset loader và roles protocol |
| `tools/`, `configs/`, `appliance/` | Dependencies nguyên trạng trong source V1, không tự bật module bổ sung |
| `train_incremental_kaggle.py` | Entrypoint training gốc |
| `eval_checkpoint.py` | Evaluation gốc |
| `requirements-candle-v1-kaggle.txt` | Requirements V1 |

Sao chép toàn bộ dependency surface của sparse checkout V1 để không thiếu imports.
Sự hiện diện của source một module không có nghĩa module đó được bật trong config.
Không sao chép data, model checkpoint, kết quả hoặc Git metadata.

## Cấu hình nền giữ nguyên

- DeNICE legacy; mỗi client có một model và native `binary_cosine` router.
- Native `paper` peer neighborhoods, ξ=0,8; age-masked aggregation V1.
- 100 clients, 34 classes; T0–T5, 20 rounds/task; seed 42; batch 2048.
- Một GPU, AMP, Adam lr=0,001 theo baseline.
- Role split V1: BASE 80%, FIT 8%, CAL 2%, validation 10%.
  Learner còn local validation split 10% trong current BASE như implementation gốc.
- Không historical raw replay; không bật APPLIANCE/CME/CGoFed.
- Checkpoint mỗi round, ZIP mỗi task; TEST cuối T5/R19 sau terminal consolidation.
- Evaluation V1: full original TEST chia disjoint receiver shards, pooled metrics;
  BinarySelf primary, MulticlassSelf secondary. Không đổi inference/evaluation protocol.

## Integrity và trạng thái

`candle_v4_runtime/v1_clone_manifest.json` ghi path, kích thước, SHA-256 và Git blob
ID của từng source file. `tools/clone_candle_v4_runtime.py` kiểm tra archive trước
khi ghi và đọc lại hashes sau sao chép. Đường dẫn runtime có `.gitattributes -text`
để checkout Windows/Linux không làm đổi line endings của source lock.

Đây là kiểm tra toàn vẹn bản sao, không phải unit tests, smoke hoặc benchmark.
Chưa chạy training/evaluation/tests V4; chưa tạo launcher/notebook full training V4.
V1 và V3 hiện có giữ nguyên. Các absolute imports bên trong bản sao cần được resolve
trong process có `candle_v4_runtime` làm import root khi tích hợp launcher sau này.

V4 hiện mới có nền V1 nguyên bản; chưa có claim cải thiện accuracy hoặc transfer.
