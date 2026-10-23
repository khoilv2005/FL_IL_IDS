# CANDLE V3 — bản nền sao chép từ V1

> Bản nền được giữ làm origin. Tiến độ triển khai tiếp theo nằm tại
> [Peer Teaching Implementation](CANDLE_V3_PEER_TEACHING_IMPLEMENTATION_20261023_VI.md).
> Runtime hiện đã có delta V3 và manifest riêng; notebook baseline vẫn pin bản clone.

## Phạm vi hoàn thành

Tạo nhánh runtime riêng cho CANDLE V3 từ đúng source commit mà notebook V1 đang pin:
`a9714ac96b3ecec3910985d6fda8224b1584b570`.

Đây là giai đoạn `v1_clone`. Thuật toán chọn peer vẫn giống V1; chưa triển khai
chọn peer theo class bổ sung. Chưa chạy training, evaluation hoặc unit tests.

V1 được giữ nguyên. Các thay đổi thuật toán V3 về sau nằm trong
`candle_v3_runtime/`, không chỉnh các module V1 tại thư mục gốc.

## Những gì được sao chép

367 file nguồn, 3.804.166 byte, gồm toàn bộ các cây source mà notebook V1 tải:

- `fed_learning/`: model, clients, training, strategies và checkpoint.
- `tools/`, `configs/`, `appliance/`: dependencies nguyên trạng của runtime V1.
  Những tính năng không bật trong V1 vẫn không được bật trong V3.
- `train_incremental_kaggle.py`, `eval_checkpoint.py` và requirements V1.

Không sao chép dataset, model checkpoint, kết quả hoặc Git metadata.
`candle_v3_runtime/v1_clone_manifest.json` ghi đường dẫn, số byte, SHA-256
và Git blob ID của từng file. Các file khớp byte với source commit gốc.

Launcher `tools/train_candle_v3.py` mở một Python subprocess riêng với runtime
V3 làm thư mục làm việc và import root. Các absolute imports `fed_learning`,
`tools`, `appliance` sử dụng bản trong runtime V3. Launcher từ chối file nguồn
không còn khớp manifest trong giai đoạn clone.

## Cấu hình giữ theo V1

| Thành phần | Giá trị |
| --- | --- |
| Engine | DeNICE legacy, mỗi client một model/router |
| Tasks / rounds | T0–T5, 20 rounds/task |
| Seed / batch | 42 / 2048 |
| Optimizer | Adam, learning rate 0,001 |
| Peer selection | `paper`, similarity threshold ξ = 0,8 |
| Aggregation | Local deltas, pairwise young masks, reserve weighting |
| Router | `binary_cosine` |
| Dữ liệu | Roles V1: BASE 80%, CAL 2%, FIT 8%, validation 10% |
| Historical raw replay | Không |
| GPU | Một GPU, AMP theo V1 |
| Checkpoint | Mỗi round, ZIP mỗi task, giới hạn output 12 GiB |
| TEST | Chỉ sau Task 5 / Round 19 và terminal consolidation |
| Primary / secondary | BinarySelf / MulticlassSelf |

TEST giữ đúng protocol V1: toàn bộ TEST 34 class được chia thành các receiver
shards không trùng nhau, báo cáo pooled metrics. Không đổi sang full TEST trên
từng client. Cùng cấu hình không đồng nghĩa tái tạo chính xác kết quả số cũ.

`configs/candle_v3.json` chỉ khác config V1 ở ba đường dẫn output/resume/roles.
`configs/candle_v3_provenance.json` lưu nguồn sao chép và trạng thái clone.

## Notebook Kaggle

File: `train_candle_v3_kaggle.ipynb`.

- Clone source từ GitHub tại một commit cố định.
- Không nhúng mã nguồn, dataset hoặc checkpoint trong notebook.
- Dataset mặc định: `/kaggle/input/datasets/khoilv2005/100-clients`.
- Output: `/kaggle/working/results_candle_v3_xi_0p8_seed_42`.
- Roles mặc định: `/kaggle/working/candle_v3_roles`; có thể trỏ đến roles V1 đã khóa.
- Source checkout nằm trong `/kaggle/temp/candle_v3_source`.

Nếu chạy notebook ở giai đoạn này, kết quả là một lần chạy baseline tương đương
V1 về cấu hình. Chưa có thay đổi thuật toán để kỳ vọng cải thiện accuracy.

## Công việc tiếp theo

Trước khi đổi lựa chọn peer, cần khóa contract cho metadata: phân biệt class
thực sự train local, class nhận qua collaboration và class mới chỉ được router
đăng ký. Không dùng ownership bitmap như bằng chứng model đã nhận diện tốt.

Sau đó mới triển khai lựa chọn peer có kiến thức bổ sung, với kiểm tra tương thích
representation, masks và maturity. Không giả định peer càng khác nhau càng tốt.
Giai đoạn đó phải cập nhật manifest/stage và ghi rõ delta so với baseline này.
