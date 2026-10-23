# Incremental Strategies

Thư mục này là phần cốt lõi của các thuật toán incremental learning.

- `incremental/` là source of truth cho các thuật toán IL thuần local.
- `fed_incremental/` chỉ giữ phần mở rộng cho bối cảnh federated:
  wrapper, aggregator, hoặc ghép với `FedAvg` / `FedProx`.

## Khác gì với `fed_incremental/`

- `incremental/`
  - chứa implementation gốc của trainer và helper IL
  - dùng được cho local incremental learning

- `fed_incremental/`
  - dùng lại logic từ `incremental/`
  - thêm aggregator hoặc wrapper dành cho federated pipeline

## Mapping giữa 2 folder

- `incremental/ewc.py` -> logic EWC gốc + `EWCTrainer`
- `incremental/lwf.py` -> logic LwF gốc + `LwFTrainer`
- `incremental/der.py` -> logic DER gốc
- `incremental/nice.py` -> logic NICE gốc + helper neuron
- `fed_incremental/ewc.py` -> wrapper `FedAvgEWCTrainer`, `FedProxEWCTrainer`
- `fed_incremental/fedlwf.py` -> wrapper `FedLwFTrainer` + aggregator
- `fed_incremental/fedcbdr.py` -> implementation FedCBDR đầy đủ cho FL-IL
- `fed_incremental/der.py` -> aggregator DER
- `fed_incremental/nice.py` -> aggregator NICE
- `fed_incremental/glfc.py` -> thuật toán GLFC đầy đủ cho FL-IL
- `fed_incremental/refed.py` -> thuật toán Re-Fed đầy đủ cho FL-IL
- `fed_incremental/cgofed.py` -> thuật toán CGoFed đầy đủ cho FL-IL

## Cách import

Ví dụ:

```python
from fed_learning.strategies.incremental import EWCTrainer, LwFTrainer, DERTrainer
```

hoặc:

```python
from fed_learning.strategies.incremental.ewc import EWCTrainer
from fed_learning.strategies.incremental.lwf import LwFTrainer
```

## Lưu ý

- Nếu bạn đọc thuật toán để hiểu IL, hãy đọc `incremental/` trước.
- Local `mode="il"` chỉ hỗ trợ: `ewc`, `lwf`, `der`, `nice`.
- Riêng `FedCBDR` chỉ tồn tại ở nhánh `fed_incremental/`, không có bản IL local riêng.
- `GLFC`, `ReFed`, `CGoFed` cũng là FL-IL-only, không có bản IL local riêng.
- Nếu bạn đọc để hiểu cách ghép vào FL pipeline, hãy xem thêm `fed_incremental/`.
