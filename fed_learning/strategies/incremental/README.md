# Standalone Incremental Strategies

Thư mục này chứa các phiên bản incremental learning dùng cho chế độ local,
không mang ý nghĩa federated trong API import.

## Khác gì với `fed_incremental/`

- `fed_incremental/`
  - dành cho bối cảnh federated learning
  - có trainer + aggregator
  - được dùng cùng client/server/worker của FL pipeline

- `incremental/`
  - dành cho incremental learning thuần local
  - chỉ export trainer và helper cần cho local training
  - không export aggregator

## Mapping giữa 2 folder

- `incremental/ewc.py` -> local `EWCTrainer`
- `incremental/lwf.py` -> local `LwFTrainer`
- `incremental/der.py` -> re-export `DERTrainer`
- `incremental/nice.py` -> re-export `NICETrainer` và các helper neuron
- `incremental/glfc.py` -> re-export `GLFCTrainer`
- `incremental/refed.py` -> re-export `ReFedTrainer`
- `incremental/cgofed.py` -> re-export `CGoFedTrainer`
- `incremental/fedcbdr.py` -> local `CBDRTrainer`

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

- Các class trong thư mục này vẫn tái sử dụng logic từ `fed_incremental/` khi phù hợp.
- Mục tiêu là tách API import và ý nghĩa sử dụng, không phải viết lại toàn bộ thuật toán từ đầu.
