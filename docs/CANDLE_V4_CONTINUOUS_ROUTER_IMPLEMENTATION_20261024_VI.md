# CANDLE V4 continuous_class_v1 — implementation riêng, V1 giữ nguyên

## Phạm vi

Đã thêm một variant V4, chưa thay cấu hình chính và chưa benchmark accuracy. Tuân thủ yêu cầu mới nhất: **không đụng CANDLE V1, không thay đổi quá nhiều training/evaluation so với các thuật toán đối chứng**.

V1 code/notebook/config và bốn helper chung (ContextDetector, native TRAIN sampler, routed inference, native checkpoint) giữ nguyên byte-for-byte. Không sửa loss, native optimizer, batch, epochs/steps, current-task filtering, graph hay expert gradient/quota. Dùng100% original current-task TRAIN, validation0%; không BASE/FIT/CAL hoặc shadow native model.

## Thay đổi duy nhất trong phương pháp

V4 giữ thêm selected continuous activations kèm class/task identity. Lấy từ đúng các samples và encoder forwards mà native router đã thực hiện; không thêm random sample draw hoặc đọc lại old TRAIN. Mỗi class giữ tối đa cùng budget router_reference_per_class hiện có. Không lưu raw inputs.

StandardScaler + class-balanced LR fit **feature memory**, dự đoán class rồi ánh xạ thành task. Không truyền true task khi inference. Class-max compatibility scores không phải calibrated task posterior. Expert vẫn dùng donor handoff/scoped threshold0,5 và native fallback như v6. Không thêm quality gate hoặc freezing.

Binary detector vẫn là native multiclass, không thay capsule/graph. Continuous memory/scaler/coefficients lưu trong **V4 service state**. Inference V4 dùng một detector view riêng trên cùng local model; không tạo bản sao model để train shadow. Các helper V1 vẫn chạy nguyên bản.

## Evaluation và tính công bằng

- Giữ6tasks×20rounds, seed42, cùng original34-class TEST và disjoint receiver partition.
- Chỉ TEST tại cuốiT5/R19 sau consolidation; không thêm per-round/per-task TEST hoặc validation.
- Headline là V4 `coverage`; không average với các readout controls.
- Native V1+multiclass control giữ nguyên. Continuous-native readout được ghi riêng từ prediction đã tính để tách đóng góp router/expert; cùng TEST rows và pooled Accuracy/Macro-F1.
- Timing phân biệt native control, continuous-native inference và expert overhead. Router fitting có thêm CPU computation và được tính trong context-fit time; không giả vờ toàn bộ compute cost giống V1.

## Kiểm tra kỹ thuật

**30/30 PASS**, chỉ synthetic data:

| Nhóm | Số tests |
|---|---:|
| Continuous router + native isolation + real CNN-GRU capture/inference | 9 |
| Donor handoff | 7 |
| Original TRAIN / activation / scoped fusion | 11 |
| Terminal evaluator | 3 |

Đã kiểm tra: amplitude/class identity bị binary làm mất; cap memory và current labels; old feature memory không bị ghi đè; snapshot/restore scaler/classifier; từ chối conversion/corrupt state; Torch RNG/native binary memory/capsule summary khớp; native encoder không đổi khi capture; terminal control V1 và tất cả TEST rows giữ nguyên. SHA256 xác nhận protected V1 files và shared helpers không đổi.

Receipts: `artifacts/candle_v4_continuous_router_regression_20261024.json`, `artifacts/candle_v4_continuous_v1_isolation_20261024.json` và các regression receipts v6 hiện có. Không mở real TEST/CAL, không chạy full training trong regression.

## Notebook riêng và giới hạn

`train_candle_v4_continuous_router_kaggle.ipynb`, output `results_candle_v4_continuous_class_seed42`, protocol `candle_v4_continuous_class_router_v7`. Notebook V1 và notebook V4 multiclass/handoff hiện có không bị thay thế. Cần fresh stream; không resume checkpoint chỉ có binary memory hoặc protocol cũ.

Chưa có benchmark chứng minh variant tốt hơn. Không gán42,91% của checkpoint scoped+multiclass cũ cho variant continuous này. First-task feature subspace và missing-task coverage vẫn có thể hạn chế chất lượng; early-task representation drift chưa được loại trừ bằng các unit tests. Không dùng TEST để tạo continuous historical memory.
