# APPLIANCE: empirical acceptance và discovery V2

## Quyết định protocol

Người dùng chọn **empirical acceptance** ngày 10/10/2026. Đây là thay đổi protocol so với lifecycle strict-scope V1.

Patch đã pass acceptance trên CAL hợp lệ được phép chạy trên toàn bộ input của application. Không yêu cầu CAL lịch sử cho mỗi class trước khi mở route. Tuy nhiên, **deployment scope không đồng nghĩa CAL evidence scope**: không có bảo đảm FAR trên class/phân bố chưa quan sát.

Giữ nguyên initial certificate, head, signature, threshold, margin guard và BASE veto. Không sử dụng test labels để mở route, chọn donor hoặc tune threshold. Không đọc lại raw CAL của task cũ.

## State và authorization

`appliance_scope_mode='appliance_empirical_current_CAL_v2'` được khóa trong service contract.

Sau acceptance, registry tạo declaration riêng gắn với patch, initial certificate, guard fingerprint, backend và phiên bản Torch. Declaration cho phép inference trên 34 class của application; evidence trong initial certificate vẫn giữ nguyên phạm vi ban đầu.

- Function/head/guard còn đúng, không có conflict: `COMMITTED`, rồi `CARRY_FORWARD` qua task mới.
- CAL mới ít hơn 32: không làm hỏng certificate cũ; vẫn ghi rõ thiếu evidence.
- Monitor phát hiện FAR > 0,1% overall hoặc theo class, hoặc break > 0: latch conflict và `SUSPENDED`.
- Dependency/head/guard drift, declaration/certificate sai: `SUSPENDED`.
- Suspend giữ lại head, registry và protection. Training không dừng vì patch bị suspend.
- Không tự động xóa conflict latch hoặc đổi threshold để cứu patch.

Chế độ mặc định của service vẫn là `initial_scope_v1` cho caller cũ; launcher mới chọn empirical mode rõ ràng. Resume không cho phép âm thầm đổi service contract.

## Discovery V2

Trước khi chọn donor, dùng metadata kiểm tra receiver:

1. Chưa có capability được cài; multi-capability trên cùng receiver chưa được xác minh.
2. Có owned BASE protection evidence cho các mature old outputs cần bảo vệ. Inherited output không tạo owned provenance.
3. Current CAL có đủ FIT/SELECTION/HOLDOUT theo gate hiện có.

Chỉ donor trong live positive-weight neighborhood được phép. Xếp donor bằng quality LCB trên FIT, tie theo client ID. Chọn receiver theo ID và luân phiên giữa các class.

Budget launcher: tối đa **16 transactions/callback**, tối đa **8 receivers/class/callback**, một receiver tối đa một transaction trong callback. Không phải 16 expert inference; APPLIANCE vẫn inference bằng một local model. Không thử donor thay thế dựa vào kết quả HOLDOUT của lần reject.

## Inference và accounting

Evaluator ghi route authorized/suspended, certificate validity và số sample thực sự kích hoạt imported route. Accuracy vẫn tính một final prediction/sample trên toàn bộ test, chia disjoint cho receiver.

Backend/Torch là một phần guard fingerprint hiện có. Patch được compile/chứng nhận trên backend thực tế của receiver. Smoke local dùng CPU; training Kaggle dùng CUDA. Launcher dùng `auto`, đọc backend từ declaration đã seal trong checkpoint. Không bỏ backend binding hoặc âm thầm fallback toàn bộ imported predictions. Chưa xác minh khả năng chuyển certificate giữa backend/phiên bản Torch.

Đây là giới hạn runtime thực tế, có thể tăng thời gian full evaluation. CUDA portability cần evidence riêng; không được gọi là đã pass chỉ vì architecture giống nhau.

## Kiểm tra và readiness

`tools/check_appliance_deployment_protocol.py` kiểm tra declaration, strict fallback, drift/conflict, metadata scheduler, cumulative receipt rejection và atomicity. Các receipt/graph trong phần kiểm tra này là synthetic; không phải scientific acceptance evidence.

`tools/run_appliance_training_smoke.py --scope-mode appliance_empirical_current_CAL_v2` chạy callback automatic trên Task 3 checkpoint/graph thật, kiểm tra imported activation trên cùng locked current donor CAL HOLDOUT, rồi native Task 4 và save/resume comparison. Activation check là functional integration, không phải test holdout mới.

Launcher `train_denice_appliance_full_kaggle.ipynb` yêu cầu smoke lock mới có đúng mode và đã xác minh imported activation. Lock cũ của strict-scope V1 không mở được campaign mới.

Không coi smoke hoặc kiểm tra protocol là accuracy gain trên full test. Run `results (12)` vẫn giữ kết luận: APPLIANCE không thay đổi predictions so với MulticlassSelf.

## Kết quả integration đã hoàn tất

Lock mới: `artifacts/appliance_production_integration_smoke.json`, **61/61 checks, 0 mismatch**.

- Callback Task 3 dùng 80 model của DeNICE gốc và graph thật: 16 transactions, **3 commit / 13 reject**, tất cả reject rollback verified.
- Patch tự động: `60→0/class21`, `14→55/class20`, `88→65/class20`. Cả ba được authorize; scheduler đã xác nhận nhiều receiver cùng class.
- Imported activation trên cùng locked current donor CAL HOLDOUT: **321/336**, **193/203**, **151/153**. Đây là functional verification, không phải test accuracy độc lập.
- Task 4 có 89 active clients, tối đa 256 BASE samples/client, hai round. Cả ba receiver được train thật; sau local training, aggregation và task finalization vẫn head exact, certificate current và `CARRY_FORWARD`.
- Current CAL âm lần lượt **10 / 33 / 65 rows**, FAR=0, break=0 tại các callback đã quan sát. Thiếu CAL không làm mất chứng nhận cũ trong empirical mode; không chứng minh safety trên toàn bộ class/phân bố.
- Resumed và uninterrupted branch khớp model, algorithm/registry, RNG, novelty, ages, old references và membership bookkeeping. Service wall-clock timing không dùng làm equality criterion.
- Historical raw CAL reads=0; final test chưa mở; fresh training T0→T5 chưa chạy.

Discovery/setup traffic ở callback Task 3: **112.134.156 bytes** (gồm rejected attempts và receiver capsules). Không gọi kích thước capability vài KiB là toàn bộ communication cost.

Full launcher vẫn checkpoint mỗi round, ZIP theo task và đánh giá toàn bộ 34-class test chỉ sau Task 5 cuối cùng. APPLIANCE evaluation chọn đúng backend chứng nhận trong checkpoint; không ép CPU từ kết quả smoke local. Training GPU không bị thay đổi bởi sửa launcher evaluation này.

Loss regression của Task 4 trong `results (12)` vẫn cần audit riêng. Integration PASS không chứng minh nguyên nhân loss đã được sửa hoặc dự báo accuracy của campaign mới.
