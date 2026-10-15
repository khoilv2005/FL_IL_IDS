# Chẩn đoán DENICE Kaggle version 44 và hướng cải thiện

Ngày kiểm tra: 15/10/2026. Nguồn: [notebook và log công khai](https://www.kaggle.com/code/phungvannamanh/denice-train-100-client/log), version ID 358085977, commit `9bc16b730a68d6407b398ed23ebe6ea7c78f7519`. Chạy thành công về mặt tiến trình trong 36.459,1 giây; điều này không có nghĩa mô hình học tốt.

## 1. Những gì xác nhận được

Log chạy runner `decentralized_denice_il.py`, thư mục `results_denice_legacy_seed_42`, routing hard, router replay bật, `appliance_enabled=False`, threshold-neighborhood xi=0,5. Đây không phải pipeline `train_denice_v2_full_kaggle.py`.

Cần phân biệt hai cờ: `appliance_enabled` là service upstream; `denice_appliance_enabled` là overlay truyền hàng phân loại. Audit cũ bỏ sót cờ thứ hai. Vì vậy chỉ nhìn `appliance_enabled=False` **chưa đủ** chứng minh overlay tắt. Tuy nhiên log threshold-neighborhood không phù hợp yêu cầu `adaptive_ap` của overlay, và không có dòng `APPLIANCE overlay`; bằng chứng phù hợp với chạy baseline. Bản sửa ghi cả hai cờ và tên phương pháp hiệu lực; không dựa vào tên thư mục để nhận dạng phương pháp.

| Task | Client đánh giá / active | Accuracy (%) | F1 log (%) | Coverage pooled (%) |
|---|---:|---:|---:|---:|
| 0 | 8/50 | 90,18 | 47,34 | 100 |
| 1 | 4/60 | 73,69 | 56,87 | 100 |
| 2 | 1/70 | 50,71 | 40,27 | 100 |
| 3 | 80/80 | 27,54 | 18,10 | 65,02 |
| 4 | 89/89 | 23,67 | 15,05 | 63,80 |
| 5 | 98/98 | 22,18 | 13,68 | 63,90 |

Mỗi task lấy 50.000 mẫu test. Task 5 có 13.505.771 mẫu khả dụng nhưng chỉ dùng panel 50.000. 4.900.000 trong audit = 98 client × 50.000, **không phải** 4,9 triệu mẫu test độc lập. Không gọi đây là full-test. F1 là giá trị tổng hợp log, không suy diễn thành pooled F1.

## 2. Nguyên nhân và mức độ bằng chứng

### A. Thiếu coverage là lỗi kiến trúc/phạm vi kiến thức

Task 5: N=4.900.000; covered C=3.130.880; correct-route R=1.459.479; correct K=1.086.834. Trong run này mọi dự đoán đúng đều thuộc correct-route. Phân rã lỗi quan sát:

\[
1-K/N=(N-C)/N+(C-R)/N+(R-K)/N.
\]

- Không có lớp trong các episode khả dụng: **36,1045 điểm phần trăm**.
- Có lớp nhưng chọn sai episode: **34,1102 điểm phần trăm**.
- Đúng episode nhưng phân loại sai: **7,6050 điểm phần trăm**.
- Đúng: **22,1803%**.

Đây là phân rã mô tả, không phải hiệu quả nhân quả của từng bản sửa. Router hoàn hảo chưa chắc sửa hết 34,11 điểm: classifier của episode đúng vẫn có thể sai. `accuracy_given_correct_route=74,4673%` là độ chính xác có điều kiện trên nhóm đã được route đúng; **không phải oracle accuracy trên toàn bộ test**.

Chỉ tăng epoch hoặc thêm server tạm không tự tạo ra kiến thức lớp mà client chưa học. Overlay hiện tại từ chối `unknown_gap` nếu không có đủ positive/negative validation tại receiver. Đây là hành vi đúng về bằng chứng, nhưng giới hạn khả năng chữa missing-class coverage. Không lấy test hoặc dữ liệu client khác làm validation để vượt qua kiểm tra này.

### B. Router là nút thắt lớn ngay khi coverage đầy đủ

Ở task 2, coverage=100% nhưng routing đúng chỉ 59,492%; conditional classifier accuracy=85,2451%. Do đó cần đối chứng router trước khi tăng kích thước backbone. Repo đã có `multiclass_balanced`: logistic regression trên activation memory địa phương, cân bằng theo số mẫu mỗi episode. Dùng lại triển khai này, không tuyên bố thuật toán mới. Mục tiêu là học ranh giới phân biệt task thay vì chỉ so cosine với prototype.

Hàm mục tiêu tương ứng: tổng weighted negative log likelihood cộng L2; trọng số episode t là n/(T n_t). Không có quyền dùng nhãn test để route khi triển khai. Router fit từ memory train/replay riêng client. Có thể không cải thiện nếu biểu diễn không tách được task; cần kiểm tra trên validation.

### C. Đánh giá thay đổi quần thể client

`_select_eval_clients` ưu tiên chỉ client có full coverage khi tồn tại; khi không còn client nào đầy đủ thì quay về tất cả. Task 2 chỉ đánh giá 1 client, task 3 đánh giá 80. Vì vậy đường accuracy không đo thuần forgetting. Bộ chạy mới đặt `denice_eval_require_full_coverage=False` và không giới hạn client. Vẫn phải báo tập active thay đổi theo task, và đo forgetting trên cùng client–task intersection nếu cần so dọc thời gian. Không so điểm baseline cũ với cấu hình mới như một ablation chuẩn.

### D. Cập nhật số học không ổn định

Có **57 bản ghi client-round** với `steps=0` và skipped>0 trong log, không phải 57 client riêng biệt. Có gradient không hữu hạn, thường liên quan `fc2.weight`; một số bản ghi danh sách sau clipping rỗng nhưng scaler vẫn bỏ step. Danh sách rỗng sau gradient filtering không chứng minh backward ban đầu hữu hạn.

FP16 có miền biểu diễn hạn chế, GradScaler có thể bỏ optimizer step để tránh cập nhật hỏng. Không được ép optimizer chạy trên NaN/Inf hoặc tắt kiểm tra. Bộ chạy FP32 loại bỏ nguồn overflow do FP16; lỗi FP32 vẫn phải dừng để điều tra. Điều này chưa chứng minh mọi gradient lỗi đều do AMP và chưa chứng minh sẽ nâng accuracy.

### E. Hợp tác quá dày và cạn capacity: cần ablation

Task 0 trung bình 47,16 peers/49 peer tối đa; task 5 khoảng 84,73 peers, peer alpha khoảng 0,9898. Cơ chế này gần trao đổi rộng toàn mạng, không thể coi sentinel K=1 là bằng chứng đã tìm được một cụm tự nhiên. Log task 5 có client 35 và 85 với min_rho0=0. Đây là dấu hiệu cần kiểm tra can thiệp giữa client/cạn neuron tự do; chưa chứng minh quan hệ nhân quả với toàn bộ suy giảm.

## 3. Hướng đề nghị

Ưu tiên **định tuyến có học + phân cụm thật + kiểm chứng chuyển giao cục bộ**, sau khi ổn định số học và chuẩn hóa phép đo. Chưa mở khóa neuron mature hoặc tăng số vòng hàng loạt: chúng có thể phá retention mà không giải quyết coverage.

Bước triển khai hiện tại dùng FP32, router logistic cân bằng đã có, adaptive AP/class-block allocation theo proposal hiện có, và overlay tùy chọn. So sánh overlay với bản không overlay nhưng cùng mọi thiết lập còn lại. Đây là giả thuyết thực nghiệm, không phải cấu hình đã chứng minh tốt nhất.

Về dài hạn, nếu mục tiêu bắt buộc mọi client nhận biết mọi lớp, cần mở rộng đơn vị chuyển giao thành **expert gồm encoder/mask/normalization/head và router metadata tương thích**, hoặc học không gian đặc trưng chung trước khi chuyển head. Không thể chỉ đăng ký class ID chưa được huấn luyện. Nếu receiver không có positive validation, kết quả phải ghi là chưa được chứng nhận cục bộ; chứng nhận tại donor không thay thế phân phối receiver. Hướng expert làm tăng bộ nhớ, chi phí truyền và rủi ro rò rỉ từ model; chưa được triển khai trong bản sửa này. Không hứa giải quyết hoàn toàn 36,10% thiếu coverage bằng row overlay.

## 4. Code đã thay đổi

- `denice_run_audit.py`: nhận dạng base/service/overlay, từ chối cờ xung đột và sai `denice_expected_method`; thống kê optimizer health.
- `train_incremental_kaggle.py`: in phương pháp hiệu lực sau toàn bộ override, trước training.
- `decentralized_denice_il.py`: lưu nhận dạng/config router/AMP vào `source_audit.json`; ghi mọi client-round vào `optimizer_health.jsonl`; có `denice_fail_on_zero_update` để dừng nếu lượt có batch nhưng không cập nhật lần nào.
- `tools/run_denice_recovery.py`: launcher rõ ràng cho các ablation, loại bỏ biến môi trường config-file cũ, fresh run, không ghi đè output đã có. Không đổi mặc định baseline lịch sử.
- Test kiểm tra nhận dạng, lỗi zero-update, tính nhất quán ablation, router cân bằng với memory lệch và checkpoint restore.

Các arm:

| Arm | AMP | Router | Overlay |
|---|---|---|---|
| amp_control | Có | binary_cosine | Tắt |
| fp32 | Không | binary_cosine | Tắt |
| balanced_router | Không | multiclass_balanced | Tắt |
| cluster_control | Không | multiclass_balanced | Tắt; cấu hình cụm/validation giống overlay |
| row_overlay | Không | multiclass_balanced | Bật |

`cluster_control` và `row_overlay` chỉ khác hai trường enable/expected-method. Không quy toàn bộ khác biệt giữa balanced_router và row_overlay cho truyền kiến thức vì còn thay đổi clustering, allocation và validation fraction.

## 5. Chạy trên Kaggle

Dùng checkout chứa **các file mới trong bản sửa này**, không chỉ commit 9bc16b7. Có thể upload source bundle dưới Kaggle Dataset hoặc cập nhật Git rồi clone commit cụ thể. Các cell sau chạy từ thư mục gốc checkout, nơi có `fed_learning/`, `configs/`, `tools/` và `train_incremental_kaggle.py`. Không chạy notebook V2 cho thí nghiệm này.

Giữ dependencies của repo (`requirements.txt`); cần PyTorch, numpy, scipy, scikit-learn. Kaggle T4 hỗ trợ FP32 nhưng thời gian chạy có thể tăng. Đây là mô phỏng FL một tiến trình, không tự sử dụng cả hai T4 và không có bảo mật hạ tầng giữa client.

```python
import subprocess, sys
from pathlib import Path
repo = Path.cwd()  # đổi thành thư mục checkout nếu cần
assert (repo / 'tools/run_denice_recovery.py').exists()
data = '/kaggle/input/datasets/khoilv2005/100-clients/100-clients'
arm = 'balanced_router'  # fp32 để có đối chứng chỉ khác router
command = [sys.executable, str(repo / 'tools/run_denice_recovery.py'),
           '--arm', arm, '--data-dir', data,
           '--output-dir', '/kaggle/working/recovery_' + arm + '_42', '--seed', '42']
subprocess.run(command + ['--dry-run'], check=True)
subprocess.run(command, check=True, cwd=repo)
```

Chạy mỗi arm một session/output riêng. Trước khi chọn phương pháp, dùng `validation_metrics.json`; test cuối mặc định panel 50.000 để hạn chế chi phí. Khóa thiết kế trên validation, rồi chạy seeds 42/43/44 và `--full-test` cho báo cáo cuối. Full-test đánh giá mọi active client trên toàn bộ test nên rất tốn thời gian. Báo coverage, route accuracy pooled, conditional accuracy, macro-F1, recall mỗi lớp, false-positive rate benign, bộ nhớ, thời gian và bytes truyền; không chỉ accuracy.

Output gồm các artifact hiện có (`config.json`, `source_audit.json`, `validation_metrics.json`, metrics/checkpoint) và `optimizer_health.jsonl`. Với row_overlay có thêm `appliance_rounds.jsonl`; kiểm tra leaders, packets, accepted và lý do unknown_gap/no_compatible_donor. Một run có zero accepted không phải bằng chứng transfer có ích.

## 6. Quyền riêng tư và giới hạn kiểm chứng

Router và validation giữ dữ liệu tại client theo mô hình mô phỏng; chuyển weight/metadata không tương đương DP hay secure aggregation. Local replay vẫn giữ mẫu cũ tại client. Không thêm dữ liệu test vào replay/validation, không dùng oracle task để quyết định dự đoán thật. Các log chi tiết và model update vẫn có thể tiết lộ thông tin về phân phối lớp.

Chưa chạy lại 100 client/6 task trên T4 từ môi trường này; chưa có số accuracy mới. Những kiểm tra CPU/synthetic chỉ xác nhận logic chạy và bất biến, không chứng minh hiệu quả IDS. Kết quả Kaggle version 44 là bằng chứng chẩn đoán, không được dùng lặp lại để chọn hyperparameter trên test.

Kiểm tra thực thi: `python -m pytest tests/test_denice_recovery.py tests/test_denice_router_replay.py tests/test_denice_appliance.py -q -p no:cacheprovider -p no:faulthandler`: **18 passed**. Dry-run arm row_overlay thành công. Source ZIP có `SOURCE_MANIFEST.json` để kiểm tra hash từng thành viên, không chứa checkpoint hoặc dữ liệu train.

## 7. Tài liệu xác minh

- [Kaggle run được phân tích](https://www.kaggle.com/code/phungvannamanh/denice-train-100-client/log).
- [PyTorch AMP](https://docs.pytorch.org/docs/2.14/amp.html): gradient scaling và giới hạn số học của FP16; tham khảo cơ chế, không ép Kaggle nâng PyTorch.
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): balanced class weights và multinomial fitting. Đây là kỹ thuật có sẵn, không phải đóng góp mới của DENICE.
