# DENICE: nâng cấp replay và functional regularization

## Mục tiêu và phạm vi

Bản nâng cấp trực tiếp DENICE/CANDLE hiện có giữ nguyên mô hình CNN–GRU,
phân tuổi neuron, structural masks, CANC, micro-adapter, Context Capsule,
Dynamic-K clustering, age-aware peer aggregation và context routing.
Không thêm server, global teacher hoặc kho dữ liệu dùng chung. Runner vẫn là
trình mô phỏng tuần tự giao thức phân tán, không phải hệ thống mạng P2P triển khai thật.

Giữ `algorithm="denice"`, `mode="decentralized"` trong script hiện tại
`train_incremental_kaggle.py`: replay và regularization đã bật sẵn trong CONFIG.
Không cần algorithm mới, script riêng hay DENICE_CONFIG_OVERRIDES.
Script mặc định phase 5: chạy mới task 0–5, không tải checkpoint cũ.
API cấp thấp vẫn mặc định capacity=0 khi thiếu khóa để giữ tương thích config
và checkpoint cũ. Muốn baseline, đặt capacity=0 và memory_policy="sketches".
Replay lưu ít mẫu huấn luyện tại client, khác ràng buộc replay-free của bài báo.

## Chẩn đoán từ code và bài báo

1. `DeNICETrainer` trước đây chỉ kế thừa CE của NICE. Không có rehearsal hoặc
   functional regularization trong local optimization.
2. Đóng băng trọng số cũ không đảm bảo lớp mới không có logit cao trên mẫu cũ.
   Cross-entropy trên dữ liệu mới không trực tiếp cung cấp các negative này.
3. `forward_output` dùng LetLearner; inference dùng `forward`. Loss phụ phải
   đi qua inference logits để học cạnh tranh giữa các lớp đã quan sát, nhưng
   vẫn chặn gradient mature trước optimizer step.
4. Eq. (14) của bản PDF mô tả penalty trên mature parameters. Trong triển khai
   hard-freeze, penalty đặt đúng các tọa độ đã đóng băng không tăng khả năng
   bảo vệ thêm qua gradient. Bản nâng cấp chọn regularization đầu ra thay vì gọi
   một penalty gần như bất động là cải tiến EWC.
5. Router vẫn có thể chọn sai context. Replay trên classifier không tự sửa
   toàn bộ lỗi routing, drift estimator hoặc thiếu capacity. Các cơ chế này
   được giữ nguyên để cô lập tác động của nâng cấp học tăng dần.

## Thuật toán

Mỗi client i có bộ nhớ riêng M_i với tối đa B mẫu. Mỗi phần tử chứa input,
label, logits đã chốt, mask các lớp đã biết lúc chốt và priority ngẫu nhiên.
Cuối task, sau aggregation và consolidation, trước khi xóa active adapters:

1. Chia B thành quota gần bằng nhau cho các lớp client đã quan sát.
2. Giữ các mẫu priority cao nhất trong từng lớp; lớp mới làm quota cũ co lại.
3. Chỉ forward các mẫu mới được chọn, ở eval mode. Không cập nhật lại target
   của mẫu cũ, tránh teacher trôi theo student.
4. Chỉ lấy `client.X_train/y_train` sau validation split. Không đưa validation,
   test hoặc dữ liệu peer vào memory. Không thêm lại dữ liệu mỗi round/epoch.

Quota chưa dùng của lớp ít mẫu không được phân phối lại, nên dung lượng thực
có thể nhỏ hơn B. B phải ít nhất bằng số lớp client từng quan sát; nếu không
sẽ báo lỗi thay vì âm thầm bỏ hoàn toàn một lớp. Priority reservoir giữ lịch
sử đồng đều trong từng lớp, không ưu tiên recency khi có concept drift.

Ở mỗi minibatch mới, lấy một replay minibatch cân bằng lớp có hoàn lại:

```text
L = L_NICE(current)
  + lambda_replay * CE(z_current_adapter(replay)[S], replay_labels)
  + lambda_dark * mean_examples(mean_known_logits((z - stored_z)^2))
  + lambda_calibration * CE(z_current_adapter(current)[S], current_labels)
```

S là hợp các lớp trong local memory và các lớp hiện tại tại client. Các label
được ánh xạ vào S, kể cả label không liên tiếp. Dark loss chỉ dùng mask lưu
tại thời điểm chốt: không ép lớp chưa từng thấy về logits chưa được học.
Đây là functional regularization lấy cảm hứng từ DER, không phải bản tái lập
nguyên xi DER++: chốt logits cuối task, dùng balanced reservoir và age masks.

Các forward phụ dùng eval statistics nhưng **vẫn giữ autograd**. BatchNorm và
dropout không làm ô nhiễm target hoặc statistics bởi replay; trạng thái
train/eval riêng của từng module được khôi phục. Sau tổng loss.backward,
`reset_frozen_gradients()` và clipping hiện có vẫn thực thi. Historical
adapters không được kích hoạt để học lại; loss bảo vệ hành vi của nhánh hiện tại.
Nếu mọi đường truyền liên quan đã bị đóng băng, dark loss có thể không có
gradient hữu ích; không hứa hẹn phục hồi tri thức đã mất do recycling.

Capsule reliability vẫn nhận current-data CE gốc, không nhận tổng loss phụ;
tránh hạ trọng số peer chỉ vì client đó có replay. Tổng objective và từng loss
phụ được ghi trong audit `imbalance_controls.clients[client_id].replay` của round.

## Cấu hình và chạy

Dùng trực tiếp CONFIG trong `train_incremental_kaggle.py`. Giải nén bundle để
script này và thư mục `fed_learning/` nằm cùng một thư mục. Script tự chọn
source cạnh nó; không cần khai báo DENICE_CODE_DIR trong trường hợp này.
Trên Kaggle, gắn dataset và kiểm tra CONFIG["data_dir"], rồi chạy:

```python
%run /kaggle/working/denice/train_incremental_kaggle.py
```

Thay đường dẫn trên bằng thư mục giải nén thực tế. Mặc định chạy task 0–5.
Muốn chia phiên, đặt DENICE_TRAIN_PHASE=1 để chạy task 0–1 rồi dùng các phase
2/3/4 với continuation tạo bởi bản nâng cấp. Không dùng continuation replay-free
cũ cho cấu hình replay mới. Script Kaggle vẫn có luồng tải checkpoint Google
Drive khi chọn phase resume; phase 5 không tải checkpoint.

| Tham số | CONFIG DENICE | Ý nghĩa |
|---|---:|---|
| denice_replay_capacity | 512 | Tổng số exemplar tối đa mỗi client |
| denice_replay_batch_size | 32 | Replay minibatch |
| denice_replay_ce_weight | 1.0 | Học nhãn cũ / giảm xâm lấn lớp mới |
| denice_replay_logit_weight | 0.2 | Giữ logits đã biết |
| denice_replay_calibration_weight | 0.2 | Cạnh tranh lớp cũ–mới trên mẫu mới |
| denice_memory_policy | local_replay | Router dùng sketches, không giữ raw bank phụ |

Các trọng số là điểm khởi đầu, chưa được tune trên CICIoT2023. CONFIG bật
structural protection và fixed task allocation, tắt refresh router bằng raw
history. Chỉnh trọng số/batch size trực tiếp trong CONFIG, hoặc gọi
`run_decentralized_denice_il(config)` với các khóa ở bảng trên.

`local_replay` giữ router trong protected feature subspace như baseline
sketches, xóa reference_input_memory và old_ref_banks. B chỉ giới hạn raw replay
exemplars; activation sketches, model/adapters, current dataset và checkpoint
vẫn có chi phí riêng. Bộ nhớ replay có chi phí xấp xỉ B*(input_bytes + 5*C + 12),
với C là số output, float32 logits, bool mask, int64 label và float32 priority.
Mỗi bước có thêm tối đa hai forward/backward phụ; đo wall time và VRAM thực tế.

## Dynamic clients, checkpoint và quyền riêng tư

- Memory thuộc client ID và ở CPU. Client tạm rời giữ memory; khi quay lại
  dùng đúng memory của mình. Client mới bắt đầu rỗng kể cả bootstrap từ peer.
- Không đưa memory vào model state, capsule hoặc peer aggregation. Không tăng
  kích thước thông điệp giao thức do truyền exemplar.
- Chỉ full continuation lưu `local_replay_states` cùng RNG để resume chính xác.
  Checkpoint phục vụ inference/round delta không thay thế continuation.
- Local continuation có chứa training inputs; đây là artifact riêng tư tại
  client/máy mô phỏng. Không xem file đó là payload được phép gửi peer.
- Resume kiểm tra cấu hình và tập client. Không âm thầm resume cấu hình replay từ
  checkpoint replay-free hoặc đổi capacity/loss weights giữa chừng.
- Lưu cục bộ không đồng nghĩa differential privacy; không thêm bảo đảm DP.

## Đánh giá và ablation

Chỉnh các trọng số trong CONFIG để chạy từng ablation từ đầu với
cùng split, seeds, số phase/round, CANC, routing và ngân sách local data.
Baseline là structural CANDLE/sketches với B=0; replay chỉ bật replay CE;
dark chỉ bật logit regularization; full bật cả ba loss phụ. Để tách riêng tác
động calibration, chạy thêm config full với calibration_weight=0 qua API.

Báo cáo ít nhất: accuracy và macro-F1 cumulative sau từng task, forgetting,
backward transfer, recall từng lớp, benign false-positive rate, route accuracy,
oracle-context và no-mask diagnostics, memory bytes, train time, communication.
Không dùng oracle task ID cho chỉ số chính. Chạy ít nhất 5 seeds trên cùng
client split, báo mean/std. So sánh thêm ngân sách wall time bằng nhau vì replay
tăng compute dù số current-data epochs không đổi. Sweep B=128/256/512/1024 và
lambda_dark=0/0.05/0.2/1 trên validation; giữ test chỉ cho đánh giá cuối.

Tests kiểm tra bound/balance, class-support masking, BN preservation, gradient
freeze, tác dụng replay trên bài toán xâm lấn lớp mới có kiểm soát, và chạy
hai client/ba task với dropout–rejoin rồi so sánh uninterrupted/resume từng
tensor. Kết quả synthetic chỉ chứng minh cơ chế và tích hợp, **không chứng
minh tăng accuracy/macro-F1 trên CICIoT2023**.

## Nguồn

- Bản PDF CANDLE trong workspace: Section 4.2–4.3, Eq. (12)–(23).
- [NICE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gurbuz_NICE_Neurogenesis_Inspired_Contextual_Encoding_for_Replay-free_Class_Incremental_Learning_CVPR_2024_paper.html).
- [Dark Experience Replay, NeurIPS 2020](https://arxiv.org/abs/2004.07211).

## Kết quả kiểm chứng implementation — 2026-09-24

- Bộ hồi quy gồm test_denice_replay, test_candle_pdf_logic, test_candle_repairs,
  test_candle_adapters, test_denice, test_denice_forensic và
  test_denice_metric_repairs: **199 passed**, 159.47 giây.
- Sau chỉnh sửa cuối tách current CE khỏi optimization loss cho capsule
  reliability: chạy lại test_denice_replay và test_denice: **131 passed**,
  73.62 giây. Đây là tập con chạy lại, không phải 330 kiểm thử khác nhau.
- Python compilation và git diff --check đạt. Kiểm thử chạy trên CPU,
  chưa kiểm chứng numerical parity của AMP/CUDA hoặc benchmark dataset thật.
- Môi trường Windows hiện tại có xung đột pyarrow khi import pandas/sklearn;
  lệnh test vô hiệu hóa pyarrow trong tiến trình test trước khi import hai
  thư viện, không sửa dependency hoặc mã huấn luyện để né lỗi này. Pytest
  cần quyền ngoài sandbox do Windows từ chối truy cập thư mục tạm tạo bởi test.

Đóng gói bằng `python tools/package_denice.py`. ZIP ở
`output/denice_source_20260925.zip` chứa fed_learning, script Kaggle, evaluator,
requirements, tài liệu và tests mới; mỗi file có SHA-256 trong SOURCE_MANIFEST.
Không chứa dữ liệu, checkpoint hay notebook đánh giá. Script Kaggle được cập
nhật trực tiếp theo yêu cầu ngày 2026-09-25. Regression tests cũ vẫn nằm trong
repo; ZIP chỉ chứa tests replay độc lập để tránh phụ thuộc notebook ngoài bundle.
