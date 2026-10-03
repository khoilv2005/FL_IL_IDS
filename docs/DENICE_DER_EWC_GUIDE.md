# Hai phiên bản DENICE: DER/DER++ và EWC

Ngày: 03/10/2026. Cả hai vẫn dùng `algorithm="denice"`, `mode="decentralized"`.
Đây là hai cơ chế học liên tục độc lập trong cùng giao thức DENICE, không thêm server.
Không có bảo đảm đạt một mức accuracy trước khi đo trên bộ IDS thực.

## 1. Chọn đúng phiên bản

| File chạy độc lập trên Kaggle | Mặc định | Lưu dữ liệu cũ để huấn luyện? | Regularization |
|---|---|---|---|
| `train_denice_der_kaggle.py` | DER++ | Có: reservoir riêng từng client | Khớp logits + CE replay |
| `train_denice_ewc_kaggle.py` | EWC nhiều task | Không: không có raw replay | Fisher và trọng số neo |

Launcher DER hỗ trợ `DENICE_VARIANT="der"` để chạy DER thuần hoặc `"derpp"`
để chạy DER++. Launcher EWC dùng `"ewc"`. Các JSON preset tương ứng nằm trong
`configs/denice_der.json`, `configs/denice_derpp.json`, `configs/denice_ewc.json`.
Tên phương pháp trong log là `denice_cl_method`; không đổi tên thuật toán.

Hai file được sinh từ launcher chung bằng `python tools/build_denice_launchers.py`.
Nếu sửa logic launcher chung, sinh lại hai file trước khi commit. Không sửa riêng
một bản rồi chạy generator vì thao tác đó sẽ ghi đè thay đổi riêng.

## 2. Cơ chế bám theo tài liệu bạn cung cấp

### DER và DER++: DER.pdf, trang 3–4, Eq. (5), (6), Algorithm 1–2

1. Forward batch hiện tại và giữ logits **trước** khi optimizer cập nhật.
2. Lấy batch ngẫu nhiên đều từ reservoir của chính client, tính squared-logit loss.
3. Với DER++, lấy **batch độc lập thứ hai** từ cùng reservoir để tính CE theo nhãn cũ.
4. Cộng các loss, backward và optimizer step.
5. Sau bước thành công, đưa batch hiện tại cùng logits đã giữ vào reservoir.
   Nếu AMP bỏ qua bước do overflow thì không ghi batch đó.

Mẫu thứ n trong luồng dùng Algorithm R: khi bộ nhớ đầy, rút số nguyên đều
trong [0,n-1]; chỉ thay một slot nếu số đó nhỏ hơn capacity. Không chia quota
theo class. Mỗi lần trình bày mẫu trong epoch/round là một lần đến của luồng;
đây không phải lấy mẫu đều theo ID dữ liệu duy nhất. Không làm mới teacher logits
ở cuối task. Bộ nhớ có thể thiếu class hiếm; tăng capacity là một ablation cần đo.

DER: `L = CE_current + alpha * mean_batch(sum_class((logit-logit_saved)^2))`.

DER++: thêm `beta * CE_replay`, lấy trên batch thứ hai độc lập.

Mặc định `alpha=beta=0.5`, squared norm cộng theo class như biểu thức trong bài.
`denice_der_reduction="mean"` chia thêm cho số tọa độ để thử biến thể MSE trung
bình; không so trực tiếp alpha giữa hai reduction mà không xét hệ số số class.
Với 34 tọa độ, mean cần alpha lớn gấp 34 để có cùng độ mạnh như sum trên cùng dữ liệu.

Các thích nghi DENICE được công khai: không augmentation ảnh trên đặc trưng IDS;
replay dùng BN/dropout ở chế độ inference để tránh làm nhiễu thống kê, nhưng vẫn
có gradient; GRU xử lý riêng để backward được với cuDNN. Mặc định khớp mọi logit
trong output cố định; `seen` là ablation chỉ khớp class đã biết lúc lưu. DER vẫn
lưu nhãn để phục vụ thống kê/router và có thể dùng DER++, dù DER thuần không dùng
nhãn cũ trong replay loss. Không gọi đây là tái hiện nguyên xi toàn bộ thí nghiệm bài báo.

### EWC: ewc.pdf, trang 3, Eq. (3)

Cuối task, sau cập nhật peer và hiệu chỉnh BN, mỗi client:

1. Lấy tối đa `denice_ewc_fisher_samples` mẫu **train riêng** của task hiện tại.
2. Tính gradient log-probability từng mẫu, bình phương rồi lấy trung bình:
   `F_i = mean_n[(d log p(y_n|x_n) / d theta_i)^2]`.
3. Lưu Fisher và trọng số neo; không giữ các mẫu dùng tính Fisher.

Task tiếp theo dùng:

`L = CE_current + lambda/2 * sum_old_tasks sum_parameters F_task * (theta-anchor_task)^2`.

Không chia penalty cho tổng Fisher hoặc số tham số như elastic loss cũ.
`fisher_labels="model"` lấy nhãn mẫu từ phân phối dự đoán để ước lượng Fisher;
`"empirical"` dùng nhãn thật, là ablation empirical Fisher. Bình phương gradient
trung bình của cả batch không được dùng thay cho trung bình bình phương gradient.

`ewc_mode="separate"` lưu từng cặp Fisher/anchor, cộng tất cả penalty task cũ.
`"online"` là biến thể tiết kiệm bộ nhớ: tích lũy Fisher có decay và neo vào trọng
số task gần nhất. Online không tương đương với tổng penalty nhiều anchor của bản gốc.
Lambda=0 là đối chứng tắt penalty, vẫn giữ các thiết lập còn lại để đo công bằng.

### Điểm khác DENICE cũ và điều giữ nguyên

- Hai preset cho phép neuron đã cấp phát, kể cả mature, học **cục bộ**; neuron
  dự phòng/retired vẫn không nhận gradient. Nếu tiếp tục đóng băng toàn bộ neuron
  cũ, phần lớn EWC penalty sẽ không giúp bảo vệ một tọa độ đang thay đổi nào.
- Age, connection masks, CANC, adapter, clustering và giao thức peer vẫn dùng
  cơ chế DENICE. Mature rows không nhận peer delta; cập nhật cục bộ trên các row
  đó được giữ lại thay vì bị aggregation quay về trọng số trước local training.
- Mất bảo đảm bất biến hoàn toàn hàm cũ; chống quên chuyển một phần sang DER/EWC.
  BN mature vẫn giữ thống kê, nhưng thay đổi tham số có thể làm router cũ lệch.
- Fisher/anchor EWC và reservoir DER là riêng tư của từng client, không nằm trong
  capsule. Client mới không nhận lịch sử EWC/replay của client cho mượn model.
- Tắt experimental plasticity, residual MLP, LDA và transfer selection để không
  trộn nhiều thay đổi. DER và EWC mặc định đều tắt router replay cho dễ so sánh.
- Ba hệ số replay cũ `denice_replay_ce_weight`, `denice_replay_logit_weight`,
  `denice_replay_calibration_weight` không điều khiển bản DER. Dùng alpha/beta mới.

## 3. Chạy song song trên Kaggle, từng bước

### Bước 1 — Push đầy đủ mã

Commit/push các module mới và các file đã sửa, không chỉ hai launcher. Bao gồm
`denice_der.py`, `denice_ewc.py`, `denice_variants.py`, hai client, runner,
checkpoint helper và launcher. Nếu dùng main, push vào main; nếu nhánh khác,
đặt `DENICE_CODE_REF` đúng tên/ref. Hai lần chạy phải checkout cùng một commit.

Có thể dùng ZIP `output/denice_source_20261003_der_ewc.zip` thay GitHub: giải nén,
đặt `DENICE_CODE_DIR` đến thư mục chứa `fed_learning/` và hai launcher. Khi dùng
GitHub, đừng đặt `DENICE_CODE_DIR` hoặc để thư mục `fed_learning/` cũ cạnh script.

### Bước 2 — Tạo hai notebook độc lập

Notebook A: DENICE DER. Notebook B: DENICE EWC. Cùng dataset 100-clients, GPU,
seed và ngân sách round/epoch. Bật Internet nếu clone GitHub. Chạy đồng thời ở
hai session riêng khi quota Kaggle cho phép; không chạy hai biến thể trong hai
thread chung kernel vì chúng chia sẻ RNG, biến môi trường và GPU.

### Bước 3 — Cell cấu hình DER++

Chạy cell này **trước** launcher trong notebook A:

```python
import os, json
os.environ.pop("DENICE_CODE_DIR", None)  # bỏ dòng này nếu chủ động dùng ZIP local
os.environ["DENICE_CODE_REF"] = "main"  # tốt hơn: SHA commit đã push của thí nghiệm
os.environ["DENICE_VARIANT"] = "derpp"
os.environ["DENICE_SEED"] = "42"
os.environ["DENICE_TRAIN_PHASE"] = "5"  # chạy mới task 0–5
os.environ["DENICE_OUTPUT_DIR"] = "/kaggle/working/derpp_a05_b05_m1024_s42"
os.environ["DENICE_CONFIG_OVERRIDES"] = json.dumps({
    "denice_replay_capacity": 1024,
    "denice_replay_batch_size": 32,
    "denice_der_alpha": 0.5,
    "denice_der_beta": 0.5,
    "denice_der_reduction": "sum",
    "denice_cl_logit_scope": "all",
    "denice_cl_train_mature": True,
    "denice_eval_local_validation": True,
})
```

Sau đó dán **toàn bộ** `train_denice_der_kaggle.py` vào cell tiếp theo và chạy,
hoặc `%run /đường/dẫn/thực/tế/train_denice_der_kaggle.py` nếu đã upload dạng file.
Không dùng nguyên đường dẫn minh họa. Không cần upload `fed_learning/` nếu mã mới
đã push và launcher clone được GitHub.

### Bước 4 — Cell cấu hình EWC

Notebook B, trước launcher:

```python
import os, json
os.environ.pop("DENICE_CODE_DIR", None)
os.environ["DENICE_CODE_REF"] = "main"  # dùng cùng SHA với notebook A
os.environ["DENICE_VARIANT"] = "ewc"
os.environ["DENICE_SEED"] = "42"
os.environ["DENICE_TRAIN_PHASE"] = "5"
os.environ["DENICE_OUTPUT_DIR"] = "/kaggle/working/ewc_l100_f128_separate_s42"
os.environ["DENICE_CONFIG_OVERRIDES"] = json.dumps({
    "denice_ewc_lambda": 100.0,
    "denice_ewc_fisher_samples": 128,
    "denice_ewc_fisher_labels": "model",
    "denice_ewc_mode": "separate",
    "denice_cl_logit_scope": "all",
    "denice_cl_train_mature": True,
    "denice_eval_local_validation": True,
})
```

Chạy `train_denice_ewc_kaggle.py` ở cell kế tiếp. Preset tự tắt replay,
router replay và elastic loss cũ. Đừng bật lại chúng nếu đang đo EWC riêng.

### Bước 5 — Kiểm tra log trước khi chờ nhiều giờ

- `Training source` và `git_commit`: cùng commit mới ở hai notebook.
- `DENICE continual-learning controls`: method là `derpp` hoặc `ewc`, cùng
  các hệ số bạn vừa đặt. `source_audit.json` và `config.json` lưu cấu hình chạy.
- Đường dẫn output khác nhau. Đổi tên output ở mỗi thí nghiệm, đừng dùng chung
  `DENICE_OUTPUT_DIR`/`resume_output_dir` cho hai notebook.
- Bản DER ghi `stream_reservoir_derpp`, `stream_seen`; cuối task 0 đã có replay
  **ngay trong task**, không phải chỉ bắt đầu hoạt động ở task 1 như buffer cũ.
- EWC ghi `ewc_consolidation` cuối task và `regularization_loss` ở task sau.
  Penalty bằng 0 tại đúng trọng số neo là bình thường; kiểm tra các bước sau.
- `skipped_optimizer_steps` phản ánh bước AMP bị bỏ qua. Nhiều bước bị bỏ qua
  kéo dài cần kiểm tra loss/gradient trước khi tăng hệ số regularization.

## 4. Tham số DER: chỉnh ở đâu, chỉnh thế nào

Chỉnh dictionary trong `DENICE_CONFIG_OVERRIDES`, rồi chạy **mới từ task 0**.
Preset được áp dụng sau CONFIG gốc; vì vậy sửa các khóa trùng ngay trong CONFIG
gốc có thể bị preset ghi đè. Cell override là cách rõ ràng nhất. Viết lại cả
dictionary ở mỗi lần chạy để không sót tham số của thí nghiệm trước.

| Khóa | Mặc định | Các giá trị thử ban đầu | Ý nghĩa |
|---|---:|---|---|
| `denice_replay_capacity` | 1024 | 512, 1024, 2048, 4096 | Số mẫu tối đa **mỗi client**; tăng giúp phủ class nhưng tăng RAM/checkpoint |
| `denice_replay_batch_size` | 32 | 16, 32, 64, 128 | Mỗi batch replay; DER++ có hai lần lấy batch |
| `denice_der_alpha` | 0.5 | 0, 0.1, 0.5, 1.0 | Độ mạnh khớp logits; quá lớn có thể giữ cả sai số cũ và cản học mới |
| `denice_der_beta` | 0.5 | 0, 0.1, 0.5, 1.0 | CE trên mẫu cũ, chỉ dùng DER++ |
| `denice_der_reduction` | sum | sum / mean | Cộng hay trung bình squared error theo class |
| `denice_cl_logit_scope` | all | all / seen | Toàn bộ output hay class đã biết; seen là biến thể riêng |
| `denice_cl_train_mature` | True | True / False | Cho học cục bộ neuron mature hay giữ hard freeze làm đối chứng |
| `denice_router_replay_enabled` | False | False / True | DER có thể dùng buffer để cập nhật router; thử riêng sau khi chốt replay |

Ví dụ DER thuần: `DENICE_VARIANT="der"`; beta bị bỏ qua, không có replay CE.
Ví dụ ER đối chứng: `DENICE_VARIANT="derpp"`, alpha=0, beta=0.5.
Ví dụ không auxiliary replay loss: alpha=beta=0; bộ nhớ vẫn được thu thập và
tiêu thụ RNG, vì vậy đây là đối chứng loss chứ không phải bản không có bộ nhớ.

Quy trình đề xuất: giữ batch32, alpha=beta=.5 → thử capacity → cố định capacity
và thử alpha → cố định alpha và thử beta → sau đó mới thử batch/reduction/router.
Mỗi bước chỉ đổi một yếu tố. Các dải trên là gợi ý thí nghiệm, không phải tham số
tối ưu IDS được chứng minh bởi bài báo.

## 5. Tham số EWC

| Khóa | Mặc định | Các giá trị thử ban đầu | Ý nghĩa |
|---|---:|---|---|
| `denice_ewc_lambda` | 100 | 0, 10, 100, 1000, 10000 | 0 là đối chứng; tăng bảo vệ trọng số cũ mạnh hơn |
| `denice_ewc_fisher_samples` | 128 | 32, 64, 128, 256 | Mẫu train để ước lượng Fisher; tăng làm chậm do backward từng mẫu |
| `denice_ewc_fisher_labels` | model | model / empirical | Nhãn lấy từ model hay nhãn thật |
| `denice_ewc_mode` | separate | separate / online | Nhiều anchor theo bài, hoặc một anchor với Fisher tích lũy |
| `denice_ewc_decay` | 1.0 | 0.9, 0.95, 1.0 | Chỉ có tác dụng trong online; thấp hơn giảm ràng buộc lịch sử |
| `denice_cl_train_mature` | True | True / False | False là ablation hard freeze, có thể khiến phần lớn EWC không còn tác dụng |

Thứ tự: chạy lambda=0 và 100 cùng seed → quét lambda theo thang log → chốt
lambda rồi tăng Fisher samples → so model/empirical → cuối cùng so online.
Không coi `denice_fisher_samples` của capsule/CANC là Fisher của EWC: khóa mới
`denice_ewc_fisher_samples` điều khiển estimator độc lập này.

Nếu task mới học kém nhưng task cũ được giữ tốt, thử giảm lambda. Nếu quên cũ
nhiều, kiểm tra router và penalty thực sự có gradient trước khi tăng lambda.
Lambda lớn hơn không mặc nhiên tốt hơn. Giá trị lambda từ một dataset/bài báo
không chuyển nguyên sang đây vì tổng Fisher phụ thuộc model và tập dữ liệu.

Separate cần khoảng hai bản sao tham số cho mỗi task/client trên CPU, cùng
bản sao tạm của các penalty đang tính trên GPU. Online chỉ cần một cặp; dùng
online như một thí nghiệm riêng khi RAM/checkpoint của 100 clients quá lớn.

## 6. So sánh kết quả và tránh chọn tham số bằng test set

`validation_metrics.json` là kết quả cuối task trên các fold validation riêng
của từng client, được dựng lại bằng đúng phép chia đã loại khỏi optimizer train.
Nó chứa accuracy, macro-F1, route accuracy, per-task accuracy và forgetting.
Các fold này vốn cũng phục vụ capsule trong paper-mode DENICE; đây là validation
phát triển, không phải test độc lập. Mẫu validation không được đưa vào DER buffer
hoặc estimator EWC mới. Chỉ metric được tổng hợp; không có bước fit model trên
validation. Số liệu validation được lấy sau consolidation.

Để tune mà không đọc test set trong quá trình chạy, có thể đặt đồng thời:

```python
"denice_post_task_eval": False,
"denice_eval_final_round": False,
"eval_every": 999999,
"denice_eval_local_validation": True,
```

Giữ `denice_validation_fraction` > 0 (launcher mặc định 0.1); fold nhỏ có thể
làm metric dao động. Chọn cấu hình trên validation qua toàn bộ task, ưu tiên
macro-F1 và forgetting bên cạnh accuracy. Chốt tham số rồi bật lại test eval
để báo cáo một lần trên test set độc lập. Không chọn cấu hình dựa riêng task 0.

`round_metrics.json` ghi round cuối **trước** consolidation nếu bật; `training_history.json`
ghi `task_accuracies` sau consolidation. Đừng trộn hai loại accuracy trong cùng
bảng so sánh. Giữ cùng split, seed, batch train, learning rate, số round/epoch,
route mode và commit. Chạy ít nhất các seed 23/42/73 cho cấu hình cuối; báo cáo
trung bình và độ lệch chuẩn, không chỉ lấy seed tốt nhất. Các hệ số làm thay đổi
ngân sách replay/backward cần đi kèm thời gian và RAM khi báo cáo.

Thay đổi method, hệ số replay hoặc EWC cần chạy mới. Resume chỉ tiếp tục đúng
cấu hình đã lưu; runner từ chối đổi các control đó để tránh so sánh hai lịch sử
học khác nhau dưới cùng một tên thí nghiệm. Task-boundary continuation giữ RNG,
reservoir, tổng số lượt đến, Fisher/anchor và trạng thái riêng từng client.

## 7. Chạy thử tự động ở local hoặc Kaggle terminal

```sh
python tools/benchmark_denice_der_ewc.py --config experiment.json --methods der derpp ewc ewc_zero --seeds 23 42 73 --output-dir output/comparison
```

`experiment.json` là CONFIG đầy đủ có đường dẫn dataset phù hợp máy đang chạy.
`ewc_zero` giữ cách train EWC nhưng đặt lambda=0. Muốn sửa hệ số, tạo JSON chỉ
chứa override và thêm `--overrides overrides.json`. Thư mục output nên mới cho
mỗi lần; tool chạy tuần tự để tái lập, hai notebook ở phần 3 mới là chạy song song.
Tool chỉ ghi kết quả, không chọn winner theo test accuracy. Để tune validation,
đặt các khóa tắt test ở mục 6 trong file override.

Ví dụ chạy chỉ validation bằng file đã cung cấp:

```sh
python tools/benchmark_denice_der_ewc.py --config experiment.json --methods derpp ewc --overrides configs/denice_validation_only.json --seeds 42 --output-dir output/validation_comparison
```

Bỏ `--config` sẽ chạy dữ liệu tổng hợp nhỏ, **không phải IDS**. Kiểm tra chức năng
ba seed được lưu trong `DENICE_DER_EWC_SMOKE.json`; không suy luận mức tăng trên
100 clients từ số này. Bộ dữ liệu IDS thực chưa có trong workspace.

## 8. Căn cứ và giới hạn

- DER: tài liệu người dùng `E:/Study/DACN_KLTN/DER.pdf`, Algorithm 1–2 và Eq. 5–6;
  [bản công bố](https://arxiv.org/abs/2004.07211).
- EWC: `E:/Study/DACN_KLTN/ewc.pdf`, Eq. 3, tổng penalty khi chuyển sang task C;
  [bản công bố](https://arxiv.org/abs/1612.00796).
- Bài báo cung cấp cơ sở cơ chế, không chứng minh hai tích hợp DENICE này sẽ đạt
  99/98/87/70/70/70%. Router, non-IID, bộ nhớ, Fisher nhiễu và allocation vẫn có
  thể giới hạn kết quả. Đặc biệt EWC không sửa router bằng dữ liệu replay.
