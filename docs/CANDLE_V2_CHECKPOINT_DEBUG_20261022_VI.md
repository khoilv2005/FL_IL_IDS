# CANDLE V2 — debug checkpoint sau results (14)

Ngày 22/10/2026. Training source `a7830e2`; kết quả full TEST đã khóa tại
Accuracy **3,1548%**, pooled Macro-F1 **2,4369%**. Báo cáo này không thay thế
hai metric đó bằng oracle hoặc bằng development accuracy.

## 1. Kết luận

**Nút thắt lớn đã xác định là cạnh tranh điểm số giữa các task trên những
subspace có fan-in khác nhau. Chưa có bằng chứng mất ranking phân biệt nội bộ
task cũ trong các cặp checkpoint được đối chiếu.**

- Đã chạy inference từ đủ sáu terminal checkpoints, tổng **447 model evaluations**.
- Với T0–T4, khi giữ chính tập class cạnh tranh lúc task vừa kết thúc, prediction
  tại T5 trùng **100%** prediction lúc đó trên toàn bộ cặp được kiểm tra.
- Hàng FC2 cũ và connection masks T0 không bị thay đổi qua các checkpoint đã so.
- Kiểm tra sâu 4 client cho thấy tọa độ FC1 T0 giữ nguyên chính xác. Tuy nhiên,
  features cấp phát về sau áp đảo độ lớn trên cả input T0.
- Một đối chứng không cần training, chuẩn hóa cosine theo fan-in từng row,
  cải thiện development accuracy nhưng vẫn gây nhiều break; chưa đạt điều kiện
  đưa vào inference chính thức.

Do đó không thể mô tả failure này đơn giản là “model quên hết class cũ”, cũng
không thể kết luận thêm distillation sẽ tự giải quyết được nó. Preservation
ranking và khả năng so sánh scores giữa các task là hai yêu cầu khác nhau.

## 2. Phạm vi và khả năng tái lập

Input: `C:\Users\khoak\Downloads\results (14).zip`. Đã trích sáu task archives,
kiểm tra terminal/graph checksums bằng loader hiện có, kiểm tra terminal tensors
FP32 và finite, rồi restore cùng đường code dùng bởi evaluator Kaggle.

Panel: **2.176 unique validation inputs = 64/class × 34 class**, chọn deterministic
từ locked validation role, owner có nhiều support trước, tie-break client ID,
khử trùng nội dung. Role-manifest hash khớp training. Mỗi checkpoint chỉ đánh giá
cumulative classes đã xuất hiện; mọi model ở cùng checkpoint dùng cùng panel.

- Không optimizer, không fitting, không điều chỉnh threshold.
- Không đọc final TEST. Historical validation chỉ được offline evaluator đọc,
  không đưa vào training, prototype bank hoặc live client.
- Đây là development source đã được sử dụng trước đây; không gọi là independent
  confirmation. Panel cân bằng không đại diện class frequencies của full TEST.
- Oracle task-mask dùng true task sau forward để chẩn đoán, không dùng trong
  inference triển khai.
- Main inference không dùng adapters, continual head hoặc donor model.
- Fingerprint weights, masks, ranks và CANDLE state trước/sau inference khớp.
- Có 260 zero-feature prediction rows trong 447 lượt model evaluation; không có
  available classifier row có effective weight bằng 0. Không suy ra mọi lỗi
  training đã được loại trừ từ những kiểm tra này.

## 3. Task-free so với task oracle trên cùng scores

Oracle available chỉ thu hẹp tập class **đã available** về đúng task; không mở
thêm class. Raw affine là đối chứng score không fitting, không phải classifier
đã train lại bằng raw logits.

| Checkpoint | Models | Task-free cosine | Oracle task, available classes | Raw affine |
|---|---:|---:|---:|---:|
| T0 | 50 | 44,65% | 44,65% | 36,52% |
| T1 | 60 | 50,55% | 62,91% | 40,21% |
| T2 | 70 | 31,61% | 62,08% | 25,97% |
| T3 | 80 | 19,19% | 56,97% | 16,20% |
| T4 | 89 | 9,60% | 50,51% | 9,88% |
| T5 | 98 | **5,16%** | **47,55%** | **5,66%** |

T5 gồm 213.248 prediction rows vì 98 model cùng chạy 2.176 input. Đây không phải
213.248 input độc lập. Membership và cumulative class scope thay đổi theo task;
không dùng riêng trend ở bảng này để tính forgetting của một cohort cố định.

### Phân rã lỗi T5

Mỗi prediction được xếp đúng một nhóm:

| Nhóm | Rows | Tỷ lệ trong errors |
|---|---:|---:|
| Correct | 11.011 | — |
| True class không available | 32.192 | 15,92% |
| True class available, native sai, oracle trong task đúng | 90.395 | **44,70%** |
| True class available, oracle trong task vẫn sai | 79.650 | 39,38% |

Cross-task competition là nhóm lỗi lớn nhất trong phân rã này, chưa chiếm đa số
tuyệt đối mọi errors. Oracle 47,55% cũng cho thấy chỉ giải quyết task selection
vẫn chưa đủ để đạt hiệu năng cao: còn within-task learning/transfer và coverage.

## 4. Paired retention: ranking cũ thực sự còn nguyên

Chỉ lấy client có cả checkpoint lúc task sinh ra và T5. Tại T5, giữ đúng class
mask lúc sinh ra để không thêm competing classes. So sánh cùng input content.

| Task của input | Paired clients | Prediction rows | Accuracy lúc sinh ra | T5 cùng mask | T5 task-free |
|---|---:|---:|---:|---:|---:|
| T0 | 48 | 18.432 | 44,49% | **44,49%** | 0,34% |
| T1 | 59 | 22.656 | 79,42% | **79,42%** | 0,98% |
| T2 | 70 | 26.880 | 76,35% | **76,35%** | 2,33% |
| T3 | 80 | 30.720 | 64,82% | **64,82%** | 4,48% |
| T4 | 87 | 33.408 | 34,79% | **34,79%** | 10,77% |

Không chỉ accuracy bằng nhau: **từng prediction đều trùng** ở cả năm hàng.
Điều này chứng minh retention của within-task decision trên panel được đo,
không chứng minh feature information của mọi input/distribution đều giữ nguyên.

## 5. Cơ chế: features mới áp đảo classifier cũ

Code `candle_v2.logits()` dùng effective weights có connection mask nhưng chuẩn
hóa feature trên toàn bộ FC1 coordinates đã allocate:

\[
s_c(h)=10\frac{(M_c\odot w_c)^\top h}
{\|M_c\odot w_c\|_2\,\|h\|_2}.
\]

Row cũ có fan-in nhỏ đã khóa; row mới có thể đọc nhiều coordinates được allocate
về sau. Feature mới có thể kích hoạt rất mạnh cả trên input cũ mà preservation
của mature features vẫn hoạt động đúng.

### Phân rã chính xác ở 4 client đã khóa trước phép đo

So T0 với T5 trên 384 input T0. Chia FC1 thành coordinates T0 và coordinates cấp
phát thêm. Tổng hai phần dot-product tái tạo original score, sai số tối đa
`3,34e-6`.

| Client | T0 / thêm FC1 coords | Max change features T0 | Mean norm phần thêm / norm toàn feature | Score winner: phần cũ / phần thêm |
|---|---:|---:|---:|---:|
| 4 | 40 / 136 | **0** | 99,9593% | 0,0155 / 7,3116 |
| 9 | 40 / 128 | **0** | 99,9753% | 0,0118 / 8,5647 |
| 65 | 32 / 104 | **0** | 99,9369% | 0,0305 / 7,7139 |
| 98 | 48 / 160 | **0** | 99,9444% | 0,0204 / 6,6703 |

Norm ratio trên không phải tỷ lệ cộng tuyến tính giữa hai norm; nó mô tả mức độ
phần thêm áp đảo norm toàn vector. Toàn vector có cosine drift lớn so với T0
không đồng nghĩa old coordinates bị thay đổi: ở đây chúng giữ nguyên chính xác.

**Cần phân biệt:** global `||h||` là hệ số chung cho mọi class trong một sample;
bỏ riêng hệ số này không đổi argmax. Vấn đề là geometry/fan-in và độ lớn activation
khác nhau giữa old/new subspaces, không phải một lỗi chia số đơn thuần.

Preservation hiện tại chỉ softmax/KL trên old classes, trên current inputs và
teacher mỗi round. Nó không đặt ràng buộc trực tiếp lên mức cạnh tranh new-vs-old
trên old inputs. Paired retention cho thấy giữ ranking cũ chưa giữ được cumulative
prediction. Từ đó chưa thể quy mọi failure cho riêng hệ số preservation.

## 6. Một đối chứng sửa geometry, không training

Khóa một thay đổi duy nhất trước inference: thay feature norm toàn vector bằng
feature norm trên fan-in của từng classifier row:

\[
s_c^{row}(h)=10\frac{(M_c\odot w_c)^\top h}
{\|M_c\odot w_c\|_2\,\|M_c\odot h\|_2}.
\]

Không đổi weights/masks/availability, không dùng true task trong prediction,
không chọn threshold, không import prototype mới. Chạy đủ 98 model cùng panel.

| Development metric | Original cosine | Cosine theo row mask |
|---|---:|---:|
| Accuracy | 5,1635% | **15,1331%** |
| Pooled Macro-F1 | 3,6441% | **15,7857%** |
| Old T0–T4 recall gộp trên panel | 3,2132% | **15,7308%** |
| New T5 recall gộp trên panel | 19,7903% | **10,6505%** |
| Predictions vào T5 | 72,5414% | 31,6467% |

Rescue **25.162**, break **3.902**, net correct **+21.260** trên **213.248**
prediction rows. Accounting khớp `delta_accuracy=(rescue-break)/N`.

Đối chứng này hỗ trợ giả thuyết mismatch geometry/fan-in có đóng góp thực sự,
nhưng **chưa phải production fix**: còn nhiều breaks, giảm new-class performance
và accuracy vẫn thấp. Không thay main runner, không đổi checkpoint, không
rerun full TEST và không tuyên bố full TEST đã tăng từ 3,15% lên 15,13%.

## 7. Còn lỗi học trong task, không chỉ cumulative scoring

Ngay tại thời điểm task vừa học, dùng oracle task-mask, current task classes đã
available nhưng không có local data chỉ đạt recall khoảng **19,19% ở T4** và
**24,98% ở T5**. Nhóm locally owned tương ứng cũng chỉ đạt **42,37% / 29,05%**.

Điều này ngăn kết luận “chỉ cần chữa scoring là toàn bộ V2 tốt”. Đồng thời nhóm
nonowned có thể nhận kiến thức qua native aggregation/inherited state; chưa có
matched training ablation để quy gain hoặc failure riêng cho virtual prototypes.

## 8. Quyết định sau debug

1. Nút thắt đầu tiên cần sửa là **cumulative score competition phù hợp với masks
   và capacity đang tăng**, không mặc định thêm distillation để giữ old ranking.
2. Giữ row-mask cosine là diagnostic variant; chưa commit nó thành inference
   mặc định vì negative transfer vẫn lớn.
3. Một sửa đổi tiếp theo cần vượt original trên cùng development protocol,
   báo riêng old/new performance và rescue/break; không dùng true task hoặc
   final TEST để chọn cấu hình.
4. Chưa full retrain. Cần một control tách cumulative inference khỏi prototype
   learning trước khi quy trách nhiệm hoặc đổi đồng thời nhiều thành phần.

Đây là debug bằng checkpoint đã có, không phải một training campaign mới.

## Receipts và lệnh tái lập

Local receipts: `audit_denice/candle_v2_checkpoint_debug_20261022_v1/`.
Public compact receipt: `artifacts/candle_v2_checkpoint_debug_20261022.json`.
Figure: `artifacts/candle_v2_checkpoint_debug_20261022.png` và `.svg`.

```powershell
.venv-denice-gpu\Scripts\python.exe -X utf8 -u tools/debug_candle_v2_checkpoints.py --inputs audit_denice/candle_v2_checkpoint_debug_20261022_v1/inputs --out <NEW_OUTPUT>
.venv-denice-gpu\Scripts\python.exe -X utf8 tools/summarize_candle_v2_debug.py --diagnostics <NEW_OUTPUT> --training-log audit_denice/candle_v2_results14_20261022_v2/candle_v2_training.jsonl --out <NEW_SUMMARY_JSON>
.venv-denice-gpu\Scripts\python.exe -X utf8 -u tools/debug_candle_v2_feature_contributions.py --inputs audit_denice/candle_v2_checkpoint_debug_20261022_v1/inputs --panel <NEW_OUTPUT>/development_panel.npz --out <NEW_CONTRIBUTIONS_OUTPUT>
.venv-denice-gpu\Scripts\python.exe -X utf8 -u tools/debug_candle_v2_masked_cosine.py --checkpoint audit_denice/candle_v2_checkpoint_debug_20261022_v1/inputs/checkpoint_task_5_all_rounds.zip --panel <NEW_OUTPUT>/development_panel.npz --out <NEW_SCORE_ABLATION_OUTPUT>
```

Các placeholder phải thay bằng path mới. Không commit raw validation inputs,
checkpoints hoặc per-sample scores vào Git.
