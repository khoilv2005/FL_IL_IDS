# DeNICE V2 — Cluster Model, Cluster Router và Full-Test Evaluation

## Trạng thái

Specification được chốt ngày 11/10/2026. P0 quản lý cluster state đã đạt 44/44
structural unit tests; xem [báo cáo implementation P0](DENICE_V2_P0_IMPLEMENTATION_20261011_VI.md).
V2 chưa được triển khai trong training runner hiện tại. DeNICE V1 và các số audit cũ được
giữ nguyên. Không tự chuyển notebook training V1 thành V2.

P1 pure aggregation engine đã đạt 30/30 tests (P0+P1: 74/74 PASS);
xem [báo cáo P1 và dry-run partition](DENICE_V2_P1_AGGREGATION_20261011_VI.md).
Native workers/BN collector chưa tích hợp; chưa có kết quả training V2.

## 1. Đơn vị aggregation và inference

- K cluster không chồng lấn; mỗi client tham gia thuộc đúng một cluster.
- Membership ban đầu cố định trong một task.
- Mỗi cluster giữ một model, một router và một coordinator tạm thời.
- Coordinator là client có nhiều class **đã học local** nhất theo Class
  Knowledge Matrix có provenance; class kế thừa/availability không tự trở
  thành local ownership. Tie-break phải deterministic và khóa trước test.
- Coordinator và aggregation weights là hai cơ chế độc lập. Leader không
  được ưu tiên weights chỉ vì được chọn làm leader.
- Không có server tổng hợp toàn federation/global model. Đây là
  **semi-decentralized**, có temporary server trong từng cluster.
- Không generative model, mimic data hoặc inter-cluster aggregation trong
  phiên bản đầu.

## 2. Một round

```
Task t / Round r
  1. Xác định partition K cluster; giữ membership trong task
  2. Cập nhật Class Knowledge Matrix từ metadata hợp lệ
  3. Chọn temporary server
  4. Broadcast cluster state của round trước
  5. Local training trên current Non-IID data
  6. Class-wise + mask-aware aggregation
  7. Tạo cluster model và router tương thích
  8. Đồng bộ toàn bộ cluster state
  9. Validation trên development hợp lệ và broadcast
 10. Checkpoint / round tiếp theo
```

Clients phải bắt đầu round từ cùng reference/version của cluster. Đồng bộ
không chỉ gồm weights: phải bao gồm masks, neuron ages, BN state, router
thresholds/memory/feature masks, class registration và metadata phiên bản.
Chỉ aggregate các contribution tương thích; bảo vệ mature parameters.
FC2 class-wise aggregation chuẩn hóa trên contributor đủ điều kiện cho row
đó; không dùng class metadata đơn thuần làm competence certificate.

K được người dùng chốt **tự động từ clustering ở đầu task**, không ép K=5.
Partition algorithm, leader tie-break, weighting, BN/maturity merge và state
handoff giữa tasks cần contract cụ thể trước implementation. Xem
[kế hoạch implementation](../plans/denice_v2_implementation_plan_20261011.md)
cho các module, engineering proposals và gates triển khai.

## 3. Cluster router và replay-free

Mỗi cluster chỉ có một router dùng cùng cluster model. Không gộp sketches
từ encoder/calibration không tương thích.

- Current-task memory có thể được tạo bằng shared encoder trên current data
  hợp lệ tại từng client, rồi trao đổi sketches và metadata phù hợp.
- Historical raw train/CAL không được đọc lại để tạo hoặc refit memory.
- Routing representation phải được bảo vệ qua tasks; giữ nguyên neuron IDs
  không đủ nếu upstream parameters, masks hoặc BN làm activation thay đổi.
- Kiểm tra activation drift, thresholds, memory schema và feature-space
  versions sau aggregation. Không xem hash/shape compatibility là bằng
  chứng functional consistency.
- Khi đổi membership/leader cần explicit state handoff và compatibility
  check; không reset/merge historical router tùy ý.

Router consistency là gate nghiên cứu/engineering chưa được giải quyết bởi
việc chốt evaluation protocol.

## 4. Evaluation protocol chính — mỗi predictor chạy cùng FULL TEST

Tại checkpoint Task t, định nghĩa:

\[
\mathcal T_{\le t}=\{(x,y)\in\mathcal T: y\in\cup_{s=0}^t C_s\},
\qquad N_t=|\mathcal T_{\le t}|.
\]

Đây là cumulative test của **tất cả class đã xuất hiện toàn hệ thống**, không
lọc theo classes mà một cluster đã học. T5 dùng toàn test của 34 class.

Mỗi cluster predictor f_k (model + router) đánh giá **đúng cùng N_t rows**:

\[
Acc_{k,t}=\frac{1}{N_t}\sum_{(x,y)\in\mathcal T_{\le t}}
\mathbf 1[f_{k,t}(x)=y].
\]

Metric chính:

\[
\boxed{Acc_{V2,t}=\frac{1}{K_t}\sum_{k=1}^{K_t}Acc_{k,t}}
\]

\[
\boxed{MacroF1_{V2,t}=\frac{1}{K_t}\sum_{k=1}^{K_t}MacroF1_{k,t}}
\]

Macro-F1 của từng predictor được tính trên fixed class universe
\(\cup_{s\le t}C_s\), zero_division=0. T5 dùng đủ 34 class. Phải báo support
và class-wise metrics, không bỏ class chưa được predictor học/đăng ký.

**Không cộng confusion matrices của K models rồi gọi Macro-F1 tính từ đó
là Mean Macro-F1.** Đây là hai đại lượng khác nhau. Accuracy tính từ tổng
correct/(K_t N_t) bằng mean accuracy khi mọi predictor có cùng N_t, nhưng
K_t N_t là số prediction events, không phải số unique test samples.

### Quy tắc dữ liệu và inference

- Không chia test theo cluster/client hoặc lấy union test shards của members.
- Không subsample, cân bằng lại hoặc lọc test theo local support.
- Tất cả cluster dùng cùng row IDs/content, preprocessing và class mapping.
- Chỉ đưa x và metadata inference được phép vào predictor; không true task
  hoặc true class. Y chỉ được evaluator đọc để tính metrics sau prediction.
- Không lọc test rows ngoài coverage. Native inference masks là phần của
  predictor; không tự mở class mask chỉ để làm đẹp evaluation.
- Không dùng test để chọn leader, partition, weights, router hoặc thresholds.
- Nếu predictor thiếu state hoặc eval không đủ N_t rows: đánh dấu incomplete;
  không âm thầm bỏ cluster lỗi rồi lấy mean trên các cluster còn lại.
- Report per-cluster metrics, K_t, member count, coordinator ID, checkpoint
  và router/model hashes. Metric chính không weight theo member count.

Client-weighted accuracy là metric phụ:

\[
Acc_{client\text{-}weighted,t}
=\frac{\sum_k m_{k,t}Acc_{k,t}}{\sum_k m_{k,t}},
\]

trong đó m_k là số members. Đặt tên riêng, không thay headline mean-cluster.

## 5. Cost accounting

Mỗi checkpoint có N_t unique samples và K_t N_t prediction events. Ví dụ
K=5 ở T5 của dataset hiện tại: 13.505.771 unique rows,
**67.528.855 prediction events**. Có 5 predictors được load/evaluate.

V1 audit cũ dùng disjoint receiver shards nên toàn run chỉ có N prediction
events; V2 full-test-per-cluster có K N. Ít model hơn không tự chứng minh
evaluation nhanh hơn protocol cũ. Đo forwards, wall time, peak memory và
communication; test data có thể được cache/load một lần và stream theo batch.

Không xem K N events là các mẫu độc lập khi tính confidence intervals: các
predictors dùng cùng test rows. Report uncertainty theo protocol đã khóa,
đặc biệt các independent training seeds trong full benchmark.

## 6. So sánh với DeNICE V1

**31,2446% BinarySelf / 37,1463% MulticlassSelf cũ dùng disjoint receiver
shards. Không dùng trực tiếp làm baseline cho headline V2 mới.**

Đánh giá lại từng local predictor V1 trên cùng \(\mathcal T_{\le t}\), giữ
nguyên inference policy của từng variant. Báo rõ mean-model accuracy/F1 của
V1 và mean-cluster accuracy/F1 của V2; số predictors và cost khác nhau là một
phần thay đổi phương pháp. Binary router gốc và multiclass variant phải tách
riêng tên. Không gọi multiclass variant là V1 binary nguyên gốc.

Nếu cluster sizes chênh lệch, nên bổ sung control V1 với cùng partition V2:
mean local-model score trong mỗi group, rồi mean đều giữa K groups. Đây là
**cluster-matched V1 summary**, không phải tạo thêm ensemble/cluster model
cho V1. Chốt cách trình bày controls trước khi đọc benchmark test metrics.

Full benchmark V1/V2 giữ cùng Non-IID split, clean data roles, tasks,
backbone, local training budget và independent seed protocol. Communication
bao gồm election/metadata, local uploads, router exchange, broadcast và
state migration; không chỉ tính classifier-row bytes.

## 7. Triển khai theo giai đoạn

| Phase | Mục tiêu | Gate |
|---|---|---|
| P0 | Partition, Class Knowledge Matrix, leader và cluster state | Membership không chồng lấn, ownership đúng, broadcast/restore cùng state |
| P1 | Class-wise mask-aware aggregation | Reference/version đúng; masks/maturity/BN nhất quán; leader role không bias weights |
| P2 | Shared cluster router | Compatible feature space, no historical raw replay, kiểm tra drift và task transition |
| P3 | Full-test-per-predictor evaluator | Mỗi predictor đủ cùng N_t rows; mean metrics đúng; label-blind; per-cluster receipts |
| P4 | V1/V2 full benchmark | Protocol/config khóa; nhiều independent seeds; báo forgetting và tổng cost |

Forgetting cần task-boundary metrics theo identity/state lineage đã xác định.
Nếu membership/cluster identity đổi giữa tasks, phải chốt lineage và công thức
so sánh trước khi gọi biến động accuracy là forgetting. Evaluation cadence
phải cấu hình trước run; document này không tự bật full test ở mọi round.

Không khởi động full training trước khi có model/router chung hợp lệ. P0 structural
tests và P1 pure engine tests đã PASS. Native P1 worker/moment collector đã PASS
one-round smoke trên hai client cùng cluster bằng BASE thật; 89/89 regression
tests PASS. Xem [báo cáo native P1](DENICE_V2_NATIVE_P1_SMOKE_20261011_VI.md).
P2 primitives/shadow tests đã triển khai, tổng regression 103/103 PASS. Capacity
preflight thực tế **NO-GO cho whole-tensor closure**: khóa 100% backbone;
[báo cáo P2](DENICE_V2_P2_ROUTER_CAPACITY_20261012_VI.md).
P2 chưa PASS, native two-task smoke chưa chạy. P3/full multi-task runner chưa
hoàn thành; chưa có empirical historical router consistency hoặc accuracy benchmark V2.
