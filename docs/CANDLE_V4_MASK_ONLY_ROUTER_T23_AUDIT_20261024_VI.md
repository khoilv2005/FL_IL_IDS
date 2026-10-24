# CANDLE V4 — Mask-only và router T2/T3

## Kết quả control trên toàn bộ TEST

100 clients, 13.505.771 mẫu, cùng partition seed523687, checkpoint cuối T5 của run cũ. Không train lại encoder/expert, không fit bằng TEST, không đổi threshold0,5. Multiclass router fit bằng saved activation memory như lượt readout trước.

| Cấu hình | Pooled Accuracy | Pooled Macro-F1 |
|---|---:|---:|
| V1 multiclass, local task class mask | 38,44% | 19,87% |
| **Task-mask-only** | **41,78%** | **21,68%** |
| V4 scoped + multiclass | **42,91%** | **23,04%** |

Task-mask-only chỉ mở tất cả class của task do input router dự đoán; không biết true task, không import expert/head, không đổi weights, neuron ranks, connection masks hoặc adapters. Ví dụ router chọn T2 thì cho FC2 cạnh tranh giữa12–17. Router estimator và thresholds giữ nguyên.

Mask-only rescue476.783 / break25.619 so với native, net +451.164 đúng, +3,3405pp Accuracy. V4 scoped rescue258.227 / break106.134 so với Mask-only, net **+152.093 đúng**, +1,1261pp Accuracy và +1,3589pp Macro-F1.

Mask-only đạt74,79% mức tăng net accuracy của V4 so với V1. Đây là tỷ lệ chênh lệch endpoint, không phải khẳng định74,79% từng prediction gain có cùng nguyên nhân. Vì vậy không được quy toàn bộ +4,47pp của V4 cho expert learning: phần lớn lợi ích đã xuất hiện khi chỉ sửa class availability. Expert vẫn có gain riêng trên control này; 62 clients tăng, 37 giảm, 1 không đổi khi so với Mask-only.

Control dùng một native forward với global predicted-task mask, rồi áp lại local mask trên cùng logits để kiểm tra baseline. Tính lại V4 prediction từ native fallback và cached expert overrides, không train/fuse lại parameters. Task confusion matrix khớp reference ở100/100 clients; native class confusion matrix khớp tuyệt đối ở99/100, client còn lại L1 discrepancy6. Pooled accuracy baseline và V4 khớp readout trước; Macro-F1 sai khác dưới1e-7. Batch2048 được khóa trước run, fallback512 nếu không tái lập trong tolerance; không client nào cần fallback. Ghi đầy đủ numerical discrepancy trong JSON, không gọi đó là tái lập mọi prediction tuyệt đối.

## Panel router được chọn bằng metadata

Chọn trước8 client ID nhỏ nhất có cả T2/T3 memory tại checkpoint cuối T3 và cuối T5: **1,3,5,6,7,8,9,11**, từ59 client đủ điều kiện. Lấy tối đa128 TEST samples/class trong12–23 theo RNG cố định42+client; tổng11.560 mẫu, gồm7.444 mẫu class từng học local và4.116 mẫu class chưa học local. Không chọn theo score/accuracy.

Panel này dùng để phân tách cơ chế, không đại diện ngẫu nhiên cho100 clients. Kết quả panel là task routing accuracy, không phải class accuracy ở bảng trên.

| Encoder | Router memory/classifier | Routing Accuracy trên T2/T3 panel |
|---|---|---:|
| Cuối T3 | T3 bank, multiclass fit | 51,82% |
| Cuối T5 | Cùng T3 bank/classifier | **51,82%** |
| Cuối T3 | T5 bank, multiclass fit | 49,26% |
| Cuối T5 | T5 bank, multiclass fit | **49,26%** |

Thresholds/feature policy giống nhau, shared T2/T3 sketches không đổi. Feature subspace thực sự sử dụng có102 coordinates ở7 clients,85 ởClient9 trong tổng548 coordinates.

**Trong effective routing subspace, continuous activation cũng như binary bits trên cùng input không đổi từ T3 sang T5:** maximum selected continuous drift0, binary flip fraction0. Các activation ngoài mask có thay đổi, nhưng bị loại khỏi router. Không có bằng chứng encoder drift làm giảm routing trên panel/interval này; không suy rộng kết luận này cho mọi task/client/interval chưa đo.

Mức giảm51,82%→49,26% đến từ thay đổi memory bank/refit classifier khi thêm các task sau, không phải encoder T3→T5 trên selected features. Chất lượng routing đã yếu ngay ởT3. Không nên thêm freezing/coordinate protection để sửa một drift không quan sát được.

## Memory có đủ tính phân biệt không?

- Train accuracy của multiclass LR trên3.546 saved sketches:75,69%.
- Holdout20% saved sketches, split cố định theo task, cùng LR config:58,32%. Đây không phải holdout raw-data độc lập.
- 789 memory rows có binary pattern trùng một pattern mang task label khác. Trên chính memory này, mọi deterministic classifier chỉ đọc pattern có trần empirical90,41% vì conflicting labels. Không coi90,41% là dự báo TEST accuracy.
- Trên panel TEST, class từng học local đạt62,64% task routing; class chưa học local chỉ25,07% khi dùng final bank.
- Nếu chỉ cho chọn giữa T2 và T3, routing đạt69,15%; đây là candidate-set oracle diagnostic, không phải global inference hoặc kết quả triển khai.

Nguồn code giải thích tradeoff: trong chế độ sketch, `decentralized_denice_il.py` chọn `stable_feature_mask` từ units đã allocate ở **task đầu của từng client**, rồi giữ permanent routing subspace. Nó bảo toàn representation, nhưng không tự thêm feature chuyên biệt cho các class xuất hiện muộn. Sau đó `binarize_layer_activations` còn nén mỗi activation thành một bit theo threshold của layer. Stability đã được giữ ở panel, discrimination/coverage vẫn yếu.

## Một nonlinear probe cố định

Sau khi thấy LR fit memory kém, chạy đúng một probe: kNN k=3, Hamming distance, distance weighting, fit saved sketches; không hyperparameter sweep, không fit/tune bằng TEST, không thay router chính thức.

| Model | Saved-sketch holdout | T2/T3 TEST panel |
|---|---:|---:|
| Multiclass LR | 58,32% | **49,26%** |
| kNN fixed3 | 60,03% | 48,70% |

kNN không tạo cải thiện TEST trên panel này. Không có cơ sở thay LR bằng kNN chỉ để cứu routing. Đây là bounded development probe, không phải benchmark hoặc bằng chứng phủ định mọi nonlinear router.

## Quyết định sau audit

1. Giữ Task-mask-only là baseline bắt buộc, cùng router/data/protocol với V4. Nó rẻ hơn expert bank và giải thích phần lớn mức gain đã quan sát.
2. Không thêm freezing và không thay sang kNN từ kết quả hiện có.
3. Nếu mục tiêu vượt50%, nghiên cứu representation/memory router đủ giàu cho class xuất hiện muộn, thay vì chỉ đổi classifier trên cùng binary sketches. Một hướng tối thiểu cần thiết kế contract riêng là lưu **continuous class-aware memory trên routing space ổn định** trong current task, thay cho việc bỏ amplitude/class identity khi chỉ lưu binary task sketches. Hướng này chưa được chứng minh; first-task subspace vẫn có thể thiếu thông tin nên không được hứa nó sẽ giải quyết toàn bộ vấn đề.
4. Continuous historical activations không thể phục hồi từ binary sketches. Không dùng TEST để tạo memory mới hoặc lấy lại old TRAIN ởcheckpoint T5. Cần thu memory hợp lệ từ một fresh stream khi task đang current.
5. Donor handoff v6 đã sửa về kỹ thuật nhưng chưa được benchmark; không gán các kết quả42,91% cho handoff mới và không loại expert theo TEST.

Không thay runtime/notebook hoặc chạy full training trong lượt audit này. Task coverage thiếu22,52% và routing ceiling45,52% của checkpoint cũ vẫn là giới hạn riêng, không phải lỗi chia dữ liệu.

## Artifacts

`artifacts/candle_v4_mask_router_readout_20261024.json`: summary, configs và hashes. Chi tiết local ở `audit_denice/candle_v4_mask_router_audit_20261024/`: `mask_only.json`, `mask_only_per_client.csv`, cached native/mask/scoped/route predictions của100 clients, `router_t23_drift.json`, `router_t23_summary.json`, `router_panel_client_<id>.npz`, `knn_memory_probe.json` và logs.

Scripts: `tools/audit_candle_v4_mask_only_full.py`, `tools/audit_candle_router_t23_drift.py`, `tools/probe_candle_router_knn_memory.py`. Source restore lấy từ package của run gốc; metadata-selected panel dùng `checkpoint_task_3.pt` trích từ task3 archive và continuation cuốiT5.
