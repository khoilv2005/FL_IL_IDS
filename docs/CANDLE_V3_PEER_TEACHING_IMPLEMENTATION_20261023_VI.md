# CANDLE V3 — triển khai peer teaching lần đầu

Cập nhật: có thêm [variant A Direct FC2](CANDLE_V3_DIRECT_FC2_VARIANT_20261023_VI.md)
để ưu tiên thử trước. Nội dung head-learning dưới đây mô tả variant C riêng biệt.

## Trạng thái

Đã viết các module P0/P1/P2 và hook vào cuối task của private runner V3.
Chưa chạy unit tests, smoke, training hoặc evaluation. Chưa có kết quả PASS hay
bằng chứng cải thiện Accuracy/Macro-F1. Không thay source/config/notebook V1.

Notebook `train_candle_v3_kaggle.ipynb` vẫn pin bản baseline clone đã công bố.
Chưa chuyển notebook này thành bản full training peer teaching.

## Các phần đã viết

Trong `candle_v3_runtime/fed_learning/strategies/decentralized/candle_v3/`:

| Module | Chức năng |
| --- | --- |
| `contracts.py` | Schema, receiver reference identity, thông số phiên học |
| `knowledge.py` | Local ownership, current support, acquired classes riêng biệt |
| `discovery.py` | Chọn donor theo marginal coverage trên graph được khai báo |
| `snapshot.py` | Frozen receiver, canonical masked-eval features, SHA-256 |
| `head_learning.py` | Class-balanced head-only SGD trên cùng reference |
| `head_aggregation.py` | Kiểm tra owner/reference/round rồi tổng hợp delta |
| `router_teaching.py` | Sketch từ receiver extractor và receiver thresholds |
| `verification.py` | Predicted task inference, accounting và acceptance current CAL |
| `transaction.py` | Shadow apply và commit/rollback head, masks, ranks, router, registry |
| `communication.py` | Serialized application messages và budget |
| `data.py` | BASE/FIT/CAL current-task view, hủy quyền truy cập khi đóng task |
| `service.py` | Phiên end-of-task có giới hạn và receipts |

## Hook trong native runner

Khi bật `candle_v3_teaching.enabled`:

1. Hook sau successful optimizer step ghi nhận đúng class có batch update thành công;
   batch AMP bị skip không được dùng để ghi nhận local learning.
2. Native training và masked aggregation V1 vẫn chạy bình thường.
3. Sau các thao tác task-finalization làm đổi topology, masks, BN và adapter path,
   tạo snapshot receiver. Không có thêm native aggregation sau phiên học này.
4. Teaching graph được khai báo là mọi directed pair giữa các active clients hiện
   đã tham gia discovery của task. Đây là lớp graph riêng; native graph ξ=0,8 không đổi.
5. Chọn tối đa hai donor, ba target; target phải thiếu trong current BASE receiver
   và FC2 row chưa allocate. Không ghi đè mature rows.
6. Availability-only FIT probe trước khi học. Thiếu FIT trả UNKNOWN; nếu một target
   đã đủ recall thì trả REGISTRATION_GAP và không học block đó. Bản đầu không tự cài
   registration-only hoặc tách/tìm donor khác để cứu block.
7. Receiver/donor học head trong receiver reference, tạo current BASE sketches,
   rồi chạy acceptance trên current CAL. Không dùng TEST/validation cho commit.
8. PASS mới commit. FAIL/UNKNOWN/REGISTRATION_GAP/DEFERRED không làm dừng native training.
9. Task-finalized checkpoint lưu cả metadata, registry, evidence và teaching ledger.

## Chi tiết head optimizer

Chỉ head tạm có gradient. Encoder ở eval mode, BN và dropout không thay đổi.
Canonical API tái sử dụng `DeNICEModel.penultimate_features()` và effective FC2
weight/bias masks theo inference; không dùng training-only learner gates.

Mỗi head round là một bước SGD từ cùng reference. Các gradient mini-batch được
cộng thành gradient local trước khi tổng hợp. Trọng số class bằng nghịch đảo số
mẫu BASE thực sự dùng trong phiên; contribution mass bảo đảm gradient gộp đúng
objective này. Receiver có 0 mẫu dương vẫn có trọng số qua mẫu âm. Không sử dụng
reserve-neuron alpha của native aggregation cho head session.

Phiên đầu dùng FC1 coordinates đã mature; không khóa thêm backbone và không mở
toàn bộ neuron chưa được bảo vệ. Head mới được copy vào FC2 row với permission
mask tương ứng và rank mature sau commit. Điều này chưa chứng minh survival của
upstream features qua task sau; native retention vẫn cần được đo.

## Controls và gate

Năm điều kiện cùng current CAL panel:

1. Availability-only: weights/masks giữ nguyên, chỉ mở class.
2. Router-only: thêm receiver-space sketches, không thay classifier weights/masks.
3. Mask-only: cùng masks/ranks/router của phiên Full, nhưng head giữ initialization.
4. Donor-only fitting: head học từ donor, cùng router policy.
5. Full: donor và receiver negatives cùng học head.

Mask-only được thêm để tránh quy cải thiện do mở connection masks thành hiệu quả
donor training. Full phải có target gain ít nhất 0,1 điểm phần trăm so với mỗi
control availability/router/mask-only, recall từng target ≥95%, observed FAR ≤0,1%
và negative break=0. Current-pool accuracy không được giảm. Donor-only là ablation
đóng góp receiver negatives; chưa yêu cầu Full phải thắng donor-only để commit.

Minimum evidence: 32 unique FIT positives/target, 32 unique CAL positives/target,
32 current CAL negatives tổng và ít nhất 32 tại receiver. CAL được deduplicate
theo content trong từng owner và giữa các participants. Report FAR theo owner/class;
không gọi 0 lỗi trên pool nhỏ là chứng nhận thống kê FAR ngoài mẫu.

Historical safety được ghi `unverified`; không truy cập raw historical CAL để bù.
Rescue/break và delta correct có invariant kiểm tra trên cùng pool.

## Profile và source lock

`configs/candle_v3_peer_teaching.json` là profile experimental, bật tính năng.
Launcher `tools/train_candle_v3.py --peer-teaching` chọn profile này. Không có flag
thì giữ profile baseline. Các tham số native và data split vẫn theo V1.

Defaults phiên học: 8 head rounds, SGD lr=0,01; tối đa 4.096 BASE, 256 FIT,
512 CAL và 50 router sketches/class/owner. Đây là budget cho phiên bổ sung, không
giới hạn training native. Các giá trị là cấu hình khởi đầu, chưa được tối ưu.

`v1_clone_manifest.json` được giữ làm immutable origin. `runtime_manifest.json`
ghi source hiện tại và delta rõ ràng so với bản clone. Tool
`tools/lock_candle_v3_runtime.py` tạo manifest này; launcher kiểm tra từng hash.
Method audit có identity riêng `candle_v3_peer_teaching`, không gắn nhãn denice_base.

## Giới hạn còn lại trước khi chạy full

- Chưa có execution receipt; code hiện chỉ được đọc/soạn, chưa kiểm thử.
- Task-boundary resume lưu trạng thái đã commit. Chưa có resume giữa một head session;
  nếu session bị ngắt, khôi phục từ boundary trước đó và tính lại theo dữ liệu task còn hợp lệ.
- Ledger đo application-wire bytes của teaching, gồm announcements, references,
  gradients, sketch, CAL content IDs và verification. Native DeNICE wire bytes chưa đo;
  `total_system_bytes=null`, chưa tuyên bố tiết kiệm bandwidth toàn hệ thống.
- Mô phỏng một process; data/features được tính trong các worker logic tại owner,
  không có raw-data message. Chưa phải network transport P2P triển khai thực tế.
- Teaching budget 128 MiB/task. Hết budget thì defer các phiên còn lại; budget này
  có thể giới hạn số receiver được hỗ trợ, phải đưa vào coverage report.
- Chưa tự sửa registration gaps, chưa mở head mature, chưa xử lý receiver không
  active trong current task. Những trường hợp này phải được báo thay vì coi đã phủ đủ class.
- Cần native smoke kiểm tra commit/reject, checkpoint restore và task transition,
  cùng independent readout chất lượng trước khi chuyển notebook full sang V3.

Không có lệnh training hoặc test nào được chạy trong lần triển khai này.
