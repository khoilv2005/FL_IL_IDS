# CANDLE V3 — Complementary Peer Teaching trên model của receiver

Ngày: 2026-10-23. Trạng thái: đề xuất triển khai, chưa thay thuật toán hoặc chạy thí nghiệm.

Cập nhật triển khai: đã viết modules P0/P1/P2 và task-end hook, chưa chạy tests/smoke.
Xem [implementation receipt](../docs/CANDLE_V3_PEER_TEACHING_IMPLEMENTATION_20261023_VI.md).
Các yêu cầu bên dưới vẫn là protocol mục tiêu; chưa phải mọi gate đã được kiểm chứng.

Cập nhật ưu tiên theo đề xuất tiếp theo: thêm **variant A Direct FC2** và thử trước
variant C học head. Hai cơ chế giữ config/method identity riêng, không fallback
theo kết quả CAL. Xem [Direct FC2 contract](../docs/CANDLE_V3_DIRECT_FC2_VARIANT_20261023_VI.md).
Smoke A đã PASS correctness, chưa có install tự nhiên: xem
[smoke receipt](../docs/CANDLE_V3_DIRECT_FC2_SMOKE_20261023_VI.md).

Nguồn yêu cầu: `C:\Users\khoak\Downloads\candle v3.txt`, đã đọc toàn bộ.
Bản nền: runtime V1 riêng tại `candle_v3_runtime/`, source gốc
`a9714ac96b3ecec3910985d6fda8224b1584b570`.

## 1. Ý tưởng chính

Giữ local training và native masked aggregation V1. Bổ sung một phiên học ngắn
ở cuối mỗi task, khi dữ liệu current task vẫn còn được phép sử dụng:

1. Receiver xác định class có thể cần bổ sung.
2. Chọn peer còn dữ liệu current của class đó.
3. Đóng băng một bản model receiver dùng cho inference.
4. Receiver và peer học cùng một head mới trên encoder receiver.
5. Receiver tổng hợp những cập nhật đó, kiểm tra model và router rồi commit/rollback.

Ví dụ A có `{6,7}`, B có `{7,8}`: phiên A nhận class 8 dùng encoder A tại cả A
và B. Phiên B nhận class 6 dùng encoder B tại cả hai bên. Hai phiên có model
reference khác nhau và không gộp updates với nhau.

Không có central server/global model. Receiver điều phối phiên của chính mình;
inference vẫn một model local. Raw data ở lại client sở hữu. Donor có thêm tính
toán training cho phiên này; đây là chi phí bổ sung cần đo.

## 2. Hai đường cộng tác có điều kiện khác nhau

| Đường | Eligibility | Thứ trao đổi |
| --- | --- | --- |
| Native V1 | Giữ similarity ξ=0,8, masks, maturity và quy tắc hiện tại | Masked local deltas |
| Peer teaching mới | Liên kết P2P được protocol cho phép, dữ liệu current, class mapping/preprocessing và khả năng chạy snapshot receiver | Receiver reference, head deltas, router sketches và verification receipts |

V1 paper mode tạo các neighborhood theo threshold, không tạo partition cluster
không chồng lấn. Không dùng `labels=0`/K sentinel làm cluster ID cho lookup.

Đề xuất teaching graph là một lớp liên kết riêng trên tập peer discovery được
phép của mô phỏng. Không dùng cùng điều kiện prototype similarity ≥0,8 để loại
teaching donor: việc đó có thể giữ nguyên giới hạn coverage của V1. Cũng không
cho phép peer ngoài tập mạng đã được protocol công bố. Log cả hai loại cạnh.

## 3. Metadata và functional gap

Mỗi client cần các trường có ý nghĩa độc lập:

- `local_seen_classes`: chỉ class từng có optimizer update từ dữ liệu local.
- `current_train_classes/counts`: dữ liệu current còn được phép training.
- `registered_classes_by_task`: class router cho phép dự đoán.
- `peer_acquired_classes`: class nhận qua một phiên đã commit, kèm provenance.
- `class_evidence`: nguồn, role, task, phiên bản model, số mẫu và metric quan sát.
- `model/preprocessing/mask/router_version`: định danh reference.

Không cập nhật local sample counts khi nhận class từ peer. Ownership, registration
và competence không thay thế nhau. Bitmap là tín hiệu discovery, không phải giấy
chứng nhận chất lượng.

Class có ở peer nhưng chưa đăng ký tại receiver chỉ là candidate. Trước fitting,
đánh giá shadow `Availability-only` bằng dữ liệu current hợp lệ ở peer. Model đã
nhận diện tốt thì xét registration; model còn yếu mới tạo functional-gap session.
Thiếu evidence thì ghi UNKNOWN. Class đã đăng ký nhưng chất lượng thấp cũng có
thể là functional gap, nếu có evidence current; không ghi đè mature row trong
phiên bản đầu.

Ưu tiên donor theo số class cần bổ sung được phủ thêm, chất lượng evidence còn
hiệu lực và chi phí reference mới. Với nhiều donor, dùng marginal coverage sau
mỗi lựa chọn để tránh chọn cả nhóm chỉ cung cấp cùng một class. Donor native
accuracy tốt chỉ dùng để xếp hạng, không chứng minh transfer sang receiver tốt.

## 4. Học head trong cùng reference

Cho snapshot receiver R là S, gồm effective encoder h_R, existing heads, masks,
BN buffers, adapter/path state, preprocessing và router thresholds. Giữ tất cả
ở chế độ eval, tắt dropout, không cập nhật BN. Chỉ module head tạm thời φ có gradient.

Chỉ dùng feature trước FC2 đúng với inference có weight/bias masks. Không gọi
`forward_output()` nguyên trạng cho fitting: hàm này có learner/young activation
gates. Cũng không mặc định `forward_inference()` là cùng đường inference vì hàm
đó bỏ một số masks. Cần một API trích feature canonical được dùng nhất quán.

Với một class c, φ=(w_c,b_c); nếu receiver cần nhiều class cùng task thì dùng một
block φ gồm các rows đó. Kiến trúc vẫn FC2 34 rows; module tạm không tạo ensemble.
Sau commit, copy đúng những tọa độ được cấp quyền vào FC2 và đăng ký masks/ranks.
Chỉ ưu tiên rows chưa được xác nhận/protect; mature rows không được tự mở khóa.

Loss tại mỗi participant j:

\[
 L_j(\phi;S)=\frac{1}{Z_j}\sum_{(x,y)\in D_{j,\mathrm{BASE,current}}}
 a_y\,CE([z_{R,\mathrm{existing}}(x),z_{\phi}(x)],y).
\]

Class weights a_y, sampling, tập logits cạnh tranh và normalization Z_j phải khóa
trước pilot. Bản đầu dùng các registered classes của current episode tại receiver
cộng targets của phiên, phù hợp native task-gated inference; không tự chuyển loss
thành phân loại all-seen routerless. Existing rows của task cũ vẫn giữ trong
snapshot để kiểm tra inference, nhưng không đưa nhãn task cũ vào training.
Labels không thuộc tập logits cạnh tranh cần quy tắc xử lý công bố trước. Bản đầu chỉ dùng
samples có labels thuộc tập logits đó và ghi số samples bị loại; không giả nhãn
chúng thành một class cũ. Mẫu âm receiver thuộc các class local vẫn đóng góp vào
loss, kể cả receiver có 0 mẫu dương của target.

Tất cả participants bắt đầu một update round từ cùng φ^r và cùng S:

\[
 \phi^{r+1}=\phi^r+\eta\sum_{j\in\{R\}\cup D_R}\omega_j\Delta_j^r,
 \quad\sum_j\omega_j=1,\quad\omega_j\ge0.
\]

Không lấy n_receiver,target=0 làm trọng số 0. Trọng số tính từ objective và các
mẫu thực sự được dùng, có cân bằng positive/negative. Không mặc định alpha native
hoặc fraction reserve neurons là trọng số đúng cho phiên này. Một bước SGD cùng
reference cho phép kiểm tra aggregation gradient của objective đã chọn; nhiều
bước optimizer local là xấp xỉ, phải ghi rõ, không gọi là tương đương nghiệm gộp.

Cache feature/logits có thể ở lại từng participant trong phiên để tránh forward
backbone mỗi head round. Cache gắn với S, chỉ dùng current task và hủy khi hết
quyền dữ liệu task; không tạo historical raw/feature replay bank.

## 5. Thứ tự cuối task và router

Thứ tự đề xuất:

```text
20 rounds native như V1
  → hoàn tất các thay đổi topology/masks/BN ảnh hưởng feature
  → khóa receiver snapshot S
  → Availability-only probe
  → học head bằng current BASE ở receiver và donor
  → tạo current-task router sketches qua chính S
  → shadow apply head + masks/ranks + router registration/memory
  → đánh giá candidate cuối cùng
  → commit/rollback
  → hoàn tất maturity/protection và kiểm tra effective feature không đổi
  → checkpoint task-finalized và ngừng truy cập dữ liệu task
```

Trong code hiện tại `_finalize_candle_task`, release capacity, cập nhật masks/ranks
và BN cuối task có thể nằm sau round cuối. Hook mới phải xác định đúng vị trí;
không học head trước một thao tác có thể làm thay đổi feature rồi lưu head đó.
Nếu vẫn phải đổi effective feature sau phiên, candidate chưa hợp lệ: cần refit/
reverify trong current task hoặc rollback, không chỉ sửa version metadata.

Donor dùng extractor và thresholds của R để tạo binary sketches từ current BASE.
Không copy donor-space memory, không OR memory tùy ý và không thay thresholds
lịch sử để làm đẹp current task. V1 router giữ per-sample binary vectors nên một
vector trung bình không tự động thay thế được format này. Truyền sketches có
giới hạn theo class, gắn extractor/mask/threshold version; kiểm tra đủ features
để router current thật sự đọc được các sketch mới.

Current-task memory và `episode_classes` được stage cùng head. Không sửa historical
task memory từ raw data cũ. Gộp sketches không bảo đảm task routing đúng: phải đo
end-to-end task prediction bằng X và các quyết định class cuối cùng.

## 6. Đánh giá cơ chế và acceptance

Các controls bắt buộc trên cùng snapshot và cùng panel:

| Control | Weights | Registration/router |
| --- | --- | --- |
| Native baseline | Giữ nguyên | Giữ nguyên |
| Availability-only | Giữ nguyên | Chỉ mở class trong episode phù hợp |
| Router-only | Giữ nguyên | Mở class + thêm receiver-space current sketches |
| Donor-only fitting | Fit head tại donor | Cùng router policy đã khóa |
| Full peer teaching | Fit head tại donor và receiver negatives | Cùng router policy |

So sánh Full với Router-only để tách đóng góp weights; so với Donor-only để tách
đóng góp negative training ở receiver. Nếu nhiều targets thì đánh giá model sau
khi kết hợp tất cả rows và router changes, không suy PASS từ từng row riêng.

BASE để training, FIT để discovery/lựa chọn đã khai báo, CAL để acceptance theo
split độc lập đã khóa. Validation dùng readout development; không quay lại chỉnh
threshold bằng panel vừa báo cáo. Giữ roles và TEST protocol V1. TEST chỉ cuối
T5/R19; không dùng TEST cho donor selection hoặc commit.

Điểm xuất phát acceptance giữ các gate đã dùng: target recall ≥95%, observed FAR
≤0,1%, negative break=0 trên evidence yêu cầu, gain vượt controls, không vi phạm
protected state. Cần khóa minimum evidence và mức gain trước thí nghiệm; không
tự xem 0 lỗi trên vài mẫu là đủ. Thiếu current evidence: UNKNOWN/skip phiên,
training tiếp tục. Không hạ gate sau khi xem kết quả.

Historical class risk có thể chưa được quan sát đầy đủ. Dùng policy empirical đã
thảo luận: báo rõ coverage và phần chưa kiểm chứng, không tuyên bố strict safety
trên mọi class. Mẫu historical development được phép đọc trong audit riêng không
được đưa vào production acceptance/training hoặc dùng để chọn candidate.

Metrics: target recall/precision, task-routing accuracy, class confusion, rescue,
break, gain so với từng control, old/new accuracy, observed FAR theo class/owner,
unknown evidence, retention qua task tiếp theo. Rescue/break và delta accuracy
phải tính trên cùng pool với Δcorrect=rescue−break.

## 7. Survival và transaction

Freeze row không tự bảo vệ head trước upstream drift. Chỉ dùng permissions/protection
V1 đã có; theo dõi các feature coordinates mà head thực sự sử dụng. Chưa thêm cơ
chế khóa toàn backbone. Sau task transition cần kiểm tra mature row, masks, BN,
upstream dependencies và prediction retention bằng readout được phép.

Transaction gồm head, ranks, masks, registry, provenance, router memory và thresholds
versions; commit nguyên tử, reject phải restore toàn bộ. Resume lưu phase/session ID,
snapshot hash, head round, accepted targets, receipts và byte ledger. Không replay
session đã commit khi resume; không giữ quyền đọc raw task cũ để chạy lại phiên.

## 8. Communication và tính toán

Đo toàn bộ native traffic và phần bổ sung: metadata, receiver reference, head
updates, sketches, verification, retry và setup. Head FP32 257 tham số chỉ 1.028
byte; đó không phải tổng packet hay tổng phiên. Không tuyên bố thêm bitmap miễn phí.

Cache/reuse receiver reference chỉ khi hash, masks, BN, preprocessing và adapter
path khớp. Một reference dùng cho nhiều targets tại cùng donor giúp amortize chi phí.
Native messages chưa chắc chứa đủ post-aggregation reference; không giả định có sẵn.

Report wall time, số receiver/donor forward batches, head optimizer steps, peak
memory, snapshot bytes và total serialized message bytes. Không coi head nhỏ là
bằng chứng training overhead nhỏ. Không truyền raw data không đồng nghĩa có DP.

## 9. Triển khai theo thứ tự

| Mốc | Deliverable |
| --- | --- |
| P0 | Metadata/provenance contract, teaching graph và snapshot identity |
| P1 | Canonical inference feature API, standalone head trainer và delta reducer |
| P2 | Receiver-space sketch builder và atomic head/router transaction |
| P3 | Pilot giới hạn trên checkpoint ở đúng current task, đủ controls và wire ledger |
| P4 | Native task transition, checkpoint/resume, retention và dữ liệu current-only |
| P5 | Kaggle full-run notebook sau khi integration correctness đạt |

Khuyến nghị P3 chỉ một receiver–donor session đơn class trước; nếu structural
correctness đúng nhưng không có gain thì ghi kết quả âm, không tự mở thêm donor,
adapter hoặc backbone training để cứu. Sau đó mới đánh giá một session nhiều class
đã đăng ký trước. Không dùng accuracy thấp để bắt đầu vòng lặp redesign không giới
hạn. Tách engineering correctness, empirical acceptance và scientific effectiveness.

Các modules mới dự kiến nằm trong `candle_v3_runtime/fed_learning/strategies/`
`decentralized/candle_v3/`: `contracts.py`, `discovery.py`, `snapshot.py`,
`head_learning.py`, `head_aggregation.py`, `router_teaching.py`, `transaction.py`,
`communication.py`. Hook vào private training runner. Giữ nguyên source V1 ở root.
Khi bắt đầu implementation, cập nhật stage và kiểm tra source manifest: launcher
clone hiện cố ý từ chối thay đổi runtime, không được vô hiệu hóa kiểm tra âm thầm.

Notebook V3 hiện có vẫn là V1 clone. Kế hoạch này chưa biến notebook đó thành
phiên bản peer teaching đã hoạt động.

## 10. Vị trí nghiên cứu và giới hạn

FedSaC nghiên cứu cân bằng similarity/complementarity; không thể coi việc chọn
peer khác class tự nó là novelty: [Yan et al., ICML 2024](https://proceedings.mlr.press/v235/yan24a.html).
FedPAC nghiên cứu feature alignment và classifier collaboration:
[Xu et al., ICLR 2023](https://arxiv.org/abs/2306.11867).
Iterative aggregation updates có tiền lệ trong
[McMahan et al., AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html).

Đóng góp cần kiểm chứng ở đây là sự kết hợp: complementary current-data peer
selection, head learning trong receiver reference có negative contribution tại
receiver, receiver-space router acquisition và survival trong no-replay continual
learning. Chưa kết luận mới hơn toàn bộ literature hay đảm bảo cải thiện.

Việc dùng chung receiver encoder giải quyết coordinate alignment của phiên học,
nhưng không tạo thêm thông tin vào encoder đã đóng băng. Nếu feature không phân
biệt được target và negatives thì head learning vẫn thất bại. Mặt khác, nếu head
đã tốt mà router không mở đúng task thì end-to-end vẫn thất bại. Hai kết quả cần
được đo riêng trước khi bàn về hiệu quả toàn hệ thống.
