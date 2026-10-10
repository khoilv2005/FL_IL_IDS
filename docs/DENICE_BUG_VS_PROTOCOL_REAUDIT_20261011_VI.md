# DeNICE: audit lại bug, protocol và hạn chế thuật toán

## Kết luận trực tiếp

**Chưa tìm thấy bug registration/checkpoint gây ra coverage thấp trong
`results (11)`.** Coverage thấp khớp allocation Non-IID, skipped tasks và
chính sách inference/aggregation của runner tại đúng commit training.

**Trong phạm vi đã đăng ký, routing là nút thắt accuracy lớn.** Đây là failure
mode thực nghiệm của router hiện tại; chưa có bằng chứng rằng nó đến từ một
phép tính bị implementation sai. Một số class còn lỗi classifier ngay khi ép
đúng task; nguyên nhân learning/forgetting/negative transfer chưa được tách.

Có một vấn đề implementation độc lập: option `old_client_data_prob` được lưu
nhưng không được allocator sử dụng. Không đủ bằng chứng để quy coverage của
dataset hiện có cho lỗi option này vì thiếu commit tạo dataset.

## 1. Phân loại từng phát hiện

| Phát hiện | Kết luận audit | Bằng chứng / giới hạn |
|---|---|---|
| Receiver không có dữ liệu một task | Ràng buộc allocation/participation | 73/74 missing-profile pairs không allocated vào task |
| Receiver allocated nhưng thiếu usable BASE | Chưa đủ thông tin để kết luận bug | Client 47/T4; role summaries rỗng, chưa phân biệt raw shard rỗng với clean-role exclusion |
| Mất profile của task đã học local | Không quan sát thấy | 0 pair có local participation nhưng thiếu profile |
| Mất class đã học local khỏi registry | Không quan sát thấy | 0 class có BASE local nhưng unreachable; reconstruction khớp 98/98 |
| 1.200 FC2 rows rank 0 | Khớp trạng thái chưa consolidate/đăng ký của protocol | 444 missing-profile + 756 missing-mask rows; rank không chứng nhận competence |
| Catch-up nhận weights nhưng không thêm profile | Behavior hiện tại; chưa chứng minh vi phạm specification | Helper chỉ synchronize target-plastic weights, caller không clone detector cho rejoin |
| Peer có class nhưng receiver không được mở class | Giới hạn của đường collaboration hiện tại | Native FC2 pair mask yêu cầu giao receiver/sender labels; không có class-registration merge |
| Router chọn sai task cho class đã đăng ký | Hạn chế thực nghiệm, chưa chứng minh bug tính toán | Native/fast-path replay khớp; binary cosine dùng score và argmax theo từng sample |
| Classifier vẫn sai với matched oracle | Lỗi nhận diện đã đo; nguyên nhân chưa xác định | 387.518 mẫu trong registered scope không được legal route nào cứu |
| `old_client_data_prob` không có tác dụng | Bất nhất contract cấu hình trong source đã audit | Attribute được Store tại line 168, không có Load trong `ClientAllocator` |

Không gọi mọi behavior trong code là đúng với paper chỉ vì code chạy như vậy.
Audit này xác nhận fixture khớp implementation đã dùng. Chưa có chứng minh đầy
đủ mọi thành phần tương đương specification/toàn bộ equations của paper.

## 2. DeNICE gốc và variant router phải được báo riêng

| Metric | BinarySelf gốc | MulticlassSelf variant |
|---|---:|---:|
| Accuracy toàn 13.505.771 test rows | 31,2446% | 37,1463% |
| Accuracy trên 7.796.356 rows có true class đã đăng ký | 54,1256% | 64,3492% |
| OracleMatched trên cùng registered scope | 95,0295% | 95,0295% |
| Routing-recoverable / lỗi trong registered scope | 3.189.016 / 3.576.534 = **89,16%** | 2.391.944 / 2.779.462 = **86,06%** |
| Macro-F1_34 trong registered scope | 26,3100% | 34,7612% |
| OracleMatched Macro-F1_34 trong registered scope | 63,3448% | 63,3448% |

Oracle dùng nhãn task thật để chẩn đoán, không phải accuracy deployable.
Registered-scope accuracy là metric có điều kiện, không thay toàn test.
95% pooled oracle accuracy cũng không chứng minh mỗi class được học tốt:
registered oracle recall class 2/28/3/0/31 lần lượt chỉ khoảng
0,69% / 1,01% / 1,30% / 2,42% / 7,21%.

67,26% lỗi MulticlassSelf liên quan coverage là decomposition **trên toàn
test**. Không diễn giải thành 67,26% lỗi code hoặc thành phần lỗi thuật toán
trong riêng valid scope.

## 3. Audit lại đã kiểm tra gì?

Script `tools/audit_denice_bug_vs_protocol.py` đạt **23/23 checks PASS**:

- Hash source `decentralized_denice_il.py`, `denice_aggregation.py`, allocator
  được đọc bằng `git show` tại training commit
  `4a45a7176ac4f8cb8671dba83f3e52071e44a3ca` khớp provenance receipts.
- Full-audit completion, verification, checkpoint/role hash và valid-scope
  summary cùng nguồn; full accuracies giữ nguyên.
- Đọc lại 588 receiver-task rows và 3.332 receiver-class rows; phân hoạch test,
  missing profiles/classes và local coverage khớp số đã công bố.
- AST xác nhận allocator lưu `old_client_data_prob` nhưng không đọc attribute.

Kế thừa các bằng chứng đã kiểm tra: prediction reproduction không mismatch,
native/fast evaluator tương đương trên phạm vi đã đối chiếu, registration
reconstruction 98/98 và valid-scope accounting 414/414.

Không chạy thêm model, fit, train, đọc raw historical train/CAL hoặc chia lại
test. PASS integrity không phải chứng minh mọi thành phần training không bug.

## 4. Bất nhất allocator có nghĩa gì?

Source khai báo option là xác suất old clients nhận dữ liệu task mới. Nhưng
`_allocate()` chọn client độc lập mỗi task và không tham chiếu option này.
Nếu experiment yêu cầu option đó điều khiển participation, implementation
hiện tại không đáp ứng contract ấy.

Dataset metadata ghi giá trị 1,0 và các tập client thực tế không lồng nhau.
Không có commit producer để xác nhận dataset được tạo từ chính source đã audit.
Vì vậy chỉ kết luận **option inert trong audited source**, chưa kết luận đây
là nguyên nhân nhân quả của accuracy thấp ở checkpoint. Không thay allocation
hay regenerate dataset; làm vậy sẽ đổi benchmark và cần chạy lại controls.

## 5. Nên làm gì tiếp?

1. Giữ nguyên DeNICE baseline và full-test reporting, không ép 34-class coverage.
2. Với class hợp lệ, ưu tiên tìm nguyên nhân router không phân biệt được task:
   kiểm tra activation/memory/masks và score separation; chỉ sửa code nếu tìm
   được invariant bị vi phạm. Đổi representation/objective là algorithm variant.
3. Đo learning và forgetting của class yếu bằng các task-end checkpoints, cùng
   receiver/class/test rows. Không đồng nhất T5 recall thấp với forgetting.
4. Hiệu ứng riêng của graph cần control khớp training. Không quy classifier
   yếu cho aggregation nếu chưa có control; bootstrap/catch-up cũng là peer
   transfer và phải được mô tả riêng.

Mục tiêu nhận diện mọi class hay chỉ class học/nhận hợp lệ chưa được tự động
đổi. Nếu chọn mục tiêu toàn class, coverage là giới hạn cần nghiên cứu mở rộng;
điều đó vẫn không biến allocation Non-IID thành bug.

Chưa sửa production, chưa train lại, chưa quay lại APPLIANCE trong lượt này.

Artifact: `artifacts/denice_bug_vs_protocol_20261011.json`.
