# DeNICE: phân biệt coverage của protocol và lỗi trong phạm vi đã học

## Kết luận

Non-IID, client không tham gia một số task và không đọc lại raw dữ liệu cũ là
ràng buộc nghiên cứu. **Coverage thiếu không tự động là lỗi implementation.**
Audit provenance đã tái dựng registration khớp 98/98 receiver, không tìm thấy
task profile đã học local bị mất. Con số 67,26% là phần lỗi dự đoán liên quan
đến coverage trên toàn test, không phải tỷ lệ lỗi code.

Mục tiêu mỗi client nhận diện toàn bộ 34 class hay chỉ những class đã học/nhận
hợp lệ cần được chốt trước khi chọn hướng sửa thuật toán. Trong lúc chưa đổi
mục tiêu, giữ nguyên yêu cầu đánh giá toàn test; các nhóm dưới đây là chẩn đoán
có điều kiện, không thay metric chính.

## 1. Cùng checkpoint, cùng predictions, khác phạm vi thống kê

Fixture: `results (11)`, DeNICE legacy, ξ=0,8, seed 42, Task 5 / Round 19.
Không CGoFed, CME, APPLIANCE hoặc donor inference. BinarySelf dùng native
`binary_cosine`; MulticlassSelf là variant router `multiclass_balanced` fit từ
binary memory. Không gọi MulticlassSelf là router nguyên gốc.

| Phạm vi receiver–sample | Samples | BinarySelf Acc | MulticlassSelf Acc | Multiclass route Acc | OracleMatched Acc |
|---|---:|---:|---:|---:|---:|
| Toàn test, 34 class | 13.505.771 | 31,2446% | 37,1463% | 42,8166% | 54,8568% |
| True class có BASE local ở receiver | 6.528.799 | 53,8791% | 63,5693% | 66,5989% | 94,3110% |
| True class nằm trong native registration | 7.796.356 | 54,1256% | 64,3492% | 67,0174% | 95,0295% |
| Đã đăng ký nhưng không có BASE local | 1.267.557 | 55,3952% | 68,3664% | 69,1728% | 98,7303% |

Mỗi sample vẫn giữ prediction đã có, kể cả dự đoán sang class ngoài nhóm.
Chỉ lọc theo true class **sau inference** để tính metric. Không đưa nhãn thật
vào model/router, không mở mask và không chia lại test cho phù hợp receiver.

MulticlassSelf đúng **5.016.894 mẫu** trên cả toàn test lẫn nhóm đã đăng ký.
64,3492% chỉ dùng mẫu số nhỏ hơn; không phải model vừa được cải thiện từ
37,1463% lên 64,3492%.

OracleMatched thay task dự đoán bằng true task trong cùng legal bank, giữ các
chính sách còn lại của audit. Đây là diagnostic có nhãn, không phải kết quả
inference triển khai được.

## 2. Trong phạm vi đã đăng ký, routing chiếm phần lớn lỗi accuracy

| MulticlassSelf | Tổng lỗi | Routing có thể cứu bằng matched oracle | Classifier vẫn sai khi matched oracle |
|---|---:|---:|---:|
| Class có BASE local | 2.378.488 | 2.007.064 (84,38% lỗi) | 371.424 |
| Class đã đăng ký | 2.779.462 | 2.391.944 (86,06% lỗi) | 387.518 |

Vì vậy routing là ưu tiên mạnh nếu mục tiêu là accuracy trong phạm vi đã học
hoặc đã đăng ký. Coverage vẫn chiếm phần lớn lỗi trên toàn test theo giao thức
hiện tại. Hai kết luận dùng hai phạm vi khác nhau và không mâu thuẫn.

## 3. Accuracy cao của oracle không chứng minh mọi class đã học tốt

Macro-F1 dùng đủ 34 class, pooled confusion matrix:

| Phạm vi | MulticlassSelf macro-F1 | OracleMatched macro-F1 |
|---|---:|---:|
| Toàn test | 19,7131% | 30,9036% |
| Class có BASE local | 34,6920% | 63,0360% |
| Class đã đăng ký | 34,7612% | 63,3448% |

Trong nhóm đã đăng ký, OracleMatched recall của class 2 chỉ **0,69%**, class 28
**1,01%**, class 3 **1,30%**, class 0 **2,42%**, class 31 **7,21%**.
Ngay nhóm có BASE local, class 28 cũng chỉ **1,03%**. Đây là bằng chứng còn lỗi
classifier/learning trên một số class; chưa xác định nguyên nhân và chưa đủ để
kết luận forgetting. Không hardcode class, hạ gate hoặc fit theo test labels.

## 4. Hai câu hỏi chưa được đo đầy đủ

**Continual forgetting — NOT_MEASURED.** Cần so cùng receiver/class/test rows
tại checkpoint cuối task khai sinh và các task sau. Accuracy class cũ ở T5
không tự nó đo forgetting. Full run trước không có intermediate evaluation.
Có thể dùng task-end checkpoints để audit, không cần retrain chỉ để đo.

**Collaboration gain/negative transfer — NOT_IDENTIFIED.** Accuracy nhóm được
đăng ký nhưng không có BASE local không phải hiệu ứng nhân quả của graph.
Cần control training khớp protocol. Tắt graph aggregation nhưng vẫn giữ
representative clone/catch-up chưa phải control hoàn toàn local; phải mô tả
riêng các đường peer transfer. Chưa chạy control training trong audit này.

## 5. Quyết định hiện tại

- Không ép mọi receiver có 34 class hoặc tự tạo profile cho task chưa học.
- Không thay allocation, graph, ξ, checkpoint, masks hoặc production defaults.
- Không triển khai class admission từ khuyến nghị coverage trước đó.
- Giữ full-test metrics và báo thêm valid-scope diagnostics có mẫu số rõ ràng.
- Nếu mục tiêu là các class đã học/nhận hợp lệ: ưu tiên audit routing, rồi đo
  learning/forgetting theo class; registration chỉ là proxy, không chứng nhận
  độc lập rằng kiến thức từ peer đã transfer an toàn.
- Nếu mục tiêu là nhận diện 34 class ở mỗi client: coverage là giới hạn năng lực
  cần nghiên cứu; mở coverage phải là variant được kiểm chứng, không gọi là
  sửa lỗi registration.

## 6. Integrity và tái lập

**414/414 checks PASS**, gồm 196 đối chiếu route confusion với predictions đã
lưu và các kiểm tra phân hoạch/counts. Không forward, fit, retrain, đọc raw
historical train/CAL hoặc tạo predictions mới. Chỉ đọc confusion matrices,
inventory và BASE class counts đã khóa của full audit.

Script: `tools/audit_denice_valid_scope.py`.
Summary: `artifacts/denice_valid_scope_20261011.json`.
ZIP local: `artifacts/denice_valid_scope_20261011.zip`, gồm full per-client
receipts và báo cáo; không chứa dataset/model.

```powershell
python tools/audit_denice_valid_scope.py `
  --audit-dir audit_denice/denice_bottleneck_20261010/full_cuda `
  --roles audit_denice/appliance_legacy_results11/inputs/roles/role_manifest.json `
  --metadata audit_denice/denice_bottleneck_20261010/inputs/metadata.json `
  --out audit_denice/valid_scope_new.json
```

Full test đã được dùng để chẩn đoán; không gọi các phân tích này là independent
final confirmation và không chọn hyperparameters dựa trên kết quả này.
