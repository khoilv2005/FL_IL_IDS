# APPLIANCE — chốt thiết kế Evidence / Certificate Scope

## Quyết định

Giữ prospective install và certificate theo evidence thực sự quan sát được. **Không bật head patch trên cumulative inference chỉ vì đã pass CAL của task hiện tại.** Không dùng task router, similarity prototype, head margin hoặc task training để suy ra true scope của một input.

Với protocol hiện có (`cumulative_dataset`, task-agnostic inference), chưa có nguồn metadata độc lập bảo đảm input chỉ thuộc các class đã có CAL. Vì vậy **chưa có scope vừa đủ căn cứ bảo vệ cumulative inference, vừa cho phép cài hữu ích hai cặp Task 1**. Dừng chuỗi thử guard/classifier và smoke của head-patch theo chính sách strict này; cần redesign transfer/protection hoặc có domain authority hợp lệ thật sự. Đây không bác bỏ functional transfer hay hiệu quả của protocol empirical riêng.

## 1. Phân biệt ba đối tượng

| Đối tượng | Nội dung | Không được suy ra |
|---|---|---|
| Evidence scope `E` | Owner, task, class đã đo bằng đúng frozen receiver function trên current CAL H; binding role/partition/coordinates/function/version | Mọi input có score giống target đều thuộc class trong `E` |
| Observable application domain `D` | Source/domain contract có provenance, preprocessing và predicate đánh giá được trước prediction từ input/metadata | Training task hoặc predicted task là true task của test sample |
| Protection summaries `P` | Finite BASE/sketch enclosures, owner/class provenance, version; dùng veto | CAL acceptance hoặc population FAR cho class cũ |

Với nguồn input hợp lệ `D`, cần biết **inventory các class có thể xuất hiện** theo một authority độc lập. Mọi class có thể xuất hiện nhưng chưa có evidence giữ trạng thái UNKNOWN. Registry hoặc hash của declaration không tự chứng minh inventory đúng; đây là ranh giới tin cậy của application.

## 2. CAL production giữ nguyên

- Donor target CAL FIT ≥32, selection positive ≥8.
- Receiver current CAL selection negatives ≥32.
- Reserved H: target positives ≥32, receiver negative **tổng** ≥32.
- Recall ≥95%; receiver FAR tổng và mỗi observed receiver class ≤0,1%; pooled break =0; rescue >0.
- Kiểm tra compatibility, mature dependencies, graph positive-alpha và protection như trước.
- ≥32 **mỗi protected owner–class** là yêu cầu qualification audit, không tự đổi thành requirement production.
- Đây là gate empirical. Không gọi nó là chứng nhận population FAR 95%; không đổi gate thành 2.995 mẫu.

## 3. Prospective install

```text
Target còn current
→ donor/receiver live snapshots của đúng task
→ compile head + imported route bằng current FIT
→ khóa candidate, predicate/domain authority và phiên bản trước CAL selection
→ selection → khóa thresholds/function
→ reserved current CAL H acceptance
→ xác nhận observable domain inventory nằm trong evidence scope
→ staging → commit hoặc rollback/skip
```

Không dùng head/guard Task 5 quay ngược Task 1. Không tái sử dụng CAL lịch sử hay gọi FIT là CAL. Acceptance trên current CAL chỉ cấp evidence đã đo; bước kiểm tra application domain vẫn độc lập. Không chuẩn bị patch thủ công để thay đường automatic install.

## 4. Inference không dùng nhãn

```text
Input x + metadata nguồn do application cung cấp
→ domain/predicate hợp lệ, preprocessing đúng, input hữu hạn?
→ mọi class có thể xuất hiện trong nguồn này có evidence hợp lệ?
→ certificate/function/dependency/receipt chain còn đúng, không conflict?
→ historical protection veto không hit?
→ imported route + margin guard pass?
    YES: imported prediction
    NO/UNKNOWN: prediction local
```

API membership chỉ nhận `x`, `source_id`, `preprocessing_sha256`; không nhận `y_true`, `true_task_id`, class của mẫu hoặc router prediction. Đây chỉ là kiểm tra domain, **không tự cấp phép route**. Có thể kiểm tra source thuộc registered contract nhưng code không thể tự chứng minh semantic inventory của nguồn đó.

Không thêm trusted task ID tại test để làm scope dễ hơn: đó là protocol có thêm task signal, không còn main task-agnostic CIL. Source contract nếu có thật phải được báo cáo là một giả định deployment, không được bịa từ label panel.

**Không dùng chính activation region làm safety certificate.** Ví dụ hai sample class 6 và class 0 có thể cùng nằm trên một phía threshold. Việc `signature > τ` chỉ chứng minh predicate đúng, không chứng minh sample class 0 đã bị loại. Audit trước không có exact collision cũng không giải quyết luận điểm này.

## 5. Pre-birth và post-birth

### Pre-birth

Historical support summaries chỉ veto những vùng được enclosure và có provenance. Thiếu summary hoặc thiếu coverage không thành PASS; giữ route tắt trong nguồn có thể chứa các class chưa có evidence. Bound của finite BASE không thành bound trên mọi sample tương lai của class đó. Không cộng samples trùng, đổi role hoặc pooling owner để che deficit.

### Post-birth

Chỉ tạo receipts cho guard đã tồn tại, đã khóa, tại owner/task còn current. Receipt binding patch/function/declaration/role/partition/coordinate/version như code hiện có. Append receipts không đổi positive certificate hay thresholds, không tạo lại raw CAL. Evidence scope mở rộng khi receipts hợp lệ; **không tự tạo nguồn domain độc lập**.

Drift hoặc conflict → SUSPENDED; giữ head/protection để chờ evidence hợp lệ. Thiếu CAL mới không làm hỏng certificate cũ, nhưng không cấp quyền inference ngoài phạm vi có căn cứ. Không âm thầm dùng empirical all-input mode để vượt điều kiện này.

## 6. Hai cặp shortlist

| Pair | Thời điểm compile | Positive CAL F/S/H theo metadata | Scope optimistic có thể đo | Scope cumulative tại T1 | Quyết định |
|---|---|---|---|---|---|
| 2 ← 62 / class 6 | Task 1 | 1176/588/589 | 6–11 (chưa chứng nhận thực tế) | 0–11 | DEFERRED |
| 2 ← 64 / class 8 | Task 1 | 245/122/123 | 6–11 (chưa chứng nhận thực tế) | 0–11 | DEFERRED |

Hai experiment phải là hai receiver clone độc lập. Mỗi cặp có 17 missing protected pools, gồm 4 pre-birth missing pools. Ngay cả giả sử current CAL chứng nhận được **toàn bộ 6–11**, source cumulative vẫn có thể đưa class 0–5 vào. Không có input-domain authority loại được nhóm này; positive/negative CAL có nhiều mẫu cũng không tự sửa authorization gap.

`scope_preflight.py` kiểm tra logic và fail closed; `plan_appliance_scoped_install.py` khóa execution decision từ artifact evidence đã có, không đọc dữ liệu, không fit/forward. Đây là contract/preflight, **chưa nối thành scope mode mới trong production runner** và không tự migrate certificate cũ. Schema mới không được dùng để đổi hash hoặc scope của certificate đã seal.

## 7. Điểm dừng và việc tiếp theo

**Không chạy hai smoke chỉ để thêm một report UNKNOWN. Không chạy full retrain hiện tại.** Thiếu scope authority là lỗi thiết kế, không cần thêm dữ liệu test để nhận ra.

Các điều kiện cần thay đổi trước khi tiếp tục head-patch strict deployment:

1. Có nguồn input với domain inventory được xác minh độc lập, không từ test label/route; protocol phải chấp nhận giả định deployment này; hoặc
2. Có evidence/protection thiết kế lại đủ kiểm soát các class cũ trong nguồn cumulative, không dùng summaries như CAL; hoặc
3. Đổi rõ claim thành empirical scoped deployment với phần ngoài evidence ghi UNKNOWN/risk, thay cho cam kết bảo vệ strict hiện tại. Đây là thay đổi methodology, không phải bug fix.

Nếu mục tiêu tiếp tục là task-agnostic cumulative IDS, không có domain restriction bên ngoài và allocation hiện tại giữ nguyên, chọn **redesign transfer/protection của head-patch**. Không tiếp tục thêm guard/classifier để vượt thiếu authority. Chưa chọn/triển khai thuật toán transfer mới hay train full trong thay đổi này.

Artifact quyết định: `artifacts/appliance_scoped_install_plan_20261010.json`.
