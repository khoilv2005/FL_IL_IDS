# APPLIANCE: audit guard đối chiếu và FIT envelope

## Trạng thái

**Chưa đạt gate để đưa guard mới vào production/native smoke.**
Đã chạy local, không train backbone, không đổi ξ, không mở lại final test.
Main training notebook và runner vẫn dùng bản đã có trước audit này.

Fixture là checkpoint cuối Task 3 của results (13), cùng bộ clean roles đã khóa.
Kết quả này là development, không phải xác nhận độc lập hay accuracy toàn hệ thống mới.
Knowledge receipts vẫn được mô phỏng từ các positive edge thật của graph;
không có bằng chứng rằng run cũ thực sự truyền các receipt này.

## 1. Code đã bổ sung

- `appliance/contrastive_protection.py`: giữ own BASE veto; đối với peer support,
  chỉ veto khi sample nằm trong vùng support và negative head thắng imported head.
  Mỗi class dùng union geometry từ các nguồn có receipt; chọn một negative head
  bằng quy tắc cố định: ưu tiên patch donor, sau đó nhiều own BASE rows hơn,
  rồi sender ID. Không chọn bằng validation/HOLDOUT.
- `appliance/fit_contrastive_protection.py`: hiệu chỉnh độ lệch logit giữa context
  bằng offset `max(0, percentile99(negative_logit - patch_logit))` từ donor current FIT.
  Offset được khóa trước SELECTION/HOLDOUT; không hạ acceptance recall 95%.
- Audit acceptance có các cờ `--contrastive`, `--fit-counter`, `--development`.
- Các script chẩn đoán FIT, positive envelope và exact feature collisions.

Negative head chỉ là counter-evidence, không được cài thành classifier output,
không biến peer provenance thành receiver ownership và không thay thế CAL.
Mọi head thêm vào đều nằm trong dependency maturity precheck và guard fingerprint.
Save/restore khôi phục đúng negative heads, offsets và provenance; runtime không nhận nhãn.
Đây chưa phải registry/installer đã chạy qua native round.

## 2. Raw counter: đủ 16 request đã khóa từ FIT

12 request có donor trưởng thành để đánh giá CAL; 4 request không có candidate.
4/12 pass CAL: receiver 0/class 21, 4/class 22, 54/class 20, 55/class 20.
**Pass current CAL không đồng nghĩa bảo vệ được class ngoài receiver support.**

Receiver 54/class 20 pass recall CAL 96,72%, nhưng trên các peer validation pool
vẫn có 1.030 false activation. Vì vậy tăng số CAL pass không đủ làm gate production.
Các validation pool được xét cho candidate pass và mọi patch cũ cài ở Task 3.
Không suy ra một accuracy chung bằng cách cộng các pool này.

Có 11 control checks cho mỗi candidate được đánh giá, tổng 132 check executions,
không có mismatch. Đây là các kiểm tra lặp trên 12 candidate, **không phải 132
native checks độc lập**. Chúng kiểm tra restore, version/receipt binding, input alignment,
nonfinite rejection, own-veto preservation, dependency maturity và model immutability.

## 3. Pair khó: receiver 65, donor 88, class 20

### 3.1. Đối chiếu logit thô không giải quyết được

Trên 306 donor current FIT positives:

- Guard cũ, shadow CPU với threshold cũ: kích hoạt 289 mẫu.
- Raw counter làm mất 288/289 activation đó.
- Riêng negative head class 13 veto 302/306 mẫu FIT class 20.
- Median `negative_logit - patch_logit` của head 13 là **+7,5166**.

Do đó không thể giả định head thuộc các context khác nhau có cùng thang logit.
CAL HOLDOUT của raw counter cho recall 0%, và reject đúng quy tắc.

### 3.2. FIT offset khôi phục recall nhưng không khôi phục protection

Offset class 13 = **8,445693**, chọn từ percentile 99% của FIT, trước CAL.
CAL HOLDOUT: **149/153 = 97,3856% recall**, receiver FAR 0, break 0.
13 integrity controls đều pass, kể cả offset fingerprint và fixed-quantile rejection.

Trên cùng các validation pool, với final tau/gamma của candidate này:

| Pool | Rows | Own-only activation | Union peer veto | FIT counter |
|---|---:|---:|---:|---:|
| Receiver validation | 1.318 | 0 | 0 | 0 |
| Donor validation, class 20 | 256 | 235 | 2 | 229 |
| Peer 52 validation, old classes | 1.052 | 252 | 0 | 247 |
| Peer 88 validation, old classes | 938 | 254 | 1 | 254 |
| Peer 89 validation, old classes | 1.294 | 256 | 0 | 252 |

Ba peer pool có tổng **753 false activation** với FIT counter.
Chứng nhận current CAL không bao gồm các old-class pool này.
**Không được lấy 97,39% recall làm bằng chứng rằng guard đã sửa xong.**

11 negative head rows cần 11.308 bytes tensor FP32, chưa gồm metadata, geometry,
receiver function setup/capsule và wire encoding. Đây không phải tổng communication.

## 4. Positive FIT envelope cũng chưa đủ

Mỗi envelope chỉ fit mean/covariance từ donor current FIT class 20, khóa radius
ở percentile 99%, shrinkage cố định 0,01 trước validation. Không sweep các tham số.
Các thử nghiệm dùng guard cũ với threshold cũ làm shadow reference; **không phải**
ablation cùng threshold với FIT-counter ở phần 3.2, và chưa chạy CAL acceptance.

| Envelope | Target activation /256 | False activation trên 3 peer pool |
|---|---:|---:|
| Unit sketch 16D, diagonal | 234 | 760 |
| Raw sketch 16D, diagonal | 231 | 751 |
| Full preprocessed input, diagonal | 230 | 748 |
| Full preprocessed input, covariance | 229 | 725 |

Guard cũ trên các pool này có 235 target activation và 762 false activation.
Các envelope vẫn để lọt phần lớn negatives; không đủ tiêu chí bảo vệ.
Không có claim CAL pass cho bốn envelope này.

## 5. Kiểm tra feature collision

Đã so hash SHA-256 của toàn bộ input FP32 sau preprocessing, chỉ canonicalize
signed zero, không làm tròn hay approximate matching:

- 306 positive FIT rows có 305 input khác nhau.
- 256 positive validation rows có 256 input khác nhau.
- Không có exact cross-label collision với các peer old-class validation rows
  của panel đã kiểm tra.

Điều này loại được một dạng lỗi nhãn/input trùng hoàn toàn trong panel này.
**Không chứng minh** rằng các class không thể phân biệt, cũng không chứng minh
chúng đủ tách biệt cho các guard thống kê hiện tại. Không suy ra bound cho full test.

## 6. Quyết định và công việc kế tiếp

Các guard đang thử chưa đạt đồng thời positive retention và peer-negative protection.
Không thay production guard bằng chúng, không sửa certificate cũ để thêm policy mới.

Bước hợp lý tiếp theo là tạo negative evidence phân biệt tốt hơn, được thu tại
owner từ **current BASE của từng task**, persist và chuyển theo live graph:

1. Lưu summary đủ cho một detector phân biệt positive/negative; hiện PCA box
   rộng không thay thế được mẫu âm hay discriminative function.
2. Guard được fit/compile từ positive FIT và negative summaries hợp lệ;
   không tạo raw historical CAL, không gọi negative summaries là CAL evidence.
3. Khóa detector và giữ nguyên recall/FAR/break acceptance.
4. Kiểm tra trên peer development negatives trước native integration.

Đây là thay đổi metadata/guard cần kiểm chứng, không phải lý do train full backbone
ngay. Cũng không có bằng chứng để hardcode class 13/14 hay donor blacklist.

## 7. Artifacts và tái chạy

Artifact nhỏ trong repo:
`artifacts/appliance_counter_guard_development_20261010.json`.
Nó lưu cả các thử nghiệm không đạt; không cherry-pick candidate tốt nhất.
Raw counter driver/helper trước khi thêm FIT offsets được snapshot trong thư mục
audit local; hash snapshot được ghi trong artifact. Run partial `_01` đã dừng
trước khi hoàn tất bị loại khỏi completed results. Pair65 chạy lặp không được
tính là independent replication.

Ví dụ acceptance với FIT offsets, từ root repo:

```powershell
python -m tools.audit_appliance_provenance_acceptance `
  --checkpoint audit_denice/appliance_provenance_v1/checkpoint_task_3/checkpoint_task_3.pt `
  --graphs audit_denice/appliance_coverage_v2/cluster_history.json `
  --decisions audit_denice/appliance_provenance_v1/receiver_aware_final/decisions.json `
  --calibration-store audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store `
  --out audit_denice/appliance_provenance_v1/new_fit_counter `
  --contrastive --fit-counter --development `
  --roles audit_denice/appliance_historical_calibration_local/_runtime_5f5bbe06cee041de/roles `
  --data audit_denice/appliance_transfer_results9/inputs/dataset
```

Code hỗ trợ cả 16 request; audit FIT-offset trong báo cáo chỉ chạy pair65 nhằm
falsify thiết kế trên failure mode đã biết. Không báo nó là campaign 16 request.
Các output directory phải mới, không ghi đè audit đã khóa.
