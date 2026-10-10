# APPLIANCE: một thí nghiệm discriminative linear guard

## Kết luận

**Chưa PASS.** Pair **receiver 65 ← donor 88 / class 20** giữ recall current CAL
**98,6928% (151/153)** nhưng vẫn có **1.507 false activation / 4.903 peer negatives
(30,7363%)**, vượt ngân sách đã khóa **0,1%**. Không tích hợp guard này vào runner,
không chạy native smoke hoặc full retrain. Không mở thêm MLP hay sweep tham số.

Đây là development trên DeNICE legacy Task 3 / round 19 từ results (13), không
phải kết quả final test hay nhiều seed. Checkpoint SHA256:
`cac0d0e08c7c60495952fbf3dda3a22088e85acba06f0317fe5f8b80d4d021bf`.

## Thiết kế đã thực hiện

1. Giữ nguyên receiver, donor, head patch, graph và sketch 16D/seed 20261008.
2. Từ current BASE của peers hợp lệ **52, 88, 89**, tạo count/mean/scatter theo
   class, tại mỗi task khi dữ liệu còn hiện hành. Sau chuyển task chỉ dùng thống kê.
   Export giới hạn ở các class có owned BASE support trong checkpoint; inherited
   binary memory không được coi là quyền sử dụng dữ liệu hoặc evidence sở hữu.
3. Positive fitting chỉ dùng **306 donor current CAL-FIT class 20**. Negative
   fitting chỉ dùng class moments BASE, gồm mọi class được peer hỗ trợ trừ class 20.
4. Fit **ridge least squares tuyến tính**, không phải Logistic Regression: positive
   và negative mỗi bên mass 1/2; negative classes có mass bằng nhau, nguồn trong
   cùng class được weight theo số BASE rows hợp lệ. Standardize từ moments,
   ridge cố định **0,001**, không penalize intercept. Không sinh synthetic examples.
5. Inference label-blind: shared sketch → `tanh(w·z+b)`; đây là ranking score,
   không phải xác suất calibrated. Có **16 weights + 1 bias FP32 = 68 bytes**.
6. Giữ cosine envelope FIT và owned BASE veto; guard cần thêm linear score > τ
   và fixed-context head margin > γ. Không cần task router chọn imported context.
7. Current CAL SELECTION chọn τ/γ, với linear-score floor **0** khóa trước.
   Threshold cuối **τ=0; γ=21,3461723; β=1**. Khóa declaration trước HOLDOUT/VAL.
   Linear threshold được lưu riêng, không đổi ý nghĩa cosine threshold của packet cũ.
8. Current CAL HOLDOUT vẫn áp dụng recall ≥95%, receiver FAR ≤0,1% theo từng
   class, break=0 và đủ ≥32 rows. BASE moments không thay thế CAL acceptance.

### Phạm vi mô phỏng

Checkpoint cũ **không chứa các moments này**. Audit tái dựng theo thời gian
Task 0→1→2→3 qua `CurrentBaseData`; raw BASE được mở khi task đó là current,
không đọc ngược sau `advance`. Đây là mô phỏng khả năng thu thập/persist, không
chứng minh run cũ đã thực sự gửi summaries hoặc knowledge receipts.

Không đọc raw CAL task cũ. Current CAL HOLDOUT từng được quan sát ở các audit
trước, nên không được gọi lần này là independent confirmation hoặc final evidence.

## Development validation

Mỗi owner/class bỏ **256 rows đầu** đã được xem trước, lấy tối đa **512 rows tiếp**
trong validation role. Không dùng validation để chọn ngưỡng hoặc fit guard.
Row roles tách biệt; audit canonical FP32 content hash cũng tìm thấy **0 exact
content overlap** giữa các mẫu đánh giá và positive FIT/negative BASE dùng fit
guard. Không suy ra độc lập thống kê theo network flow/source từ phép kiểm tra này.

| Pool | Rows | Original route activation | Linear guard activation | False activation |
|---|---:|---:|---:|---:|
| Receiver validation tail | 1.789 | 0 | 0 | 0 |
| Donor positive class 20 tail | 512 | 464 | 464 | 0 |
| Peer 52 old-negative tail | 1.536 | 502 | 501 | 501 |
| Peer 88 old-negative tail | 1.276 | 505 | 501 | 501 |
| Peer 89 old-negative tail | 2.091 | 505 | 505 | 505 |
| **Peer negatives tổng** | **4.903** | **1.512** | **1.507** | **1.507** |

Original route và linear guard được đo trên **chính cùng rows**. Guard chỉ loại
được **5/1.512** false activation cũ (~0,33%). Không so trực tiếp 1.507 với 753 ở
report trước vì panel và số rows khác nhau.

- Donor positive validation recall **90,625% (464/512)**; thấp hơn CAL, dù guard
  không làm giảm thêm so với original route trên panel này.
- Toàn bộ **1.507** peer false activations đều thuộc **class 13**:
  501/512 ở peer 52, 501/512 ở peer 88, 505/512 ở peer 89.
- **Break=0** trên các pool này không phải bằng chứng protection tốt: receiver
  đã đoán sai nhiều mẫu class 13 trước override; đổi từ một class sai sang class
  20 vẫn là false activation nhưng không thuộc `correct → wrong`.
- Linear-only AUROC **0,859999**, AP **0,302454**, trên 512 positive / 4.903 negative.
  Aggregate AUROC không thể thay FAR gate của từng class. Median score class 20
  **0,714734**; class 13 của ba peer **0,682865 / 0,676870 / 0,688115**.
- Có class thiếu validation tail vì các rows đã hết sau exclusion. Chi tiết
  `missing_tail_classes` được lưu trong JSON; không claim coverage đủ mọi class.

## Communication và integrity

Mô phỏng bằng actual serialized application messages:

| Loại payload | Bytes |
|---|---:|
| Receiver function capsule, gồm model/router/masks/state | 6.829.408 |
| Peer BASE moments + provenance, lossless gzip | 68.352 |
| Negative support receipts | 1.253 |
| Linear detector + binding | 2.031 |
| Locked threshold declaration | 2.171 |
| CAL SELECTION quantiles + counts | 9.910 |
| CAL HOLDOUT counts | 748 |
| Experimental capability, gồm head/signature/detector/declaration | 12.835 |
| **Tổng** | **6.926.708 (~6,61 MiB)** |

68-byte coefficients không phải tổng transfer cost. Capability dùng JSON/hex
để audit, chưa là tối ưu deployment codec. Không đo TLS/network traffic, latency
thực hoặc baseline graph aggregation của full round.

**17/17 integrity checks PASS**: exact capsule/head/detector restore, coefficient
tamper, dependency/version/role/feature-space changes, nonfinite input, unauthorized
peer, historical BASE runtime rejection, CAL role separation, label-blind inputs,
receiver unchanged, không lưu từng negative example trong packet. Đây là kiểm
tra audit, không phải native survival checks.

## Artifacts và tái chạy

- Summary trong repo: `artifacts/appliance_discriminative_guard_development_20261010.json`.
- Raw locks/results/wire log/source snapshot local:
  `audit_denice/appliance_provenance_v1/discriminative_guard65_04/`.
- `_01/_02` là rerun để bổ sung instrumentation; không phải independent seeds.
- `_03` dừng do hàm hash gặp empty slice; đã sửa. Không lấy partial output làm result.
- `_04` là audit hoàn tất với code snapshot/hashes. Không đổi ridge, features,
  weighting hay threshold-selection policy giữa các rerun.

```powershell
.venv-audit\Scripts\python.exe -m tools.audit_appliance_discriminative_guard --checkpoint audit_denice/appliance_provenance_v1/checkpoint_task_3/checkpoint_task_3.pt --graphs audit_denice/appliance_coverage_v2/cluster_history.json --calibration-store audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store --base-store audit_denice/appliance_current_runtime_data_local/full100_v1/base_store --roles audit_denice/appliance_historical_calibration_local/_runtime_5f5bbe06cee041de/roles --data audit_denice/appliance_transfer_results9/inputs/dataset --out audit_denice/appliance_provenance_v1/discriminative_guard65_NEW
```

Out phải là thư mục mới. Không cần chạy lại Kaggle cho audit đã hoàn tất này.

## Quyết định tiếp theo

Linear 16D theo thiết kế này **chưa chứng minh protection**. Không được kết luận
mọi linear classifier/shared sketch đều bất khả thi: mục tiêu ridge, weighting,
donor representation và coordinate projection cũng có thể góp phần. Nhưng dữ
liệu hiện tại không đủ để mở guard vào production hoặc tiếp tục tăng classifier.

Ưu tiên tiếp theo là kiểm tra khả năng phân biệt **positive–protected negative**
của shared sketch và donor compatibility trên development hợp lệ. Không hardcode
class 13, không tune margin từ các validation tail vừa đọc, không bỏ gate 95%.
Runner production và backbone training giữ nguyên.
