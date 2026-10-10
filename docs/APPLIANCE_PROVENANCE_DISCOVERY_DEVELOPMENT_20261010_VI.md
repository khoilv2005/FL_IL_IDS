# APPLIANCE: provenance protection và receiver-aware discovery

## Trạng thái

Đã implement các thành phần thử nghiệm P0/P1/P2 và chạy audit offline trên
checkpoint hiện có. **Chưa tích hợp vào training runner, chưa chạy native smoke,
chưa chạy full training mới.** Cấu hình production và certificate cũ giữ nguyên.

Gate vẫn là recall CAL ≥95%, FAR ≤0,1% theo class, break = 0, đủ support tối thiểu.
Không tăng transaction budget, không sửa ξ=0,8, không CGoFed/CME.

Kết quả hiện tại chưa đạt điều kiện chuyển cả policy mới sang native smoke:
protection đủ rộng để chặn class ngoài support cũng có thể chặn class import.

## 1. Dữ liệu và phạm vi

- Fixture: terminal Task 3 / Round 19 của `results (13).zip`, DeNICE legacy + APPLIANCE.
- SHA256 terminal: `cac0d0e08c7c60495952fbf3dda3a22088e85acba06f0317fe5f8b80d4d021bf`.
- Graph: đúng snapshot Task 3 / Round 19 trong `cluster_history.json`.
- Roles SHA256: `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.
- Development: clean validation của receiver, donor và peer hợp lệ; tối đa 256
  mẫu/class/owner, tất cả patch được cài lần đầu tại Task 3, không chọn pair theo test.
- Donor ranking: chỉ **CAL FIT của task hiện tại**. Threshold của candidate mới
  chọn trên CAL SELECTION, khóa trước CAL HOLDOUT; không fallback donor theo HOLDOUT.
- Không đọc final test hoặc raw CAL lịch sử. Không train backbone.
- CPU FP32 / Torch 2.14. Đây là guard thử nghiệm mới, không tái chứng nhận guard CUDA cũ.

Checkpoint không có receipt theo class-row của lần aggregation thực tế. Vì vậy
audit mô phỏng nguồn support qua cạnh positive-alpha, không chứng minh run cũ đã
truyền provenance hay một class-row cụ thể. Số coverage dưới đây là tiềm năng.

## 2. Code đã thêm

### P0 — `appliance/provenance_protection.py`

- Tách own BASE support và evidence peer; không biến peer evidence thành CAL
  hoặc ownership của receiver.
- Kiểm tra owner, role/preprocessing/sketch, task cutoff, positive edge,
  source/model/dependency version và receipt checksum.
- Receipt `aggregation`/`bootstrap` cần caller ghi class-row contribution thật.
  Audit hồi cứu chỉ mô phỏng, không thay thế instrumentation này.
- Receipt `protection_only` chỉ ghi nhận summary âm đã nhận. Nó **không** chứng
  minh receiver đã nhận classifier cho những class đó.
- Veto không cần nhãn; snapshot cố định gắn với guard mới. Không append thêm veto
  vào certificate đã cài.
- Snapshot bind implementation và NumPy version; restore/veto fail khi binding đổi.
- Geometry cache dùng content digest, checkpoint không lặp geometry theo từng receipt.
- Truyền source lossless zlib một lần/receiver/source version; các lượt sau chỉ
  gửi receipt nhỏ. Quota failure rollback ledger, bytes đã gửi vẫn được tính.
- Không relay peer evidence dưới tên own BASE. Checksum không phải chữ ký Byzantine.

### P1/P2 — `appliance/receiver_aware_discovery.py`

- Donor gửi head metadata trước; receiver kiểm tra dependency maturity trước capsule.
- FIT probe kiểm tra khả năng head hoạt động trên receiver feature và nguy cơ
  activation trên current receiver FIT, có support veto.
- Rank bằng compatible-positive LCB trừ receiver FIT FAR; donor quality LCB làm
  tie-break. Không dùng aggregation alpha làm competence score.
- Bind head/reference/dependency version; từ chối model thay đổi sau precheck.
- Donor FIT positives chạy qua receiver replica không tạo receiver-owned recall.
  CAL acceptance 95% vẫn là gate riêng, bắt buộc.
- Setup capsule và chi phí probe chưa được đo như một production transaction.

## 3. Metadata coverage và chi phí

| Task 3 / 80 receiver | Kết quả |
|---|---:|
| Cặp receiver–class thiếu own protection | 329 |
| Cặp còn thiếu sau mô phỏng peer support | 1 |
| Receiver thiếu own support | 44 |
| Receiver có đủ support tiềm năng cho old mature classes | 79/80 |
| Wire gửi nguyên summary | 145,74 MiB |
| Wire zlib + cache, mode class-row | 66,64 MiB |
| Wire zlib + cache, mode protection-only | 67,65 MiB |

Đây là một snapshot, không phải overhead cả campaign. Không gọi toàn bộ transfer
là vài KiB: summary/cache, capsule và probe phải được tính thêm vào payload head.
Đủ support geometry không đồng nghĩa đã đủ CAL hoặc đủ điều kiện install.

## 4. Development: lợi ích và failure mode

Các pool giữ nguyên sample, head, router và threshold đã cài. Chỉ thay protection
trong phép tính shadow. Mỗi pool có transition accounting riêng; không gộp các
pool khác distribution thành headline accuracy toàn hệ thống.

| Trên 9.658 peer-validation negatives | Guard cũ | Peer class-row veto | Protection-only veto |
|---|---:|---:|---:|
| False activation | 772 | 759 | 1 |

| Trên 768 donor-validation positives | Guard cũ | Peer class-row veto | Protection-only veto |
|---|---:|---:|---:|
| Target correct | 715 | 714 | 482 |

Receiver-validation thêm 4.364 mẫu không có imported activation trong panel này.
Không được xem nó là bằng chứng đã kiểm tra đủ class ngoài local support.

Điểm quyết định: receiver 65 / class 20, khi bảo vệ cả negative support từ peer:

- Donor-validation positive correct **235/256 → 2/256**.
- Peer-negative false activation **762 → 1** trên 3.284 mẫu.

Veto rộng chặn lỗi nhưng đồng thời chặn gần hết target positives. Đây là failure
của khả năng phân biệt trong **shared 16D + union box veto đang dùng**; chưa đủ
bằng chứng để kết luận mọi portable signature hay mọi 16D representation đều thất bại.

### Đính chính giả thuyết về class 13/14

Ở checkpoint Task 3:

- Receiver 0: `fc2[14]` rank 0; live peers không có own BASE summary class 14.
- Receiver 55: `fc2[13]` rank 0; live peers không có own BASE summary class 13.
- Receiver 65: `fc2[13]` rank 0; peers 52/88/89 có own summary class 13.

Vì vậy chưa có bằng chứng các receiver đã sở hữu classifier cho các class gây
false activation. P0 chỉ bảo vệ inherited mature class rows sẽ bỏ sót regime này.
Không suy ra backbone hoàn toàn không encode chúng; class-level competence chưa
được chứng minh. Failure ngoài support đã được xác nhận bằng validation riêng,
không hardcode class 13/14 hoặc chọn threshold bằng full-test labels.

## 5. Receiver-aware discovery và current-CAL acceptance

Audit toàn bộ 16 request vốn được chọn tại Task 3, cùng graph và budget cũ:

- 47 donor candidates.
- 22 bị loại bởi maturity precheck trước capsule; 25 được probe FIT.
- 4 request không có donor mature hợp lệ.
- 2 lựa chọn donor thay đổi: receiver 4/class 22 chọn 13 thay 8;
  receiver 54/class 20 chọn 4 thay 86.

Không so 22 donor candidates này với 19 transaction rejects của cả campaign:
hai mẫu số khác nhau.

Sau khóa donor trên FIT, đánh giá CAL SELECTION/HOLDOUT của candidate mới:

| Policy | Request | CAL được đánh giá | Pass CAL | Install/native |
|---|---:|---:|---:|---:|
| Run cũ Task 3 | 16 | theo log cũ | 3 commit | run cũ |
| Receiver-aware + peer class-row protection | 16 | 12 | 4 | 0 |
| Receiver-aware + protection-only toàn negative support hợp lệ | 16 | 12 | 2 | 0 |

Bốn candidate pass ở mode class-row: receiver 0/c21, 4/c22, 55/c20, 65/c20.
Hai candidate pass ở mode protection-only: receiver 4/c22 và 55/c20.

Receiver 0/c21 đạt CAL recall 94,9405% ở mode protection-only: **reject**, không
nới gate 95%. Receiver 65/c20 đạt 0%: cũng reject. Các gate vẫn hoạt động đúng.

Đây là retrospective current-CAL development của checkpoint cũ. Guard declaration
mới không grant quyền deploy, không ghi đè certificate cũ, không phải independent
final test. Class-row mode tăng pass nhưng chưa sửa false activation ngoài support;
mode rộng hơn giữ safety bằng rejection và chưa chứng minh tăng coverage hữu dụng.

## 6. Kiểm tra và quyết định

- 15/15 controls về provenance/snapshot; 12/12 controls về transport/cache,
  rollback, ownership separation, implementation drift, maturity và ranking.
- Full FIT probe kiểm tra model/router state không bị thay đổi.
- 62/62 source hash trong production smoke lock cũ khớp.
- Notebook training full hiện tại **chưa sử dụng** các module thử nghiệm này.

**Chưa đạt gate để chạy full mới.** Không cần retrain backbone để kiểm tra tiếp.
P1/P2 có thể giữ làm nền, nhưng P0 còn phải phân biệt target và peer negative
support thay vì veto toàn union box. Cần giữ ≥95% CAL recall và xác nhận trên
development ngoài support; sau đó mới bind guard mới, instrumentation receipt
thật vào aggregation/bootstrap, chạy native smoke và save/resume.

Không giảm gate, tăng budget hoặc đưa policy over-veto vào production chỉ để
tăng coverage trên giấy. Số full-test hiện đã đo vẫn là APPLIANCE **37,8620%**;
audit này không tạo một accuracy toàn hệ thống mới.

## 7. Chạy lại audit

Checkpoint đã được giải nén và checksum-verified ở:
`audit_denice/appliance_provenance_v1/checkpoint_task_3/checkpoint_task_3.pt`.
Các command chạy từ root repo, dùng Python environment có dependencies của dự án.

```powershell
python -m tools.audit_appliance_provenance_protection `
  --checkpoint audit_denice/appliance_provenance_v1/checkpoint_task_3/checkpoint_task_3.pt `
  --graphs audit_denice/appliance_coverage_v2/cluster_history.json `
  --out audit_denice/appliance_provenance_v1/new_development `
  --development --protection-only `
  --roles audit_denice/appliance_historical_calibration_local/_runtime_5f5bbe06cee041de/roles `
  --data audit_denice/appliance_transfer_results9/inputs/dataset

python -m tools.audit_appliance_receiver_aware_discovery `
  --checkpoint audit_denice/appliance_provenance_v1/checkpoint_task_3/checkpoint_task_3.pt `
  --graphs audit_denice/appliance_coverage_v2/cluster_history.json `
  --history audit_denice/appliance_coverage_v2/stored/automatic_history.json `
  --calibration-store audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store `
  --out audit_denice/appliance_provenance_v1/new_receiver_aware

python -m tools.audit_appliance_provenance_acceptance `
  --checkpoint audit_denice/appliance_provenance_v1/checkpoint_task_3/checkpoint_task_3.pt `
  --graphs audit_denice/appliance_coverage_v2/cluster_history.json `
  --decisions audit_denice/appliance_provenance_v1/new_receiver_aware/decisions.json `
  --calibration-store audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store `
  --out audit_denice/appliance_provenance_v1/new_acceptance --protection-only
```

Bỏ `--protection-only` để chạy riêng mode class-row. Output directory phải mới;
script không ghi đè một audit đã khóa. Artifact nhỏ được lưu trong repo:
`artifacts/appliance_provenance_discovery_development_20261010.json`.
