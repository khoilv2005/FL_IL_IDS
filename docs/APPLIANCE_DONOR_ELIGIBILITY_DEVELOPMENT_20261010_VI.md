# APPLIANCE — Donor eligibility và kiểm tra patch active

## Phạm vi

Checkpoint DeNICE legacy + APPLIANCE Task 5/round 19 của results (13). Graph thật với alpha > 0; không CGoFed, CME, retrain hay cài patch mới.

Dữ liệu là role FIT 8% đã tách khỏi BASE/CAL/VAL, chia theo content hash thành signature-FIT / selection / evaluation; tối đa 64 nội dung khác nhau mỗi owner–class–split. Phần cũ trong audit separability được loại theo content. Đây là development hồi cứu, không thay thế CAL acceptance và không phải final confirmation.

Gate giữ nguyên: recall ≥95%, FAR từng owner–class ≤0,1%, break=0, positive và mỗi negative pool ≥32 mẫu. Thiếu evidence không được tính là pass. 32 mẫu mỗi owner–class là quy tắc đủ evidence bảo thủ của audit này; không thay gate production CAL vốn đếm negative aggregate. 0 FP trên panel nhỏ chưa chứng nhận FAR population ≤0,1%. Native donor screen chỉ đo recall/confusion, không có receiver override nên không đo break; break=0 được kiểm tra ở shadow patch và transfer probe.

## Toàn graph

- Clients: **98**.
- Missing-owned receiver–class requests: **1630**; có donor sở hữu class trong graph: **1034**; không có: **596**.
- Owned donor edges: **4020**; loại vì receiver output không free: **1023**. Class có output mature không tự bảo đảm competence, nhưng installer hiện không thay các row này.
- Metadata candidates: **2997**; unique donor–class: **1025**.
- Maturity funnel: `{"precheck_pass": 2205, "precheck_rejected": 792}`.
- Selection: `{"failed": 1275, "insufficient_evidence": 930}`.
- Evaluation: `{"failed": 1277, "insufficient_evidence": 928}`.
- Donor pairs qua cả hai screen: **0**.
- Transfer probes thực sự chạy: **0**; pass trong scope quan sát: **0**. Không có certificate mới hoặc quyền install.

Native screen dùng **patch-free self view + binary_cosine lưu trong checkpoint**, không chạy APPLIANCE của donor và không refit multiclass. Đây là điều kiện eligibility bảo thủ, không phải upper bound của head + imported route. Donor native router sai vẫn có thể chứa head transfer được. Vì vậy zero pass của screen này không chứng minh tất cả parameter transfer bất khả thi.

Có **3** cặp đạt recall/FAR trên phần selection đã quan sát nhưng thiếu evidence cho toàn scope. Chúng vẫn là unknown, không được nâng lên safe donor và không được chọn bổ sung sau khi xem evaluation.

| Trục lỗi trên evaluation | Số cặp |
|---|---:|
| Recall positive <95% (n≥32) | 1113 |
| FAR negative vượt ngân sách | 164 |
| Phạm vi thiếu/ít mẫu | 1092 |

Các trục lỗi chồng lắp; không cộng để suy ra số cặp reject. Dừng sớm nếu positive <32 hoặc recall <95%; FAR chưa được đo cho các cặp đó. Negative screen vẫn chạy đầy đủ cho cặp có positive đạt gate và tất cả patch đã cài.

## 7 patch active và 1 patch suspended

| Receiver ← donor / class | State | Recall evaluation | FAR max owner–class | Break | Kết luận shadow |
|---|---|---:|---:|---:|---|
| 0 ← 60 / 21 | CARRY_FORWARD | 58/64 (90.62%) | 0.00% | 0 | failed |
| 9 ← 89 / 12 | SUSPENDED | 61/64 (95.31%) | 92.19% | 163 | failed |
| 13 ← 30 / 6 | CARRY_FORWARD | 59/64 (92.19%) | 0.00% | 0 | failed |
| 55 ← 14 / 20 | CARRY_FORWARD | 59/64 (92.19%) | 2.50% | 0 | failed |
| 64 ← 4 / 9 | CARRY_FORWARD | 61/64 (95.31%) | 0.00% | 0 | insufficient_evidence |
| 65 ← 88 / 20 | CARRY_FORWARD | 63/64 (98.44%) | 100.00% | 0 | failed |
| 76 ← 30 / 6 | CARRY_FORWARD | 59/64 (92.19%) | 0.00% | 0 | failed |
| 97 ← 30 / 6 | CARRY_FORWARD | 59/64 (92.19%) | 0.00% | 0 | failed |

Packet, threshold, head, reference classes và shield-at-install giữ nguyên. Shadow chạy CPU FP32 trên checkpoint hiện tại, không revalidate certificate CUDA. Baseline để tính break/rescue ở đây là binary router của receiver; không so trực tiếp với break của full-test multiclass. Negative scope gồm receiver, live peers và donor gốc. Các receiver khác nhau có thể query cùng nội dung; không cộng rescue/break giữa patch để suy ra accuracy toàn hệ thống.

### Diễn giải

- Trong 7 patch active: **6 không đạt gate quan sát, 1 thiếu evidence**. Patch 64 ← 4/class 9 đạt 61/64 positive và không false activation trên phần quan sát, nhưng không đủ số mẫu ở các protected pools để được tính safe.
- Patch 65 ← 88/class 20 có 63/64 target hits, nhưng kích hoạt trên **64/64 class 13 của owner 89 và 63/64 của owner 88**. Đây là lỗi phân biệt positive/protected negative tái xuất hiện trên development khác role.
- **Break=0 không có nghĩa FAR=0**: nếu baseline đã sai trên negative, patch nhận nhầm không tạo thêm break nhưng vẫn là false activation.
- Patch suspended 9 ← 89/class 12 có 163 break trong shadow; route này đang tắt, các break đó không phải thiệt hại đã xảy ra trong full-test deployed.
- Các patch class 6 dùng cùng donor 30 và cùng positive pool, không phải ba replication độc lập. 59/64 là point estimate trên panel nhỏ; không được suy ra population recall chắc chắn <95% hoặc toàn head-patch thất bại.
- Scope ở đây gồm các class có owned BASE support trong live neighborhood và donor gốc. Không chứng nhận mọi class ngoài scope hoặc toàn provenance lịch sử của aggregation.

## Quyết định

- Reject fixture **65 ← 88/class 20 tại Task 3** theo audit `3bda3f5`; không biến thành blacklist mọi checkpoint hoặc mọi donor cho class 20.
- Chưa chạy full training. Chưa tăng transaction budget, giảm gate, đổi guard hoặc production runner.
- Tách **donor native competence**, **receiver transfer compatibility**, **protected-negative evidence coverage** và **current CAL acceptance**. Không coi metadata ownership/maturity hoặc zero FP do thiếu mẫu là safe transfer.
- Nếu thiếu scope hoặc native router làm screen quá bảo thủ, cần evidence/qualification phù hợp với imported route trước khi kết luận phải thay toàn bộ transfer design.

## Integrity

- Models/routers bất biến: `True`.
- Content splits disjoint: `True`.
- Installed head exact + query dependencies mature: **8/8**.
- Raw historical CAL reads=0; test reads=0; installations=0; new certificates=0.
- Không chọn lại donor/threshold trên evaluation.
- Communication mới chưa đo; CPU/local simulation không phải triển khai decentralized mới.

Artifact tổng hợp: `artifacts/appliance_donor_eligibility_development_20261010.json`.
