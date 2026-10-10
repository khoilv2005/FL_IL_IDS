# APPLIANCE — Actual Transfer Eligibility Audit

## Protocol đã khóa

Task 5/round 19, checkpoint DeNICE legacy + APPLIANCE của results (13), SHA256 `3003593d64e3d63a81200b90d1a6e7487fff6c0875e3518cd58c2ccad22fe391`. Dùng graph thật với alpha dương, ξ=0,8. Không train backbone, cài patch, sửa runner hoặc thay gate.

Native donor `binary_cosine` không còn sàng lọc candidate. Đo donor FC2 head trên **receiver features**, shared sketch 16D và guard `signature > tau AND fixed-context margin > gamma AND NOT own BASE veto`. Router của receiver chỉ tạo baseline để đếm break/rescue; donor router không quyết định imported activation.

Hai nhánh: (1) control có tau=donor-FIT p95 floor, gamma=0; (2) cùng guard production, quantile grid 31 điểm và tie rule chọn trên **development selection** của receiver/donor. Không thêm classifier hoặc feature mới. Nhánh (2) dùng đầy đủ owned development classes của hai endpoint; không giả danh đây là current CAL production.

Dữ liệu là các original FIT panels đã khóa ở audit trước: signature-fit / selection / evaluation tách content, tối đa 64 nội dung/owner/class/split. Đây là **development hồi cứu đã được dùng**, không phải untouched confirmation. Toàn bộ guard selection khóa trước actual-transfer evaluation. Gate: recall ≥95%, FAR từng owner–class ≤0,1%, break=0; positive và từng protected pool ≥32 mẫu. Đây là sufficiency rule bảo thủ của audit, không đổi production aggregate CAL gate.

## Funnel và kết quả

| Bước | Số candidate |
|---|---:|
| Metadata candidates | 2997 |
| Qua maturity | 2205 |
| Thiếu usable signature FIT → UNKNOWN, chưa probe | 942 |
| Đã chạy actual head+route+guard | 1263 |

| Evaluation | Guard control | Guard chọn trên development |
|---|---:|---:|
| PASS trong scope quan sát đủ evidence | 0 | 0 |
| FAIL | 1204 | 1196 |
| UNKNOWN do evidence | 59 | 67 |

Candidate được chọn trước evaluation và pass cả hai split: **0**. Không cộng số PASS evaluation-only thành donor đủ điều kiện.

Trục lỗi evaluation (có chồng lắp):

| Trục | Control | Calibrated |
|---|---:|---:|
| Recall positive <95%, n≥32 | 1121 | 1143 |
| Head thắng task-reference ≥95% positive (chưa có routing/veto) | 899 | 899 |
| FAR protected negative vượt gate | 767 | 452 |
| Break >0 | 558 | 239 |
| Thiếu/ít evidence positive hoặc negative | 1263 | 1263 |

## Candidate có tín hiệu nhưng chưa đủ evidence

Có **23** candidate đạt toàn bộ metric **đã quan sát trên cả selection và evaluation**, thuộc 12 receiver; vẫn UNKNOWN nếu còn protected pools dưới 32 mẫu. Trong đó **23** candidate bị native evaluation đánh FAIL nhưng actual function đạt metric quan sát. Đây là bằng chứng native screening không tương đương transfer eligibility; không phải chứng minh safe install.

Không cộng rescue/break trên các receiver/candidate dùng trùng samples thành accuracy hệ thống. Chi tiết candidate và missing pools có trong artifact JSON. Scope chỉ bao gồm class sở hữu BASE của receiver/live peers; class ngoài scope không được chứng nhận. 0 false activation trên ≤64 mẫu không chứng minh population FAR ≤0,1%.

23 cặp có tín hiệu này thiếu tổng cộng **408 owner–class pools khác nhau**; phân bố class: `{30: 31, 33: 36, 29: 24, 27: 23, 2: 12, 32: 15, 10: 27, 5: 8, 16: 28, 28: 30, 31: 31, 13: 4, 26: 20, 0: 12, 25: 14, 17: 11, 3: 11, 12: 3, 7: 5, 18: 21, 24: 10, 23: 6, 19: 3, 15: 8, 22: 3, 21: 2, 1: 1, 9: 1, 20: 3, 11: 2, 4: 2, 8: 1}`. Toàn bộ có <32 unique contents khả dụng trong evaluation role đã khóa: tăng cap 64 hoặc batch size không giải quyết được. Cần evidence hợp lệ bổ sung hoặc phạm vi chứng nhận có thể kiểm chứng; không lấy test/BASE/provenance giả làm CAL.

Ví dụ đầu theo thứ tự ID (không chọn donor mới bằng evaluation):

| Receiver ← donor / class | Recall selection | Recall evaluation | FAR max evaluation | Pool thiếu evidence |
|---|---:|---:|---:|---:|
| 1 ← 3 / 6 | 95.31% | 95.31% | 0.00% | 122 |
| 1 ← 67 / 6 | 95.31% | 95.31% | 0.00% | 122 |
| 1 ← 98 / 6 | 98.44% | 95.31% | 0.00% | 122 |
| 2 ← 62 / 6 | 96.88% | 95.31% | 0.00% | 17 |
| 2 ← 64 / 8 | 96.88% | 96.88% | 0.00% | 17 |
| 11 ← 1 / 8 | 98.44% | 95.31% | 0.00% | 85 |
| 16 ← 7 / 6 | 98.44% | 96.88% | 0.00% | 113 |
| 16 ← 11 / 9 | 100.00% | 95.31% | 0.00% | 113 |

## Current CAL và patch đã cài

Current CAL status: `{'not_opened_development_not_passed': 2205}`. Chỉ guard đã pass development mới được mở CAL holdout Task 5. Class cũ không có positive CAL trong Task 5 → UNKNOWN; không đọc lại CAL cũ. Nếu có phép đo current CAL thì signature/guard vẫn khóa, không tune CAL holdout. Không tạo certificate hoặc install; prototype development-FIT không được gọi là một production current-CAL-FIT compile mới.

7 patch active và 1 suspended giữ nguyên checkpoint. Tái sử dụng exact frozen shadow kết quả audit trước, cùng checkpoint/panels/packet, có source SHA; **không phải replication mới**. Active shadow status: `{'failed': 6, 'insufficient_evidence': 1}`. 6/7 active fail observed gate, 1/7 thiếu evidence vẫn là cảnh báo. Patch suspended không được tính thành thiệt hại deployed. Không mở rộng activation scope, không sửa certificate/registry trong audit.

## Integrity và giới hạn

- Model/router bất biến: `True`.
- Cache imported features chỉ cho checkpoint không adapter; so với `stable_signals` ở 81 receiver: `True`, prediction/activation match. Numerical checks dùng selection, không sửa threshold trên evaluation.
- Guard lock không đổi sau evaluation: `True`.
- Recompute accounting 5052 bộ metric và 15 synthetic controls (12 so selector production + 3 missing-evidence): PASS.
- Historical raw CAL reads=0; final test reads=0; installations=0; certificate mới=0.
- Số CAL reads ở đây là **truy cập role evidence**: FIT loader phải giải nén original train shard để trích đúng FIT indices. Không dùng nhãn/mẫu CAL lịch sử để fit hay chấm; không khẳng định zero byte-level I/O của các row khác trong shard nén. Đây là audit hồi cứu, không phải runtime replay-free đã triển khai.
- CPU shadow không revalidate CUDA certificate. Baseline break là receiver checkpoint binary router, không phải full-test multiclass refit.
- Không đo communication deployment mới; audit local CPU không được gọi là protocol decentralized đã triển khai.

## Quyết định tiếp theo

Tách ba tầng: functional metric quan sát, đầy đủ protected evidence và current CAL authorization. Không dùng 0 native pass làm lý do bác bỏ head transfer; cũng không dùng partial success để cài patch. Chưa chạy full training. Nếu có candidate giữ recall/FAR tốt nhưng thiếu scope, ưu tiên coverage evidence hợp lệ; nếu có đủ evidence mà actual function vẫn FAIL thì loại đúng cặp/checkpoint đó trước khi xem xét redesign.

Artifact: `artifacts/appliance_actual_transfer_eligibility_development_20261010.json`. Script: `tools/audit_appliance_actual_transfer.py`.
