# APPLIANCE V2 — Chiến dịch functional gap có giới hạn

## Quyết định

**Đã tìm được ca donor learning có đóng góp vượt Availability-only, nhưng chưa có update đạt acceptance. Dừng chiến dịch tại ngân sách đã khóa; chưa native smoke hoặc full training.**

Receiver 3/class 7 khác ca receiver 2/class 6 trước đây: Availability-only nhận đúng **0/98**, sau donor training + receiver integration nhận đúng **92/98 = 93,8776%** trên donor CAL-HOLDOUT. Như vậy không thể kết luận mọi cải thiện chỉ đến từ mở availability. Tuy nhiên recall vẫn thấp hơn gate 95%, và independent cumulative old-risk vẫn chưa có.

## Phạm vi khóa trước prediction

Tool: `tools/run_appliance_bounded_gap_campaign.py`.

- `prepare`: chỉ đọc checkpoint, graph và role/store metadata; không đọc CAL arrays, BASE arrays hoặc chạy model prediction.
- `run`: kiểm tra protocol SHA256, checkpoint/graph SHA256, role/store SHA256 và source hashes trước execution.
- Task 1 và Task 2, round 19, cùng lineage `results (13).zip`; config base method `legacy`, ξ=0,8.
- Receiver 2 bị loại khỏi search mới. Receiver/donor có adapter hoặc imported-head registry bị loại bằng checkpoint metadata. Không suy ra đây là một chiến dịch fresh training hay independent replication.
- Tối đa **6 receiver–class cases, 12 donor–class probes, 2 trial updates**, application egress **128 MiB**.
- Time budget 900 giây, kiểm tra tại ranh giới case/trial; đây không phải hard process timeout giữa một phép tính.
- CAL-FIT và CAL-HOLDOUT mỗi class tối đa 512 unique rows theo sampling seed khóa trước. BASE giữ cap 1024/class và learning rules cũ.
- Chọn class/task tăng dần, receiver ID hợp lệ đầu tiên chưa dùng trong task. Donor xếp theo breadth của current negative classes, capped positive FIT count, rồi donor ID. Không dùng prediction hoặc test labels để xếp metadata.
- Positive BASE ownership, maturity, current refresh provenance, positive graph alpha, optimistic FIT/HOLDOUT counts được kiểm tra trước capsule. Metadata counts không tự chứng minh unique evidence; actual pools vẫn dedup và kiểm tra lại.
- Không bổ sung case/donor sau outcome; không thử lại learning rate, regularization, threshold; không thay donor sau HOLDOUT.

Protocol execution: `audit_denice/appliance_bounded_gap_campaign_v1/locked_02/protocol_before_CAL.json`. Lần `locked_01` chỉ chuẩn bị metadata, chưa chạy CAL; sau bổ sung BASE-negative precheck và ghi đúng learning trace, khóa lại `locked_02` với **cùng danh sách cases** trước prediction đầu tiên.

| Task | Receiver | Target | Donors khóa trước |
|---|---:|---:|---|
| 1 | 1 | 6 | 22, 23 |
| 1 | 3 | 7 | 54, 51 |
| 1 | 11 | 8 | 34, 54 |
| 2 | 6 | 12 | 96, 74 |
| 2 | 1 | 13 | 74, 34 |
| 2 | 3 | 14 | 74, 26 |

Metadata enumeration tìm thấy 79/73 structural receiver–class options và 1515/679 donor–class pairs ở Task 1/2. **Chỉ 12 pair đã được probe**, không gọi các options chưa đo là functional gaps.

## Quy tắc trial giữ nguyên

Không thay `appliance/functional_gap_discovery.py` hoặc production runner:

1. Shadow Availability-only; weights/masks/ranks giữ nguyên.
2. Donor và receiver kiểm tra bằng current CAL-FIT của từng owner.
3. Phân loại registration/functional/unknown gap.
4. Mọi negative veto đã quan sát trong **hai donor được khóa của case** vẫn có hiệu lực. Không chọn một donor benign để bỏ qua evidence bất lợi từ donor còn lại.
5. Chỉ functional gap đủ evidence và không veto mới trial; chọn theo FIT, tối đa hai trials trong thứ tự cases khóa trước.
6. Donor huấn luyện target readout trong receiver coordinates trên **current owned BASE**. Receiver tích hợp bằng **own current negative BASE**. Frozen encoder và other classifier rows; donor steps 128, integration steps 64, lr 0,02 giữ nguyên.
7. Khóa updated function trước HOLDOUT. Gửi integrated row tới cả hai donor để đo **function mới**, không tái sử dụng negative score của Availability-only.
8. So learned update với Availability-only trên **cùng HOLDOUT inputs/labels tại từng endpoint**. Inference chỉ nhận x, y chỉ dùng tính metric sau prediction.

Absence of veto trong hai donor **không phải safety trên toàn graph hoặc cumulative old classes**. Có current empirical qualification cũng chưa cấp quyền install.

## Kết quả FIT và quyết định

| Task/receiver/target | Gap tại donor probes | Kết quả |
|---|---|---|
| T1 / 1 / 6 | 2 registration gaps | Veto: false activation class 11 ở cả hai donor |
| T1 / 3 / 7 | 2 functional gaps | Trial donor 54 |
| T1 / 11 / 8 | 2 functional gaps | Veto peer 34; không bỏ veto bằng cách chọn peer 54 |
| T2 / 6 / 12 | 2 functional gaps | Veto false activation class 13/14 |
| T2 / 1 / 13 | 2 functional gaps | Trial donor 74 |
| T2 / 3 / 14 | 2 functional gaps | Không trial: đã hết ngân sách 2 updates |

Tổng: **2 registration, 10 functional, 0 unknown donor–class probes**. Không phải số client được cải thiện hay được cài patch.

Veto đáng chú ý:

- Receiver 1/class 6: donor 22 ghi **117/512** class 11 thành class 6; donor 23 ghi **83/359**. Availability-only có target recall tốt vẫn không được đăng ký tự động.
- Receiver 11/class 8: donor 34 ghi **1/3** class 10 false activation, có 1 negative break; donor 54 benign không xóa evidence này.
- Receiver 6/class 12: false activation class 13/14 tại cả donor 96 và 74; giữ gate FAR 0,1% theo từng observed owner–class.

## Trial trên HOLDOUT đã khóa

Các số primary dưới đây thuộc **donor được chọn**, không phải full-test accuracy hay receiver population recall (receiver không có positive target trong current CAL).

| Trial | Positive HOLDOUT | Availability-only | Learned update | Rescue / break | Current HOLDOUT gate |
|---|---:|---:|---:|---:|---|
| T1, 3 ← 54 / class 7 | 98 | 0/98 = 0% | 92/98 = **93,8776%** | 92 / 0 | FAIL recall <95% |
| T2, 1 ← 74 / class 13 | 136 | 58/136 = 42,6471% | 59/136 = **43,3824%** | 1 / 0 | FAIL recall <95% |

### Class 7

- Protected donor 51: Availability-only **0/51**, updated **48/51 = 94,1176%**, 48 rescue / 0 break.
- Primary donor negatives: **0 FP / 667**.
- Receiver 3 negatives: **0 FP / 442**.
- Protected donor 51 negatives: **0 FP / 540**.
- Tổng đã quan sát: **140 rescue / 0 break** trên ba endpoint HOLDOUT pools; không suy ra 140 globally distinct samples giữa owners, không dùng pooled result để bỏ gate của primary donor.
- Old Task 0 inputs không nằm trong current Task 1 CAL. Một số negative owner–class chỉ có 1–4 mẫu. Không có chứng nhận broad safety hay population FAR ≤0,1% từ các số 0 FP này.

### Class 13

- Protected donor 34: **15/34** trước và sau, không có gain.
- Negatives của donor 74/receiver 1/donor 34 lần lượt **395/107/47**, đều 0 FP và 0 break trên current pools.
- Improvement primary chỉ **1 positive sample**; không đủ recall gate và không có bằng chứng khái quát độc lập.

**0/2 trial updates đạt current empirical HOLDOUT gate. 0 install/native smoke được cấp phép.** Không hạ 95% để đổi tên class 7 thành PASS, không lấy kết quả này làm lý do thử thêm donor hoặc hyperparameter trong chiến dịch đã đóng.

## Communication và thời gian

- Execution CPU local: **62,79 giây**, không gồm extraction Task 2 archive và metadata preparation.
- Tổng simulated application egress **82.358.578 bytes = 78,5433 MiB**.
- Full receiver capsules: **82.216.864 bytes**, 12 capsule transfers.
- Current metadata requests/responses: **97.925 bytes**, 254 exchanges mỗi chiều.
- Còn lại: query/evidence summaries, 2 learned updates và 4 integrated-row HOLDOUT queries/responses.
- Không gửi raw examples hoặc per-example features/labels trong message payloads. Không có claim differential privacy.

Con số thấp hơn scan 41 donor trước (**261,49 MiB**) vì **giới hạn search**, không phải vì đã làm receiver capsule nhẹ đi. Hai scan có scope khác nhau; không dùng tỷ lệ này làm một accuracy/communication benchmark ngang hàng. Capsule vẫn chiếm >99% egress; không gọi toàn transfer là vài KiB.

Byte budget là federation-wide cho **chiến dịch mô phỏng này**, kể cả metadata và setup, không chỉ một quota riêng cho mỗi donor. Chưa tính socket/TLS hoặc baseline FL communication.

## Integrity và evidence limits

- Đối chiếu lại 30 evidence summaries: rows=positive+negative; FAR/recall đúng numerator/denominator; net rescue=rescue−break; per-class FP và negative counts khớp tổng.
- Tổng ledger bytes khớp sent totals, dưới budget 128 MiB.
- Protocol checksum khớp; source receiver/replica hashes không đổi sau probes/trials.
- Không mở original FIT/VALIDATION/final test, không đọc raw historical CAL. BASE chỉ đọc ở hai trial thực sự chạy.
- CAL-HOLDOUT loại nội dung trùng FIT/SELECTION tại cùng owner. **Cross-owner independence và prior exposure independence chưa được xác lập**. Đây là retrospective development, không phải confirmation mới.
- Independent cumulative old-risk receipt authority chưa có; không dùng boolean, provenance metadata hoặc current summaries để thay thế.

Aggregate cùng locked protocol: `artifacts/appliance_bounded_gap_campaign_20261010.json`.

## Chốt sau điểm dừng

1. **Discovery không còn là blocker chính:** tìm được actual functional gap đủ current precheck, donor training đã tạo gain vượt việc mở mask ở class 7.
2. **Cơ chế chưa đủ acceptance:** class 7 thiếu recall gate; class 13 vẫn nhận biết kém; independent cumulative old-risk thiếu ở cả hai.
3. **Đóng bounded campaign tại đây.** Không mở rộng lên toàn graph, không tune để cứu class 7, không chuyển notebook cũ sang full training dưới tên V2.
4. Không kết luận train-time learning vô ích hoặc APPLIANCE chắc chắn thất bại: gain 0→93,88% là evidence cơ chế có thật. Nhưng cũng chưa đủ để coi transfer là contribution deployable.
5. Nếu chọn ưu tiên functional availability repair, phải giữ risk acceptance: receiver 1/class 6 cho thấy repair vẫn có thể gây hại. Đây là một hướng đóng góp khác, không được ghi nhận như donor knowledge learning.

### Cách chạy lại trên một output mới

Tại repo root, dùng `.venv-audit/Scripts/python.exe -m tools.run_appliance_bounded_gap_campaign`:

```text
--phase prepare
--task1 <checkpoint_task_1_all_rounds.zip>
--task2 <checkpoint_task_2_all_rounds.zip cùng lineage>
--roles <locked role directory>
--cal-store <current calibration_store>
--base-store <current base_store>
--out <new output directory>
```

Sau đó `--phase run` với cùng `--roles`, `--cal-store`, `--base-store`, `--out` và `--publish <aggregate.json>` nếu cần. `run` dùng checkpoint coordinates trong locked protocol, không enumerate/chọn lại. Không ghi đè output execution cũ.
