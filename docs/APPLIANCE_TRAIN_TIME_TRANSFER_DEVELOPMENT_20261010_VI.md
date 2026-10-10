# APPLIANCE — prototype Train-time Knowledge Transfer

> Báo cáo này giữ protocol All-seen lịch sử. Entry point đã đổi mặc định sang Functional-Gap-Aware Discovery; chỉ dùng `--protocol legacy_all_seen` nếu cần tái tạo diagnostic cũ. Xem `APPLIANCE_FUNCTIONAL_GAP_DISCOVERY_20261010_VI.md` cho luồng hiện tại.

## Kết luận

Đã chạy local hai clone độc lập của receiver 2 tại **Task 1 / round 19**, đúng lineage `results (13)` của audit trước. Không dùng weights Task 5 hoặc imported route. Cơ chế train-time **readout-only** chuyển được class 6 vào classifier local, nhưng **0/2 cặp đạt đồng thời gate đã khóa**. Chưa tích hợp runner, chưa native survival hoặc full training.

Đây là thay đổi phương pháp: empirical risk-controlled parameter learning/integration, không strict per-class safety certificate. Prototype hiện chỉ cập nhật một row `fc2`, không cập nhật representation; chưa thể suy rộng kết quả sang mọi dạng train-time transfer.

## Protocol đã khóa trước BASE/CAL

| Thành phần | Quy tắc |
|---|---|
| Checkpoint | Task1 terminal `b057f27a763b63bd5ffff936926f5e7d4bb0696c07b35ada8c2d4de209799a87` |
| Graph | Round19 `0c4c7acd41b31627ef32591c31aa72d561adfc2721028dead91e459d697d3a22` |
| Roles | `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15` |
| Pair | `2 ← 62 / 6` và `2 ← 64 / 8`, không thay donor sau outcome |
| Model | Legacy DeNICE, binary_cosine; không CGoFed/CME/imported routes |
| Donor | Trưởng thành ở target, có current owned BASE và positive-alpha live edge |
| Learning | Frozen receiver features và tất cả row khác; học row target bằng donor current BASE |
| Integration | Receiver-current BASE anchors + proximal preservation của donated row |
| Steps | Donor 128, receiver 64; Adam lr 0,02; L2 0,001; anchor weight 1 |
| BASE cap | 1.024 mỗi class / owner, seed cố định; không đọc BASE task cũ |
| Primary inference | All-seen 0–11, cùng label space trước/sau; một local backbone |
| Secondary inference | Router cũ, chỉ đăng ký class availability tại Task1; không refit router |
| CAL | Task1 replay current CAL H; positive ≥32, receiver negatives tổng ≥32; recall ≥95%, receiver FAR tổng/per-observed-class ≤0,1%, negative break=0, rescue>0 |
| Development risk | Old accuracy drop ≤1pp và target FP trên old FIT ≤0,1% |

All-seen là phép thử classifier được chọn trước để tách tác động của task mask. **Không được so trực tiếp accuracy All-seen trong report với headline routed DeNICE, hoặc đổi secondary thành primary sau khi đã xem outcome.** Task1 được replay offline; đây không phải một run prospective theo thời gian thật. Hai pair đã được chọn từ development Task5 nên đây là retrospective fixed-pair case studies, không chứng minh automatic discovery không dùng future information.

## Update được tạo như thế nào

```text
Receiver gửi function capsule cho donor hợp lệ trong graph
→ donor chạy frozen receiver encoder trên CURRENT BASE của chính donor
→ học một row target trong tọa độ receiver
→ gửi row + metadata/function/role binding, không gửi raw examples
→ receiver dùng CURRENT BASE của mình để regularize donated row
→ shadow model + đăng ký class availability trong router hiện có
→ khóa update trước CAL và development
→ đo acceptance/risk → giữ shadow hoặc rollback
```

Loss donor cân bằng target-vs-rest theo cạnh tranh logit trên toàn bộ seen classes. Với `r = logsumexp(other_logits)`, positive dùng `softplus(r-z_c)` và negative dùng `softplus(z_c-r)`, kèm proximal L2. Receiver dùng `softplus(z_c-r)` trên own-current anchors để hạn chế new-class probability mass, đồng thời giữ row gần donated row.

Regularization chỉ có thông tin từ current anchors; nó **không bảo đảm old data unseen không bị logit competition**. Giữ nguyên old parameter rows cũng không đồng nghĩa giữ nguyên argmax prediction.

Inference không cần donor model, capsule, competence/gate, imported detector hoặc threshold override. Class availability được đăng ký bằng task/class của dữ liệu training hợp lệ, không bằng nhãn hay task thật của test sample.

## Kết quả primary All-seen

| Pair | Target recall CAL trước → sau | Receiver current FAR | Old FIT accuracy trước → sau | Old target FP | Kết luận |
|---|---:|---:|---:|---:|---|
| 2 ← 62 / 6 | 0 → **99,8302%** (588/589) | 0/565 | 83,2512 → 82,9228% | **5/609 = 0,8210%** | CAL PASS; old-risk FAIL |
| 2 ← 64 / 8 | 95,9350 → **78,0488%** (96/123) | 0/565 | 83,2512 → 83,2512% | 0/609 | CAL FAIL; old-risk PASS |

Class 6 có 255/256 recall đúng trên donor development FIT, từ baseline 0. Current receiver CAL/FIT prediction giữ nguyên. Old witness có 0 rescue và 2 break nên accuracy delta = `(0-2)/609 = -0,3284pp`; 5 false positives gồm 2 ở class 1 và 3 ở class 3. FAR class 1 = 2/256, class 3 = 3/79. FAR và break khác nhau vì baseline có thể đã sai.

**Class 8 không phải case classifier chắc chắn thiếu knowledge:** All-seen baseline đã có recall CAL 95,9350% dù metadata `unit_rank=0` và target absent khỏi local task mask. Update readout còn phá 22 positive CAL đúng của baseline. Không kết luận nguồn latent ability là aggregation hay initialized weights chỉ từ quan sát này; nhưng có thể kết luận support/age metadata không đủ để suy ra functional absence. Donor development FIT recall cũng giảm 95,3125 → 86,3281%.

## Secondary: router bình thường, không imported route

| Pair | Target CAL recall trước → sau | Old witness FP / break | Ý nghĩa |
|---|---:|---:|---|
| Class 6 | 0 → **98,1324%** | **0 / 0** | Signal đáng giữ cho local model + router hiện có |
| Class 8 | 0 → **91,0569%** | 0 / 0 | Chưa đạt 95%; availability có tác động, chưa tách khỏi update |

Old witness routed accuracy cả hai giữ 51,5599%; class 6 donor FIT routed recall 96,0938%. Những số này cho thấy đường deployment routed có tín hiệu tốt hơn về old protection so với All-seen trong case 6. **Không gọi đó là PASS của primary đã khóa**, không tune inference policy hoặc regularization trên chính panel này để lấy headline đẹp hơn.

## Old-class qualification và giới hạn evidence

Panel đầu của receiver 2 và donor 62 không có Task0 FIT; code đã trả risk UNKNOWN/FAIL, không lấy pool rỗng làm zero-FAR PASS. Vì vậy bổ sung old-class development bằng graph witness, **không fit lại update hoặc đọc lại CAL**:

- Chọn witness bằng metadata: greedy old-class coverage, capped FIT capacity, tie theo ID. Không chọn bằng prediction quality.
- Peer **78**, alpha `0,019677100422437872`, bao phủ 0–5. Panel gồm class0=9, class1=256, class2=5, class3=79, class4=256, class5=4.
- Rebuild shadow từ row đã seal và verify **candidate function hash khớp tuyệt đối** trước khi đánh giá.
- Old FIT chỉ là offline qualification, không đưa vào donor/receiver optimization, không trở thành CAL. Physical source train NPZ chứa nhiều task; chỉ selected FIT rows của scope đã thấy được forward.
- Class 0/2/5 ít hơn 32 mẫu; không suy ra population FAR hoặc strict safety. Pooled data/content không được coi là các mẫu độc lập.
- Negative CAL Task0 không mở. CAL Task1 không đọc lại trong bổ sung witness. CAL counts tái sử dụng là kết quả của **cùng function**, không phải independent replication.

Đây là local endpoint emulator. Historical FIT qualification chưa được triển khai thành decentralized runtime evidence service; không claim có thể đọc các shard đó trong native training. Development đã được dùng ở những audit trước; không gọi nó là untouched final confirmation.

## Communication

Mỗi pair: trained readout packet **2.921 bytes ≈2,85 KiB**, nhưng receiver function capsule **6.685.240 bytes**, tổng transfer **6.688.161 bytes ≈6,38 MiB**. Đây là application-wire simulation, không phải socket/TLS hoặc toàn FL round cost.

Không có raw examples/per-example receiver features trong transfer packet. Binary router memory có sẵn đi theo function capsule; raw reference memory bị chặn. Chi phí offline old-FIT witness qualification chưa có transport implementation, được ghi riêng là chưa đo, không nhập vào tổng 6,38 MiB để làm giả “total system communication”.

## Quyết định và bước tiếp theo

**Không full train, không production install, không tự động chạy native smoke.** Cơ chế readout-only đã đạt positive transfer ở class 6 nhưng chưa qua primary old-risk gate. Class 8 bị reject do không có primary benefit và recall CAL giảm.

1. Giữ class 6 làm candidate nghiên cứu; loại class 8 khỏi đường readout-overwrite tại snapshot này.
2. Nếu tiếp tục với deployment DeNICE router có sẵn, khóa protocol đó **trước** một evaluation độc lập hợp lệ, dùng cùng update đã freeze; secondary hiện chỉ là signal, chưa đủ headline/native approval.
3. Nếu mục tiêu là All-seen, phải redesign **training regularization bảo vệ old-function competition**. Current-only negative anchors của receiver không đại diện các class cũ được inherit. Không tăng lambda hoặc đổi budget sau khi nhìn panel hiện tại.
4. Nếu sau evidence độc lập có benefit/risk chấp nhận được mới native survival → runner integration → full một seed. Không thêm imported route/guard/certificate để cứu prototype này.

Files: `appliance/train_time_transfer.py`, `tools/run_appliance_train_time_transfer.py`, `tools/complete_appliance_train_time_development.py`. Artifacts: `artifacts/appliance_train_time_transfer_development_20261010.json` và `artifacts/appliance_train_time_transfer_old_witness_20261010.json`.
