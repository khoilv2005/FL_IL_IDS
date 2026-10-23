# CANDLE V4 — matched positive/negative aggregation pilot

## Quyết định

**Có tín hiệu đóng góp của receiver-negative gradients vượt donor-only threshold control
trên FIT của ca này. GO cho bước native shadow có giới hạn. Chưa GO cho full/ACTIVE.**

Class balancing chưa chứng minh lợi ích thêm: hai nhánh joint hòa ở endpoint và
unbalanced đạt tiêu chí FIT tại threshold 0,5 sớm hơn trong run này.

Đây là expert-only replay với graph V1 được ghi lại và một base predictor cố định.
Native local training/aggregation không được chạy lại. Không gọi đây là native gain,
retention PASS, độc lập CAL acceptance hoặc một full benchmark.

## 1. Protocol khóa trước

- Receiver 30 ← donor 16, class 13, current Task 2, MLP 39→64→1.
- Ba nhánh: donor-only, joint negative class-balanced, joint negative unbalanced.
- Cùng init seed/hash, 880 Adam steps, lr=0,001, clip aggregated gradient norm=5.
- Mỗi step 128 positives donor +128 negatives; mỗi nhánh xử lý **225.280 examples**.
- Joint branches có cùng actual sample sequence; chỉ khác negative loss weights.
- Positive draws giống nhau ở cả ba nhánh. Receiver đóng góp **0 positive samples**.
- Class balancing: inverse negative-class frequency từ BASE counts đã stage,
  `N_negative/(number_negative_classes*N_class)`, không tính từ FIT/CAL.
- Positive/negative groups có expected loss mass 0,5/0,5. Không proximal regularization
  trong pilot này (mu=0), để không thêm confound vào contribution của receiver negatives.
- Expert parameters đều young/trainable; không sửa native backbone hoặc router.
- Final model ở expert round 19 được khóa trước; không early stop/chọn round thắng.
- Threshold 0,5 và rule donor FIT recall≥95% được báo cáo cho mọi nhánh.
- Không mở CAL examples, TEST hoặc ACTIVE gate. Tất cả fusion metrics là shadow.

Matched donor-only được train lại vì sampling schedule của aggregation khác standalone
trước. Historical standalone vẫn báo riêng, không sửa số liệu cũ: 880 steps nhưng
222.940 example forwards, so với 225.280 của các matched branches mới.

## 2. BASE và reference correctness

Receiver current BASE có **4.096 unique class 12 samples** sau cap, không có class 13.
Không dùng 516 receiver FIT class 12 để train. BASE/FIT toàn bộ participants không
trùng content; pooled FIT loại thêm **43** bản trùng giữa hai owners.

Mọi donor gradient tính trên reference receiver hiện tại, packet bind step/round/task/
class, input fingerprint và expert parameter hash. Donor trả một tensor contribution
đã gắn global group weights; receiver cộng, không FedAvg thêm lần nữa.
Adam/moments chỉ nằm tại reference owner; clip một lần sau khi cộng gradients.

| Correctness check | Kết quả |
| --- | --- |
| Same initialization | Cùng SHA-256 ở cả ba nhánh |
| Positive sample sequence | Cùng hash ở ba nhánh |
| Joint negative sample sequence | Cùng hash, balancing là thay đổi loss duy nhất |
| Nonzero donor-positive gradient steps | 880/880 |
| Nonzero receiver-negative gradient steps | 880/880 ở hai joint branches |
| Receiver positive samples | 0 |
| Pooled-gradient equivalence, max abs difference | D: 2,70e-8; balanced: 1,86e-8; unbalanced: 3,91e-8 |

Pooled-gradient check là offline algebra audit trên đúng sampled BASE mini-batch,
không phải raw-feature packet của giao thức. Training dùng actual summed peer gradients.
Các lần chạy lại chỉ bổ sung audit telemetry với nguyên cấu hình; final expert hashes
khớp giữa lần trước và lần ghi đủ loss/margin, không phải hyperparameter search.

## 3. Donor availability và lịch 880 steps

Cạnh positive-alpha 30←16 có ở **18/20 native graph snapshots**: rounds 1–17 và 19,
vắng rounds 0 và 18. Không gửi gradient/verification trên cạnh bị thiếu.

Rule quota causal: thêm 44 updates mỗi scheduled round, carry quota khi donor vắng,
tối đa 88 updates tại một round có donor. Vì vậy:

- Round 0: 0 updates, giữ 44 pending.
- Round 1: 88 updates.
- Rounds 2–17: 44 updates/round.
- Round 18: 0 updates, giữ 44 pending, expert không đổi.
- Round 19: 88 updates, tổng 880.

Round 18 chỉ reuse FIT receipt đúng expert/base/data version không đổi; không tạo
query qua edge không hợp lệ. Round 0 chưa có donor FIT receipt.
Donor-only matched control cũng dùng lịch này để giữ work/clock parity; donor local
thực tế có thể train sớm hơn, nên không gọi đây là timing superiority so với donor-only
tối ưu hoặc end-of-task timing ablation đã hoàn thành.

Case được chọn từ development và graph cuối task ở nghiên cứu trước. Replay tôn trọng
edge availability từng round, nhưng chưa phải chứng minh autonomous online discovery
từ round 0 mà không biết future graph. Native integration vẫn cần kiểm tra riêng.

## 4. Kết quả endpoint trên FIT

Donor FIT: 5.242 positives và 750 negatives. Receiver FIT: 4.237 negatives,
trong đó 516 class 12. Threshold-FIT95 là calibration diagnostic/development rule,
không phải gate được chứng nhận. Ties khiến recall thực tế hơi lớn hơn 95%.

| Nhánh/threshold | Donor recall | Donor FP | Receiver FP | Receiver class12 FP |
| --- | --- | --- | --- | --- |
| Historical standalone /0,5 | 99,771% | 1 | 506 | 505 |
| Historical standalone /FIT95 | 95,021% | 0 | 2 | 2 |
| Matched donor-only /0,5 | 99,771% | 0 | 507 | 506 |
| Matched donor-only /FIT95 | 95,040% | 0 | 2 | 2 |
| **Joint balanced /0,5** | **99,771%** | **0** | **0** | **0** |
| Joint balanced /FIT95 | 95,002% | 0 | 0 | 0 |
| **Joint unbalanced /0,5** | **99,771%** | **0** | **0** | **0** |
| Joint unbalanced /FIT95 | 95,021% | 0 | 0 | 0 |

Như vậy giảm false activation không chỉ là so với threshold 0,5: joint còn giảm
**2→0 class12 FP** so với donor-only đã FIT calibration, trong khi vẫn giữ recall≥95%.
Tại threshold 0,5, hai joint branches giữ recall cao 99,771% và đạt 0 FP trong hai FIT pools.
Không suy rộng thành lợi thế trên mọi class, mọi donor hoặc một population FAR guarantee.

### Rescue/break phải đọc cùng false activation

Base là native receiver30 từ checkpoint **Task2 terminal**, task được dự đoán từ input.
Không dùng oracle task mask. Cùng base function này được giữ cố định suốt expert replay.

| Nhánh/threshold | Rescue trên pooled unique FIT | Break | Đổi sai→sai |
| --- | --- | --- | --- |
| Matched donor-only /0,5 | 5.230 | 0 | 507 |
| Matched donor-only /FIT95 | 4.982 | 0 | 2 |
| Joint balanced /0,5 | 5.230 | 0 | 0 |
| Joint balanced /FIT95 | 4.980 | 0 | 0 |
| Joint unbalanced /0,5 | 5.230 | 0 | 0 |
| Joint unbalanced /FIT95 | 4.981 | 0 | 0 |

Pooled unique panel gồm 10.186 samples, 5.242 positives; baseline đúng 3.810 mẫu.
Ở threshold 0,5, donor-only và joint cùng đúng 9.040 mẫu: **không có thêm accuracy gain
giữa các nhánh tại cùng threshold trên panel này**, dù joint giảm target false activation.
Nguyên nhân: V1 đã sai cả 516 receiver class12 samples; donor-only chỉ đổi nhãn sai
thành nhãn sai khác. Không được gọi `break=0` là không có false activation hoặc an toàn.

Recall95 comparisons có chênh 1–2 target đúng do ties; không claim joint tăng net
rescue tại cùng recall95. Đóng góp quan sát được rõ nhất là giảm false activations
trong khi duy trì target recall, không phải final system accuracy hay V4 hơn A/C.

## 5. Learning curve và first observed useful round

Rule quan sát: donor recall≥95%, observed pooled và per-owner/per-negative-class FAR≤0,1%,
negative break=0, rescue>break. Dùng FIT lặp theo rule đã khóa; không cấp phép ACTIVE.

| Nhánh | First expert round đạt rule ở 0,5 | First ở FIT95 |
| --- | --- | --- |
| Donor-only matched | Không có | Không có |
| Joint balanced | 5 | 1 |
| Joint unbalanced | 3 | 1 |

Đây là round indices của expert replay trên frozen base, không phải native learning
curve hoặc một chứng nhận sequential. Class balancing chậm hơn ở 0,5 trên run này.
BASE đã có cap 4.096/class, nên mức imbalance nhỏ hơn full uncapped allocation;
không suy ra balancing vô ích nói chung. Không đổi variant chính bằng cách chọn
nhánh thắng hoặc round tốt hơn sau khi thấy FIT.

Report từng round còn ghi sampled weighted BCE, positive-score margins (score−threshold),
gradient norms, expert change L2, số updates/pending, recall/FP/rescue/break và bytes.

## 6. Communication thực sự được mô phỏng

| Thành phần | Joint balanced hoặc unbalanced |
| --- | --- |
| Expert references, 880 lần | 9.592.428 bytes |
| Gradient messages, 880 lần | 9.699.788 bytes |
| BASE count receipt | 63 bytes |
| Frozen V1 verification base reference, 1 lần | 4.673.763 bytes |
| FIT content-ID dedup | 401.544 bytes |
| Expert reference cho FIT, 18 lần | 191.304 bytes |
| FIT metric receipts | 21.330 bytes |
| **Tổng mỗi joint trial** | **24.580.220 bytes = 23,44 MiB** |

Matched donor-only có 5.287.941 bytes verification/monitoring/export, không có peer
gradient training messages. Các independent trials đều bị charge một verification
base snapshot; không miễn phí setup vì benchmark tái dùng cùng base để tính metrics.

Raw tensor lower bound cho 880 reference+gradient = 18.480.000 bytes, đúng bằng
44 lần 420.000 bytes từng ước lượng cho 20 single-update rounds. Payload thực tế có
headers/receipt/hash và verification. Đây là serialized application-wire simulator,
không phải measured network benchmark. Không gửi raw BASE/FIT examples trong packets.

Verification reference chỉ chứa weights/masks/adapter/BN inference state và binary
router memories, không reference inputs hoặc training/old-data caches. Predictions
của payload tối thiểu sau restore khớp native source trên cả hai FIT panels.

Native traffic chưa đo; total-system traffic chưa biết. Nếu native base đổi mỗi round,
verification reference có thể phải gửi lại; không dùng 23,44 MiB frozen-base setup này
làm ước lượng chắc chắn cho live integration hoặc full 100-client campaign.

## 7. Coverage qua 20 rounds

| Rounds | Reachable metadata gaps | Evidence-count eligible gaps |
| --- | --- | --- |
| 0–1 | 124/126 = 98,41% | 87/126 = 69,05% |
| 2–12, 19 | 121/126 = 96,03% | 85/126 = 67,46% |
| 13–18 | 117/126 = 92,86% | 85/126 = 67,46% |

Union theo thời gian: 124/126 gaps từng có donor, 87/126 từng đủ count evidence.
Không có bằng chứng từ Task2 này để đổi toàn bộ graph hoặc thêm đường cross-neighborhood.
Đây là ownership metadata opportunities; functional gap, expert competence và
deployment coverage chưa suy được từ các tỷ lệ đó.

## 8. Giới hạn và mốc tiếp theo

- FIT đã dùng nhiều lần trong development; chưa có independent CAL acceptance.
- 0 FP trên 516 class12 samples không chứng nhận population FAR≤0,1%.
- Chưa có receiver positives class13, nên target recall vẫn chỉ đo ở donor domain.
- Chưa đo old-task false activation/forgetting hoặc expert-bank interactions.
- Chưa native integration: graph trace và mature V1 base là inputs của simulator.
- Core V4 vẫn 367 source V1 nguyên bản; code pilot nằm trong tools ngoài core.

**GO tiếp:** native DRAFT/shadow integration có bounded budget, current-only data,
isolate expert RNG/parameters khỏi V1, checkpoint/resume toàn expert state, dùng actual
neighbors của run mới. Kiểm tra shadow gain với current native predictor từng round.
**NO-GO hiện tại:** full training, ACTIVE expert gate và claim V4 thắng A/C.

Artifacts: `artifacts/candle_v4_joint_pilot_20261023.json`,
`artifacts/candle_v4_round_coverage_20261023.json`.
Local full telemetry/models/optimizer checkpoints nằm trong
`audit_denice/candle_v4_joint_complete_20261023/`; không push raw data hoặc checkpoints.
