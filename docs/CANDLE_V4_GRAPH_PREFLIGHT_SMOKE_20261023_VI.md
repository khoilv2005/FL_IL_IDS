# CANDLE V4 — graph-aware preflight và smoke hợp lệ T2 → T3

Tiếp nối: [coverage-aware negative development](CANDLE_V4_COVERAGE_NEGATIVES_DEVELOPMENT_20261023_VI.md).
Không thay lại CAL REJECT của smoke dưới đây; experiment mới chỉ đánh giá FIT.

**Preflight và engineering PASS; acceptance bank REJECT; chưa chạy full seed 42.**
Lần này hai expert thực sự học, mỗi expert 352 Adam updates. Nguyên nhân reject
là 3 false activation của expert class12 trên class13 tại owner3, không còn là
fixture thiếu cạnh hoặc draft chưa train. Không đổi candidate, threshold hoặc gate
sau khi CAL mở.

Receipts: [preflight](../artifacts/candle_v4_graph_preflight_20261023.json),
[native smoke](../artifacts/candle_v4_graph_transition_20261023.json).

## 1. Preflight trước training, không dùng future graph

Quy tắc được ghi vào `selection_rule_lock.json` trước khi tạo snapshot:

1. Restore V1 T1; thực hiện native T2 task preparation/allocation trên current BASE.
2. Trước optimizer step đầu tiên, dùng đúng native Context Capsule +
   `paper_context_cluster()`, β theo V1 và ξ=0,8, để tạo current context graph.
3. Selector chỉ nhận graph và metadata current BASE ownership, actual post-split
   BASE counts, FIT/CAL counts. Không có API FIT/CAL sample hoặc quality scores.
4. Lọc donor có cạnh hiện tại, ≥32 training positives, ≥32 FIT positives và ≥32 CAL
   positives theo metadata. Required owners phải có ≥32 CAL negatives ngoài hai target.
5. Chọn theo receiver ID → class-pair IDs → donor-tuple IDs → filler IDs tăng dần.
   Chọn đủ 4 owners trong receiver neighborhood; không ưu tiên bằng recall/accuracy.
6. Khóa fixture/config/seed/content hashes trước khi materialize FIT của selected owners.
7. Fork đúng prepared state của các owner đã chọn, tiếp tục từ round0; không prepare
   lại hoặc dùng model/graph ở round tương lai. Recheck eligibility theo actual graph
   sau local training + aggregation ở từng round.

Snapshot có 70 active clients T2; 126 ownership gaps, 103 gaps có donor đạt source
count conditions, 2.220 donor–class offers; 27 receivers có ít nhất hai source-count
eligible gaps. Những count này chưa bao gồm mọi joint-owner feasibility condition,
không phải số safe transfers hoặc functional gaps đã được chứng minh.

Graph preflight được tạo sau native allocation nhưng **0 native/expert optimizer
steps**. Snapshot dùng positive uniform placeholders để biểu diễn eligibility từ
native context edges; không dùng chúng làm aggregation weights. Native alpha thật
được tính lại mỗi training round. Deterministic replay giữ nguyên lựa chọn khi đảo
thứ tự metadata dictionary.

## 2. Fixture đầu tiên theo quy tắc

| Thành phần | Giá trị |
| --- | --- |
| Receiver | 6 |
| Expert class12 | Donor1 |
| Expert class13 | Donor2 |
| Required owners | `{1,2,3,6}` |
| Native tasks | T2 → T3, 8 rounds/task, finalization thật |
| Expert training | Balanced joint, MLP39→64→1, Adam0,001, clip5, seed42 |
| Scheduling | Quota44/round, max88 catch-up; không ép cạnh |
| Threshold rule | Donor FIT recall95 order statistic, khóa trước CAL |
| Acceptance | Recall≥95%, min32 unique positive/negative counts; observed FAR và per-negative-class FAR≤0,1%; break0; atomic bank; net gain>0 |
| Full rule | Giữ nguyên yêu cầu hai ACTIVE, không đổi hồi tố |

Metadata tại thời điểm chọn:

| Target/donor | Post-split BASE positives | FIT count | CAL count |
| --- | ---: | ---: | ---: |
| Class12 / donor1 | 1.272 | 128 | 43 |
| Class13 / donor2 | 3.687 | 1.888 | 428 |

Receiver6 chỉ có current BASE classes15/17. Đây là ownership-gap preflight, chưa
phải một phép chứng minh độc lập rằng mở availability của native model không đủ.
Không dùng kết quả CAL cũ để xếp hạng hoặc loại candidate.

Trong cả 8 round T2, actual donor edges đều hợp lệ. Mỗi expert nhận **352 updates**,
90.112 lượt mẫu (45.056 positive + 45.056 negative). Không thay optimizer, balancing,
expert architecture hoặc budget để cứu endpoint.

## 3. Kết quả CAL một lần tại T2

Thresholds đã khóa trước CAL: class12 `0.9854978323`; class13 `0.9858531952`.
Counts dưới đây là sau owner-level content dedup và current selected-owner BASE/FIT exclusion.

| Metric | Kết quả |
| --- | ---: |
| Class12 trên required donor1 CAL | **43/43 = 100% recall** |
| Class13 trên required donor2 CAL | **409/427 ≈ 95,78% recall** |
| Receiver6 false activation, cả bank | **0/127 negatives** |
| Candidate rescue, tổng prediction events của 4 owner pools | **3.312** |
| Candidate break | **0** |
| Candidate multi-gate overlap | **0** |
| Class12 false activation tại owner3 | **3/732 negatives ≈ 0,410%** |
| Class12 false activation riêng class13 tại owner3 | **3/574 ≈ 0,523%** |
| Class13 false activation | **0 trên các observed owner pools** |
| Bank decision | **REJECT**, atomic nên 0/2 ACTIVE |

Recall cả hai required targets đạt gate. Vi phạm duy nhất trong acceptance receipt
là class12 FAR tại owner3, gồm pooled owner FAR và per-class FAR. Không bỏ owner3,
gộp pool để che lỗi hoặc thay threshold sau kết quả này.

`rescue=3312` là tổng events trên các owner CAL pools; chưa dedup cross-owner thành
một test population. Nó không phải full TEST accuracy, và break0 không đồng nghĩa
false activation0: ba lỗi này xảy ra trên các mẫu native đã dự đoán sai.

## 4. Phân tích rejection bằng metadata đã có

Donor1 và receiver6 — hai phía tạo gradients cho expert12 — đều **không có current
BASE class13**. Expert13 thì có class12 negatives từ donor2. Ba FP của expert12 đều
thuộc class13 ở owner3, phù hợp với một khoảng trống hard-negative coverage.

Đây là giả thuyết có căn cứ về dữ liệu/objective coverage, chưa phải kết luận nhân
quả đã có ablation. Không thêm negative donors, đổi candidate sang donor2, train
thêm hoặc tune margin trong run này. Kết quả không chứng minh MLP thiếu capacity,
và cũng không chứng minh V4 đủ an toàn để deploy.

## 5. Transition, retention và engineering

| Kiểm tra | Kết quả |
| --- | --- |
| Native T2 → T3, checkpoints mỗi boundary | PASS |
| Native weights + algorithm/router state + RNG on/off tại T2 và T3 | Exact |
| Bank checkpoint restore | Exact |
| Cả hai expert khác initialization, có actual updates | PASS |
| Learned expert weights giữ nguyên qua T3 | Exact |
| Expert không có T2 updates tại T3 | PASS |
| Không duplicate round/step history sau resume | PASS |
| Historical raw CAL access API bị chặn | PASS |
| Current graph được recheck và gate owners có cạnh hợp lệ tại T2 | PASS |
| Native parameters hữu hạn | PASS |
| ACTIVE inference/arbitration của accepted bank | **Chưa kiểm chứng: bank REJECT** |

T3 không mở CAL vì không có bank đã được chấp nhận: `SKIPPED_NO_ACTIVE_BANK`.
Không dùng lại raw CAL/FIT T2 để đo retention trong T3 hoặc trong summarizer;
retention weights và update counters lấy từ checkpoint receipts.

Một lỗi device-placement của native round-resume xuất hiện tại T2 finalization:
reference tensors đã deserialize về CPU trong khi model ở CUDA. Đã sửa restore
đưa references/labels về model device, rồi resume từ checkpoint R7. Không train lại
T2, không có CAL outcome trước sửa, không thay candidate/gate. Source revisions,
checkpoint hash và no-CAL assertion được lưu trong receipt. Native baseline sau
sửa khớp tuyệt đối với nhánh bank đã resume.

## 6. Communication đầy đủ setup

| Thành phần | Modeled application bytes |
| --- | ---: |
| Native graph descriptor discovery, 70 clients | 27.784.368 |
| Role-count metadata advertisements | 1.414.983 |
| Expert training + one-shot gate | 37.567.625 |
| **V4 incremental, gồm toàn bộ setup** | **66.766.976 ≈ 63,67 MiB** |
| Receiver6/class12, gồm phân bổ setup | 33.383.753 |
| Receiver6/class13, gồm phân bổ setup | 33.383.223 |
| Native T2/T3 capsules + neighbor state | 891.452.976 |
| **Combined modeled total** | **958.219.952 ≈ 913,83 MiB** |

Trong budget đã khóa: bank128MiB, receiver/class64MiB, combined2GiB. Toàn bộ discovery
setup của 70 clients được tính cho fixture này, không coi dữ liệu đã trong RAM là
giao tiếp miễn phí. Đây là explicit application serialization models, không phải
real transport benchmark hoặc dự báo chính xác cho full 100-client training.

## 7. Giới hạn và quyết định

- CAL có lịch sử development, không phải untouched confirmation holdout.
- Preflight dùng BASE của cả70 clients để xây graph. CAL content exclusion kiểm tra
  BASE/FIT của selected owners; không claim independence rộng hơn từ toàn bộ footprint.
- Preparation giải nén NPZ shard chứa nhiều task/role để lọc current rows; không
  claim physical I/O chỉ materialize current rows. Trainer/selector không dùng historical CAL API.
- Đây là observed empirical risk gate, không có population FAR≤0,1% guarantee.
- Không mở TEST, không đo full-system accuracy/forgetting hoặc ACTIVE inference latency.
- Một fixture được chọn bằng quy tắc đã khóa; không có post-CAL donor/threshold search.

**Giữ NO-GO cho full seed42 theo protocol hiện tại.** Khác lần trước, blocker không
còn là feasibility/0-update fixture. Ta đã có một phép thử hợp lệ: hai expert học,
recall đạt, nhưng bank không vượt protection gate tại một owner–class pool.

## 8. Chạy lại

```powershell
python tools/run_candle_v4_graph_transition.py --out audit_denice/new_graph_smoke --mode preflight
python tools/run_candle_v4_graph_transition.py --out audit_denice/new_graph_smoke --mode bank
python tools/run_candle_v4_graph_transition.py --out audit_denice/new_graph_smoke --mode baseline
python tools/summarize_candle_v4_graph_transition.py --out audit_denice/new_graph_smoke
```

`resume-bank` chỉ dùng khi resume một round checkpoint đang có; không phải nhánh
chạy thêm để chọn kết quả. Không có full-training notebook/run mới ở mốc này.
