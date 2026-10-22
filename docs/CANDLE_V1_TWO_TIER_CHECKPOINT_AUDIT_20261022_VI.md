# CANDLE V1 — audit Bitmap / Two-Tier / Head / Episode Memory

Ngày: 22/10/2026. Audit development từ checkpoint DeNICE V1 legacy trong
`results (11)`, được dùng làm nguồn cấu hình baseline CANDLE V1 mới. Đây không
phải kết quả của một run CANDLE V1 vừa train lại.

## 1. Kết luận

1. Discovery hai tầng có lý do thực nghiệm: ở Task 5, 662/1.630 cặp
   receiver–class thiếu dữ liệu local chỉ có donor ở ngoài neighborhood.
2. Donor coverage chưa phải functional transfer: probe trong neighborhood có
   head chuyển được khi ép task, nhưng inference bình thường vẫn không nhận đúng.
3. Prototype/episode memory donor không tự dùng được trên receiver. Cặp ngoài
   neighborhood có FC1 prototype khác rõ; cả raw pooling, thresholded mean và OR
   đều không cứu target và làm hỏng prediction vốn đúng trên negative panel.
4. Chưa có cơ sở triển khai nguyên đề xuất hoặc khẳng định cải thiện Accuracy/F1.
   Đây là hai case khóa bằng metadata, không phải benchmark toàn bộ candidate.

Không optimizer, không train backbone/head, không final TEST, không commit vào
model thật. Hai source models/routers có complete-state hashes giữ nguyên qua
mọi shadow variant. Chưa thực hiện Gaussian prototype calibration hoặc transfer
young neurons/dependency closure; kết quả chỉ kiểm tra head row và episode memory.

## 2. Nguồn và protocol

- Checkpoint: Task 4/5, Round 19, legacy, seed42, ξ=0,8, binary_cosine.
- File nguồn: `audit_denice/appliance_legacy_results11/inputs/checkpoint_task_{4,5}_all_rounds.zip`.
- Load full FP32 terminal, kiểm tra internal SHA-256 và finite tensors; graph
  lấy đúng round19 trong cùng archive.
- Role manifest SHA-256:
  `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.
- Data runtime: `appliance_current_runtime_data_local/full100_v1/calibration_store`.
  Mỗi audit chỉ mở CAL partition của task tương ứng checkpoint. Không mở raw CAL
  của task trước, original BASE, validation hoặc final TEST.
- Construction/evaluation chia bằng hash nội dung với seed20261022; cùng content
  có cùng partition giữa các owner. Dedup theo content và cap128/owner/class/partition.
- Case selection không đọc predictions: receiver thiếu BASE/availability,
  donor mature, donor CAL và receiver CAL negatives mỗi phía có ít nhất64 records
  trước content split; ưu tiên donor support rồi negative support và IDs.
  Con số64 là điều kiện đủ dữ liệu cho probe hai phần, không phải production
  acceptance gate hoặc chứng nhận FAR thống kê.
- Hai panel gồm donor target positives và receiver current negatives. Receiver
  không có target positives trong panel. Recall target là trên donor-origin data
  chạy bằng receiver model, không chứng minh generalization lên môi trường receiver.
- Head probe copy một donor row/bias với eta=1 cố định; mask-only chỉ đổi FC2 masks.
  Không average nhiều donor, không allocate thêm FC1/CNN/GRU neurons, không fit head.
  Đây là phép tách cơ chế, không phải bản triển khai đầy đủ intra young-neuron
  transfer hoặc inter Gaussian prototype calibration của proposal.
- `probe_plan.json` ghi trước inference; SHA kiểm tra không đổi khi kết thúc.
- Dữ liệu/checkpoint này từng được audit; không gọi evaluation partition mới là
  untouched confirmation. Không chọn eta, sigma, margin hoặc threshold từ kết quả.

## 3. Coverage trên toàn terminal graph

Paper mode tạo neighborhood chồng lấn, không phải K cluster disjoint. Intra nghĩa
là peer có positive alpha trong recorded neighborhood. Extra nghĩa ngoài tập đó.
Probe extra là counterfactual nghiên cứu cho đề xuất mới, không phải edge hợp lệ
của baseline V1 và không cấp phép triển khai production.

| Metadata | Task 4 | Task 5 |
|---|---:|---:|
| Terminal receivers | 89 | 98 |
| Receiver–class pairs đã xuất hiện | 2.670 | 3.332 |
| Cặp thiếu local BASE presence | 1.319 | 1.630 |
| Có mature owner-donor trong neighborhood | 1.007 | 968 |
| Chỉ có mature owner-donor bên ngoài | 312 | 662 |
| Availability gaps | 954 | 1.200 |
| Availability gaps có donor trong neighborhood | 676 | 686 |
| Availability gaps chỉ có donor bên ngoài | 278 | 514 |
| Current-task bitmap gaps | 162 | 133 |
| Current gaps: intra / extra-only | 144 / 18 | 106 / 27 |
| Mean positive-alpha peers | 9,8202 | 5,4694 |

Đây là cặp receiver–class, không phải số class độc lập hoặc tỷ lệ lỗi code.
Local BASE presence được dựng từ task-start training metadata và không tự chứng
nhận successful optimizer/competence. Mature rank cũng không chứng nhận accuracy.
Historical bitmap có thể lấy từ metadata; không giả định prototype của các class
cũ hiện có trong capsule của current task.

Task 5 không có cặp current-class đáp ứng đồng thời donor positives ≥64 và
receiver negatives ≥64 theo selection đã khóa. Không hạ điều kiện để chọn một
case đẹp; Task 4 có302 donor–receiver–class candidates và chọn đúng2 cases.

## 4. Hai case Task 4: head và routing

Donor11/class24 được chọn vì metadata CAL support lớn nhất, không dựa trên test
hoặc kết quả probe. Cùng một donor được dùng ở cả hai regime:

| Kiểm tra | Receiver7 ← donor11, intra | Receiver65 ← donor11, extra |
|---|---:|---:|
| Alpha của edge native | 0,184805 | 0 |
| Mature owner-donors trong neighborhood | 15 | 0 |
| Target evaluation positives | 128 | 128 |
| Receiver evaluation negatives | 83 | 121 |
| Baseline / Availability-only recall | 0% / 0% | 0% / 0% |
| Mask-only recall | 0% | 0% |
| Head transfer: native recall | 0% | 0% |
| Head transfer: forced-task recall, diagnostic | **99,2188%** | **0%** |

Forced task chỉ là oracle diagnostic. Baseline/availability/mask-only forced-task
recall đều0%; head-preserve-mask và head+mask đều99,2188% ở receiver7. Như vậy
gain classifier trong case này cần weights donor, không chỉ mở availability/mask.
Native router không đưa target positives tới đúng task, nên native gain vẫn0.

Donor11 trên cùng128 donor-origin positives: forced-task recall100%, nhưng native
recall và target task-route accuracy chỉ2,3438%. Donor có classifier function tốt
nhưng router hiện tại kém; presence/maturity không đồng nghĩa native competence.

Ở receiver65, chính donor head đó không nhận đúng target ngay cả khi ép task.
Điều này phù hợp với vấn đề function/representation compatibility; chưa đủ để
quy mọi lỗi cho một tensor hoặc bác bỏ mọi cơ chế prototype calibration.

## 5. Episode memory synchronization

Ba phép gộp chỉ sửa shadow current-episode bank:

- Pool: nối nguyên binary sketch rows; binary_cosine dùng mean prototype như cũ.
- Mean: threshold mean của bank đã gộp ở0,5 thành một binary prototype.
- OR: OR mọi sketch trong bank đã gộp thành một prototype.

Weights head và availability giữ cùng một recipe head+mask giữa ba variants.
Không import thresholds hoặc đổi backbone để ép memory tương thích.

| Case / Variant | Target recall native | Rescue | Break |
|---|---:|---:|---:|
| Intra: head+mask | 0% | 0 | 0 |
| Intra: + raw memory pool | 0% | 0 | 0 |
| Intra: + thresholded mean | 0,7813% | 2 | 2 |
| Intra: + OR | 0% | 0 | 6 |
| Extra: head+mask | 0% | 0 | 0 |
| Extra: + raw memory pool | 0% | 0 | 30 |
| Extra: + thresholded mean | 0% | 0 | 30 |
| Extra: + OR | 0% | 0 | 30 |

Mọi transition dùng cùng panel của case; `delta correct = rescue - break` khớp.
Extra memory variants phá27 predictions class25 và3 predictions class26; overall
panel accuracy giảm12,0482% →0%. Đây không phải global TEST accuracy.

Target FAR vẫn0 trong những extra variants này: damage xuất hiện vì routing đổi
sang các class sai khác, không phải vì mọi mẫu bị đổi thành class24. Chỉ kiểm tra
FAR của imported class sẽ bỏ sót loại damage này.

Negative panel chỉ thuộc Task4; rủi ro với Task0–3 vẫnUNKNOWN. Không diễn giải0
break trên panel nhỏ thành protection tuyệt đối hoặc FAR population ≤0,1%.

## 6. Prototype / routing-space compatibility

Đo hai model trên cùng128 construction target inputs; evaluation target inputs
không trùng nội dung construction. Không tạo Gaussian samples hoặc fit classifier.

| Metric | Intra | Extra |
|---|---:|---:|
| FC1 prototype dimension | 256 | 256 |
| Cosine prototype donor vs receiver | 0,79284 | 0,32371 |
| Donor prototype → receiver evaluation features, mean L2 | 38,1336 | 69,1438 |
| Receiver prototype → receiver evaluation features, mean L2 | 5,4566 | 4,0946 |
| Context binary agreement trên cùng input | 100% | 94,3744% |
| Context binary Jaccard | 100% | 63,7149% |
| Thresholds bằng nhau | Có | Không |
| Stable feature masks bằng nhau | Có | Không |

High binary agreement extra phần lớn phản ánh nhiều zero bits; Jaccard cho thấy
khác biệt rõ hơn. Intra có routing sketches giống nhau trên construction inputs
nhưng không đồng nghĩa FC1 features bằng nhau hoặc episode memory phân biệt task tốt.

Prototype donor cách receiver target features xa hơn reference prototype của
receiver trên panel này. Đây là cảnh báo về việc dùng trực tiếp donor prototype
để calibrate head receiver; không phải phép thử đầy đủ PP A và không kết luận
mọi noise level/mapper đều thất bại. Không tune thêm để cứu hai cases.

Payload tham chiếu: bitmap34 class packed5 bytes; một FC1 prototype FP32
1.024 bytes. Context sketch548D: packed69 bytes, storage float32 hiện tại2.192
bytes/row. Donor current episode bank82 rows:179.744 bytes (~175,53 KiB) ở storage
hiện tại, chưa gồm metadata/network overhead. Đây là kích thước dữ liệu, không
phải tổng communication wire traffic hoặc chi phí tăng bằng0.

## 7. Quyết định sau audit

Giữ discovery hai tầng như một candidate ablation riêng của CANDLE V1. Chưa có
bằng chứng để gộp head/prototype/router memory theo proposal vào main training.

Nút thắt intra đã cụ thể: classifier row có tín hiệu transfer, nhưng router của
cả donor và receiver đều không đưa target tới đúng episode. Memory pooling/OR
đã thử không giải quyết được. Nút thắt extra gồm cả classifier compatibility và
routing-space incompatibility. Không đánh đồng hai failure modes.

Không train full lại ở bước này. Cần một contract representation/router hợp lệ
trước khi triển khai calibration hoặc memory synchronization; không tiếp tục
thử donor/hyperparameter không giới hạn trên các panel vừa quan sát.

## 8. Tái lập và artifacts

```powershell
.venv-denice-gpu\Scripts\python.exe -u tools/audit_candle_v1_two_tier.py `
  --task 4 --out audit_denice/candle_two_tier_NEW_task4
.venv-denice-gpu\Scripts\python.exe -u tools/audit_candle_v1_two_tier.py `
  --task 5 --out audit_denice/candle_two_tier_NEW_task5
```

Output sử dụng cho báo cáo:

- `audit_denice/candle_v1_two_tier_20261022_task4_v2/`
- `audit_denice/candle_v1_two_tier_20261022_task5_v2/`
- Summary portable: `artifacts/candle_v1_two_tier_audit_20261022.json`.

Mỗi directory có coverage CSV/JSON, probe plan lock, per-case metrics/geometry và
completion. Không sửa input checkpoint, native runner hoặc notebook training.
Thực thi bằng local RTX4060 Laptop / Torch2.10 cu128, inference FP32. Không chạy
unit tests trong nhiệm vụ này; các kết quả trên là diagnostic inference/audit.
