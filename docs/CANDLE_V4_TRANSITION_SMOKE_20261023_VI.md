# CANDLE V4 — smoke cuối T2 → T3: chưa đủ điều kiện full

Tiếp nối: [graph-aware preflight và fixture đã học](CANDLE_V4_GRAPH_PREFLIGHT_SMOKE_20261023_VI.md).
Báo cáo dưới giữ nguyên kết quả của fixture cũ; không thay bằng kết quả case mới.

**NO-GO cho full seed 42.** Đã chạy đúng một fixture 4 clients qua T2 → T3,
nhưng fixture không tạo được hai expert đã học: donor 16 nằm ngoài neighborhood
thực tế của receiver 18 trong cả 8 round T2. Hai draft có **0 optimizer updates**,
không có expert ACTIVE. Không đổi donor, ξ, threshold hoặc gate sau CAL; không
chạy thêm fixture để tìm một kết quả PASS.

Receipt: [artifacts/candle_v4_transition_20261023.json](../artifacts/candle_v4_transition_20261023.json).

**Đã sửa sau audit:** gate trả `DEFERRED_NO_AUTHORIZED_TRAINING` khi draft chưa học;
`DEFERRED_GATE_GRAPH_SCOPE` nếu required CAL owner ngoài current graph; sang task
mới mà chưa có bank được chấp nhận thì `SKIPPED_NO_ACTIVE_BANK`. Các trường hợp
này không mở FIT/CAL hoặc gửi packet. Sáu guard checks PASS trong
[receipt preflight](../artifacts/candle_v4_gate_preflight_fix_20261023.json).
Đây chỉ là kiểm tra guard, không phải chạy thêm native smoke hoặc chứng nhận ACTIVE.

## Thiết lập đã khóa

- Client IDs `{8,16,18,30}`; receiver 18, declared donor 16, target classes 13 và 17.
- Native V1 T1 checkpoint, giữ bốn model đã tồn tại; native graph tự tính từng round,
  `paper`, ξ=0,8. Không ép cạnh, không dùng graph tương lai để train.
- T2: 8 native rounds và task finalization thật; sau đó T3: 8 rounds + finalization.
- Current BASE dedup/cap 4.096/class, rồi native local-validation split 10%.
  Expert chỉ được train từ post-split current BASE.
- MLP 39→64→1, balanced joint, Adam 0,001, clip5; quota44/round, max88 catch-up.
- Threshold rule khóa trước CAL: donor FIT order statistic với recall ≥95%.
  Không chọn lại threshold bằng CAL, không chọn round thắng.
- Cả bank xét atomic acceptance trên current CAL; recall ≥95%, ≥32 unique positives
  tại required donor, ≥32 negatives/owner, observed FAR ≤0,1% và per-observed-negative-class
  FAR ≤0,1%, break=0, net gain dương.
- Arbitration khóa trước: normalized excess `(score−threshold)/(1−threshold)`, tie
  theo class ID; nếu không expert ACTIVE nào kích hoạt thì dùng native prediction.
- Bank budget 128 MiB, mỗi receiver/class 64 MiB, modeled native+bank ≤2 GiB.
  Không mở TEST. Full training chỉ được phép nếu qualifying smoke PASS.

Đây là fixture development, không phải fresh 100-client benchmark hoặc blind
online discovery. Checkpoint nhiều-GPU được ánh xạ về CUDA0 của máy local.

## Root cause: preflight fixture thiếu kiểm tra reachability

Mình đã khóa fixture bằng ownership và evidence-count metadata trước khi xác nhận
donor thuộc live neighborhood. Đây là thiếu sót của harness/discovery preflight,
**không phải bug của native clustering** và không chứng minh MLP V4 không học được.

Trong cả 8 round T2, peers có alpha dương của receiver 18 là `{8,30}`; self là 18.
Donor 16 không có cạnh hợp lệ. Cuối T2, alpha lần lượt:

| Contributor | Alpha |
| --- | ---: |
| 8 | 0,259906 |
| 18, self | 0,424330 |
| 30 | 0,315764 |

| Target | Declared donor16 | Peer hiện tại có current BASE | Peer đủ count CAL positive ≥32 |
| --- | --- | --- | --- |
| 13 | Ngoài neighborhood | Không có | Không có |
| 17 | Ngoài neighborhood | 8 và 30 | 8 |

Không chuyển class17 sang donor8 sau khi xem CAL. Đây chỉ là phân tích metadata
sau run, không phải một trial mới. Class13 vẫn không có nguồn hợp lệ trong fixture này.

## Những gì đã và chưa kiểm chứng

| Kiểm tra | Kết quả thực tế |
| --- | --- |
| Native T2 → T3 + task-boundary checkpoints | Hoàn thành |
| Native tensors/masks/router/algorithm state/RNG on/off tại T2 và T3 | Exact |
| Bank checkpoint/restore | Exact về state |
| Hai draft còn tồn tại, không train lại tại T3 | Có, nhưng cả hai chưa học bước nào |
| Learned-expert retention | **Chưa kiểm chứng** |
| Hai ACTIVE experts trong inference thật | **Chưa kiểm chứng** |
| Collision arbitration | Nhánh candidate được gọi; controlled handler check PASS, không phải ACTIVE quality proof |
| Gate | Reject cả hai draft; không commit |
| Historical CAL API sau transition | Bị chặn |
| Native parameters cuối task | Hữu hạn |
| Budget smoke | Trong budget; không đủ cơ sở dự phóng full vì không có expert training |

Receipt gốc có các cờ `expert_retention_weights_exact` và `inference_restore_exact`.
Chúng chỉ chứng minh lưu trữ draft/khôi phục fallback trong run này. Không được diễn
giải thành khả năng giữ kiến thức hoặc inference của hai ACTIVE experts.

## Gate và lỗi cần chặn trước CAL

Code đã chạy vẫn mở FIT/CAL cho draft chưa có update; việc này là thừa và cần
defer ở preflight. Ngoài ra, gate-reference simulation gửi tới cả bốn owner, gồm
donor16 đang ngoài neighborhood: gate path chưa tuân thủ đầy đủ ràng buộc graph.
Vì vậy **không gọi phần acceptance này là PASS kỹ thuật đầy đủ**.

CAL của hai task reject do recall/FAR/break. Scores của hai draft ngẫu nhiên còn
cùng vượt gate trên 2.763 CAL rows T2. Những số này không được dùng để kết luận
chất lượng hoặc khả năng phân biệt của hai expert V4 đã được train.

V4 vẫn không ảnh hưởng native training hoặc bật expert: cả hai status ở checkpoint
T2/T3 là `REJECTED`. Không có live prediction bị override bởi các draft này.

CAL hiện có cùng role lineage từng được dùng bởi các nghiên cứu APPLIANCE. Không
có đầy đủ per-row access history để chứng nhận là untouched confirmation holdout.
Smoke này chỉ là acceptance development. Không dùng raw CAL T2 để train/refresh
ở T3. Preparation có giải nén NPZ nguồn để lọc current role, không claim không
materialize bất kỳ byte của task khác khi giải nén.

## Communication và reproducibility

Tổng modeled application bytes, gồm native capsules/neighbor state và gate:
**1.097.975.489 bytes ≈ 1,023 GiB**. Phân rã và receiver/class bytes nằm trong receipt.
Đây là serialization model, không phải số đo mạng thực. Training traffic expert
bằng 0; không lấy chi phí này để dự báo một V4 full run đang học bình thường.

Một lỗi JSON serialization của return metadata xảy ra sau khi cả T2/T3 và các
checkpoint đã lưu xong. Chỉ sửa conversion output arrays; không train lại bank,
không đổi optimizer, candidate, threshold hay CAL outcome. Source hashes trước/sau
output fix được giữ trong receipt.

```powershell
git checkout c313aa6f5a114a61f2cec9a345988a3f2a4a86e7
python tools/run_candle_v4_transition.py --out audit_denice/new_v4_transition --mode prepare
python tools/run_candle_v4_transition.py --out audit_denice/new_v4_transition --mode bank
python tools/run_candle_v4_transition.py --out audit_denice/new_v4_transition --mode baseline
python tools/run_candle_v4_transition.py --out audit_denice/new_v4_transition --mode summarize
python tools/summarize_candle_v4_transition.py --out audit_denice/new_v4_transition
```

Commit trên giữ code của smoke đã ghi. Main sau preflight fix sẽ defer fixture
không đủ điều kiện trước CAL; không được gán kết quả CAL cũ cho code đã sửa.

Không chạy thêm full smoke trong phiên này. Việc cần sửa trước một lần kiểm chứng
ACTIVE tiếp theo là **preflight theo actual graph trước khi mở FIT/CAL**, và chọn
fixture có ít nhất hai missing classes có donor/evidence hợp lệ. Không sửa ξ hoặc
hạ risk gate để biến fixture hiện tại thành PASS. Full seed42 chưa bắt đầu.
