# APPLIANCE-Head-Agg — P1 implementation và P2 bounded pilot

Ngày: 2026-10-10. Protocol: `appliance_direct_head_v1`.

**P1 PASS; P2 NO-GO, 0/3 case đạt acceptance. Không chuyển P3 hoặc full training.**

## 1. Code đã triển khai

| Module | Trách nhiệm |
|---|---|
| `appliance/direct_head_contract.py` | Shape, architecture/class/role/graph/function binding, maturity và rules cố định |
| `appliance/direct_head_codec.py` | Raw FC2 row + bias FP32, bit-packed masks, bounded framing và SHA256; không donor optimizer |
| `appliance/direct_head_aggregation.py` | Absolute effective weights, alpha-only q=1, normalization theo từng contributor coordinate |
| `appliance/direct_head_transaction.py` | Isolated shadow, effective-old blending, Mask-only, protected-state check, commit/rollback |
| `appliance/direct_head_verification.py` | Owner-local current CAL, x-only inference, FIT chọn eta, H receipts và empirical qualification |
| `tools/run_appliance_direct_head_feasibility.py` | Metadata-only lock, ba fixtures, message reconstruction, wire ledger, hard budgets |

`tools/check_appliance_direct_head_mechanics.py`: **37/37 checks**, gồm exact codec roundtrip, corruption/nonfinite/mask rejection, duplicate donor, normalization, learned-zero contributor, no-contributor preservation, eta=0 exact no-op, Mask-only, maturity/ownership/task-freeze protection, backbone invariance, rollback, isolated commit, checkpoint restore, receipt authority và cấm chọn eta bằng HOLDOUT. Các checks acceptance dùng synthetic counts chỉ để kiểm tra logic, không phải experimental PASS.

`tools/audit_appliance_direct_head_feasibility.py`: **693 consistency checks PASS** trên artifacts, không mở CAL lại. Đây là audit counters/bindings/ledger, **không** phải independently recomputed predictions.

### Quyền ghi vào reserve slot

Fixed allocation của DeNICE dùng `freeze = ranks != 1`, nên rank-0 reserve cũng bị chặn gradient. Direct-head permission chỉ cấp cho slot **rank 0, không có current BASE ownership**; không suy gradient freeze của reserve thành mature knowledge. Mature slot, owned target và FC2 có task-wide freeze bị reject. Không thay native SGD policy.

Tọa độ không donor nào đóng góp giữ nguyên **raw weight và connection mask** của receiver. Với tọa độ được phép, blend từ old **effective** value, không mở raw inactive weights một cách vô thức. Mask-only giữ original raw row nhưng có cùng mask/rank/availability với final Head-Agg.

Commit hiện chỉ tạo model clone của pilot và provenance receipt; chưa nối native runner, chưa chứng minh optimizer/aggregation protection qua round. Không gọi đây là production install.

## 2. Protocol và phạm vi

- DeNICE legacy, graph paper ξ=0,8, round 19, native restored `binary_cosine` router.
- Không CGoFed/CME/imported route/mapper, không donor hoặc receiver optimizer, không BASE/test reads.
- Fixtures từ plan và các development audit trước; **không phải independent confirmation**.
- Target còn current tại checkpoint Task 1/2; mỗi owner mở đúng CAL partition của task đó. Không đọc task cũ so với thời điểm đang mô phỏng.
- FIT eta grid `{0.25,0.5,1}`; donor IDs cố định. Single controls cũng chọn eta trên FIT, rồi khóa cùng aggregate trước H.
- Mỗi selected donor nhận đúng một cold receiver function reference. Head/spec requests được serialize và reconstructed trên exact reference cache; chỉ summary counts quay về receiver, không raw examples.
- Gate: ≥32 unique positives trên mỗi required donor H; ≥32 receiver negatives; recall ≥95%; FAR ≤0,1% ở mọi observed required owner/class; negative break=0; vượt Availability-only và Mask-only về target hits và net correct trên cùng rows.
- Dedup và FIT/selection–H content exclusion thực hiện trong từng owner; cross-owner independence **chưa được chứng minh**.
- CAL-H là logical split của current CAL materialization; partition file chứa cả splits. Không sử dụng H outcomes để chọn eta/variant. H này từng là development source ở những audit trước, không gọi untouched test.

Hai lần đầu dừng trước đọc CAL: lần prepare dùng nhầm config key `denice_cluster_mode` thay vì `denice_clustering_mode`; lần execution reject nhầm rank-0 reserve do freeze semantics. Lần thứ hai đã ghi 13.531.682 bytes setup, không có FIT/H outcome. Cả hai được giữ ở local `locked_01/02`, không dùng để chọn hyperparameter. Successful bounded execution là **`locked_03`**.

## 3. HOLDOUT kết quả đã khóa

Mỗi ô recall là **receiver model** chạy trên CAL của hai donor; không phải donor-model accuracy hoặc full federation accuracy.

| Case | Head-Agg eta | Availability / Mask-only recall | Head-Agg recall | Target hits thêm vs cả hai controls | Kết luận |
|---|---:|---|---|---:|---|
| T1: receiver 3, class 7, donors 54/51 | 1 | 0% / 0% | **92/98 = 93,88%; 49/51 = 96,08%** | +141 | FAIL: owner 54 dưới 95% |
| T2: receiver 1, class 13, donors 74/34 | 0,5 | 58/136 = 42,65%; 15/34 = 44,12% | **59/136 = 43,38%; 15/34 = 44,12%** | +1 | FAIL: cả hai dưới 95% |
| T2: receiver 3, class 14, donors 74/26 | 0,25 | 73/104 = 70,19%; 44/68 = 64,71% | **73/104 = 70,19%; 45/68 = 66,18%** | +1 | FAIL: cả hai dưới 95% |

Các case đều đủ empirical count threshold. Head-Agg ghi **0 false positives và 0 negative breaks** trên current H negatives ở từng endpoint đã kiểm tra:

- Class 7: receiver 442, donor 54: 667, donor 51: 540 negatives.
- Class 13: receiver 107, donor 74: 395, donor 34: 47 negatives.
- Class 14: receiver 347, donor 74: 427, donor 26: 117 negatives.

Không cộng các pool thành IID population certificate hoặc cumulative old-risk guarantee. Một số class/owner có rất ít witness. Old-risk giữ **UNKNOWN**.

### Multi-donor contribution

Head-Agg bằng single donor 1 trên predictions correct-count của cả ba cases. Class 7 hơn single donor 2 đúng **1** target sample; không hơn cả hai single controls. Các case 13/14 bằng cả hai singles. **Chưa chứng minh lợi ích multi-donor aggregation.**

### Native routing limitation

Target positives tới đúng task context lần lượt là `92/98,49/51`, `59/136,15/34`, `73/104,45/68`. Head-Agg nhận đúng tất cả target samples đã tới context đó trong các pool đang xét. Các counts phù hợp với bottleneck native task routing; chúng không chứng minh compatibility trên mọi input. Không có forced-context protocol hoặc imported route được thêm để cứu pilot.

Class 7 chứng minh direct weights có gain thật vượt Mask-only trên fixture này. Nhưng khả năng nhận diện hữu ích đó vẫn chưa đạt acceptance, và không chứng minh Head-Agg hơn single-head.

## 4. Communication và thời gian

Execution thành công: **200,90 giây CPU**, single Torch/BLAS thread; preparation và failed setup tính riêng.

| Application-wire category | Bytes |
|---|---:|
| Cold receiver function references, 6 lần | 41.554.512 |
| Existing classifier-row packets, 6 lần | 10.502 |
| Metadata requests/responses | 79.914 |
| FIT candidate requests / summary receipts | 178.017 |
| HOLDOUT candidate requests / summary receipts | 110.278 |
| **Tổng bounded execution** | **41.933.223 = 39,99 MiB** |

Packet class 7: 1.747 bytes/donor; class 13/14: 1.752 bytes/donor. Raw weights/bias/mask không phải whole transaction cost. Setup references chiếm khoảng **99,10%** traffic. Sáu cold copies được reuse đúng version cho FIT và H; không giả định native graph messages đã cung cấp cache hợp lệ.

Failed pre-CAL setup thêm 13.531.682 bytes; tổng traffic thực sự của các local executions là **55.464.905 = 52,90 MiB**, vẫn dưới 128 MiB. Successful-run budget không được dùng che failed setup cost.

Đây là simulated application payload, không socket/TLS traffic và chưa cộng whole native DeNICE training communication. Không tuyên bố phương pháp giảm tổng bandwidth.

Metadata messages hiện chứa static role-allocation counts kèm neuron ranks, bao gồm allocations của task tương lai; không dùng chúng làm learned ownership/requests, không đọc future raw partitions. Đây là một giới hạn của prototype announcement: production cần lọc learned/current metadata theo thời điểm, không quảng bá allocation như kiến thức đã học. Nó không thay weights hoặc lựa chọn fixtures trong pilot này.

## 5. Kết luận / stopping rule

**Đóng P2 với NO-GO cho APPLIANCE-Head-Agg theo protocol đã khóa.** Không thêm case/donor, không tune eta sau H, không hạ 95%, không thêm mapper/imported route hoặc train lại backbone để cứu kết quả này.

P1 implementation hoạt động; tín hiệu transfer trực tiếp class 7 có thật. Tuy nhiên, không case nào đạt final acceptance và aggregation chưa tạo lợi ích vượt single donor tốt nhất. Không chuyển P3, không nối production runner, không tạo/chỉnh full-training notebook.

Nếu nghiên cứu tiếp single-head hoặc native routing, đó là quyết định protocol mới; không đổi nhãn control hoặc secondary diagnostic thành Head-Agg PASS.

## 6. CLI và artifacts

Kiểm tra P1:

```powershell
.venv-audit/Scripts/python.exe -m tools.check_appliance_direct_head_mechanics
```

Metadata lock và execution dùng hai phases. Tham số `--task1/--task2` là task archive ZIP hoặc thư mục có `checkpoint_archive_manifest.json`; `--roles` là locked clean roles; `--cal-store` là current task-scoped materialized store, không original dataset fallback. `run` kiểm tra source/role/store/checkpoint hashes và từ chối overwrite execution đã mở H.

```powershell
python -m tools.run_appliance_direct_head_feasibility prepare --out <new-output> --roles <roles> --cal-store <store> --task1 <task1-archive> --task2 <task2-archive>
python -m tools.run_appliance_direct_head_feasibility run --out <same-output> --roles <roles> --cal-store <store>
python -m tools.audit_appliance_direct_head_feasibility <output>
```

Commands để reproducibility, không phải đề xuất chạy thêm một pilot sau NO-GO.

Local raw run artifacts: `audit_denice/appliance_direct_head_v1/locked_03` (gitignored). Public summary/protocol/checks nằm ở `artifacts/appliance_direct_head_*20261010.json`. P1 code và báo cáo được commit/push; không upload checkpoint hoặc dữ liệu CAL lên GitHub.
