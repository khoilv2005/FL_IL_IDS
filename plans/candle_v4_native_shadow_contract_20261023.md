# CANDLE V4 — contract cho native shadow integration tiếp theo

Kết quả tiếp theo: [transition smoke](../docs/CANDLE_V4_TRANSITION_SMOKE_20261023_VI.md)
NO-GO full do candidate reachability/preflight; chưa có hai ACTIVE experts.

Trạng thái: bounded native hook đã triển khai và chạy 20 rounds; xem
[receipt native shadow](../docs/CANDLE_V4_NATIVE_SHADOW_20261023_VI.md).
On/off và resume exact; chưa kiểm tra native task transition hoặc ACTIVE.
Phần dưới giữ contract thiết kế. Kết quả P1 nằm tại
`docs/CANDLE_V4_POSITIVE_NEGATIVE_PILOT_20261023_VI.md`. Chưa mở CAL, ACTIVE hoặc full.

## Điểm gắn vào V1

Trong `candle_v4_runtime/fed_learning/training/decentralized_denice_il.py`, thêm service
sau native `_aggregate_round()` và các router/BN refresh của round, trước round metrics
và round checkpoint. Không gắn vào cuối task như Direct FC2.

Service nhận current task/round, actual positive-alpha native neighborhood, current
BASE views của active owners, metadata/input authority và per-task communication budget.
Không truyền cho service API đọc CAL/TEST hoặc historical role fallback.
Fixed-pair integration smoke dùng config fixtures, không hardcode client/class trong engine.

## Tách state và optimizer

- Native model, masks, ages, BN, adapters và router giữ V1.
- Expert parameters/Adam state là một bank riêng; không register bank như child module
  của native model rồi để lọt vào native state_dict/aggregation hoặc base optimizer.
- Init/sampling dùng private RNG, hoặc fork/restore tất cả RNG được chạm. Không reseed
  native training/dropout/DataLoader. Service không đổi global precision/thread flags.
- DRAFT không sửa native availability/ownership và không ảnh hưởng capsule reliability
  trong engineering smoke; native predictor vẫn là output chính thức.
- No ACTIVE API trong mốc này. FIT blending chỉ là offline/shadow diagnostics.

## Expert round contract

1. Metadata: owner current BASE support, request và input fingerprint.
2. Eligibility: actual current edge có alpha>0, source có positives, receiver có negatives.
3. Quota 44/round, max88 khi available; carry không dùng future graph, không học từ stale
   gradient khi donor vắng. Tất cả skipped/deferred budgets phải log rõ.
   Không ép đủ 880 khi graph run mới không cho phép; báo actual steps và giữ controls
   matched theo work thực sự. Pending thuộc đúng task/class/generation, hết quyền khi
   chuyển task; không dùng cached old positives/gradients để tiêu quota ở task mới.
4. Gửi một current reference và global contribution weights/counts; mỗi peer trả một
   weighted gradient packet bind receiver, donor, class, task, round, inner-step,
   expert-generation, parameter/layout/mask/input hashes.
5. Validate, sum once, global clip, receiver-owned optimizer step. Không average Adam states.
6. Persist draft/optimizer, completed-step IDs, pending quota và wire ledger atomically.
7. FIT diagnostics versioned theo exact current base; không tái dùng frozen T2 terminal
   predictor để giả lập accuracy của native rounds mới.

Original pilot dùng capped/deduped BASE riêng. Native integration phải ghi chính xác
expert lấy từ client post-local-validation BASE nào, có smoke cap hay không; không
đánh tráo FIT thành train hoặc âm thầm tăng expert data budget để giữ kết quả đẹp.

## Checkpoint/resume

Expert state phải lưu ngoài base tensors nhưng trong algorithm continuation:
owner, input/schema/config IDs, draft weights, Adam moments, update counter, seed/RNG,
pending quota, last applied packet/reference IDs và serialized ledger counters.
Restore cùng protocol cho cùng expert scores, updates, bytes và base state như chạy liền.
Đổi optimizer, balancing hoặc scheduling policy phải tạo experiment/version mới.
Bootstrap/catch-up từ một client khác không tự thừa kế expert ownership/certificates.

## Engineering và quality gates tách riêng

Engineering smoke:

- Real native local optimizer và aggregation chạy; expert hook được gọi tại round boundary.
- Một nhánh có receiver-negative gradients thật, version/finite/gradient algebra đúng.
- Shadow on/off cùng seed/data cho base params/masks/router/graph khớp, hoặc ghi rõ
  mọi khác biệt được giải thích. Không gọi RNG drift là algorithm gain.
- Save/restore giữ expert và native behavior; không duplicate steps/bytes.
- Có no-peer/late-peer case; không cố ép cạnh để tạo update.
- Truy cập ngoài current task hoặc CAL/TEST làm smoke fail.

Quality:

- So fixed-threshold và FIT-calibrated donor-only với joint, budget khớp.
- Đo per-owner/class FP, changed-wrong, rescue/break, convergence, earliest observed
  utility và setup+round traffic với base predictor thực sự thay đổi.
- DRAFT scores tốt chưa cấp quyền ACTIVE. Old-task risk và multi-expert bank chưa được
  chứng nhận bằng current task negatives. Chỉ dùng policy empirical có scope đã khóa.
- Không mở full chỉ vì structural tests pass. Không chọn round thắng bằng CAL.

Để tiến lên acceptance/retention, trước hết cần duyệt data-access history và một
protocol holdout hợp lệ; CAL từng xuất hiện ở development cũ không được gọi là bộ
confirmation mới độc lập.
