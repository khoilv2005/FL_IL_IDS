# cnn_coordinate_router_v1 — protocol pilot khóa trước kết quả

## Contract

- Fresh legacy model, shared cluster9/30, ξ0,8; P0/P1 giữ nguyên.
- Target Conv2 trước. Nếu routing FAIL, cho phép đúng một fresh Conv3 run,
  cùng budget/seed/loss; không thử GRU, không tune feature subset/threshold.
- Binary-cosine scorer giữ làm control; policy mới là CNN coordinate closure.
- Selected features = allocated units của target tại T0; mask và thresholds cố
  định qua T1. Chọn Conv3 nếu cần là variant công khai, không sửa memory đã commit.
- Mỗi selected output row khóa tất cả kernel coordinates/bias và connection-mask
  bits, kể cả inactive edges. Active incoming mask edges quyết định upstream
  channel closure; recursively freeze upstream Conv/BN channels cần thiết.
- BN affine, running mean/variance của những channels này protected; canonical
  counters giữ nguyên. Không hash/freeze unrelated rows hoặc GRU chỉ vì tên tensor.
- Dense Conv2 fan-in có thể yêu cầu toàn Conv1. Đây là closure thật, không cắt
  upstream để tạo capacity giả. Existing protected coordinates không được bỏ.
- Cumulative CE/full forward cùng routerless: cluster seen union + current
  authorized BASE classes; chỉ successful optimizer receipts đăng ký class.
- Tasks0→1, seed42, one round/task, one epoch, Adam FP32 LR0,001, batch16,
  cap32 BASE/class/client, cap64 VALIDATION/class/client. No distillation/replay.

## Data và evaluation

Router fit/current sketches chỉ đọc current BASE. Không đọc historical BASE/CAL
để refresh thresholds/router. Historical binary memory giữ nguyên.
Read-only evaluator được đọc old VALIDATION để đo class accuracy/forgetting và
so selected activations trước/sau trên cùng rows; không đưa VAL vào fit/training.
Panel đã là development; không untouched test và không final TEST.

Tại mỗi checkpoint dùng **cùng model** đo:

1. Cumulative argmax trên registered class union.
2. Actual predicted-task gating (X-only).
3. Oracle true-task gating chỉ diagnostic.

Không ghép shallow routing accuracy với routerless weights. Oracle80,75% từ
checkpoint routerless cũ không là target/claim cho checkpoint mới. Báo routing
confusion/balanced accuracy, class accuracy/Macro-F1 fixed scope0–11, old/new
recall, forgetting, class2 retention, rescue/break actual vs own cumulative.
Missing validation classes0/9/11 ghi rõ; không bỏ chúng khỏi fixed Macro-F1.

## Gate

- Capacity: actual protected backbone<90%, còn learner parameters; Conv2 target
  selected rows stable, các unselected/current rows hoặc tầng sau còn học được.
- Native optimizer thật; coordinate/mask/BN/version protection; old memory và
  thresholds byte exact; selected activations max absolute drift≤1e-6 trên current
  và read-only historical development; model/router restore giống nhau.
- Minimum routing feasibility: balanced task accuracy≥60%, each task recall≥25%.
- Để chuẩn bị P3: balanced task accuracy≥75%, each task recall≥60%; actual gated
  class Accuracy và Macro-F1 vượt own-checkpoint cumulative; old accuracy loss
  so T0≤10pp. Per-class degradation phải báo rõ dù overall gate PASS.
- GO chỉ bounded P3 preparation, không full training/multi-seed claim. Nếu FAIL,
  không tune guard/threshold/LR hoặc lấy oracle làm actual metric.

## Command

```powershell
.\.venv-audit\Scripts\python.exe tools/run_denice_v2_coordinate_smoke.py `
  --cache audit_denice/denice_v2_native_p1 --feature-layer conv2 `
  --out artifacts/denice_v2_coordinate_conv2_smoke_20261012.json
```
