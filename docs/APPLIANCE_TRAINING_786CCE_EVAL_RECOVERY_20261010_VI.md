# Audit log 786cce và khôi phục full evaluation

## Trạng thái thực tế

Nguồn: `C:\Users\khoak\Downloads\fl-il-lk-na-786cce.log`, code training `92fe73219b05f4cd10ab06beb90964d7a279b9fc`.

- Fresh DeNICE legacy + APPLIANCE, ξ=0.8, seed42, hoàn tất Task0–5, 20 round/task.
- Task5 round19: train loss `1.708402964540943`.
- Callback `task_finalized`: 10 requests, 7 offers, 0 commit mới, **7 authorized**; counter communication `428974131` bytes. Không coi 7 offers là 7 install mới.
- `checkpoint_task_5_all_rounds.zip` đã seal đủ 20 round, khoảng **1.64 GiB**, lúc 23689.6 giây.
- Lỗi lúc 23715.4 giây, trước router fitting/full-test prediction:

```text
APPLIANCE guard requires cuda / Torch 2.10.0+cu128;
requested cpu / 2.10.0+cu128
```

Launcher đã hardcode CPU dựa trên fixture smoke CPU. Trong training Kaggle, model/patch thực tế ở CUDA. Certificate kiểm tra đúng và chặn runtime khác; launcher chọn sai backend. Không phải lỗi checkpoint, không cần train lại. Chưa có accuracy cuối của lần chạy này.

AMP vẫn có optimizer skip và một số client có capacity warning/critical trong log. Những dòng đó chưa đủ để suy ra tỷ lệ skip toàn run hoặc nguyên nhân accuracy; không thay training/guard trong sửa lỗi evaluation này.

## Sửa lỗi

`resolve_evaluation_device` đọc declaration đã seal trong checkpoint. `auto` dùng backend đã chứng nhận và kiểm tra Torch version. Không thay certificate, guard, threshold, lifecycle, weights hay optimizer. CUDA certificate cần GPU và Torch đúng phiên bản; không fallback âm thầm về CPU.

Full training launcher dùng `device='auto'` cho lần chạy sau. Lock integration lưu amendment evaluator riêng, giữ nguyên bằng chứng 61 checks từ native smoke; **không tuyên bố đã chạy lại native smoke hoặc full evaluation**.

## Chạy tiếp lần này

Dùng **`eval_denice_appliance_full_kaggle.ipynb`**, không chạy lại notebook training.

1. Mount output lần training này hoặc cung cấp link Drive ZIP của output.
2. Giữ dataset gốc `/kaggle/input/datasets/khoilv2005/100-clients/100-clients`.
3. Cần `checkpoint_task_5_all_rounds.zip` và `denice_clean_roles/role_manifest.json`, `role_lock.json` của đúng lần training. Evaluator không đọc raw CAL hoặc role indices; không tạo lại split.
4. Set `RESULTS_PATH` tới output đã mount, hoặc `RESULTS_DRIVE_URL`. Nếu role files mount riêng, set `DENICE_CLEAN_ROLES_DIR`.
5. Bật GPU; checkpoint trong log yêu cầu Torch `2.10.0+cu128`.

Notebook clone GitHub main; không nhúng source hoặc yêu cầu runtime ZIP. Outer ZIP chỉ giải nén Task5 và role metadata, không nhân đôi cả sáu task. Output mới có timestamp riêng; không ghi đè thư mục eval lỗi.

Đánh giá toàn bộ test một lần, chia disjoint cho receiver, ba decision policies trên cùng checkpoint và samples: BinarySelf, MulticlassSelf, APPLIANCE. Self là reference inference với backup head, **không phải một baseline training độc lập không APPLIANCE**.

Đọc `completion.json` (`completed=true`), `client_metrics.json`, `confusion_matrices.npz`, `pipeline_lock.json`. `appliance_activity` báo authorized/suspended và số row thực sự kích hoạt; không dùng authorized count thay cho test gain.
