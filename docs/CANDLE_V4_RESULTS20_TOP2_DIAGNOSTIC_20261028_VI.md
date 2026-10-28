# CANDLE V4 results20: Top-2 task diagnostic local

Đây là chẩn đoán trên checkpoint đã train, không phải inference mới được triển khai.
Không train/refit, không chọn threshold, không sửa runtime hay V1.

## Protocol

- Checkpoint results20 cuối T5/R19, sau consolidation.
- 100 client shards chính thức, seed partition 523687.
- Mỗi client lấy ngẫu nhiên không hoàn lại 2.048 rows, seed 20261028 + client ID.
- Tổng 204.800 TEST rows. Đây là TEST development diagnostic, không phải holdout
  độc lập để chứng nhận cơ chế sẽ được thiết kế sau đó.
- Đọc trực tiếp scaler/LR coefficients đã lưu; không refit với sklearn local.
- Decoder hệ số được assert khớp task Top-1 của production router trên toàn panel.
- Hai nhánh dùng cùng model, native task masks/adapters và expert bank threshold0,5.
- Baseline dùng Top-1. Oracle chỉ đổi sang Top-2 nếu task thật đứng thứ hai;
  các mẫu khác giữ baseline. True task chỉ dùng cho counterfactual này.
- Verify native weights và expert/Adam state không thay đổi.

## Kết quả trên cùng panel

| Metric | Baseline thực tế | Oracle Top-2 diagnostic |
|---|---:|---:|
| Accuracy | 48,0718% | 61,2993% |
| Macro-F1 | 27,5621% | 35,2000% |
| Task đúng có trong lựa chọn | 51,0513% Top-1 | 65,1992% Top-2 |

- True task đứng thứ hai: 28.975 rows.
- Đổi đúng task thứ hai + giữ nguyên V4 classification cứu 27.090 rows.
- Accuracy gain oracle: +13,2275 điểm phần trăm.
- 93,4944% các rows có true task thứ hai được phân loại đúng sau đổi task.
- 1.885 rows vẫn sai dù đã chọn đúng task thứ hai.
- Native classification riêng nhận đúng 17.357 trong nhóm true-task-second;
  V4 gồm experts nhận đúng 27.090. Không dùng con số này làm gain toàn hệ thống.
- Task hoàn toàn thiếu trong receiver memory: 45.994 rows; Top-2 không bổ sung memory.
- Broken=0 là hệ quả của việc oracle chỉ đổi các mẫu có Top-1 task sai. Không phải
  bằng chứng một router thực tế có thể đạt zero break.

Client có nhiều rescue nhất trên panel: 39 (666), 79 (570), 95 (508), 75 (504),
16 (498), 3 (497), 36 (491), 88 (488), 87 (482), 63 (480).

Client19: 1,7090% -> 4,6875%; client38: 3,5156% -> 4,1504%.
Hai client này không được cứu nhiều bởi Top-2; không nên chỉ chọn các client thấp
nhất để phát triển routing.

## Diễn giải và giới hạn

Có dư địa đáng kể để nghiên cứu phân xử Top-1/Top-2. Chưa có quy tắc input-only
chọn task thứ hai, chưa đo rescue/break của một phương pháp mới và chưa chứng minh
vượt CGoFed. 61,30% là oracle trên panel, không phải full TEST metric hoặc dự báo.

Full results20 chính thức vẫn là Accuracy48,0823%, Macro-F1 27,6110%.
Không chỉnh ngưỡng theo panel này rồi gọi đánh giá lại là xác nhận độc lập.

Script: `tools/audit_candle_v4_results20_top2.py`.
Artifact: `artifacts/candle_v4_results20_top2_20261028.json`.
Full local receipt: `audit_denice/candle_v4_results20_top2_20261028/audit.json`.
