# Audit full run DeNICE + APPLIANCE — results (12)

Ngày: 10/10/2026. Nguồn: `C:\Users\khoak\Downloads\results (12).zip`.
Code chạy: `db5199ba8c2f0fcf317c047da900a3d99c34abdf`.

## 1. Kết luận

Training và full-test đã hoàn tất. Automatic install có hoạt động: 17 transaction, 3 commit. Tuy nhiên cả 3 imported routes đều bị suspend ở cuối run. Toàn bộ 13.505.771 prediction APPLIANCE giống MulticlassSelf, không chỉ bằng nhau về accuracy.

Đây là một run integration hoàn tất nhưng **chưa chứng minh được gain end-to-end của APPLIANCE**. Không có cơ sở gọi 37,18% là phần cải thiện do transfer. Gain quan sát được trong các nhánh inference là do thay router trên cùng checkpoint.

Production smoke trước đó chứng minh install/protection/resume, nhưng chưa giải quyết quyền sử dụng patch trên cumulative domain. Chốt full campaign khi blocker này còn tồn tại là quá sớm nếu mục tiêu là chứng minh accuracy gain của APPLIANCE. Phần scope đã được cảnh báo trước run; kết quả thực tế xác nhận nó làm toàn bộ patch không thể tham gia final decision.

## 2. Integrity và metrics

Audit đọc trực tiếp archive, không giải nén checkpoint hoặc fit model. Prediction CSV được đọc theo chunk để recompute confusion matrices và metrics.

- 6 task, 20 round/task: 120 native training rounds.
- 126 APPLIANCE callbacks: 120 round + 6 task-finalized callbacks.
- Full test: 13.505.771 rows, 98 receivers, 34/34 classes.
- Global row ID không trùng; toàn bộ rows được tính đúng một lần.
- Confusion matrices recompute khớp cả 3 nhánh.
- Accuracy/Macro-F1 recompute khớp `completion.json`.
- APPLIANCE vs MulticlassSelf: **0 prediction mismatch / 13.505.771**.

| Policy | Pooled accuracy | Macro-F1 trên 34 class |
|---|---:|---:|
| BinarySelf | 31,2826% | 15,4646% |
| MulticlassSelf | 37,1847% | 19,7499% |
| APPLIANCE | 37,1847% | 19,7499% |

MulticlassSelf tăng 5,9021 điểm accuracy, 4,2853 điểm Macro-F1 so với BinarySelf. Đây là self-router ablation của checkpoint APPLIANCE; không thay thế một fresh DeNICE control run độc lập.

`final_report.json` có accuracy null vì runner đã tắt eval nội bộ. Full-test thật chạy ở launcher sau training, kết quả đúng nằm trong `full_test_task_5/completion.json`.

## 3. Discovery và transaction

| Task | Requests cuối task | Offers cuối task | Cặp graph/FIT khả dụng trước install preflight | Thử install | Commit |
|---|---:|---:|---:|---:|---:|
| 0 | 53 | 38 | 221 | 2 | 0 |
| 1 | 104 | 137 | 1.492 | 5 | 0 |
| 2 | 101 | 80 | 470 | 4 | 1 |
| 3 | 130 | 98 | 372 | 5 | 2 |
| 4 | 80 | 6 | 1 | 1 | 0 |
| 5 | 10 | 7 | 0 | 0 | 0 |

Tổng cặp graph/FIT là 2.556, chưa đồng nghĩa tất cả đều pass provenance/acceptance. Selector chỉ chọn một receiver cho mỗi class; receiver ID nhỏ được ưu tiên, donor theo Wilson quality LCB. Không chọn receiver thay thế theo HOLDOUT khi cặp đầu tiên thất bại.

120 round callbacks đều có zero offers. Code yêu cầu output donor `rank >= 2`; class hiện tại còn young trong round và chỉ được maturation lúc cuối task. Offers xuất hiện ở 6 callback `task_finalized`. Vì vậy automatic discovery chạy mỗi round nhưng transfer thực tế của run này chỉ được thử cuối task.

Outcome của 17 transaction:

- 11 reject `CURRENT_INSTALL_OLD_BASE_SUPPORT_MISSING`.
- 3 reject `STABLE_CURRENT_AGGREGATE_ACCEPTANCE_REQUIRED`.
- 3 committed.

Ba reject do acceptance có target recall 92,8775%, 90,6250%, và 0%, dưới gate 95%; FAR/break của các acceptance này đều zero. Không nên gọi mọi reject là lỗi code hoặc tự hạ threshold theo test.

Old-support preflight hiện lấy mọi output mature của task cũ làm required classes, rồi yêu cầu owned BASE shield bao phủ chúng. Mature output có thể được inherit từ bootstrap/aggregation, trong khi shield chỉ có owned provenance. Dữ liệu run có `representative_clone` cho nhiều receiver bị reject. Đây là mismatch giữa knowledge availability và owned protection evidence; thiếu owned evidence không được sửa bằng cách bịa provenance.

Đề xuất: kiểm tra provenance/support/capacity bằng metadata trước selector, để không chọn receiver đã biết chắc sẽ bị reject. Không chọn cặp khác dựa vào HOLDOUT của cặp thất bại. Muốn phục vụ nhiều receiver cho cùng class cần định nghĩa lại scheduling/budget một cách rõ ràng.

## 4. Ba patch đã commit và sống qua training

| Donor → receiver / class | Task install | Recall trên CAL acceptance | Rescue / break | Trạng thái cuối |
|---|---:|---:|---:|---|
| 89 → 9 / class 12 | 2 | 96,00% | 48 / 0 | SUSPENDED: new protection conflict |
| 60 → 0 / class 21 | 3 | 95,54% | 321 / 0 | SUSPENDED: scope |
| 72 → 4 / class 23 | 3 | 98,90% | 90 / 0 | SUSPENDED: scope |

Cuối Task 5, cả 3 có `head_exact_after=true`, `certificate_current=true`, `function_current=true`. Không thấy head bị ghi đè hoặc certificate/function drift trong metadata đã lưu.

Certificate scope:

- Class 12: `[12,13,14,15,17]`.
- Class 21: `[19,20,21,23]`.
- Class 23: `[18,19,20,21,23]`.

Trong khi runtime scope luôn là toàn bộ classes đã thấy, cuối run là `0..33`. Cả 3 đã bị suspend vì scope ngay ở callback commit. Điều kiện này chặn toàn bộ route, không chỉ những samples thuộc class ngoài scope. Không có task-ID hoặc label-blind domain resolver giúp chứng minh input nằm trong scope cũ.

Patch class 12 còn xuất hiện conflict thực tế ở Task 3 Round 0: current negative FAR **18,4332%**, **40 break**. Conflict bị latch. Đây là lý do riêng để không bật lại patch này chỉ bằng việc sửa scope. Cuối Task 5, monitor có FAR 0,9091%, 1 break.

Không được bỏ `route_authorized()` hoặc dùng true label/task để bật patch. Cần thiết kế chứng nhận cho application domain bằng evidence hợp lệ; CAL ít không tự cấp quyền mở rộng domain. BASE support veto hiện chỉ bảo vệ finite owned support, không thay thế chứng nhận CAL cho toàn bộ class distribution.

## 5. Coverage nhỏ hơn nhiều so với mục tiêu global

Chỉ 3 receiver/class pairs đã nhận patch. Trong test shards của đúng 3 receivers có tổng **23.825** samples thuộc target classes tương ứng:

- Receiver 9 / class 12: 11.896.
- Receiver 0 / class 21: 9.726.
- Receiver 4 / class 23: 2.203.

Ngay cả giả sử cả 3 patch hoạt động hoàn hảo, chỉ sửa target-class predictions và không gây break, gain tuyệt đối tối đa trên cùng frozen baseline là **0,1764 điểm accuracy**. Đây là bound lạc quan; thực tế còn thấp hơn nếu Self đã đúng một phần. Vì vậy chỉ bật 3 patch này không thể đưa 37,18% lên 50%. Phải sửa scheduling/provenance/coverage, không chỉ authorization.

## 6. Loss Task 4 có dấu hiệu bất ổn

| Task | Loss Round 0 | Loss Round 19 |
|---|---:|---:|
| 0 | 3,2847 | 0,8263 |
| 1 | 1,1617 | 0,0172 |
| 2 | 2,0395 | 0,4146 |
| 3 | 2,1117 | 0,8079 |
| 4 | 3,0906 | 3,7647 |
| 5 | 2,5836 | 1,5032 |

Task 4 giảm đến 1,6605 ở Round 4, sau đó tăng gần liên tục: Round 10 = 2,1865, Round 19 = 3,7647. Client train-loss max tăng từ 5,53 lên 32,28. Đây là xu hướng xấu, không còn là nhận định từ một loss point riêng lẻ.

Diagnostics trên tối đa 64 BASE samples/client cho thấy client 48 ở Round 19 có full-output CE và learner-only CE đều khoảng 49,95; client 90 khoảng 19,87; client 60 khoảng 16,22. Vì learner-only CE cũng cao, không thể giải thích toàn bộ bằng competing logits của task cũ. Diagnostics này là sample nhỏ, không thay thế full local evaluation.

Chưa đủ evidence xác định AMP hay aggregation là nguyên nhân. Debug per-client training records bị tắt, không có tổng optimizer skip-rate trong các artifacts đã đọc. Không suy ra global AMP failure từ vài dòng log. Cần audit riêng client 48/90/60 trên checkpoint Task 4; chưa tự thay LR, xi hoặc architecture.

## 7. Class 28

648 true test rows, zero true positive. Model có **19.805 predictions class 28**, tất cả false positives. Vì vậy vấn đề không phải model hoàn toàn không thể xuất class 28 hoặc dataset test thiếu class này.

648 samples thật class 28 bị đoán nhiều nhất sang class 32 (181), class 1 (149), class 25 (69). Recall, precision, F1 class 28 đều zero. Không có APPLIANCE patch class 28 được cài trong run này.

## 8. Chi phí và ZIP

- APPLIANCE application-payload traffic: **88.224.490 bytes ≈ 84,14 MiB**, gồm discovery/setup/failed attempts.
- Receiver function capsules: 39.827.792 bytes ≈ 37,98 MiB.
- Capability packets tổng: 25.646 bytes ≈ 25,04 KiB; không phải toàn bộ transfer cost.
- Tổng APPLIANCE callbacks: 193,69 giây ≈ 3 phút 14 giây.
- Tổng native round timings: 18.322,66 giây. Không có matched control để suy ra causal slowdown; callback timings chỉ là phần được instrument.
- Full-test: 3.104,27 giây ≈ 51 phút 44 giây.
- Archive tải về khoảng 6,316 GiB; tổng member size khi giải nén khoảng 6,783 GiB.
- Sáu task archives: 5,266 GiB; latest full continuation: 0,753 GiB.
- Eval folder và eval ZIP cùng tồn tại: khoảng 0,075 GiB + 0,074 GiB.

ZIP chạy thành công cho run này. Chưa xác minh peak disk usage hoặc bảo đảm quota cho cấu hình/seed khác; output vẫn có bản eval trùng và latest full snapshot.

## 9. Việc tiếp theo

1. Khóa run này làm evidence integration/fallback; không ghi nhận APPLIANCE accuracy gain.
2. Giải quyết cumulative authorization bằng CAL/evidence hợp lệ. Không bỏ safety gate, không dùng test labels để chọn route/scope.
3. Prefilter receiver bằng support/provenance trước lựa chọn donor; định nghĩa scheduling cho nhiều receiver/class và giới hạn communication.
4. Kiểm tra patch class 12 có conflict bằng development/current CAL; không re-enable vì head vẫn nguyên vẹn.
5. Audit loss Task 4 riêng, không thay cả backbone/router/xi trong cùng experiment.
6. Chỉ sau khi install có route được authorize và coverage đủ ý nghĩa mới chạy full tiếp. Checkpoint run này giữ để debug; không cần chạy lại 6 tasks chỉ để đọc nguyên nhân.

Artifacts audit: `audit_denice/appliance_results12/summary.json`, `patches.json`, `transactions.json`, `task_summary.json`, `per_class_metrics.csv`, `audit_output.py`. Audit không sửa code training hoặc checkpoint, không fit từ test.
