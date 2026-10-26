# Notebook cách2 — giữ checkpoint, sửa router memory, đánh giá toàn bộ

File dùng: `eval_candle_v4_historical_repair_kaggle.ipynb` (companion `.py`). Đây là **evaluation-only**, không phải notebook fresh training hoặc resume Task5.

## Chạy

1. Kaggle GPU + Internet, gắn dataset `100-clients` gốc.
2. Điền `RESULTS17_DRIVE_URL` sau khi results17 upload xong. Hoặc gắn ZIP results17 vào Kaggle Input và điền `RESULTS17_ZIP`; notebook cũng tự nhận nếu chỉ có một ZIP chứa Task5 archive.
3. results16 có link Drive điền sẵn; có thể thay bằng `RESULTS16_ZIP` nếu đã mount.
4. Run All. Save Version để giữ output riêng `results_candle_v4_historical_repair_seed42`.

Notebook resolve results17 trước khi tải results16 để tránh tải4,7GB khi link17 chưa sẵn sàng. Source clone theo full public SHA, sparse runtime + hai utility files. sklearn pin1.6.1 như Kaggle run cũ. Download/cache ở `/kaggle/temp`; khoảng14GiB disk tạm cho hai ZIP, continuations và nested archive đang giải nén. Không giải nén toàn bộ20 round models/task.

## Quy trình

- Kiểm tra source/run lock của hai ZIP, completion, SHA256 final checkpoint results17; từng historical continuation T0–T4 kiểm tra manifest/checksum.
- Đọc lineage **từ final checkpoint**, không hardcode Client35/43 hoặc lựa chọn theo Accuracy.
- Truy missing task về đúng client tạo memory và snapshot cuối task đó, gồm clone chain trong cùng task. Kiểm tra sketches, thresholds, masks, coordinates và episode class list.
- Stage tất cả repair hoặc báo lỗi; không bỏ qua client structural FAIL để lấy kết quả đẹp. Chỉ fit class-balanced LR/scaler theo công thức cũ trên TRAIN feature memory đã lưu. Không mở raw TRAIN, không dùng TEST fit/calibrate. Router không thiếu memory giữ nguyên estimator đã lưu, không refit.
- Native model/masks/router và expert weights được restore từ cùng final checkpoint. Giữ threshold0,5, scoped fusion; không acceptance gate mới.
- Chạy lại native terminal evaluator nguyên bản trên100 client models, cùng full34-class TEST phân bổ disjoint theo seed523687. Mỗi row tính một lần; pooled Accuracy/Macro-F1. Không thay thành mỗi client full TEST hoặc oracle task routing.
- Báo V1 native control, continuous-native sau phục dựng và V4 fusion sau phục dựng; metrics V4 cũ trích riêng từ results17. Kiểm tra native control confusion matrix có khớp cũ không.

V1 code/config/notebook và toàn bộ runtime training/evaluation không sửa trong lần đóng gói này. Utility chỉ gọi evaluator có sẵn. Evaluator kiểm tra native/service fingerprints trước/sau; runner kiểm tra thêm expert tensor hashes.

## Output

- `repair_run_lock.json`: source, checkpoint hash, environment và phạm vi readout.
- `repair_provenance.json`: toàn bộ client/task/source/clone-chain và kích thước memory.
- `historical_router_repair_bundle.pt`: router states phục dựng, gắn đúng checkpoint gốc; **không phải full training continuation**, không resume train bằng file này.
- `candle_v4_final_test.json`: full metrics/per-client/confusion, router accuracy, rescue/break so native, so sánh V4 cũ.
- `repair_completion.json`: completion, số rows, Accuracy/F1 và baseline match.

Results16/17, checkpoint gốc và notebook train/resume cũ không bị ghi đè. Không cập nhật backbone/expert, không train lại task nào. Router LR fit trên saved TRAIN feature memory là thao tác duy nhất có fit trong quy trình.

## Đã kiểm tra trước phát hành

- Local6/6 technical tests PASS: exact-clone chain, giữ local/unaffected state, no alias, incompatible descriptors bị reject, member/checksum, extraction chỉ continuations.
- Preflight thật toàn bộ100 clients: **50/50 affected clients phục dựng và fit được**, 50 unaffected giữ nguyên; không mở raw TRAIN hoặc chạy TEST inference trong preflight này.
- Trước đó đã readout thật35/43, đạt70,24% và67,86% Accuracy sau phục dựng. Đây không phải kết quả toàn hệ thống.
- Syntax/companion/pin và V1 isolation kiểm tra riêng. Chưa chạy notebook full readout trên Kaggle trong bước tạo file.

Đây là **development counterfactual sau audit TEST**, không phải fresh V8 training hoặc independent benchmark. Native/sketch metadata equality là structural evidence; không tự tuyên bố formal encoder identity trên mọi input. Phần readout cuối sẽ đo rủi ro thực tế trên tất cả classes, không chọn/hạ threshold từ TEST.

Receipts: `artifacts/candle_v4_historical_repair_tests_20261026.json`, `artifacts/candle_v4_historical_repair_all_client_preflight_20261026.json`.
