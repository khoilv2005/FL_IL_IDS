# Resume CANDLE V4: Task4 → Task5 từ results16

Notebook: `resume_candle_v4_task5_from_task4_kaggle.ipynb`, companion cùng tên `.py`.

Chạy Kaggle GPU + Internet, mount dataset `100-clients`, Run All. Tải trực tiếp Drive file `12m7ZyoowmKNhKPiSQQ0FH2mzhAweD7tw`; có thể điền `MOUNTED_RESULTS_ZIP` nếu đã mount ZIP. Download/source/cache nằm dưới `/kaggle/temp`; output giữ nguyên đường dẫn đã khóa `/kaggle/working/results_candle_v4_continuous_class_seed42`.

Đã đọc archive local `C:\Users\khoak\Downloads\results (16).zip` và kiểm tra:

- Source `37a277c8f6ae85e74c9553d4d30e5e27f052ca93`, protocol `candle_v4_continuous_class_router_v7`.
- Chọn `checkpoint_task_4_all_rounds.zip` → `continuation_state_task_4.pt`, không dùng outer `continuation_state_latest.pt`.
- Manifest Task4 có đủ rounds 0–19. Metadata: boundary `task`, completed_task 4, resume_from_task 5, round_state null.
- Continuation 800.923.403 bytes; SHA256 `1336e1d91f4301bac73b78005c9ecd195bbd3ec36b9d4832e7b3b63967bed075`, khớp manifest.
- State lưu 99 client models và 99 continuous routers; giữ nguyên tập state, không tự tạo thêm client. Cấu hình benchmark vẫn 100 clients.

Notebook giữ đúng source commit, original run lock, router, seed42, batch2048, dữ liệu current-task TRAIN và protocol evaluation. Resume model/algorithm/masks/router/expert/optimizer/RNG/history từ full continuation theo runtime gốc. Chỉ train Task5 rounds 0–19, consolidation, rồi Full TEST 34 classes. Không sửa V1, không thêm validation, không thay training/evaluation code. Không copy toàn bộ round checkpoints T0–T4 sang output mới; các archive cũ vẫn ở results16.

Kiểm tra chuẩn bị: 7/7 PASS (chọn Task4 thay latest; chuẩn bị lặp không đổi state; từ chối thiếu round; từ chối sai checksum; không ghi đè output Task5 đã tồn tại; notebook/companion syntax; các notebook V1/V4 có sẵn không thay đổi). Receipt: `artifacts/candle_v4_task5_resume_preflight_20261025.json`.

Chưa chạy training/TEST hoặc thử tải Drive từ Kaggle trong bước chuẩn bị. Notebook cài đúng sklearn1.6.1 theo requirements runtime gốc trước khi nạp router checkpoint. Dùng session mới và Save Version để giữ output Task5.
