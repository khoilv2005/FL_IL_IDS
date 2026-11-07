# Pilot task-versus-background routing

Thử hai biến thể lấy cảm hứng từ task/OOD separation trong
[MORE,CoLLAs2022](https://proceedings.mlr.press/v199/kim22a.html).
Đây là auxiliary binary logistic classifiers trên saved features, không phải
reproduction MORE transformer/replay hoặc bằng chứng OOD detection tổng quát.

## Protocol

-100clients, cùng204800-row development panel trước,2048/client,
  seed20261028+clientID, official shard seed523687.
-Fit tất cả bộ chấm điểm trước mởTEST, chỉ saved receiver TRAIN features,
  existing StandardScaler. Không historical raw replay, không train backbone.
-OVR-task: từng task là positive, các task còn nhớ là negatives.
-OVR-class-max: từng class là positive, các class còn nhớ là negatives;
  task lấy max class decision score.
-C1,balanced,lbfgs,max_iter1000,random_state0 cố định, không sweep.
-Decision scores chưa calibrated; so chỉ trong hai task gốc. Task không có memory
  không được đưa vào candidates; ties giữTop1. Đây là một điểm rủi ro đo bằng
  end-to-end test, không coi score là task posterior.
-Giữ đúng một task cho adapter, mask, expert scope, threshold0,5.
-Binary classifiers chỉ thấy observed negatives; không tự biết classes chưa
  lưu hoặc task chưa có memory. Chưa giải quyết coverage.

## Verification

-Original baseline confusion matrix bắt buộc khớp prior panel.
-Hash native weights và toàn bộ original service state giữ nguyên.
-8protected V1/native files khớp manifest.
-2OVR tests PASS: simple separation, single-class/task, absent-task exclusion,
  source feature memory bất biến.2legacy five-method scorer tests PASS.
-Shared evaluator thêm optional scorer/method/output arguments, default cũ giữ
  nguyên, không sửa production runtime hoặc notebook.
-Extra peer communication0; fit thêm nhiều binary models tốn compute/storage.
  Combined evaluation wall time không phải latency riêng từng method.
-TEST đã dùng development nhiều lần; không gọi confirmation độc lập.

## Kết quả

Xem receipt `artifacts/candle_v4_ovr_task_router_20261107.json` và bảng cập nhật bên dưới.

| Cơ chế | Accuracy | Macro-F1 | Rescue | Break | Net |
|---|---:|---:|---:|---:|---:|
|Top1|48,0718%|27,5621%|0|0|0|
|OVR-task|47,4541%|26,8494%|7.694|8.959|-1.265|
|OVR-class-max|47,9458%|27,4032%|2.096|2.354|-258|

Task binary đổi32.013 routes; class binary đổi10.103. Route đúng lần lượt
103.157 và104.293, thấp hơn baseline104.553/204.800. NO-GO cho thay thế
inference chính thức bằng hai biến thể này. Không suy từ kết quả này rằng MORE
nguyên bản thất bại hoặc OOD learning không thể giúp CANDLE.

Trong các phép thử gần đây, SLDA-inspired vẫn tốt nhất trên cùngpanel:
48,1724% accuracy,27,6606% Macro-F1; gain nhỏ và bootstrap client chứa0,
chưa là improvement ổn định hay full-system benchmark.

Kết quả OVR không tự chứng minh coverage là nguyên nhân duy nhất. Nhưng tiếp
tục fit classifier từ cùng remembered classes không cung cấp bằng chứng về
missing classes. Đề xuất tiếp theo là cohort metadata chọn trước cho receiver-space
donor features, đối chứng shared-covariance và production router, ghi cả negative
transfer/reference bytes. Không hạ threshold hoặc dò thêm binary settings trênTEST.
Full kết quả chính thức vẫn48,08%/27,61%; không đổi default notebook/training.
