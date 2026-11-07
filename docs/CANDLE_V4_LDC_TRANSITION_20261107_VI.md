# Kiểm chứng LDC-inspired: Task4 → Task5

## Kết luận

NO-GO cho việc thêm compensation vào V4 dựa trên pilot này. Ở ba client2,5,6,
routing activations của cùng current TRAIN input trước/sau Task5 khớp tuyệt đối
trên các cặp đã lấy mẫu: rawMSE=0. Hai phép bù học lại làm accuracy giảm.
Chưa có bằng chứng temporal drift T4→T5 là nút thắt ở ba ca này.

Đây không phải kết luận LDC nói chung thất bại, hoặc mọi client/mọi transition
đều không drift. Historical memory có thể đã lệch trướcT4; pilot không đo được
điều đó. Một window và current-domain samples không chứng minh toàn bộ historical
activation distribution được bảo toàn.

## Fixture và protocol

- Checkpoint cuốiTask4 từ fresh-run results19, checksum khớp archive manifest;
  checkpoint cuốiTask5 results20/21. Không dùng old-v7 Task4.
- Ba client đầu theoID có ít nhất2 task memory lịch sử, có Task5 memory,
  cùng selected indices và historical feature arrays byte-equal trước/sau.
- Bổ sung điều kiện có actual local Task5 TRAIN>1280 theo native receipt.
  Preflight ban đầu dừng khi client0 chỉ có bootstrap memory, chưa fit hoặc mởTEST;
  sửa selection theo data availability, không theo chất lượng.
- Chỉ đọc Original TRAIN class30..33. Mỗi class lấy tối đa256fit và64assess,
  seed42. Client2:537fit/128assess; client5:279/64; client6:456/64.
- Current TRAIN chạy qua frozen old/new encoder để có paired features.
- Ridge affine,lambda1,intercept không penalty. MLP d→128→d,ReLU,Adam.001,
  100full-batchsteps,seed42; scaler fit riêng hai phía từfit pairs.
- Map chỉ historical feature memory; giữTask5 memory hiện tại, refit cùngLR.
- Controls: production saved router và refit memory nguyên bản.
- Mọi fit hoàn tất trước mởTEST. Dùng128samples/class×34=4352/client;
  cùngpanel cho3client, tổng13056lượt prediction. Không phải official disjoint
  pooled benchmark, không so con số34,87% này với full-system48,08%.
- Giữ một task duy nhất cho adapter/mask/expert scope; thresholdexpert0,5.
- Đây là hai bản thích nghi từ [LDC,ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01192.pdf),
  không tái lập nguyên paper hay training-time integration.

## Drift assessment trên current TRAIN chưa dùng fit

| Client | Identity standardizedMSE | Ridge | MLP |
|---|---:|---:|---:|
|2|0|0,000294|0,098810|
|5|0|0,000710|0,071512|
|6|0|0,006307|0,182763|

Identity đã đúng trên input pairs; regression có shrinkage và MLP có approximation
error tạo sai lệch không cần thiết. Chưa phải bằng chứng mọi old feature sẽ có
transport error giống current samples.

Old/new router predictions khác nhau45,31%,43,75%,57,81% trên current assess
samples của2,5,6 dù input activation không đổi. Hai router có class support và
classifier/scaler fit quaTask5 khác nhau, nên disagreement không thể được dùng
đơn độc làm chỉ số representation drift.

## Kết quả class prediction

| Cơ chế | Accuracy | Macro-F1 | Old classes0..29 accuracy | Task5 accuracy | Rescue / Break |
|---|---:|---:|---:|---:|---:|
|Baseline|34,8729%|30,8583%|37,5434%|14,8438%|0/0|
|Refitcontrol|34,8575%|30,8415%|37,5347%|14,7786%|0/2|
|Ridgecompensation|34,6124%|30,6108%|37,3177%|14,3229%|68/102|
|MLPcompensation|26,7999%|23,7979%|28,3333%|15,2995%|331/1385|

MLP tăng nhẹTask5 nhưng phá nhiều old predictions. Không chọn thông số/epochs
khác từTEST để cứu phương pháp. TEST ở đây là development sau nhiều vòng audit,
không phải independent confirmation. Local sklearn1.9.0 khác saved1.6.1;
refitcontrol vì vậy có thể khác baseline vài rows, đã báo riêng.

## Verification và hành động

- Hash native model và toàn bộ original service(expert/Adam/router) giữ nguyên.
-8 protected V1/native files khớp manifest.
- Unit test PASS: ridge phục hồi affine shift ở unseen pairs, query không mutate
  ridge, MLP output hữu hạn. Native encoder không train; chỉ projector/routercopy fit.
- Chưa tạo production shadow predictor; old/new model copies chỉ là temporary
  diagnostic references. Không thay training hoặc notebook, không yêu cầu full lại.
- Mappings học local; không phát sinh P2P transfer trong pilot. Tăng compute và
  cần temporary previous encoder, chưa benchmark latency/storage riêng.

Không ưu tiên cycle-consistency ở window đã đo zero drift. Hướng kiểm chứng
tiếp nên tập trung coverage và task/OOD discrimination, hoặc đo thêm window nếu
có metadata nghi ngờ drift. Không giả định cứ tăng độ phức tạp projector là tăng
accuracy. Script:`tools/pilot_candle_v4_ldc_transition.py`;
receipt:`artifacts/candle_v4_ldc_transition_20261107.json`.
