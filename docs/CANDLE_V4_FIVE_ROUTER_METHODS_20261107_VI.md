# Năm hướng từ literature để chọn task trong Top2

Đã chạy đủ năm phương pháp trên100 client và204.800 mẫu panel cố định.
SLDA-inspired tốt nhất trong năm bản thích nghi đã thử: Accuracy48,1724%,
Macro-F1 27,6606%. Gain accuracy chỉ0,1006 điểm so Top1; chưa đủ chứng minh
cải thiện ổn định. Không đổi inference chính thức hoặc yêu cầu train full lại.

## Paper và phạm vi thích nghi

1. NCM Euclidean: dùng centroid class và khoảng cách Euclidean trong feature
   đã StandardScaler. Tham khảo nearest-mean classification trong
   [iCaRL](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html).
   Không thực hiện herding, distillation hay raw exemplar rehearsal của iCaRL.
2. Shared covariance: khoảng cách Mahalanobis với covariance residual chung
   LedoitWolf. Tham khảo [Deep SLDA](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf).
   Fit batch trên memory có sẵn, chưa phải update streaming như paper.
3. Covariance riêng class: Mahalanobis với LedoitWolf từng class; singleton
   dùng covariance chung. Tham khảo [FeCAM](https://arxiv.org/abs/2309.14062).
   Chưa có Tukey transform/correlation normalization của paper.
4. Random projection: Gaussian256 chiều seed42, ReLU rồi ridge lambda1
   để phân biệt class. Tham khảo [RanPAC](https://arxiv.org/abs/2307.02251).
   Không sử dụng pretrained ViT hay first-session adaptation như paper.
5. kNN5: năm láng giềng gần nhất trong standardized saved feature memory,
   vote đều. Tham khảo hướng feature-neighbor của
   [DkNN](https://arxiv.org/abs/1803.04765); không thực hiện multilayer
   conformal calibration đầy đủ của DkNN.

Đây là năm prototype lấy cảm hứng từ paper để phân xử task trong CANDLE,
không phải benchmark tái lập nguyên thuật toán hoặc kết luận paper đó thất bại.

## Protocol

- Checkpoint results20/21 finalT5/R19; saved receiver TRAIN activation memory.
- Fit cả năm trước khi mở TEST; scaler có sẵn; không raw TRAIN lịch sử.
- Cùng2048 mẫu/client từ official disjoint shards, seed20261028+client ID.
- Chỉ chọn một trong hai task gốc; score task là max score class thuộc task đó.
  Tie giữTop1, task không có memory không được thêm vào candidate set.
- Adapter, class mask và expert scope đúng task đã chọn, threshold expert0,5.
- Không sweep hyperparameter. TEST đã dùng development nhiều lần, không phải
  confirmation độc lập. Không dùng true task để chọn prediction.

## Kết quả

| Cơ chế | Accuracy | Macro-F1 | Route accuracy | Rescue | Break | Net |
|---|---:|---:|---:|---:|---:|---:|
| Top1 |48,0718%|27,5621%|51,0513%|0|0|0|
| NCM Euclidean |47,3916%|26,7009%|50,3301%|6.989|8.382|-1.393|
| SLDA-inspired |**48,1724%**|**27,6606%**|**51,1616%**|4.262|4.056|**+206**|
| FeCAM-inspired covariance |43,2188%|23,7000%|45,8740%|7.168|17.107|-9.939|
| RanPAC-inspired |48,1626%|27,3996%|51,1519%|3.415|3.229|+186|
| kNN5 |47,8984%|27,4845%|50,8516%|6.211|6.566|-355|

SLDA cải thiện60 client, giảm39, hòa1. Paired bootstrap resample100 clients,
10.000 lần seed42: khoảng95% mô tả gain accuracy là[-0,1831;+0,3574] điểm.
RanPAC cải thiện51, giảm45, hòa4; khoảng[-0,2842;+0,4639] điểm.
Cả hai khoảng chứa0. Đây không phải uncertainty qua training seeds, cũng không
điều chỉnh selection bias khi chọn tốt nhất trong năm; không gọi gain chắc chắn.

## Correctness và chi phí

- Top1 confusion matrix khớp tuyệt đối panel oracle diagnostic trước.
- Native weights và toàn bộ service state hash giữ nguyên.
-8 file V1/native được bảo vệ khớp manifest; không sửa V1/training/notebook.
-2 unit tests PASS: năm scorer trên geometry đơn giản, missing-task exclusion,
  input memory bất biến, singleton covariance fallback.
- Network bytes bổ sung0 vì fit và inference dùng local saved features.
  Không có nghĩa compute/storage bằng0: covariance riêng tốn O(Cd²), shared
  covariance O(d²), random projection O(256d+256C), kNN dùng saved feature bank.
- Combined evaluation94,71s trên máy local, chưa tính fit/loading/checkpoint;
  đây là thời gian chung chia sẻ forward giữa5 phương pháp, không phải latency
  riêng của từng phương pháp. Chưa đủ để xếp hạng inference overhead chính xác.

## Hướng nên giữ

Giữ SLDA-inspired làm ứng viên development ưu tiên vì tăng cả accuracy và F1
với shared covariance gọn hơn covariance từng class. RanPAC là đối chứng thứ hai.
Chưa có bằng chứng đủ mạnh để thay Top1 chính thức48,08% fullTEST, hoặc dự báo
>50%/61%.61,30% vẫn là oracle ở panel trước, không phải kết quả thực tế.

Năm phép thử cùng thiếu thông tin về class không có trong receiver memory.
Chúng thay metric trên memory hiện hữu, không tự tạo coverage hay sửa drift.
Bước có giá trị tiếp theo là so baseline/shared-covariance trên cohort bổ sung
receiver-space donor features được chọn trước bằng metadata, rồi kiểm tra
retention qua task; không tiếp tục dò nhiều công thức trên TEST để tìm số đẹp.
Nếu mở rộng training phải khóa protocol và giữ V1 làm đối chứng riêng.

Script:`tools/pilot_candle_v4_five_router_methods.py`.
Receipt:`artifacts/candle_v4_five_router_methods_20261107.json`.
Plan:`plans/candle_v4_five_router_methods_20261107.md`.
