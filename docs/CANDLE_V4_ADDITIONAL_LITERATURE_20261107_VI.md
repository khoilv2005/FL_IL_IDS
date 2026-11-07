# Research bổ sung: drift, task inference và missing-class transfer

Ngày đọc:2026-11-07. Đây là literature review và đề xuất kiểm chứng, chưa phải
implementation hoặc kết quả của CANDLE. Chỉ dùng paper/trang proceedings gốc.
Không thay V1/training/default notebook trong lượt research này.

## 1. Learnable Drift Compensation — ECCV2024

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01192.pdf)

LDC học projector giữa feature spaces liên tiếp để cập nhật prototype cũ khi
backbone thay đổi. Dùng current-task data và old/new models để có các cặp
biểu diễn, không cần giữ raw exemplars của task cũ.

Liên hệ CANDLE (suy luận): feature memory được ghi ở các thời điểm khác nhau,
trong khi encoder tiếp tục cập nhật. Bảo toàn feature indices không chứng minh
giá trị activation không drift. Cần đo trước khi kết luận đây là nguyên nhân.

Pilot đề xuất: tại transition, cho cùng current TRAIN qua old/new encoder;
fit projector rồi cập nhật bản sao historical memory. So Top1/SLDA với và không
compensation, cùng data/steps. Extrapolation từ current classes sang old classes
có thể sai. Một previous model tạm thời để tạo cặp TRAIN là chi phí bổ sung;
nếu giữ ràng buộc cấm mọi model copy cần thống nhất lại thiết kế trước implementation.
Không cần inference thêm model hoặc replay raw lịch sử.

## 2. Bidirectional Alignment with Cycle Consistency — ICLR2026

[Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c1e16692b361782abafe936bd7ac2d75-Abstract-Conference.html)

Học cả old→new và new→old maps trong lúc training, thêm cycle consistency
và stop-gradient gating. Paper nghiên cứu sai lệch tích lũy của mapping một chiều.

Suy luận cho CANDLE: phù hợp làm đối chứng mạnh hơn LDC nếu compensation một
chiều bị lỗi qua nhiều task. Khác với mean-shift từ2 anchor đã thử: cần paired
features trên nhiều current samples, không suy full256D map từ vài class means.
Chưa bảo đảm cải thiện cross-client alignment, paper nghiên cứu temporal drift.
Ưu tiên sau baseline LDC vì phức tạp hơn và cần training-time integration.

## 3. Resurrecting Old Classes with New Data — CVPR2024

[Paper](https://arxiv.org/abs/2405.19074)

Perturb current inputs để embedding gần old prototypes trong old encoder;
dùng các inputs đã perturb để ước lượng drift và bù vị trí prototypes.

Suy luận: có thể hỗ trợ khi current-task examples ít đại diện vùng class cũ.
Tuy nhiên paper trên ảnh.39 network traffic features có ràng buộc miền giá trị,
quan hệ giữa cột; perturb tùy ý có thể tạo input vô nghĩa. Cần constraint/domain
validation trước khi dùng. Tăng forward/backward; không ưu tiên pilot đầu.

## 4. MORE — CoLLAs2022; task inference/OOD

[Paper](https://proceedings.mlr.press/v199/kim22a.html)
[Lý thuyết liên quan, NeurIPS2022](https://papers.nips.cc/paper_files/paper/2022/file/20f44da80080d76bbc35bca0027f14e6-Paper-Conference.pdf)

MORE học task-specific heads vừa phân loại trong task vừa từ chối dữ liệu
ngoài task. Bản gốc dùng saved samples làm OOD replay, nên không đáp ứng trực
tiếp ràng buộc no historical raw replay của CANDLE.

Suy luận: nên phân biệt khả năng nhận class với khả năng từ chối ngoài task.
Điều này liên quan việc experts hiện nhầm khi mở union, nhưng chưa chứng minh
thiếu OOD training là nguyên nhân duy nhất.

Pilot có thể thích nghi: học task discriminator từ saved TRAIN activation
memory và current-task features hợp lệ, dùng task-balanced negatives và
single-task scoped inference. Không được gọi là MORE reproduction, không lấy
raw lịch sử, không hiệu chỉnh threshold trên TEST. Feature drift vẫn là rủi ro.
Nếu detector học trực tiếp raw39features thì historical activation memory
không dùng được làm raw negatives; không đánh đồng hai representation.

## 5. FedTGP — AAAI2024

[Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/29617)

Học trainable global prototypes bằng adaptive-margin contrastive objective
để cải thiện separability thay vì chỉ average prototypes. Bản gốc có server.

Suy luận: local receiver có thể duy trì prototypes có margin trong feature
space đã xác thực, học từ eligible peer messages. Đây là decentralized redesign,
chưa phải FedTGP nguyên bản. Prototype separability không tự bảo đảm sample
features của receiver nằm gần đúng prototype, nhất là class chưa học. Nên thử
chỉ sau khi receiver-space feature transfer/alignment được kiểm chứng.

## 6. FedGen — ICML2021

[Paper](https://proceedings.mlr.press/v139/zhu21b/zhu21b.pdf)

Server học lightweight generator từ ensemble knowledge rồi gửi về clients
để regularize local training, không cần proxy dataset có sẵn theo thiết kế paper.

Suy luận: receiver-owned feature generator có thể tạo missing-class features
đa dạng hơn một centroid. Nhưng cần giải quyết representation compatibility,
mode coverage, extra compute/communication và thiết kế lại server thành P2P.
Không thể xem Gaussian noise quanh prototype là đã tái lập FedGen.
Không ưu tiên trước drift/OOD vì thay đổi hệ thống lớn hơn và chưa có evidence
generator tạo được đúng receiver-space distribution.

## Tài liệu căn chỉnh representation liên quan

[FedPAC](https://arxiv.org/abs/2306.11867) explicit local-global feature alignment
và classifier collaboration. Đây là nền tảng cho hướng alignment trong lúc
training; shared/global semantic machinery không tự biến thành decentralized
continual protocol. Cần old-class retention và full communication accounting.

## Ưu tiên và giới hạn

1. Đo temporal drift trước: paired current inputs ở old/new encoder, so feature
   displacement và router disagreement. Không cần đánh giá lại nhiều công thức TEST.
2. Nếu drift rõ, LDC là pilot có phạm vi nhỏ nhất; shared covariance hiện tốt
   nhất trong5variant đã thử làm đối chứng. Không mặc định LDC sẽ có gain.
3. Nếu coverage là chính, receiver-space feature transfer + task/OOD
   discrimination trên current TRAIN/saved features là hướng riêng cần đo.
4. Cycle consistency là nâng cấp có cơ sở khi projection một chiều thất bại.
5. FedTGP/FedGen/perturbation là lựa chọn sau, không gộp ngay cả6 vào một model.

Không thể chọn phương pháp tốt nhất từ accuracy trên dataset của paper khác.
Các paper không chứng minh CANDLE sẽ đạt>50% hoặc61%. Thử nghiệm tương lai
cần khóa cohort/config trước và tách TEST-development khỏi confirmation độc lập;
giữ100% current-task TRAIN, không dùng TEST để fit/tune, V1 giữ nguyên.
