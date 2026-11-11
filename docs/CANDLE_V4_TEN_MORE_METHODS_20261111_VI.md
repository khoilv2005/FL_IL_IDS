# Research thêm10phương pháp cho CANDLE

Ngày truy cập2026-11-11. Nguồn gốc gồm CVF,NeurIPS,AAAI và author arXiv.
Đây là research shortlist: kiểm tra mechanism/assumptions, không phải tái lập
thực nghiệm. Không thay V1/training/notebook trong lượt này. Các đánh giá phù hợp
với CANDLE bên dưới là suy luận đề xuất, không phải kết luận từ paper về CANDLE.
Không đếm lại LDC,SLDA,FeCAM,RanPAC,MORE,FedTGP,FedGen,FedPAC đã bàn.

## 1. FeTrIL — WACV2023

[Nguồn](https://openaccess.thecvf.com/content/WACV2023/html/Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper.html)

Giữ feature extractor cố định sau initial training, lưu centroid của class cũ.
Dịch features class mới để tạo pseudo-features class cũ rồi train linear
classifier trên real new features và pseudo old features.

Ứng dụng đề xuất: học classifier phân xử giữa các class/tasks bằng phân phối
features thay vì chỉ centroid. Phải có centroid đúng receiver-space. Không giải
quyết việc donor prototype không aligned; không thể lấy donor mean thô rồi
gọi đó là receiver-space knowledge. Fixed feature extractor là giả định khác
native CANDLE; pilot post-hoc chỉ là variant thích nghi, không full FeTrIL.
Ưu tiên cao cho thử nghiệm head trên feature space đã xác thực.

## 2. PASS — CVPR2021

[Nguồn](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html)

Prototype augmentation trong feature space giữ decision boundary của old tasks;
self-supervision cải thiện transferability. Không lưu raw exemplars task cũ.

Ứng dụng: pseudo-feature negatives có thể giúp classifier/expert tránh cướp
class cũ, nếu cùng representation. Không phải cứ thêm Gaussian noise quanh
prototype donor là có phân phối đúng. Self-supervised image augmentations cần
thiết kế lại cho39trafficfeatures; không rotate/permute các cột tùy ý.
Ưu tiên phần prototype augmentation trước, báo rõ bỏSSL thì không full PASS.

## 3. IL2A — NeurIPS2021

[Nguồn](https://papers.nips.cc/paper_files/paper/2021/file/77ee3bc58ce560b86c2b59363281e914-Paper.pdf)

Dual augmentation: classAug để học representation đa dạng, semanAug để giảm
classifier bias bằng thông tin ngữ nghĩa/phân phối old classes. Không replay
raw old samples; paper phân tích covariance đầy đủ và diagonal approximation.

Ứng dụng: covariance-aware loss hoặc pseudo-feature objective là một hướng
khác với dùng Mahalanobis trực tiếp như FeCAM pilot. Cần thống kê đủ đáng tin,
không suy full covariance từ vài mẫu; chi phí O(Cd²) có thể lớn. Phần classAug
cho dữ liệu mạng cần kiểm tra semantics. Ưu tiên vừa, không chỉ đổi threshold.

## 4. ACIL — NeurIPS2022

[Nguồn](https://papers.nips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html)
[Fullpaper](https://proceedings.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf)

Analytic/recursive least squares classifier duy trì thống kê, không cần raw
historical inputs. Equivalence với joint analytic solution dưới điều kiện của
paper; không đồng nghĩa bằng joint end-to-end neural training hoặc không sai.
Backbone được freeze trong incremental phases; feature expansion tăng chiều.

Ứng dụng đề xuất cho P2P: khi cùng feature reference, có thể cộng incremental
sufficient statistics G+=ZᵀZ,B+=ZᵀY rồi giải W=(G+λI)^(-1)B. Đây là suy luận
redesign, không phải ACIL gốc đã chứng minh decentralized CANDLE. Phải chống
đếm trùng packet/sample, bind version và giữ reference ổn định; khi encoder đổi
thì thống kê cũ không mặc nhiên còn hợp lệ. Không có formal DP chỉ vì dùngstats.
Ưu tiên cao nếu muốn thay routing/head bằng joint class decision có memory gọn.

## 5. DS-AL — AAAI2024

[Nguồn](https://ojs.aaai.org/index.php/AAAI/article/view/29670)

Main stream analytic learning và compensation stream với dual activation để
giảm underfitting của mapping tuyến tính; concatenated recursive least squares.
Không cần raw exemplars cũ theo thiết kế paper.

Ứng dụng: đối chứng nâng cấp ACIL nếu analytic main stream chứng minh bị
underfit. Không nên thêm trước khi có ACIL control. Thêm matrices/storage và
compute; cùng reference/alignment và provenance vẫn bắt buộc. Khác RanPAC
pilot hiện có: cần thực hiện đúng recursive/compensation protocol, không chỉ
đổi activation trong một batch ridge rồi gọi fullDS-AL. Ưu tiên vừa.

## 6. SLCA — ICCV2023

[Nguồn](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_SLCA_Slow_Learner_with_Classifier_Alignment_for_Continual_Learning_on_ICCV_2023_paper.html)

Slow learner giảm learning rate representation; classifier alignment dùng
class-wise distributions để điều chỉnh head sau học. Bối cảnh paper là mạnh
pretrained models, không phải CNN-GRU train from scratch Non-IID.

Ứng dụng: thử classifier alignment theo distribution khi reference compatible,
không tiếp tục chỉ cân một bias vô hướng. Slow-learning cần fresh training
variant và không được gọi là checkpoint-only fix. Pilot LDC vừa đo zero drift
ở3client T4→T5 nên chưa có lý do giảmLR chỉ để chữa drift. Ưu tiên phần
classifier alignment, không dự báo gain từ kết quả image benchmark của paper.

## 7. SSRE — CVPR2022

[Nguồn](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Self-Sustaining_Representation_Expansion_for_Non-Exemplar_Class-Incremental_Learning_CVPR_2022_paper.html)

Representation expansion, structure reorganization, side-branch update và
main-branch distillation; prototype selection giúp phân biệt old/new classes.
Không giữ raw exemplars theo problem setting của paper.

Ứng dụng: bổ sung trainable capacity có kiểm soát để học class thiếu thay vì
ép head trên backbone chưa có khả năng phân biệt. Cần fresh training, can thiệp
CNN/GRU/BN và maturity; không phải sửa eval. Không mặc định prototype selection
sẽ giải quyết P2P feature misalignment. Ưu tiên thấp hơn các thử head nhẹ.

## 8. EASE — CVPR2024

[Nguồn](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Expandable_Subspace_Ensemble_for_Pre-Trained_Model-Based_Class-Incremental_Learning_CVPR_2024_paper.html)

Task-specific adapters tạo expanding subspaces và joint decision trong các
subspaces; semantic-guided prototype complement bổ sung old-class features
ở không gian mới khi không có old raw instances.

Ứng dụng: kiến trúc có thể tránh bắt buộc hardTop1 routing bằng joint subspace
classifier. Không đồng nghĩa mở union expert sigmoid hiện tại đã fail. Cần
adapter/prototype complement trained theo cùng cơ chế; paper dùng PTM mạnh.
Thêm adapter paths tăng local inference cost, phải đo so một local predictor
hiện tại; inference vẫn có thể local nhưng nhiều computation branches.
Thay đổi lớn, cân nhắc riêng nếu cải tiến nhỏ không đủ.

## 9. Supermasks in Superposition — NeurIPS2020

[Nguồn](https://papers.nips.cc/paper/2020/hash/ad1f8bb9b51f023cdc80cf94bb615aa9-Abstract.html)

Giữ random base network cố định, học task supermasks. Khi không có taskID,
gradient-based optimization trên mask mixture giảm entropy để tìm task.

Ứng dụng: gợi ý lấy tín hiệu từ phản ứng classifier/adapter trước input, khác
fit thêm LR trên router memory. Nhưng lấy entropy thấp nhất của native CANDLE
không phải SupSup reproduction: mask learning/base network/output design khác.
Task có một valid class cho entropy0 vô điều kiện; số class khác nhau và
overconfidence có thể làm score lệch. Phải xử lý degenerate cases trước pilot.
Test-time gradient selection thay đổi eval compute/protocol, cần khai báo riêng;
không update weights/threshold bằng TEST rồi coi như static benchmark.

## 10. FedETF — ICCV2023

[Nguồn](https://arxiv.org/abs/2303.10058)
[Officialcode](https://github.com/ZexiLee/ICCV-2023-FedETF)

Tên paper: No Fear of Classifier Biases: Neural Collapse Inspired Federated
Learning with Synthetic and Fixed Classifier. Dùng synthetic fixed ETF classifier
để giảm classifier bias và tạo mục tiêu representation chung giữa clients.
Không nhầm với paper khác arXiv2303.16066 có tên tương tự và tác giả khác.

Ứng dụng: cùngseed/class mapping xác định classifier geometry từ đầu có thể
giảm alignment khó khăn trong transfer. Đây là inference đề xuất cho CANDLE,
không được xem ETF đảm bảo features của missing class receiver tự đúng.
Bản gốc federated aggregation có server; P2P continual+maturity cần redesign.
Fresh training thay đổi loss/head, giữ V1 thành baseline riêng. Cần đo forgetting,
capacity và false activations; không retroactively cài ETF vào checkpoint cũ.

## Thứ tự đề xuất

Không chọn best từ accuracy của các benchmark khác nhau. Shortlist dựa vào
phù hợp failure mode và mức thay đổi, chưa phải xếp hạng empirical CANDLE:

1. FeTrIL-inspired và SLCA classifier-alignment: pilot trên saved receiver-space
   feature distributions đã có, single-task selection/cumulative classifier controls.
   Chỉ thực hiện nếu source feature coordinate/version hợp lệ, không dùng
   receiver TEST target features để tạo prototype. Có thể đánh giá khả năng
   calibration, chưa tái tạo được full training của paper từ finalcheckpoint.
2. ACIL: main candidate cho protocol training-time tích lũy stats ở reference
   ổn định, DS-AL là upgrade khi có underfitting evidence. Missing class vẫn cần
   positive evidence từ peer hợp lệ; stats từ donor encoder riêng không cộng được.
3. FedETF: training redesign có nguyên tắc nếu ưu tiên alignment từ gốc;
   thay đổi lớn hơn, chưa phải bước tiếp theo tự động hoặc có bảo đảm accuracy.
4. PASS/IL2A giúp xây objective bổ sung; SSRE/EASE là capacity/subspace redesign;
   SupSup là hypothesis về routing bằng classifier response, chưa có quality gain.

Mọi pilot giữ100%current-task TRAIN, không replay rawtaskcũ, không fit/tune từTEST,
khóa fixture/config trước, đo rescue/break cùng comm/reference/inference cost.
Sử dụng synthetic features hoặc sufficient stats không tự bảo đảmDP.
Không claim>50%/61% từ literature. Chưa chạy10phương pháp trong lượt research này.
