# Quyết định nghiên cứu — NO-GO APPLIANCE V2 readout-only

Ngày: **2026-10-10**. Trạng thái: **CLOSED / NO-GO** theo quyết định của người dùng sau bounded campaign tại commit `e7ba5df`.

## Phạm vi quyết định

Đóng phiên bản APPLIANCE V2 hiện tại: donor học một classifier readout row trong feature space của receiver, receiver tích hợp bằng current BASE negatives, rồi dùng native router với class availability được đăng ký.

Không tiếp tục tìm donor, mở rộng budget hoặc chỉnh learning rate/regularization/threshold để cứu một candidate. Không hạ gate recall 95% hoặc FAR 0,1%. Không đưa phiên bản này vào native integration/full training.

Đây là quyết định về **phiên bản V2 readout-only đang xét**; không phải kết luận mọi kiến trúc knowledge transfer đều bất khả thi.

## Bằng chứng được giữ nguyên

| Câu hỏi | Kết luận tại thời điểm đóng |
|---|---|
| Functional-gap discovery có phân biệt registration và functional gap? | Có, trên các development fixtures đã đo |
| Donor learning có lợi ích vượt Availability-only? | Có ở receiver 3/class 7: 0/98 → 92/98 = 93,8776% trên selected donor HOLDOUT |
| V2 đạt primary current HOLDOUT acceptance? | **0/2 trial updates đạt gate** |
| Cumulative old-class protection đã được chứng minh độc lập? | Chưa đủ evidence |
| Native integration/full training đã được cấp phép? | **Chưa; NO-GO** |

Class 7 là **functional feasibility evidence**, không phải một accepted transfer. Class 13 chỉ cải thiện 58/136 → 59/136; không đủ gate. Không đổi nhãn các kết quả development thành independent confirmation hay production success.

Bounded campaign đã đóng ở 6 receiver–class cases, 12 donor–class probes và 2 trial updates. Tổng application egress 78,5433 MiB vẫn chủ yếu là receiver capsules. Không chạy thêm experiment khi ghi quyết định này.

## Code và artifacts

- Giữ nguyên học thuật và số liệu của [bounded campaign](APPLIANCE_BOUNDED_FUNCTIONAL_GAP_CAMPAIGN_20261010_VI.md).
- Giữ [aggregate artifact](../artifacts/appliance_bounded_gap_campaign_20261010.json), checkpoints và trial rows local để phân tích/reproduce.
- Entry point `tools/run_appliance_train_time_transfer.py` mặc định chặn execution của prototype đã đóng. `--historical-replay` chỉ phục vụ tái tạo diagnostic; không mở lại hướng nghiên cứu hoặc cấp quyền install.
- Giữ nguyên core helpers và scripts audit khác như historical tools. Không gọi chúng là pipeline training V2 được khuyến nghị. Tái tạo exact implementation cũ cần checkout commit tương ứng, chẳng hạn `e7ba5df`.
- Không xóa hay thay metrics trong các báo cáo cũ. Notebook full APPLIANCE cũ không trở thành implementation hợp lệ của V2 nhờ quyết định này.
- Machine-readable status: [appliance_v2_research_status.json](../artifacts/appliance_v2_research_status.json).

## Hướng ưu tiên tiếp theo: DeNICE routing / availability

Mục tiêu trước mắt là một phương pháp DeNICE có protocol training/evaluation nhất quán, với khả năng dùng đúng kiến thức model đã có.

1. Phân biệt **local ownership**, **support nhận qua aggregation**, **class availability** và **functional competence**. Thiếu ownership/availability không tự chứng minh classifier chưa biết class.
2. Audit lifecycle availability từ training → aggregation/bootstrap → router registration → checkpoint/restore. Trước hết xác định entry nào sai hoặc thiếu provenance, thay vì tự mở toàn bộ classes.
3. Mọi sửa availability/routing phải có đối chứng cùng weights và cùng inference protocol; đo rescue/break và risk trên development hợp lệ. Registration gain không được gọi là donor learning gain.
4. Giữ frozen final-test protocol sau khi chọn thiết kế trên development. Không dùng các full-test errors đã quan sát để hardcode class guard hoặc chọn threshold.

Đây là hướng công việc tiếp theo, **chưa phải một experiment hoặc full run đã bắt đầu**. Class availability repair cũng có thể gây false activation: receiver 1/class 6 bị veto trên class 11 trong bounded campaign. Vì vậy không tự động mở mọi class và không chuyển thiếu risk evidence thành PASS.

## Điều kiện nếu muốn quay lại APPLIANCE

Cần một redesign thực chất và protocol nghiên cứu mới, nêu rõ cơ chế transfer, baseline Availability-only, evidence allocation, acceptance/risk scope, communication budget và điểm dừng. Không coi thêm donor, hạ gate hoặc đổi hyperparameter của V2 là redesign. Chỉ triển khai sau một quyết định nghiên cứu mới.
