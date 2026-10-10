# APPLIANCE — Evidence Feasibility Audit

## Kết luận

**Không thể chứng nhận đủ broad protected scope cho 23 candidate bằng clean CAL allocation hiện có và quy tắc audit ≥32/owner–class.** Cả 408 missing pools đều có CAL HOLDOUT row capacity <32; **133 pool có tổng toàn bộ original train roles <32 rows**. Không thể giải quyết bằng cap, batch size hoặc capture summary nhiều lần trên cùng samples.

Đây không phải kết luận mọi head transfer thất bại. Nó xác định giới hạn dữ liệu và thời gian của **qualification đang dùng**. Production yêu cầu receiver CAL HOLDOUT aggregate ≥32, khác với quy tắc conservative audit per-owner/class ≥32; audit này **không đổi** một trong hai gate và không áp requirement 3.000 mẫu vào production.

## Phạm vi và nguồn

23 candidate của actual-transfer development audit `fdaf47a`; checkpoint Task5/round19 SHA `3003593d64e3d63a81200b90d1a6e7487fff6c0875e3518cd58c2ccad22fe391`. Chỉ đọc artifact, role/store manifests, graph history và checkpoint state. **Không mở NPZ hoặc raw CAL/FIT/BASE/VAL/test; không forward, classifier fit, install, smoke hoặc sửa production.** Counts là số row tối đa từ metadata, chưa kiểm chứng unique/independent content.

## Ma trận evidence

| Kiểm tra 408 pool | Số |
|---|---:|
| CAL HOLDOUT row capacity ≥32 | 0 |
| Toàn bộ CAL rows ≥32 (không phải reserved HOLDOUT) | 2 |
| Tổng original train BASE+FIT+CAL+VAL <32 | 133 |
| VAL row capacity ≥32 | 123 |

VAL có thể là nguồn qualification/development khi có authority/content split hợp lệ; ở đây chưa đọc, chưa kiểm chứng unseen/unique và **không phải CAL acceptance**. BASE/provenance chỉ giúp bảo vệ/kiểm kê, không bù số mẫu CAL. Ma trận đầy đủ gồm owner, class, task, role counts, CAL FIT/SEL/H counts, partition version, nguồn summary và các candidate liên quan trong artifact JSON.

408 pool được dùng ở **2290 candidate–pool incidences**. Theo task của negative so với birth task của target:

- Pre-birth: 211.
- Cùng task: 229.
- Sau birth: 1850.

Không cộng các incidence trùng owner–class thành số mẫu độc lập.

## Production CAL authorization

Target của cả 23 candidate thuộc T1/T2/T4; current T5 chỉ có classes 30–33. **0/23 có current positive CAL cho target**, và **0/23 có certificate đúng candidate persist trong checkpoint**. Không thể cấp phép cài ở T5 bằng cách mở lại CAL cũ, đổi tên FIT thành CAL hoặc tái sử dụng native donor quality.

Metadata cho thấy **20/23** cặp có cửa sổ prospective ở đúng birth task: donor FIT≥32/SEL≥8/H≥32, receiver current SEL/H≥32, target absent và live positive-alpha edge. Đây chỉ là **khả năng về count/graph**. Maturity, head compatibility, routing, FAR/break, BASE coverage và phiên bản certificate ở birth chưa được kiểm tra; Task5 weights/guard không được backdate vào T1/T2/T4. Ba cặp bị loại ngay theo metadata: `63 ← 71 / 13` thiếu edge ở birth; `16 ← 44 / 24` có receiver negative SEL/H = 26/27; `84 ← 4 / 6` có H = 32 nhưng SEL chỉ 30.

## Evidence có thể giữ mà không lưu raw CAL

1. **Generic protection summary**: tạo khi owner/class còn current; lưu role/partition/coordinate hashes, sketch version, count và moments/enclosure. Giúp bảo vệ/ranking, không tự trở thành CAL pass cho guard mới.
2. **Exact guard receipt**: chỉ khi patch/guard đã tồn tại, đã khóa và endpoint CAL vẫn current. Lưu per-class rows/activated/break cùng patch, function/declaration, precision/device và provenance. Scope chỉ mở theo evidence thực sự đã đo; receipt không dùng lại nếu head, route, threshold, dependency hoặc shield khác phiên bản. Code `current_scope_evidence.py` / `cumulative_certificate.py` đã có binding cho installed guards.
3. **Positive acceptance**: cần đúng receiver function trên donor current target CAL HOLDOUT. Certificate của donor-native hoặc old moments không chứng minh positive recall của receiver mới.

**Pre-birth gap**: guard của class mới chưa tồn tại lúc các class cũ còn current. Không thể tạo receipt cho một function tương lai. Summary cũ chỉ là protection evidence; muốn derive một bound/receipt mới từ summary phải có một protocol kiểm chứng riêng, hiện chưa được authorize/implement. Cumulative receipt chỉ giải quyết negative classes đến **sau** install, không tự giải quyết các class trước install.

## Shortlist cho prospective smoke

Chọn theo metadata: ít missing pools, ít pre-birth gaps, tie theo ID; không dùng evaluation accuracy để xếp hạng. Hai cặp dưới đây là **đề xuất có điều kiện**, chưa được chạy:

| Receiver ← donor / class | Task phải compile | Donor CAL F/S/H | Receiver current negative H | Missing pools | Status |
|---|---|---:|---:|---:|---|
| 2 ← 62 / 6 | T1 | 1176/588/589 | 565 | 17 | DEFERRED |
| 2 ← 64 / 8 | T1 | 245/122/123 | 565 | 17 | DEFERRED |

Hai cặp cùng receiver 2 phải chạy **hai clone/experiments tách biệt**, không cài đồng thời: installer hiện chưa chứng minh multi-capability acceptance. Compile guard mới bằng snapshot và CAL thực sự current của birth task; không chuyển Task5 head/threshold trở ngược. Chỉ khởi chạy khi có legitimate positive acceptance và protected scope được xác định đủ. Khi scope chưa đủ, giữ DEFERRED/UNKNOWN, không coi pass current-only là safe cumulative deployment.

## Statistical FAR và empirical gate

Với 0 lỗi và n mẫu độc lập, upper bound một phía 95% là `1 - 0.05^(1/n)`. n=64 cho **4.573%**, không phải ≤0,1%. Cần **2995** mẫu âm độc lập để bound một pool ≤0,1%; chưa hiệu chỉnh nhiều pool/candidate. Metadata row counts và content dedup không tự chứng minh independence. Gate FAR empirical 0,1% hiện tại giữ nguyên, không được viết thành population guarantee.

## Mốc quyết định

- **Không train full hoặc chạy smoke ở T5 cho 23 class cũ này.** Chỉ bổ sung negative evidence không khôi phục positive CAL đã đóng cửa.
- Muốn thử tiếp bằng cơ chế hiện tại: chốt prospective install **khi target còn current**, coverage contract/scope rõ và evidence capture từ thời điểm thích hợp. Pre-birth protection vẫn phải giải quyết; không lấy summary thay CAL. Metadata-only shortlist chưa cho phép cài.
- Với dataset/roles và broad ≥32/owner–class rule giữ nguyên, mở thêm audit guard/classifier sẽ không khắc phục được 408 deficit. Cần nguồn CAL mới hợp lệ hoặc thiết kế evidence/transfer và phạm vi chứng nhận khác được nêu rõ; không âm thầm đổi allocation/gate, pooling owner hay cho missing=PASS.
- Sau khi thực sự có đủ legitimate evidence, nếu acceptance vẫn FAIL thì reject cặp/design; không kéo dài tuning guard. Hiện chưa đo được bước đó, nên chưa kết luận head-patch toàn bộ thất bại.

Artifact: `artifacts/appliance_evidence_feasibility_development_20261010.json`. Script: `tools/audit_appliance_evidence_feasibility.py`.
