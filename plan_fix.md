
```markdown
# ProtoTail — Kiểm toán Long-tail: nguyên nhân, confidence, code diff và impact

**Bản hợp nhất duy nhất | Tiếng Việt | 08/11/2026**

Đối chiếu paper ProtoTail, nguồn BaCon, branch `fix/prototail-tailaware-audit` (SHA `0a3b36952a9caaecd1123211045458f87f4c3fcb`) và các training logs. Phân biệt **mức chắc chắn của bằng chứng code/paper mismatch** với **mức chắc chắn lỗi gây giảm accuracy**. Các diff là *đề xuất*, chưa được áp dụng hoặc kiểm chứng trên GPU.

**Điều hướng:** Phần I — Root cause & confidence · Phần II — Code diff & impact

---

## Phần I — Đối chiếu paper, code và logs

> **Bản báo cáo ưu tiên lỗi có bằng chứng cao; KHÔNG yêu cầu chạy thêm ma trận ablation.**  
> **Ngày kiểm tra:** 08/11/2026. **Repository:** `vuhungtvt2018/prototail`.  
> **Code audit:** branch `fix/prototail-tailaware-audit`, SHA `0a3b36952a9caaecd1123211045458f87f4c3fcb`. So với `tam` tại `0ae0001487cbcb9ee9f1d32516e3f6f25468a305`.  
> **Nguồn:** paper *ProtoTail* 17 trang (mục 3.3–3.5, 4.1–4.2; Tables 1/2/4; Appendix B), `log-prototail.zip` gồm 24 runs cũ, 2 log v3 CUB/Cars LT20. Code đối chiếu thêm: BaCon chính thức `model/dist_est.py`.  
> **Bản hợp nhất:** Phần II ngay bên dưới gồm toàn bộ code diff, integration contract và tiêu chí kiểm tra. Các diff chưa áp dụng hoặc được kiểm chứng bằng GPU training.

## 0. Kết luận điều tra — đọc trước

**Không đúng nếu kết luận tất cả run Long-tail xấu do Tail-aware.** Dữ liệu cũ cho thấy ở **CUB LT20** Tail-aware đóng góp **+1.36 pp** vào Full, ở **Cars LT20** **+0.82 pp**, nhưng ở **Cars LT50** **−5.83 pp**. Trường hợp khó giải thích nhất về *đóng góp riêng của capacity regularization* chính là **Cars LT50**. Cars LT20 có thêm lỗi lớn hơn: đưa **Prototype** vào làm Base **55.70% → 44.21%** trước khi thêm Tail-aware. Hiện tượng này không được sửa chỉ bằng chuyển KMeans sang GPU.

**Ba kết luận có bằng chứng mạnh nhất:**

1. **[100% code–paper mismatch]** `Full` hiện tại **không implement dual-branch, persistent pseudo-label selection** như Eq. (9)–(11), Algorithm 1, Appendix B. Nó sử dụng CE tạm thời theo minibatch và threshold; paper nói **không có pseudo loss riêng**.
2. **[100% quan sát; ~90% cơ chế gradient]** Gate **không đạt adaptive capacity target**. Sau 200 epochs, tổng gate mỗi class chỉ **~0.82**, trong khi mục tiêu lý thuyết **[1, 3]**; tất cả giá trị gate dưới 0.5. Khác biệt giữa estimated frequent/tail classes khó tác động đến part budget như paper tuyên bố.
3. **[100% đối chiếu source]** Feedback cũ về **rank matching known frequencies** cần **đính chính**: BaCon gốc thật sự dùng rank matching theo known labeled support; branch mới đổi sang direct Hungarian class→cluster count. Branch này *phù hợp cách diễn đạt literal* trong Appendix B của paper nhưng **khác estimator BaCon gốc mà paper tuyên bố kế thừa**. Không thể khẳng định một bên đúng cho reported runs khi chưa có original training commit.

Các vấn đề đáng chú ý khác: **novel class frequency không gắn chắc với output neuron**; **primary KMeans evaluation ghép CLS chưa L2-normalize với pooled parts đã normalize**; `weight_decay` trong YAML là `1e-4` thay vì **`5e-5`** ghi trong paper; và backend FAISS không đồng nhất với sklearn về nghiệm KMeans.

---

## 1. Trước khi gọi là reproduction failure: đối chiếu đúng protocol

### 1.1. Paper đã báo cáo cái gì?

| Nguồn paper | Setup | CUB Base → Full | Cars Base → Full | Đóng góp Full − Base |
|---|---|---:|---:|---|
| **Table 1 / Table 2, trang 8** | **BaCon, LT ratio = 10** | **73.72 → 75.42%** | **66.86 → 68.29%** | +1.70 / +1.43 pp |
| **Appendix A.1 / Table 4, trang 13** | **SimGCD, LT ratio = 10** | **60.55 → 68.06%** | **70.79 → 73.80%** | +7.51 / +3.01 pp |
| **Log cũ (24 runs)** | **SimGCD, LT ratio = 20** | **62.31 → 68.12%** | **55.70 → 49.84%** | +5.82 / **−5.86** pp |
| **Log v3** | **SimGCD, LT ratio = 20** | — → **64.93%** | — → **49.11%** | Không có Base trên cùng commit v3 |

**Paper còn có Table 5 (Appendix A.4, CIFAR100-LT, BaCon)**: Base **77.50%**, `Full w/o Tail 82.00%`, `Full +Tail 83.80%` (**+1.80 pp** riêng Tail); riêng **U-Few 60.20 → 82.20% (+22.00 pp)**. Đây là claim Tail-aware rất mạnh trong paper nhưng **không có CIFAR100-LT run hoặc checkpoint đối ứng** trong 24 logs / hai log v3 để kiểm chứng. Không được mặc định các gains này tái hiện trên CUB/Cars LT20/LT50.

**Lưu ý bắt buộc:** BaCon-LT10 và SimGCD-LT20 **không phải thí nghiệm đồng nhất**. Ngay cả Table 4 SimGCD-LT10 cũng khác LT20. Không được lấy `68.29 − 49.11 = 19.18 pp` hay `73.80 − 49.11 = 24.69 pp` rồi quy toàn bộ cho implementation error. Số đó gồm khác biệt baseline, ratio, data count, estimator và code version.

### 1.2. Cách tách riêng ảnh hưởng của Tail-aware từ log cũ

Số liệu là **test All tại epoch 200, seed 0**, đơn vị **điểm phần trăm**.

| Dataset | Proto không Tail → +Tail | Full không Tail → +Tail | Full so với Base | Nhận định |
|---|---:|---:|---:|---|
| CUB LT20 | **+4.04** | **+1.36** | +5.82 | Tail có lợi trong log này |
| Cars LT20 | **+4.73** | **+0.82** | **−5.86** | Proto/Pseudo combination làm Full kém Base; **không thể đổ riêng cho Tail** |
| CUB LT50 | +0.78 | −0.10 | +2.21 | Tail gần trung tính |
| **Cars LT50** | **−1.57** | **−5.83** | −4.04 | **Tail-aware failure rõ nhất** |

Đối chiếu paper Table 2 (BaCon LT10): CUB `Full_wo_Tail 73.20 → Full 75.42` (**+2.22 pp**), Cars `67.40 → 68.29` (**+0.89 pp**). Paper còn report K-Few và U-Few tăng mạnh, đặc biệt Cars U-Few **52.00 → 61.36**. Điều đó **không tự tái hiện** bằng code SimGCD hiện tại tại LT20/LT50.

### 1.3. Hai log v3 có gì đáng tin để chẩn đoán?

| Log v3 | Final All | Best test All (không dùng chọn model hợp lệ) | Gate mean đầu/cuối | κ thực tế cuối | Lcap cuối | Parametric Novel Global → Fused |
|---|---:|---:|---:|---:|---:|---:|
| CUB LT20 | **64.93%** | 66.12% (epoch 160) | 0.269 → **0.273** | **~0.82** | **2.003** | **42.87 → 34.51** |
| Cars LT20 | **49.11%** | 51.13% (epoch 115) | 0.269 → **0.275** | **~0.82** | **2.349** | **40.90 → 34.98** |

Cả hai chạy đủ 200 epochs, KMeans FAISS GPU hoạt động, không thấy NaN/fatal error. `best.pt` đang được chọn theo test All: **đó là vấn đề reporting protocol riêng, không giải thích vì sao final accuracy thấp**. Log không đính kèm `*.tail_debug.jsonl` per-class hoặc model checkpoint; không thể xác minh class-level estimated frequencies, slot attention, hay trùng SHA tuyệt đối giữa Kaggle bundle và branch đang xem.

---

## 2. Danh sách lỗi ưu tiên: confidence được hiểu thế nào?

- **`Confidence tồn tại`**: mức chắc chắn (0–100%) rằng khác biệt/triệu chứng **thật sự hiện diện** trong source và logs; các giá trị **không phải posterior xác suất thống kê**.
- **`Confidence ảnh hưởng accuracy`**: đánh giá kỹ thuật có điều kiện về khả năng nó **góp phần** vào kết quả thấp, **không** phải phần trăm accuracy sẽ hồi phục hay dự báo speedup.
- Phân biệt **BUG/SAI PAPER**, **METHOD GAP**, **NUMERICAL/REPORT MISMATCH** và **DESIGN LIMIT**. Thấy sai mô tả không đồng nghĩa chứng minh nguyên nhân duy nhất.

| ID | Mức | Vấn đề | Confidence tồn tại | Confidence góp phần | Impact dự kiến nếu sửa |
|---|---|---|---:|---:|---|
| **LT-01** | **P0** | D1/D2 persistent pseudo-selection của paper **chưa có**, thay bằng minibatch CE | **100%** | **90%** đối với Full | **Rất lớn / chưa định lượng**; có thể đảo hướng Known/Novel |
| **LT-02** | **P0** | Adaptive gates không tiến gần capacity target | **100%** | **90%** đối với đóng góp Tail | **Rất lớn về cơ chế**, không bảo đảm All tăng |
| **LT-03** | **P0** | Frequency estimator mới **khác BaCon gốc** ở known rank matching | **100%** | **70%** đối với khác biệt v3 / target Tail | **Trung–lớn**, có thể tăng **hoặc giảm** accuracy |
| **LT-04** | **P1** | Novel KMeans cluster-size không bảo đảm ánh xạ vào đúng novel output ID | **100%** structural | **80%** với severe LT | **Lớn về target correctness**; cần thiết kế mapping mới (khác BaCon) |
| **LT-05** | **P1** | Đánh giá CLS/part không cân bằng norm | **99%** về phép ghép | **75%** đối với metric chính | **Trung–lớn trên reported All**, chưa biết dấu |
| **LT-06** | **P1** | Paper và hai log **khác baseline / ratio** | **100%** | **100%** đối với *không thể so trực tiếp* | Không tăng accuracy; ngăn kết luận sai |
| **LT-07** | **P2** | `weight_decay` YAML sai với paper | **100%** | **20%** là nguyên nhân chính | Nhỏ–trung, cần chỉnh cho reproduction |
| **LT-08** | **P2** | FAISS GPU KMeans không tương đương sklearn KMeans của log cũ | **100%** về backend | **55%** góp phần v3-vs-v2 | Ảnh hưởng cả metric/ước lượng frequency, không biết dấu |

## 3. Phân tích từng lỗi và ý nghĩa đối với Tail-aware

### LT-01 — Full hiện tại **không phải Full của paper**

**Paper:** §3.4, Eq. (9)–(11), Algorithm 1, Appendix B: **known branch** đánh giá per-class accuracy trên **supervised pool hiện tại**, mỗi round chỉ chọn một class có độ tin cậy đã xác nhận, giữ top 80% tối đa 500 mẫu, tối đa 20 rounds. **Novel branch** chạy hai KMeans seeds, Jaccard ≥0.6, Silhouette ≥median, ≥70% agreement một novel output, cluster size ≥10, DBI circuit breaker, no-remap; chọn tối đa 200/cluster, tối đa sáu rounds trong epochs 50–100. Sample sau khi chọn phải **được lưu và ở lại supervised pool qua các epoch**. §3.5 nói rõ: **pseudo-labeled samples không tạo separate loss**.

**Code:** `src/prototail_gcd/baselines/simgcd/adapter.py`, `compute_simgcd_losses()`: chỉ tạo pseudo nhãn theo `softmax confidence >= 0.7` trên **hai views của minibatch hiện tại**, tính **CE view0→view1 + view1→view0** rồi `total_loss += pseudo_weight * pseudo_loss`. `TrainingEngine` và dataloader không có persistent promoted `sample_id→label` map. Cờ `D1+D2 ACTIVE` chỉ chứng minh cả known/novel predictions có thể qua threshold, **không chứng minh D1/D2 paper được implement**.

**Log độc lập:** Cars xuất hiện `n_pseudo_novel > 0` từ **epoch 19**, CUB từ **epoch 25**, **trước novel selection bắt đầu epoch 50 của paper**; pseudo CE vẫn chạy và được log đến epoch 200. Đó là bằng chứng method/schedule khác hẳn.

**Vì sao gây hụt Long-tail:** Paper chủ trương *chỉ giữ ứng viên khó nhưng đáng tin* để không khuếch đại lỗi ở tail. CE minibatch không có cluster stability/silhouette/no-remap nên nhận các pseudo nhãn sai không qua paper gates. Novel prototypes có thể cập nhật bằng provisional labels, nhưng thiếu promotion khiến class supervision không có nguồn nhãn đáng tin tích lũy. **Confidence code mismatch 100%; chưa có counterfactual để chứng minh mức giảm cụ thể.**

**File phải sửa:** `data/{cub,cars,aircraft,manifest_gcd}.py` và/hoặc wrapper train sample, `training/engine.py`, `baselines/simgcd/adapter.py`, `train.py`; thêm `pseudo/selection.py`, `pseudo/promotion_store.py`; BaCon cần tích hợp tương ứng. **Ưu tiên khôi phục đúng Full của paper trước khi chạy lại Full.** Không có một-line patch nào khôi phục được toàn bộ selection algorithm: xem diff/integration contract ở Phần II.

### LT-02 — Tail-aware đang chỉ là một regularizer không đạt mục tiêu

**Paper:** Eq. (6)–(8), `M=3`: desired capacity `κ*_c = 1+2ρ_c`, luôn trong `[1,3]`; dự kiến lớp support lớn có gate mass lớn hơn tail. `Lcap = mean_c (κ_c−κ*_c)^2` với `κ_c = Σ_m sigmoid(u_cm)`.

**Code/log:** gate logits khởi tạo **−1**, `sigmoid(−1)=0.26894`, nên `κ≈0.807`. Sau 200 epochs vẫn chỉ `κ≈0.82`, **thấp hơn minimum target 1**; 0% gates vượt 0.5. `Lcap` vẫn ~2 ở CUB/Cars. Tức mô hình **không hiện thực được phân bổ capacity theo support như giải thích trong paper** trên các run này.

**Cơ chế gradient (đã đối chiếu công thức, không chỉ phỏng đoán):** với `C≈200`, `M=3`, `λcap=λg=λb=0.05`, lúc khởi tạo `a≈0.269, κ≈0.807`, đạo hàm phần capacity trên *mỗi gate logit* xấp xỉ

`∂(λcap Lcap)/∂u_cm = 2λcap/C a(1−a)`.

Với **κ*=3** ở class head, độ lớn ≈ **2.2×10⁻⁴**; SGD learning rate **0.01**, bước cập nhật direct khoảng **2.2×10⁻⁶** trước momentum, trong khi cosine giảm LR về sau. Vì vậy kết quả gate gần như không dịch chuyển **phù hợp định lượng** với gradient quá nhỏ. `Lgate` theo Eq. (5) cũng yếu; riêng thành phần min-bound `λb=.05` **không loại bỏ local all-zero minimum của riêng Lgate** khi `M=3`, bởi gradient gần zero hướng về zero nếu `λb < 1/M` (lưu ý **tổng objective có Lcap** nên không thể kết luận total loss có local minimum ở zero).

**Impact:** cao về **tính hợp lệ của cơ chế Tail-aware**: nếu κ của mọi class ~0.82 thì target head/medium/tail khác nhau hầu như không được thực thi. Điều này giải thích vì sao các gain K-Few/U-Few lớn trong paper khó lặp lại bằng implementation này. **Không thể từ đó suy ra chỉnh LR sẽ tự động tăng All accuracy**.

**Sửa:** trước tiên giữ nguyên paper-default và công khai cảnh báo `capacity_mae`/`κ_target vs κ_actual`; nếu mục tiêu là *làm mechanism hoạt động* (thay vì faithful reproduction), thêm **opt-in** `gate_warm_start_from_estimated_support`, đưa κ ban đầu gần target dựa trên **estimated frequencies, tuyệt đối không phải novel GT**. Đây là **method deviation phải ghi trong paper**, không được sửa âm thầm. Diff ở Phần II.

### LT-03 — Đính chính feedback cũ: new frequency fix **không faithful với BaCon gốc**

**Paper Appendix B:** nói estimator “follows BaCon” và “known categories receive sizes of their aligned clusters”. Hai mệnh đề này **không hoàn toàn mô tả cùng một implementation**, vì:

- BaCon `model/dist_est.py:70–78` thực sự lấy `known_cluster_ins_num`, **sort theo kích thước**, rồi gán theo **rank labeled-class support** (`idx1=argsort(known_labeled_ins_num); idx2=argsort(known_cluster_ins_num)`): **known rank-matching**.
- `tam` ban đầu **có rank matching tương tự BaCon**. Feedback trước từng gọi đây là bug và đề nghị thay bằng direct Hungarian mapping.
- Branch `fix/prototail-tailaware-audit` đã đổi sang `estimate[class_id] = cluster_counts[cluster_for_class[class_id]]`: **direct alignment**.

**Đính chính bắt buộc:** Không thể gọi bản sửa mới “đúng hơn BaCon” như feedback cũ. Nó đúng với *câu literal Appendix B* nhưng **khác source BaCon chính thức**. Một phần sai khác v3-vs-v2 có thể tới từ thay đổi estimator này (cùng lúc FAISS thay sklearn, optimizer/EMA cũng đổi). Tuy nhiên, **log v3 không đủ chứng minh đổi estimator gây giảm −3.19 pp CUB hay −0.73 pp Cars**.

**Vì sao quan trọng cho Long-tail:** cách Hungarian dùng **cluster purity** (`contingency / cluster_count`) có thể chọn cluster nhỏ nhưng purity cao; direct mapping biến class head thành `κ*` kiểu tail nếu head được match vào cluster nhỏ. Rank matching giữ ít nhất **thứ tự support đã biết trong labeled known data**, như logic BaCon. Chuyển giữa hai cách khi distribution skew mạnh có thể đảo capacity targets.

**Sửa có điều kiện:** nếu mục tiêu **reproduce estimator BaCon mà reported setup nói dùng**, khôi phục `known_support_rank_match` như BaCon và ghi explicit `frequency_mapping_mode` trong config. Nếu mục tiêu **faithful literal Appendix B**, giữ `direct_hungarian` và phải giải thích vì sao khác BaCon source. Không đổi ngầm. Diff ở Phần II.

### LT-04 — Novel frequency target chưa định danh được theo class logits

Code: `estimate[num_known_classes:] = sort(cluster_counts[remaining_clusters], descending=True)`. Đây là **cách làm từ BaCon reference** nhưng **không có phép nối** cluster novel thứ `j` với novel classifier output `num_known+j`. KMeans IDs vô danh và một permutation có thể thay đổi mà không làm thay đổi clustering quality; trong khi `κ*_c` dùng **theo class index** khi cập nhật gate.

**Chắc chắn về vấn đề ánh xạ, không chắc về magnitude:** với severe LT, rank sorting tự gán lớp novel support cao nhất vào neuron novel đầu tiên, dù neuron ấy có thể đang đại diện một tail category. Vậy Tail-aware đặt budget không dựa trên *support của đúng class* ở novel part. Không nên gọi đây là một “implementation bug so với BaCon” vì bản gốc cũng như vậy; đây là **giới hạn structural của estimator khi dùng làm per-class capacity**.

**Sửa:** tạo correspondence **không cần novel GT** giữa KMeans cluster và *model's current novel output dimensions*, dựa vào overlap của assignments với model predictions trên **unlabeled training data**; cần check độ tin cậy ánh xạ và khả năng collapse trước khi áp dụng. **Đây là algorithm change**, không phải quick one-line fix. Có thể giữ nguyên BaCon mapping khi cần reproduce và phát ra cảnh báo/diagnostic minh bạch. Diff ở Phần II.

### LT-05 — Primary KMeans có khả năng gần như không nhìn thấy part branch

Trong `methods/prototail.py`: `pooled_parts` là weighted average của **các parts đã L2-normalize**, nên `||pooled_parts||₂≤1`. Nhưng `global_features` lấy từ CLS **sau backbone LayerNorm**, chưa L2-normalize trước concat. Với DINOv2 768d, CLS norm thường cỡ `sqrt(768)≈27.7` (đây là **ước lượng**, chưa đo checkpoint), làm tỷ lệ squared-energy part ≤ khoảng `1/(768+1)≈0.13%`. `F.normalize(cat(global,part))` **không sửa mất cân bằng tương đối đó**.

**Liên hệ reported results:** paper mô tả cải thiện do local parts, nhưng code primary KMeans features có khả năng gần như global-only. Chưa có quy định rõ trong paper về feature concatenation/normalization của evaluation, nên không thể khẳng định paper sử dụng “normalize từng branch”; **đây là mismatch nghi ngờ giữa claim và evaluation implementation**, không phải lỗi đã chứng minh so với một dòng cụ thể trong paper.

**Sửa metric có thể làm trên checkpoint hiện có, không retrain:** chuẩn hóa riêng `global_features` và `pooled_parts` **trước** concat, ghi tên metric mới `balanced_concat` để không ghi đè kết quả gốc. Nếu re-report primary metric mới thì phải mô tả protocol đổi. Diff ở Phần II.

### LT-06 — Protocol paper và v3 khác nhau: không phải cùng target benchmark

**Paper:** Tables 1–2 **BaCon LT10**; Appendix A.1 Table 4 **SimGCD LT10**, 200 epochs, batch 128, SGD, pretrained DINOv2 ViT-B/14, weight decay 5e-5. **Log v3:** `--baseline simgcd --imb-ratio 20 --kmeans-backend faiss_gpu --eval-interval 5`. Không thể yêu cầu v3 “khớp trực tiếp 75.42 / 68.29”, vì đó là **BaCon LT10**.

**Sửa:** ghi explicit metadata `baseline`, `imb_ratio`, `split_id`, KMeans backend, optimizer hyperparams, SHA. Khi so bảng, chỉ cho phép đối sánh cùng protocol. Diff ở Phần II.

### LT-07 — Hyperparameter `weight_decay` sai theo paper

**Paper §4.1:** `weight_decay=5×10⁻⁵`. **Branch mới**: cả `configs/experiment_simgcd.yaml` lẫn `configs/experiment_bacon.yaml` ghi `weight_decay: 0.0001` (**gấp 2**). Đây là mismatch chắc chắn, dễ sửa, nhưng **không có cơ sở cho rằng một thay đổi này sẽ phục hồi hơn 10–20 pp**. Diff ở Phần II.

### LT-08 — KMeans backend và cách select checkpoint cũng khác reproduction path

**v3:** frequency KMeans + evaluation KMeans dùng **FAISS GPU** (`nredo=10`, `niter=300`); old pipeline dùng `sklearn.KMeans(n_init=10, max_iter=300)`. Cả hai đều chạy Euclidean KMeans, nhưng initialization, convergence và nghiệm cụ thể **không có exact parity**. Thêm vào đó `best.pt` hiện vẫn chọn bởi `test All`, không phải validation độc lập. **KMeans khác backend** có thể tác động `estimated_frequencies` rồi `κ*` trong training; **test-based checkpoint** làm lựa chọn checkpoint lạc quan chứ **không giải thích scores thấp**.

**Sửa:** báo cáo solver/backend trong result; nếu mục tiêu reproduction, chuẩn bị một **offline comparison trên fixed embeddings** hoặc dùng đúng backend của reported implementation. Đừng đổi tiêu chí test checkpoint ngầm khi so log cũ. Diff ở Phần II.

---

## 4. Các lỗi đã sửa trên branch mới: KHÔNG gán nhầm làm root cause hiện tại

| Lỗi từng tồn tại ở `tam` | Trạng thái tại `0a3b369` | Nhận xét |
|---|---|---|
| Known EMA và novel EMA đều dừng ở epoch 60 | **Đã sửa**: known tiếp tục, novel dừng sau 60 | Hợp hơn với Appendix B |
| Một parameter có thể nằm trong 2 optimizers (shared backbone) | **Đã sửa**: optimizer groups disjoint, engine check overlap | Không còn là ứng viên root cause trong v3 |
| KMeans exception fallback sang `class_frequencies` có novel GT | **Đã sửa**: fail-fast, bỏ fallback | Không thấy evidence oracle leakage ở v3 |
| Chia Many/Med/Few không ổn định theo support | **Đã sửa**: sort theo counts và class tie-break | Tốt hơn cho reporting; không quyết định learning |
| KMeans chỉ chạy CPU | **Đã sửa**: FAISS GPU, log profile seconds | Giải quyết performance; **không bảo đảm numerical equivalence** |
| Known frequency rank matching (đã gọi bug trước đây) | **Đã thay** direct mapping | **Cần phân loại lại** do BaCon gốc dùng rank matching |

Chưa có CI check-runs ở commit mới theo GitHub khi kiểm tra. Các unit tests trong source cho thấy có ý định cover thay đổi; **không đồng nghĩa toàn bộ integration và reproduction đã pass**.

## 5. Nếu không có thời gian ablation: ưu tiên hành động thực tế

1. **Chặn sự nhầm lẫn Full (LT-01):** Không dùng kết quả `Full` hiện tại để kết luận paper dual-branch đúng/sai. Code phải implement persistence + selection rounds thật sự nếu muốn reproduce Table 1/2/4. Đây là sửa lớn nhưng là điều kiện *bắt buộc*, không phải optional hyperparameter tuning.
2. **Khóa estimator contract (LT-03/LT-04):** Chọn rõ `bacon_rank` **hoặc** `direct_hungarian` theo source đã tạo số liệu paper. **Không nói direct mapping luôn là bug fix.** Novel mapping chưa có semantic correspondence phải ghi hạn chế.
3. **Đánh giá mechanism gate (LT-02):** Không dùng `capacity_regularization_weight>0` làm bằng chứng mechanism hoạt động. Từ log hiện tại, κ không học. Sửa gradient/init là thay đổi thuật toán hoặc training recipe; nếu quyết định sửa, công bố khác biệt với setup reported.
4. **Khóa reproduction hyperparams (LT-06/LT-07):** Không so BaCon LT10 với SimGCD LT20; sửa weight decay 5e-5 nếu đang target paper.
5. **Sửa representation evaluation một cách kiểm soát (LT-05):** Với checkpoint đã train, metric bổ sung normalize per-branch chỉ là postprocessing, không cần retrain; giữ lại metric cũ để so, không đổi định nghĩa All âm thầm.

### Phân định độ chắc chắn cuối cùng

- **Chắc chắn 100%:** implementation `Full` khác paper; gates thực tế không đạt `[1,3]`; paper và logs khác protocol; source BaCon dùng rank matching; YAML weight decay khác paper; FAISS khác sklearn.
- **Rất có cơ sở nhưng chưa có chứng cứ counterfactual:** cụ thể lỗi nào tạo ra số pp bị mất. Nhiều lỗi cùng tồn tại, không có checkpoint gốc/reporting script hoặc matched-protocol rerun nên **không thể trung thực hứa “sửa X sẽ đạt 75.42/68.29”**.
- **Không nên làm:** tăng ngẫu nhiên `λcap`, sửa `M`, chuyển sang Adam hoặc normalize features rồi nói đó là *faithful reproduction*. Nếu làm, phải ghi rõ **method revision**, không trộn vào kết quả Table 1/2/4 cũ.

---

## 6. Nguồn tái kiểm tra nhanh

- Paper: *ProtoTail*, §3.3 Eq. (4)–(8) (trang 4–5); §3.4 Eq. (9)–(11) (trang 5–6); §3.5 và Algorithm 1 (trang 6–7); Tables 1/2 (trang 8), Table 4 (trang 13), Appendix B (trang 16–17).
- Commit đã audit: https://github.com/vuhungtvt2018/prototail/commit/0a3b36952a9caaecd1123211045458f87f4c3fcb
- Frequency estimator: https://github.com/vuhungtvt2018/prototail/blob/0a3b36952a9caaecd1123211045458f87f4c3fcb/src/prototail_gcd/baselines/bacon/losses.py
- SimGCD pseudo: https://github.com/vuhungtvt2018/prototail/blob/0a3b36952a9caaecd1123211045458f87f4c3fcb/src/prototail_gcd/baselines/simgcd/adapter.py
- Gate / prototype / KMeans features: https://github.com/vuhungtvt2018/prototail/blob/0a3b36952a9caaecd1123211045458f87f4c3fcb/src/prototail_gcd/methods/prototail.py
- Source BaCon gốc: https://github.com/JianhongBai/BaCon/blob/main/model/dist_est.py#L61-L78
- Các log: `log-prototail.zip`, `tonytonychopperr_...cub...v3_full_log.txt`, `zoroswordsman_...cars...v3_full_log.txt` (đính kèm conversation).

---

## Phần II — Đề xuất sửa code và tiêu chí nghiệm thu

> **Code target**: `vuhungtvt2018/prototail`, branch `fix/prototail-tailaware-audit`, SHA `0a3b36952a9caaecd1123211045458f87f4c3fcb` (đọc ngày 08/11/2026).  
> **Báo cáo nguyên nhân và confidence:** đã tích hợp đầy đủ ở Phần I trong cùng file.  
> **Mục tiêu**: hướng dẫn teammate sửa những điểm **đã xác minh từ paper, BaCon gốc và logs**, không yêu cầu chạy lại 24–30 ablations.  
> **Quan trọng**: Một số đoạn dưới đây là **patch hunk đủ cụ thể để triển khai**, một số là **integration contract** (thuật toán selection có nhiều bước nên không thể giả vờ vài dòng diff là faithful implementation). Mọi đề xuất đều **chưa apply/test trên repo**, không được báo là code đã sửa hay bảo đảm accuracy tăng. **Không gộp các hạng mục vào một PR nếu mục tiêu còn là tìm root cause.**

## A. Bảng quyết định nhanh: sửa gì trước?

| ID | Confidence nguồn | Phải làm để reproduce reported paper? | Scope | Impact/điểm cần lưu ý |
|---|---:|---|---|---|
| **LT-01** Dual selection | **100%** paper/code lệch | **CÓ**; Full hiện không phải Full paper | Logic pseudo manager + dataloader + engine | Potential impact lớn; không được đoán pp |
| **LT-02** Gate κ không đạt target | **100%** log, **90%** gradient | **PHẢI đánh giá/giải thích**; warm-start là method revision | Prototype, before_epoch, config | Tăng matching κ, không chắc All acc |
| **LT-03** BaCon rank vs direct | **100%** ref/code khác | **CHỐT theo source tạo số liệu paper** | `bacon/losses.py`, adapters | Có thể đảo support ordering ở known |
| **LT-04** Novel support map | **100%** unidentifiable | Không sửa âm thầm: BaCon cũng rank-sort | estimator/audit | High impact logic nhưng patch phải đổi estimator |
| **LT-05** CLS/part scale | **99%** code | Tùy reported eval pipeline | `methods/prototail.py` | Chỉ sửa evaluation; metric có thể lên/xuống |
| **LT-06** Protocol mismatch | **100%** | **CÓ** | scripts guard | Chống so bảng sai, không nâng accuracy |
| **LT-07** Weight decay | **100%** | **CÓ** | hai YAML | Một dòng/config; không giải thích khoảng cách lớn |
| **LT-08** FAISS vs sklearn | **100%** backend khác | Ghi rõ backend/report | CLI & experiment metadata | Không phải bitwise-equivalent |

---

<a id="patch-lt-01"></a>

## LT-01 — Dual-branch persistent selection

### Sai ở đâu?

- **Paper**: §3.4 Eq. (9)–(11), Algorithm 1, Appendix B; `D_l_tilde` **mở rộng vĩnh viễn**, D1 và D2 chọn mẫu theo những reliability criteria khác nhau. §3.5: **không có pseudo loss riêng**.
- **Code mới**: `src/prototail_gcd/baselines/simgcd/adapter.py` (`compute_simgcd_losses`, khoảng lines 350–445): mỗi minibatch tính `pseudo_loss = 0.5 (CE(view1, pseudo_view0)+CE(view0, pseudo_view1))`, thêm riêng vào `total_loss`; không lưu `sample_id→pseudo label`; không có KMeans multi-seed/Silhouette/DBI/no-remap cho novel branch.
- **Bằng chứng từ log**: Cars novel pseudo bật ở epoch **19**, CUB epoch **25**, nhưng paper chỉ chọn novel ở epochs **50–100**. Hai log vẫn có per-batch `pseudo_loss` ở epoch 200.
- **Confidence mismatch**: **100%**. **Confidence gây hại Full accuracy**: **90%** (đánh giá nguyên nhân hợp lý, không phải phép đo counterfactual).

### Patch 1 — Integrity guard để *không tiếp tục gọi nhầm Full là reproduction*

Đây là patch **chặn sai protocol**, không phải code hoàn thiện selection. Nó giúp fail-fast và ngăn kết quả “Full D1+D2 paper” bị report sai. Chỉ áp dụng khi nhóm thống nhất đang reproduce paper; *bỏ cờ guard khi chạy compatibility legacy*.

```diff
diff --git a/src/prototail_gcd/train.py b/src/prototail_gcd/train.py
--- a/src/prototail_gcd/train.py
+++ b/src/prototail_gcd/train.py
@@ def main(allowed_baseline: str | None = None) -> None:
     parser.add_argument("--enable-novel-pseudo", "--enable_novel_pseudo", dest="enable_novel_pseudo", action="store_true", default=False, help="Enable novel pseudo labeling")
+    parser.add_argument(
+        "--require-paper-dual-selection", action="store_true",
+        help="Fail unless the persistent paper D1/D2 selector is integrated",
+    )
@@
     enable_pseudo = bool(arguments.enable_pseudo_labeling)
     enable_novel = bool(arguments.enable_novel_pseudo)
+    if arguments.require_paper_dual_selection and (enable_pseudo or enable_novel):
+        raise NotImplementedError(
+            "Paper Full requires persistent known/novel promotion "
+            "(Eq. 9-11, Algorithm 1, Appendix B). Current per-batch "
+            "pseudo-CE is NOT the paper selector."
+        )
```

**Impact**: không cải thiện accuracy, nhưng **ngăn false reproduction**. **Không xóa `pseudo_loss` rồi gọi là đã sửa D1/D2**: làm vậy chỉ biến `Full` thành một cấu hình khác, không đúng paper.

### Patch 2 — Data/engine integration contract để teammate triển khai D1/D2 *đúng*

Tối thiểu phải có **persistent accepted pseudo-label store** với identity ổn định, và apply những mẫu được promotion vào **cả supervised CE và supervised contrastive**, không chỉ CE tạm thời. Một implementation store độc lập có thể theo diff dưới đây.

```diff
diff --git a/src/prototail_gcd/pseudo/__init__.py b/src/prototail_gcd/pseudo/__init__.py
new file mode 100644
--- /dev/null
+++ b/src/prototail_gcd/pseudo/__init__.py
@@ -0,0 +1 @@
+"""Persistent pseudo-label selection and promotion."""

diff --git a/src/prototail_gcd/pseudo/promotion_store.py b/src/prototail_gcd/pseudo/promotion_store.py
new file mode 100644
--- /dev/null
+++ b/src/prototail_gcd/pseudo/promotion_store.py
@@ -0,0 +1,65 @@
+"""Stable accepted pseudo-labels. No ground-truth novel IDs are read here."""
+
+from __future__ import annotations
+
+from collections.abc import Iterable
+
+import torch
+
+from prototail_gcd.contracts import TrainingBatch
+
+
+class PromotionStore:
+    def __init__(self, total_classes: int, known_classes: int) -> None:
+        self.total_classes = int(total_classes)
+        self.known_classes = int(known_classes)
+        self._labels: dict[int, int] = {}
+
+    def add_accepted(self, pairs: Iterable[tuple[int, int]]) -> int:
+        """Called ONLY by a selection round that passed the paper criteria."""
+        n_added = 0
+        for raw_sample_id, raw_class_id in pairs:
+            sample_id = int(raw_sample_id)
+            class_id = int(raw_class_id)
+            if not 0 <= class_id < self.total_classes:
+                raise ValueError("Promoted class ID is outside the classifier")
+            previous = self._labels.get(sample_id)
+            if previous is not None and previous != class_id:
+                raise ValueError("Cannot remap an already-promoted sample")
+            if previous is None:
+                self._labels[sample_id] = class_id
+                n_added += 1
+        return n_added
+
+    def inject(self, batch: TrainingBatch) -> TrainingBatch:
+        """Make accepted unlabeled samples supervised whenever sampled."""
+        ids = batch.sample_ids.detach().cpu().tolist()
+        targets = batch.known_targets.clone()
+        mask = batch.labeled_mask.clone()
+        for i, sample_id in enumerate(ids):
+            if bool(mask[i]):
+                continue
+            promoted = self._labels.get(int(sample_id))
+            if promoted is not None:
+                targets[i] = promoted
+                mask[i] = True
+        return TrainingBatch(
+            views=batch.views, known_targets=targets,
+            labeled_mask=mask, sample_ids=batch.sample_ids,
+        )
+
+    def export_state(self) -> dict[int, int]:
+        return dict(self._labels)
```

**Đây mới là lớp *storage/injection*. Chưa phải Full algorithm.** Cần thêm những thay đổi dưới đây (integration contract, **không phải patch hoàn chỉnh**):

```diff
diff --git a/src/prototail_gcd/training/engine.py b/src/prototail_gcd/training/engine.py
--- a/src/prototail_gcd/training/engine.py
+++ b/src/prototail_gcd/training/engine.py
@@ def train_batch(self, batch: TrainingBatch, epoch: int) -> dict[str, float]:
         device = next(self.model.parameters()).device
+        # Apply on CPU first; avoid GPU scalar synchronization per sample.
+        # The selector is run in scheduled rounds, not inside train_batch.
+        store = self.context.get("persistent_promotion_store")
+        if store is not None:
+            batch = store.inject(batch)
         batch = batch.to(device)
         self.model.train()
```

**Còn phải triển khai trong `src/prototail_gcd/pseudo/selection.py` và `train.py`:**

1. `select_known_round(...)`: forward `D_l_tilde` bằng global logits, tính **per-known-class accuracy**, chọn **một class tốt nhất còn eligible**, trên unused unlabeled **predicted vào class đó** lấy top **80%** confidence, tối đa **500**, tối đa **20 rounds**. Chỉ nhận known truth từ supervised pool.
2. `select_novel_round(...)`: cluster **backbone CLS features**, `K=C_known+C_novel`, **hai seeds** (`n_init=10`), fit trên tối đa **15.000** rồi assign toàn bộ; align known clusters bằng labeled known targets và Hungarian; xét candidate novel clusters có `n>=10`, Jaccard ≥0.6 giữa seeds, mean Silhouette ≥ median clusters (sample 2.000), output agreement ≥70%, DBI tăng ≤5%, no-remap Jaccard ≥0.3; lấy top **200**/cluster, tối đa **6 rounds**, epochs **50–100**. Selection chỉ từ train+unlabeled, **không novel GT**.
3. Sau mỗi selection round: `store.add_accepted(...)`, **lưu store vào checkpoint**, load lại khi resume; rebuild/adjust sampler để promoted samples thật sự tham gia supervised pool với sampling policy rõ ràng. Dataloader có `sample_id` bất biến. `TrainingEngine` inject store vào **cả CE và supervised contrastive**. **Bỏ `pseudo_weight * pseudo_loss`** khi paper selector hoạt động (paper không có separate pseudo loss); giữ tách `legacy_pseudo_ce` thành ablation riêng nếu cần.
4. Cập nhật prototype EMA từ các mẫu đã accepted theo paper, đồng thời provisional EMA có thể dùng confidence-only; đảm bảo lịch known/novel đúng Appendix B. Cấm thay novel GT vào store hoặc estimator.

**Acceptance tests cần có, không cần GPU training:** `promote` idempotent; không nhận lại cùng sample; không remap; accepted sample có supervised mask True ở mọi epoch; rejected vẫn `UNKNOWN_TARGET=-1`; known có 1 class/round; novel không chọn trước epoch 50; novel filter phải fail nếu thiếu 1/4 tiêu chí; budget và max rounds; không đọc ground truth novel. **Cho tới khi có đủ các mục trên, tiếp tục gọi config là `legacy_full`, KHÔNG là Full paper.**

---

<a id="patch-lt-02"></a>

## LT-02 — Gate capacity cơ chế không học

### Chứng cứ / mức tin cậy

- `methods/prototail.py`: gate logits initialized `−1`, loss `MSE(Σ sigmoid(u), target_capacity)` với `capacity_weight=0.05`, optimizer SGD LR `.01`, loss mean qua **~200 classes**.
- Log CUB/Cars LT20 v3 epoch200: κ~0.82, `target κ∈[1,3]`, `Lcap≈2.00/2.35`, 0% gates >0.5; full training 200 epochs.
- **Confidence**: 100% observation, 90% gradient dilution là cơ chế chính. **Impact**: Tail-aware regularization không điều khiển đúng head/tail capacity; không có bảo đảm fix xong All accuracy sẽ tăng.

### Patch A — Không thay paper; thêm invariant cảnh báo rõ ràng

Branch đã ghi per-class `capacity_mae` trong `*.tail_debug.jsonl` (epoch bội 10). Có thể thêm cảnh báo khi κ chưa vào được range, **không tự chuyển optimizer**:

```diff
diff --git a/src/prototail_gcd/train.py b/src/prototail_gcd/train.py
--- a/src/prototail_gcd/train.py
+++ b/src/prototail_gcd/train.py
@@
                             data["capacity_mae"] = float(
                                 (gates.sum(-1) - desired).abs().mean().item()
                             )
+                            if epoch + 1 >= 30 and float(data["capacity_mae"]) > 0.5:
+                                print(
+                                    "[TAIL_WARNING] Capacity target not reached: "
+                                    f"epoch={epoch+1} "
+                                    f"capacity_mae={data['capacity_mae']:.4f} "
+                                    f"actual_mean={active_s:.3f} "
+                                    f"target_mean={desired.mean().item():.3f}",
+                                    flush=True,
+                                )
```

### Patch B — Chỉ khi quyết định **sửa phương pháp**, không phải faithful reproduction

**Opt-in support warm-start**: dùng **estimated** frequencies của train KMeans tại epoch 0 để đặt gate mass gần desired capacity trước khi tối ưu, tránh hàng nghìn SGD steps với tiny gradients. **Không dùng GT novel counts.** Cần báo trong paper là *training recipe change*. Không bật mặc định trong các run dùng so với Table 1/2/4.

```diff
diff --git a/src/prototail_gcd/methods/prototail.py b/src/prototail_gcd/methods/prototail.py
--- a/src/prototail_gcd/methods/prototail.py
+++ b/src/prototail_gcd/methods/prototail.py
@@ class PartPrototypeBank(nn.Module):
     def gates(self) -> torch.Tensor:
         return torch.sigmoid(self.gate_logits)
+
+    @torch.no_grad()
+    def warm_start_from_estimated_support(self, frequencies: torch.Tensor) -> None:
+        """NON-PAPER-DEFAULT: approximate the target gate mass at init."""
+        target = self.target_capacities(frequencies.to(self.gate_logits.device))
+        # Avoid exact 0/1: otherwise the sigmoid gradient would vanish.
+        per_slot = (target / self.num_slots).clamp(0.02, 0.98)
+        logits = torch.logit(per_slot).unsqueeze(-1)
+        self.gate_logits.copy_(logits.expand_as(self.gate_logits))
```

```diff
diff --git a/src/prototail_gcd/baselines/simgcd/adapter.py b/src/prototail_gcd/baselines/simgcd/adapter.py
--- a/src/prototail_gcd/baselines/simgcd/adapter.py
+++ b/src/prototail_gcd/baselines/simgcd/adapter.py
@@ def before_epoch(self, model, epoch, context, config):
         finally:
             model.train()
+        if epoch == 0 and bool(context.get("gate_warm_start_from_support", False)):
+            if not hasattr(model, "prototail"):
+                raise RuntimeError("Gate warm-start requires an attached ProtoTail module")
+            model.prototail.prototypes.warm_start_from_estimated_support(frequencies)
         return {"estimated_frequencies": frequencies}
```

```diff
diff --git a/src/prototail_gcd/train.py b/src/prototail_gcd/train.py
--- a/src/prototail_gcd/train.py
+++ b/src/prototail_gcd/train.py
@@ def main(allowed_baseline: str | None = None) -> None:
     parser.add_argument("--kmeans-max-iter", type=int, default=300)
+    parser.add_argument(
+        "--gate-warm-start-from-support", action="store_true",
+        help="NON-PAPER-DEFAULT: init gates from estimated training supports",
+    )
@@
     context["needs_frequency_estimate"] = (
         baseline_name == "bacon"
         or (
             method_enabled
             and float(method_settings.get("capacity_regularization_weight", 0)) > 0
         )
     )
+    context["gate_warm_start_from_support"] = arguments.gate_warm_start_from_support
+    if arguments.gate_warm_start_from_support and not context["needs_frequency_estimate"]:
+        raise ValueError("Gate warm-start requires tail-aware frequency estimation")
```

**BaCon adapter cũng cần một hook tương tự** nếu áp dụng cho BaCon. **Cảnh báo:** với head target κ*=3, `clamp(0.98)` cho κ≈2.94, không đúng 3 tuyệt đối; cố ý để gradient không bão hòa. Expected impact *chỉ có thể dự báo* là `Lcap` thấp hơn, gates phân biệt class support ngay tại epoch1; **accuracy tăng/giảm chưa biết**. Test bằng CPU 3 classes: target capacities `[1,2,3]`, sau init `gate mass≈[1,2,2.94]`; không cần train 200 epochs để xác nhận invariant.

---

<a id="patch-lt-03"></a>

## LT-03 — Chốt frequency estimator theo reference (đính chính feedback cũ)

### Nguồn gốc và mức tin cậy

- **BaCon gốc** `model/dist_est.py:70–78`: `idx1=argsort(known_labeled_ins_num)`, `idx2=argsort(known_cluster_ins_num)`, rồi gán theo rank.
- **Branch mới** `baselines/bacon/losses.py`: `estimate[class_id]=cluster_counts[cluster_for_class[class_id]]` (direct mapping).
- **Paper Appendix B** nói follows BaCon nhưng mô tả known aligned size (có thể được hiểu direct), do đó **nguồn paper và reference code không nhất quán ở mức lựa chọn estimator**.
- **Confidence có khác biệt 100%; confidence gây thay đổi accuracy v3 70%** (chưa biết dấu).

### Patch — Thêm mode rõ ràng, default giữ current để không ngầm đổi kết quả

```diff
diff --git a/src/prototail_gcd/baselines/bacon/losses.py b/src/prototail_gcd/baselines/bacon/losses.py
--- a/src/prototail_gcd/baselines/bacon/losses.py
+++ b/src/prototail_gcd/baselines/bacon/losses.py
@@ def estimate_class_frequencies(
     kmeans_max_iter: int = 300,
+    known_frequency_mapping: str = "direct_hungarian",
 ) -> torch.Tensor:
@@
     estimate = np.ones(num_classes, dtype=np.float32)
     if len(known_cluster_ids) != num_known_classes:
         raise RuntimeError("known-class frequency alignment is incomplete")
-    for class_id in range(num_known_classes):
-        # Preserve the Hungarian class -> cluster assignment.
-        estimate[class_id] = cluster_counts[cluster_for_class[class_id]]
+    if known_frequency_mapping == "direct_hungarian":
+        # Literal reading of paper Appendix B.
+        for class_id in range(num_known_classes):
+            estimate[class_id] = cluster_counts[cluster_for_class[class_id]]
+    elif known_frequency_mapping == "bacon_rank":
+        # BaCon model/dist_est.py: known support rank matching.
+        known_labeled_support = np.bincount(
+            all_targets[all_labeled], minlength=num_known_classes,
+        )[:num_known_classes]
+        known_by_support = np.argsort(known_labeled_support, kind="stable")
+        sorted_known_cluster_sizes = np.sort(cluster_counts[known_cluster_ids])
+        estimate[known_by_support] = sorted_known_cluster_sizes
+    else:
+        raise ValueError("Unsupported known_frequency_mapping mode")
```

**Forward tiếp mode qua cả hai adapters** (`bacon/adapter.py`, `simgcd/adapter.py`) từ `context["known_frequency_mapping"]`, đặt CLI/config option và log nó trong `*.config.yaml`. Mẫu hunk:

```diff
diff --git a/src/prototail_gcd/baselines/simgcd/adapter.py b/src/prototail_gcd/baselines/simgcd/adapter.py
--- a/src/prototail_gcd/baselines/simgcd/adapter.py
+++ b/src/prototail_gcd/baselines/simgcd/adapter.py
@@ def before_epoch(self, model, epoch, context, config):
                 kmeans_max_iter=int(context.get("kmeans_max_iter", 300)),
+                known_frequency_mapping=str(context.get("known_frequency_mapping", "direct_hungarian")),
```

```diff
diff --git a/src/prototail_gcd/train.py b/src/prototail_gcd/train.py
--- a/src/prototail_gcd/train.py
+++ b/src/prototail_gcd/train.py
@@ def main(allowed_baseline: str | None = None) -> None:
     parser.add_argument("--kmeans-max-iter", type=int, default=300)
+    parser.add_argument(
+        "--known-frequency-mapping", choices=["direct_hungarian", "bacon_rank"],
+        default="direct_hungarian",
+    )
@@
     context["kmeans_max_iter"] = arguments.kmeans_max_iter
+    context["known_frequency_mapping"] = arguments.known_frequency_mapping
+    config.setdefault("frequency_runtime", {})["known_mapping"] = arguments.known_frequency_mapping
```

**Cần thêm cùng một argument ở `bacon/adapter.py`**, không để BaCon và SimGCD dùng estimator khác nhau mà không ghi metadata. **Không chọn mode theo score test** (data leakage/selection bias). Chọn theo provenance của reported run. Nếu original code thực sự dùng direct mapping thì giữ mode direct; nếu claim BaCon ref implementation, dùng rank.

**Unit test không cần GPU:** fake 2 known/2 novel, đặt `cluster_count` theo một contingency mà rank-order và direct trả ra khác nhau; assert mỗi mode trả đúng giá trị *theo định nghĩa*, không trộn mode ngầm.

---

<a id="patch-lt-04"></a>

## LT-04 — Novel class support mapping: hạn chế kiến trúc của estimator

**Chứng cứ:** `bacon/losses.py`:

```python
novel_cluster_ids = sorted(set(range(num_classes)) - set(known_cluster_ids))
novel_cluster_counts = np.sort(cluster_counts[novel_cluster_ids])[::-1]
estimate[num_known_classes:] = novel_cluster_counts[:novel_count]
```

Cách này **giữ đúng kiểu rank sorting từ BaCon**, nhưng `κ*_c` của novel neuron `c` **không được chứng minh** tương ứng cluster novel nào. **Confidence vấn đề tồn tại: 100%; impact severe LT: có cơ sở 80% nhưng chưa đo được.**

**Không khuyến nghị một-line patch gán novel labels từ ground truth.** Hai lựa chọn phải viết tường minh:

- `novel_mode="bacon_rank"` → tương thích baseline ref, báo `novel_support_alignment=unverified`; không giải quyết identifiability.
- `novel_mode="model_aligned"` → lấy **current classifier posterior** của các *unlabeled training* samples, xây `cluster × novel_output` weighted-contingency, Hungarian match chỉ trên remaining novel clusters/output dims, gán counts theo matching. Kiểm tra coverage, entropy và confidence trước khi dùng; nếu collapse/ambiguity phải báo **unsupported estimate**, không lấy novel GT. **Đây là thuật toán mới, không được báo là exact ProtoTail reproduction**.

**Gợi ý code hunk giúp ngăn report sai provenance** (instrumentation, không đổi result):

```diff
diff --git a/src/prototail_gcd/baselines/bacon/losses.py b/src/prototail_gcd/baselines/bacon/losses.py
--- a/src/prototail_gcd/baselines/bacon/losses.py
+++ b/src/prototail_gcd/baselines/bacon/losses.py
@@
 from typing import Any
+import warnings
@@
     novel_cluster_counts = np.sort(cluster_counts[novel_cluster_ids])[::-1]
     novel_count = num_classes - num_known_classes
+    if novel_count:
+        warnings.warn(
+            "Novel support sizes are rank-assigned to classifier output IDs "
+            "without cluster-to-logit correspondence (BaCon reference heuristic). "
+            "Do not interpret per-novel-class capacity targets as verified support.",
+            RuntimeWarning,
+            stacklevel=2,
+        )
     estimate[num_known_classes:] = novel_cluster_counts[:novel_count]
```

Để triển khai `model_aligned`, tạo `estimate_class_frequencies(..., predict_logits_fn: Callable)` và lưu logits trong feature pass. **Không đưa sẵn patch giả tạo cho BaCon và SimGCD** vì hai model khác đường classifier; cần signature contract theo từng adapter và unit test mapping permutation invariance. Nếu teammate không có thời gian, giữ `bacon_rank` minh bạch và thừa nhận đây là limitation.

---

<a id="patch-lt-05"></a>

## LT-05 — Cân bằng feature norm khi evaluate, không cần retraining

**Chứng cứ:** `PartPrototypeBank.forward()` nhận parts đã L2-normalize; trong `ProtoTailPlugin.predict`, `pooled_parts` là weighted mean (norm ≤1) nhưng `global_features` từ DINOv2 CLS đã LayerNorm, chưa L2-normalize. Concat toàn vector rồi normalize một lần không làm equalize hai nhánh. **Confidence evidence: 99%; impact lên metric chưa biết dấu.**

**Để không thay historic metric ngầm:** thêm mode `legacy_concat` (default) và `balanced_concat` (mới). Các số reported từ mode mới phải ghi nhãn khác.

```diff
diff --git a/src/prototail_gcd/methods/prototail.py b/src/prototail_gcd/methods/prototail.py
--- a/src/prototail_gcd/methods/prototail.py
+++ b/src/prototail_gcd/methods/prototail.py
@@ def predict(self, output: ModelOutput) -> PredictionOutput:
         pooled_parts = pooled_parts / gates.sum(dim=-1, keepdim=True).clamp_min(
             1e-6
         )
-        features = F.normalize(
-            torch.cat((global_features, pooled_parts), dim=-1),
-            dim=-1,
-        )
+        feature_mode = self.config.get("eval_feature_mode", "legacy_concat")
+        if feature_mode == "balanced_concat":
+            # Evaluation-only change: scale each branch before concatenation.
+            global_for_eval = F.normalize(global_features, dim=-1)
+            part_for_eval = F.normalize(pooled_parts, dim=-1)
+        elif feature_mode == "legacy_concat":
+            global_for_eval, part_for_eval = global_features, pooled_parts
+        else:
+            raise ValueError(f"Unknown eval_feature_mode: {feature_mode}")
+        features = F.normalize(
+            torch.cat((global_for_eval, part_for_eval), dim=-1), dim=-1,
+        )
         return PredictionOutput(class_ids=None, features=features)
```

`method.config.eval_feature_mode: balanced_concat` chỉ dùng **khi re-evaluate frozen checkpoint**, không làm đổi training loss. Để kiểm chứng *không retrain*: so `legacy_concat`, `CLS-only`, `balanced_concat` trên cùng frozen embeddings và cùng KMeans backend/seed; đo `||CLS||₂`, `||pooled_parts||₂`. Không được ghi đè metric All cũ bằng số mode mới nếu paper chưa mô tả protocol đó.

---

<a id="patch-lt-06"></a>

## LT-06 — Ngăn so sánh khác protocol báo cáo

**Chứng cứ:** paper Tables 1–2 BaCon LT10; Appendix Table 4 SimGCD LT10; Kaggle v3 SimGCD LT20. Confidence **100%**.

Thêm một script guard đơn giản (không thay model) để bắt so sánh sai ngay từ saved config.

```diff
diff --git a/scripts/verify_paper_protocol.py b/scripts/verify_paper_protocol.py
new file mode 100644
--- /dev/null
+++ b/scripts/verify_paper_protocol.py
@@ -0,0 +1,43 @@
+"""Reject comparisons against paper tables from incompatible run configs."""
+
+from __future__ import annotations
+
+import argparse
+from pathlib import Path
+
+import yaml
+
+PAPER_PROTOCOL = {
+    "table1": ("bacon", 10),
+    "table2": ("bacon", 10),
+    "table4": ("simgcd", 10),
+}
+
+
+def verify(config: dict, table: str) -> None:
+    expected_base, expected_imb = PAPER_PROTOCOL[table]
+    observed_base = str(config["baseline"]["name"])
+    observed_imb = int(config["data"]["protocol"]["imbalance_ratio"])
+    if (observed_base, observed_imb) != (expected_base, expected_imb):
+        raise ValueError(
+            f"NOT comparable with paper {table}: "
+            f"observed baseline={observed_base}, LT={observed_imb}; "
+            f"expected baseline={expected_base}, LT={expected_imb}"
+        )
+
+
+if __name__ == "__main__":
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--saved-config", type=Path, required=True)
+    parser.add_argument("--paper-table", choices=sorted(PAPER_PROTOCOL), required=True)
+    args = parser.parse_args()
+    config = yaml.safe_load(args.saved_config.read_text(encoding="utf-8"))
+    verify(config, args.paper_table)
+    print("BASELINE + IMBALANCE RATIO MATCH. Now also verify split IDs, "
+          "source commit, optimization settings, and evaluation backend.")
```

**Expected:** v3 `--paper-table table1` và `--paper-table table4` phải **fail với thông báo rõ** do khác baseline hoặc ratio. Một PASS chỉ chứng minh 2 trường baseline+ratio trùng, **không tự động xác thực cùng split, commit hay model**.

---

<a id="patch-lt-07"></a>

## LT-07 — Weight decay phải theo paper

**Paper §4.1:** `5e-5`; **code 2 YAML:** `1e-4`. Confidence 100%, impact riêng lên accuracy không định lượng; không được cam kết sẽ phục hồi chênh lệch lớn.

```diff
diff --git a/configs/experiment_simgcd.yaml b/configs/experiment_simgcd.yaml
--- a/configs/experiment_simgcd.yaml
+++ b/configs/experiment_simgcd.yaml
@@
-  weight_decay: 0.0001
+  weight_decay: 0.00005

diff --git a/configs/experiment_bacon.yaml b/configs/experiment_bacon.yaml
--- a/configs/experiment_bacon.yaml
+++ b/configs/experiment_bacon.yaml
@@
-  weight_decay: 0.0001
+  weight_decay: 0.00005
```

YAML mới chỉ ảnh hưởng **run mới**, không chữa lại checkpoint cũ. Cùng với LR, seed, split, batch sampler, snapshot commit và optimizer schedule phải được lưu vào provenance khi report.

---

<a id="patch-lt-08"></a>

## LT-08 — Backend parity và checkpoint reporting

**KMeans backend v3 FAISS, logs cũ sklearn:** numerical method khác, không bitwise match. Cả training frequency estimator và evaluation metrics đều chịu tác động. **Confidence difference 100%, impact riêng không đo được**. Tăng tốc là có thật nhưng không tự suy ra cùng kết quả paper.

Để tránh regression khó theo dõi:

```diff
diff --git a/src/prototail_gcd/train.py b/src/prototail_gcd/train.py
--- a/src/prototail_gcd/train.py
+++ b/src/prototail_gcd/train.py
@@
     config["kmeans_runtime"] = {
         "backend": arguments.kmeans_backend,
         "n_init": arguments.kmeans_n_init,
         "max_iter": arguments.kmeans_max_iter,
+        "parity_note": (
+            "FAISS GPU and sklearn KMeans do not guarantee identical "
+            "initialization or assignments"
+        ),
     }
```

Bản `train.py` hiện dùng `best_test_acc` để save `.best.pt`. Nếu cần kết quả khoa học đúng quy định, chuyển sang final checkpoint **hoặc validation độc lập**; không dùng test labels để chọn best epoch. Việc này **không phải nguyên nhân cho final score thấp**, nên không đưa vào P0 Tail-aware debugging.

---

## B. Tiêu chí nghiệm thu tối thiểu, không cần 200-epoch ablations

| Check                                | Cách kiểm tra ít tốn tài nguyên                     | Pass                                                                            |
| ------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Paper-vs-run protocol                | `scripts/verify_paper_protocol.py` trên saved YAML     | Reject nếu baseline/ratio khác                                                |
| D1/D2 paper vs legacy CE             | Source + unit tests của PromotionStore                   | Accepted sample persistent; không random CE thay cho selection                 |
| No novel GT leakage                  | Fake train batch + selector / estimator unit test         | Novel target train luôn -1 trừ promoted IDs có evidence                      |
| Known estimator mode                 | Fake contingency `cluster_counts` + `labeled_support` | `direct_hungarian` và `bacon_rank` trả output đúng theo 2 định nghĩa |
| Tail capacity hoạt động?          | `*.tail_debug.jsonl` từ v3 nếu còn artifact          | κ thực tế và κ* rõ; không đánh đồng weight>0 với learned allocation |
| Balanced feature eval                | Frozen checkpoint / stored feature tensors                | Norm ratio có log; metric variant ghi riêng                                   |
| Logging provenance                   | Config, commit SHA, KMeans backend, ratio, split hash     | Không có tham số trọng yếu nào chỉ biết từ nickname của Kaggle run    |
| Confidence của tác động accuracy | Không có counterfactual thì giữ “chưa biết”       | Không tự tạo expected +X pp                                                  |

## C. Thứ tự giao việc cho teammate (không yêu cầu ablation)

**Bắt buộc để trung thực với reported paper:**

1. **Khóa protocol**: tìm script/checkpoint/config gốc đã tạo số Table 1/2/4; ghi SHA, backbone, split, baseline, imbalance ratio. Không có provenance thì phải ghi “cannot verify original reported scores”.
2. **Khôi phục đúng D1/D2** (LT-01) hoặc đổi tên run hiện tại thành `legacy_pseudo_ce_full` và **ngừng tuyên bố đã reproduce Full paper**. Không thể sửa Full bằng một dòng threshold.
3. **Chốt frequency estimator** `bacon_rank` vs `direct_hungarian` dựa theo source tạo reported runs; ghi mode. Acknowledge novel support heuristic không định danh class.
4. **Chỉnh weight decay theo paper**, kiểm tra gates chưa học rồi quyết định công khai một method-revision (`gate warm-start`) nếu mục tiêu mới là tăng efficacy chứ không phải reproduction.
5. **Đo feature norms trên frozen checkpoint** và chỉ mở rộng reporting với balanced-concat nếu thật sự phù hợp evaluation protocol; không lặng lẽ đổi official All.

**Không sửa/push gì từ báo cáo này.** Đây là source-audited implementation specification để team review, không phải patch đã được chạy qua `git apply` hay GPU integration test.

```

```
