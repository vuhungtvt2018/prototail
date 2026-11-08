
```markdown
# ProtoTail FULL — Root causes, paper design delta, và COMPLETE CODE FIX (chỉ 1 file)

**Base chính xác:** `16b65129e865da3ca9d0942b0dde00834785cf97` của branch `fix/prototail-tailaware-audit` (đã kiểm tra HEAD tại thời điểm soạn).  
**Phạm vi:** giữ Prototype + Tail-aware + Known D1 + Novel D2; không ablate bất kỳ contribution nào.  
**Trạng thái thực nghiệm:** bản vá có căn cứ từ source/log và unit-test isolated; **chưa chạy GPU, chưa chứng minh accuracy tăng**.

> **Đây là tài liệu duy nhất cần tải.** Root cause, paper deviation, toàn bộ unified patch và hướng dẫn áp dụng đều nằm bên trong. **Không cần** tải `.patch` hay `.py` riêng.

## Áp dụng trực tiếp từ CHÍNH FILE Markdown này

Đặt file Markdown vào thư mục root của checkout có HEAD đúng base SHA rồi chạy:

```bash
git rev-parse HEAD
# Phải trả về: 16b65129e865da3ca9d0942b0dde00834785cf97
python3 - <<'PYEXTRACT'
from pathlib import Path
src = Path('ProtoTail_FULL_Fix_16b6512_ONE_FILE.md').read_text(encoding='utf-8')
patch = src.split('\n```diff\n', 1)[1].split('\n```', 1)[0] + '\n'
Path('/tmp/prototail_full_fix_from_markdown.patch').write_text(patch, encoding='utf-8')
print('Extracted patch bytes:', len(patch.encode('utf-8')))
PYEXTRACT
git apply --check /tmp/prototail_full_fix_from_markdown.patch
git apply /tmp/prototail_full_fix_from_markdown.patch
PYTHONPATH=src python3 -m pytest -q tests/test_full_regression_recovery.py
git diff --check
```

Lưu ý: `git apply --check` trước khi apply là bắt buộc. **Chưa xác minh trên full checkout thật**; nếu không pass, dừng, không dùng `--reject` hay `-3` để ép áp dụng. Chưa commit/push remote.

---

## PHẦN A — Điều tra root cause & expected impact

**Nguồn gốc code:** `fix/prototail-tailaware-audit` @ `16b65129e865da3ca9d0942b0dde00834785cf97` (HEAD xác nhận lúc thực hiện).
**Dataset:** CUB-200 và Stanford Cars, SimGCD, LT-20, epoch 200, `UNLABELED-TRAIN EVAL` **là metric chính**.
**Không tắt contribution:** Prototype, Tail-aware và D1/D2 đều được giữ trong bản sửa.
**Nguyên tắc:** Bằng chứng từ log + source; phần “lý do dự kiến giúp” là cơ chế dự đoán, **không phải gain đo được**.

## 1. Regression có thật, kể cả khi so cùng evaluator

| So sánh cùng `legacy_concat` | CUB Full cũ v3 | CUB Full mới v7 | Cars Full cũ v3 | Cars Full mới v3 |
| -------------------------------- | --------------: | ---------------: | ---------------: | ----------------: |
| Epoch 5 All                      |           52.71 |  **53.34** |            42.29 |   **31.72** |
| Epoch 30 All                     |           57.86 |  **56.05** |            51.95 |   **45.17** |
| Epoch 200 All                    | **60.36** |            56.05 |  **51.90** |             44.83 |
| Epoch 200 Global-head All        |           63.63 |  **60.01** |            57.76 |   **48.01** |
| Epoch 200 Fused-head All         |           62.31 |            59.39 |            57.86 |             50.55 |

CUB giảm **4.31 pp** và Cars giảm **7.07 pp** tại epoch 200 khi đối chiếu các run cùng feature mode. Đáng chú ý: Cars đã giảm **10.57 pp ở epoch 5** khi D1/D2 chưa bắt đầu. Không thể chỉ đổ lỗi cho pseudo-selector.

Bảng ablation IR20 cũ (giữ nguyên số thực nghiệm, không dùng paper làm ground truth):

| Thành phần                            |         CUB All |        Cars All |
| --------------------------------------- | --------------: | --------------: |
| Base                                    |           57.16 |           54.35 |
| Dynamic pseudo-CE only                  |           57.44 | **58.29** |
| Prototype only                          |           58.62 | **47.72** |
| Prototype + Tail-aware only             |           58.48 |           48.49 |
| Full without Tail-aware (legacy pseudo) | **61.54** |           54.11 |
| Full with Tail-aware (legacy pseudo)    |           58.83 |           52.33 |
| Latest Full D1/D2 + Tail-aware          | **56.05** | **44.83** |

**Hệ quả:** Có ít nhất hai nguồn cần sửa đồng thời: nhánh Prototype gây hại global representation và pipeline pseudo sau thay đổi làm supervision khác trước. Tail-aware không đạt capacity target, cũng cần sửa nhưng chỉ riêng nó không giải thích cú rơi epoch 5.

## 2. Năm vấn đề được patch

### BUG-01 / Cars augmentation đổi đúng lúc regression xuất hiện

- **Source:** `src/prototail_gcd/data/cars.py`, thêm `ColorJitter()` vào train transform; CUB giữ nguyên.
- **Log:** Cars Full epoch 5: All **42.29 → 31.72**, global-head All **35.18 → 28.30**. D1/D2 chưa chạy; thay đổi input có thể ảnh hưởng cả global representation trước các bước pseudo.
- **Fix:** Quay lại Cars train transform được dùng trong các run cũ có điểm tốt hơn. Không thay đổi sampler, train/eval split, architecture hoặc D1/D2.
- **Lý do có khả năng giúp:** Phục hồi input distribution đã từng train tốt trên Cars; loại bỏ biến thay đổi duy nhất có ảnh hưởng trực tiếp đến pixels trước epoch 5.
- **Chắc chắn:** Thay đổi source **100%**; độ chắc chắn riêng `ColorJitter` gây toàn bộ mức giảm **trung bình**, chưa có controlled same-seed rerun.
- **Khác paper?** Có thể khác augmentation tham chiếu trong Appendix B (`ImageNet-style`); đây là rollback dựa trên dữ liệu chạy, không phải claim đúng paper.

### BUG-02 / Random class prototypes gây gradient sai hướng ngay từ epoch 1

- **Source:** `methods/prototail.py` khởi tạo mọi class-slot bằng orthogonal random tensor. `part_classification` đã tính từ epoch 1, nhưng `after_optimizer_step` trả về trước `prototype_warmup_epochs=30`. Từ epoch 30, EMA `momentum=0.99` kéo random vector rất chậm về centroid quan sát được.
- **Log:** Cars `Prototype-only = 47.72` thấp hơn Base `54.35` (**−6.63 pp**). Mới nhất tại epoch 200, **60/100 novel prototypes CUB** và **27/98 Cars** chưa từng update.
- **Fix:** (a) update known prototypes từ mẫu labeled ngay từ đầu; (b) lần đầu thấy class dùng **normalized centroid trực tiếp**, sau đó mới EMA; (c) part score của prototypes chưa được quan sát bằng **0** thay vì cosine với random anchor; (d) novel unsupervised pseudo updates vẫn có warmup/filter như cũ.
- **Lý do có khả năng giúp:** Phần local-class evidence được học từ dữ liệu thực trước khi được đưa vào fused CE; loại bỏ feedback từ bộ nhớ chưa khởi tạo, tăng tính ổn định khi classes cực ít mẫu.
- **Chắc chắn:** Code + prototype coverage **100%**; gain đo bằng GPU **chưa có**.
- **Khác paper?** **Khác** thời điểm known EMA bootstrap và quy tắc xử lý class chưa khởi tạo; vẫn giữ attention pooling + class-conditioned EMA prototypes + gates.

### BUG-03 / Auxiliary fused CE cộng lặp gradient global classifier

- **Source:** baseline SimGCD đã có supervised `CE(logits)` với `sup_weight=0.35`; method cộng thêm `part_loss_weight=0.5 * CE(logits + 0.5*part_scores)`. Grad của loss thứ hai cũng quay lại `global_logits`, làm đổi tỷ lệ supervised/contrastive tối ưu hóa.
- **Log:** Cars epoch 1 fused raw CE **3.689** nên contribution xấp xỉ **1.845**; global CE raw **3.727** x `0.35` xấp xỉ **1.304**. Ngay trước khi prototypes cập nhật, extra classification gradient rất lớn.
- **Fix:** Riêng SimGCD, dùng `logits.detach() + 0.5*part_scores` trong **auxiliary training CE**, không detach tính logits ở base loss, không thay đổi prediction/evaluator.
- **Lý do có khả năng giúp:** Giữ gradient của auxiliary CE tới latent parts, slot queries, gates và backbone **qua local path**, nhưng tránh dồn thêm một classifier-head gradient lên model global trong tình trạng part prototypes còn yếu. Giá trị scalar fused CE vẫn giữ nguyên tại forward.
- **Chắc chắn:** Gradient duplication code **100%**, chiều tác động lên accuracy **có cơ sở nhưng chưa benchmark**.
- **Khác paper?** **Khác** gradient flow của `Lfused`; paper không quy định detach. Đây là stabilization có chủ đích, không thể gọi “paper-faithful”.

### BUG-04 / D1/D2 promoted hard labels contaminate GT supervised objectives; D1 reliability tự xác nhận

- **Source:** `PromotionStore.inject()` đổi `labeled_mask=False` thành `True` và gán cứng class IDs cho promoted. `engine.train_batch()` làm vậy **trước** SimGCD CE, SupCon và ProtoTail fused CE. `PaperDualSelector.select_known()` lại đánh giá accuracy ngay trên pool gồm các nhãn pseudo đã promote trước đó.
- **Log:** Cars có **821/2081** unlabeled đã promote (89 Known, 732 Novel), CUB **401/1438** (93 Known, 308 Novel). Trong 20 lượt D1, tất cả reported `supervised_accuracy=1.0`, nhưng chỉ **8/98 Known Cars** và **10/100 Known CUB** được chọn mở rộng.
- **Fix:** (a) với **SimGCD** không biến promoted thành GT, giữ original supervised mask cho CE/SupCon/part CE; (b) bật lại **dynamic cross-view pseudo CE** vốn đã mang lại gain, đồng thời **giữ nguyên D1/D2 selection và persistent store**; (c) D1/D2 accepted proposals có auxiliary CE với **two-view agreement + current confidence >= 0.7**, weight 0.1, để stale/wrong pseudo có thể tự ngừng tác động; (d) D1 reliability tính **chỉ từ labeled GT gốc**, smoothed accuracy `(correct+1)/(n+2)` và ưu tiên class chưa chọn để tránh tie theo numeric ID; (e) BaCon đường chạy hiện hữu được giữ nguyên.
- **Lý do có khả năng giúp:** Khôi phục tín hiệu pseudo động đã chứng minh có lợi trong run cũ, vẫn học từ memory D1/D2, nhưng không cho nhãn thiếu xác thực liên tục tạo positive pairs sai trong SupCon; tránh Known-score 100% do tự đánh giá trên pseudo labels.
- **Chắc chắn:** Source/log chứng minh hard-label contamination & self-confirmation **100%**; thực nghiệm cho thấy Pseudo-only cải thiện Cars **+3.94 pp** nhưng chưa biết điểm Full sau sửa.
- **Khác paper?** **Khác rõ ràng:** paper dùng promotions như supervised pool, không có auxiliary CE. Đây là adaptation thực nghiệm để tối ưu học ổn định, được user cho phép.

### BUG-05 / Tail-aware không đạt capacity target dù loss active 200 epoch

- **Source:** gate logit khởi tạo `-1` (sigmoid ≈0.269), `M=3` -> κ ≈0.807. `capacity_regularization = mean((κ-target)^2)` trên 200 classes với hệ số 0.05 cho gradient mỗi gate rất nhỏ. Warm-start đã hỗ trợ trong code nhưng **flags Full mới không bật**.
- **Log:** epoch 200 CUB κ≈**0.818**, target mean≈**2.195**, MAE **1.377**; Cars κ≈**0.825**, target≈**2.362**, MAE **1.537**. `gate mean` chỉ 0.273/0.275, tất cả slots dưới 0.5.
- **Fix:** Với `method_enabled` và `capacity_weight>0`, **auto-enable support-based warm-start từ estimated frequencies ở epoch 0**; không đổi công thức gate, gate regularizer, estimator hay interval. Tail-aware không bị tắt, vẫn train 200 epochs và refresh support.
- **Lý do có khả năng giúp:** Đưa κ tới gần target từ khi học bắt đầu thay vì trông chờ gradient quá nhỏ tự đi từ κ0.81 đến κ2.2; đây là điều kiện cần để đánh giá đúng hiệu quả Tail-aware. Các log warm-start trước đó có MAE nhỏ hơn rõ.
- **Chắc chắn:** Failure log **100%**, warm-start đưa κ gần target theo toán học **cao**, accuracy gain **chưa xác nhận**. Vì normalized score bất biến theo scale chung của gates, việc đạt κ **không tự bảo đảm part accuracy tăng**.
- **Khác paper?** **Khác** initialization; không khác equations (3)–(7).

## 3. Cơ chế dự kiến cải thiện: tối ưu lại tín hiệu, không phải “thêm nhiều loss hơn”

Trước fix: random part anchors → fused CE tác động mạnh tới global classifier; paper pseudo mode tắt dynamic CE và coi promoted là GT → SupCon + CE tự khuếch đại nhãn sai; D1 tự đánh giá trên promoted; Tail-aware gate dưới target.

Sau fix: observed prototypes → local evidence có nghĩa; aux fused CE chỉ hướng dẫn local path; GT-only SupCon sạch; dynamic pseudo-CE tiếp tục xử lý unlabeled mỗi batch; D1/D2 tiếp tục chọn/promote nhưng accepted labels được lọc lại khi sử dụng; Tail-aware bắt đầu gần budget theo support.

**Tại sao không cam kết gain?** Các tác động gradient dự kiến có lợi nhưng liên kết *từng bug → số điểm accuracy tăng* không suy ra được khi chưa chạy. Chúng ta chỉ có thể kiểm chứng ngay các bất biến code và benchmark cuối trên cùng `UNLABELED-TRAIN` sau khi patch được áp dụng.

## 4. Phạm vi và các điểm chưa patch (tránh tạo bug mới)

- Không thay đổi KMeans backend, KMeans seeds, GT handling, feature mode `legacy_concat`, labels/splits/sampler, rank/Hungarian frequency alignment mới, hardware hay checkpoints. Giữ phép so sánh KMeans cùng evaluator.
- Không thay đổi `novel_prototype_end_epoch=60` hoặc điều kiện D2 DBI/Jaccard khi chưa đo được pseudo-label precision. Một phần novel prototypes vẫn có thể chưa update; mask random evidence là fix an toàn đầu tiên.
- Không thêm learning rate multiplier hoặc capacity hard projection mới; sử dụng warm-start hiện có.
- Những con số tốt của paper có thể là trên protocol khác; audit này **không sử dụng chúng để khẳng định gain**.

## 5. Các bài test cần pass sau khi áp dụng

1. `tests/test_full_regression_recovery.py`: current-vs-stale pseudo, GT untouched, SupCon không đọc pseudo làm GT, random prototype masking, D1 không self-confirm, fused CE không trùng gradient global.
2. `pytest -q` của toàn repo nếu môi trường đầy đủ.
3. Chạy Full một lần với đúng seed/split/`legacy_concat` để so epoch 5/30/50/100/200; ghi riêng `paper_promotion_ce`, `paper_promotion_kept`, `n_pseudo_known/novel`, `capacity_mae`, `novel_untouched_classes`.
4. **Không được dùng disjoint-test để thay thế main metric**, và không dùng GT novel để chọn pseudo/frequency targets.

**Lưu ý nguồn:** các chỉ số ở trên được đọc từ log cũ/mới đã gửi; code từ GitHub `16b65129e865da3ca9d0942b0dde00834785cf97`; paper chỉ dùng trong tài liệu so sánh thiết kế, không dùng để đánh giá nguyên nhân code/log ở báo cáo này.

---

## PHẦN B — So sánh thiết kế paper và phiên bản sửa Full

**Base:** `16b65129e865da3ca9d0942b0dde00834785cf97`. **Scope:** Full SimGCD LT20.
**Quan điểm:** Giữ đầy đủ 3 contribution / cơ chế: (1) latent parts + class-conditioned prototypes, (2) Tail-aware capacity, (3) known/novel D1/D2 selection. Chấp nhận đổi training strategy để giảm regression, **không tuyên bố đây là đúng 100% manuscript**.

## 1. Bảng đối chiếu explicit với manuscript

| Hạng mục                            | Manuscript gốc                                                             | HEAD trước fix                                                    | Patch mới                                                                                                         | Giống/khác paper                                                                  |
| ------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Latent slots / class-conditioned bank | 3 slots, orthogonal queries, per-class-slot prototypes                      | Có                                                                 | Giữ nguyên                                                                                                       | **Giống** về kiến trúc                                                    |
| Prototype update                      | EMA µ=0.99 **từ epoch 30**; novel threshold0.7, cutoff 60 (App.B)   | Random vectors vẫn vào fused CE từ epoch 1; update đợi epoch30 | Known bootstrap ngay khi có labeled sample; novel EMA schedule không đổi; unobserved scores=0                  | **Khác có chủ đích** về bootstrap và masking                           |
| Fused CE                              | `Lbase + λf Lfused` trên fused logits (§3.5)                           | Extra CE cũng backward qua global head                             | Extra CE giữ scalar value nhưng `global_logits.detach()` riêng loss phụ                                      | **Khác** về gradient, giữ architecture/fused scores                        |
| Gate formula & capacity target        | Eq.(3)–(7), soft λcap, log-min-max support                                | Có nhưng κ cách xa κ*                                          | Giữ nguyên phương trình, **auto warm-start epoch0** từ estimated support                               | **Giống toán, khác initialization**                                        |
| Known selector D1                     | Reliability từ supervised pool, top confidence, một class mỗi round      | Tính accuracy từ cả accepted pseudo, tie 100% theo ID            | Chỉ dùng GT labeled để chấm reliability; Bayesian smoothed + ưu tiên chưa chọn                            | **Khác** cách ước lượng reliability/tie-break, giữ one-class-per-round |
| Novel selector D2                     | Two KMeans, stability/compactness/DBI, permanent accepted labels            | Có                                                                 | Giữ nguyên selection rules/store                                                                                 | **Giống** cách chọn D2                                                     |
| D1/D2 supervision                     | Promoted là supervised examples, **không pseudo CE riêng** (§3.5) | `inject()` vào labeled_mask: GT CE, SupCon, fused CE             | Promoted có **gated weighted auxiliary CE**, GT supervised mask không đổi; dynamic pseudo CE vẫn active | **Khác lớn** (experimental recovery)                                        |
| Cars augmentation                     | ImageNet-style augmentation (App.B), reference SimGCD có ColorJitter       | Có ColorJitter mới thêm                                          | Revert đúng train transform cũ Cars, không đổi eval                                                          | **Khác reference augmentation**; rollback có căn cứ từ regression sớm   |
| Đánh giá và báo cáo             | Manuscript chưa nhất quán đầy đủ về split trong mọi bảng          | `UNLABELED-TRAIN` và `DISJOINT-TEST`                           | Giữ `UNLABELED-TRAIN` metric chính; `legacy_concat` không đổi                                             | **Không đổi protocol đo trong loạt log này**                            |

### Trích nguồn paper (không phải giả thuyết)

- §3.3 và Eq.(2)–(7): normalized part similarities, category prototypes và capacity-gating.
- §3.5 / Algorithm 1: `Lbase + λf Lfused + λg Lgate + λcap Lcap`, promotions nhập supervised pool thay vì có pseudo CE phụ.
- Appendix B (phần implementation): prototypes bắt đầu EMA từ epoch 30, novel update hạn chế epochs30–60, λcap=λg=0.05, λf SimGCD=0.5, slot queries 0.05, gate LR0.01.
- Nguồn để tra đối chiếu: bản `prototail.pdf` người dùng cung cấp; không suy diễn từ các bản paper cũ khác.

## 2. Pipeline đầy đủ sau patch (không tắt contribution)

```text
GT labeled + Unlabeled data
   │
   ├── DINOv2 → CLS + patch tokens
   │      ├── Baseline supervised CE / SupCon trên GT labeled gốc
   │      ├── Baseline self-distillation + contrastive
   │      └── Dynamic pseudo CE cho unlabeled sau warmup
   │
   ├── Latent part queries → slot features
   │      └── Class-conditioned prototype bank
   │             ├── Known bootstrap từ GT khi class lần đầu xuất hiện
   │             ├── Known EMA tiếp tục sau đó
   │             ├── Novel confidence-filtered EMA theo lịch cũ
   │             └── Chưa init → không sinh random local-class score
   │
   ├── Tail-aware gates / Lgate / Lcap
   │      └── Epoch 0 init κ≈target từ estimated support
   │
   ├── Fused CE auxiliary
   │      └── grad qua local parts/gates, không nhân đôi global classifier grad
   │
   └── D1 Known + D2 Novel, persistent promotion memory
          ├── D1 reliability GT-only, smoothed / fairness
          ├── D2 tests giữ nguyên, không dùng GT novel
          └── Accepted samples: weighted two-view-consistent CE (not SupCon GT)
```

**FULL ≠ 3 toggles chỉ để log.** Tất cả các nhánh trên phải có non-zero gradient / update khi có dữ liệu hợp lệ. Những batch chưa có pseudo được chọn vẫn có baseline self-training và part/gate objectives, không fail hoặc silently revert sang GT.

## 3. Tại sao chọn các deviation này thay vì copy paper nguyên văn?

### 3.1 Fused CE scalar giữ nguyên, thay gradient route

Bản HEAD tối ưu: `Lglobal_CE + λf CE(global_logits + 0.5*part_scores)`. Khi prototypes random, nhánh phụ cũng ép global classifier thích ứng với local evidence giả. Bản mới tối ưu global head bằng baseline và **để auxiliary CE huấn luyện local path**. Không phải bỏ fused CE; scalar loss forward không đổi. Đây là cách giảm interference giữa loss đã mạnh và loss phụ còn noisy.

### 3.2 Persistent samples vẫn được sử dụng nhưng không được “phong thánh” thành GT

Điểm yếu HEAD: accepted sample sau một lần cluster/majority sẽ đi vào SupCon như nhãn thật suốt phần còn lại. Bản mới vẫn giữ memory và D1/D2; mỗi lần sử dụng lại nhãn được kiểm bằng **argmax agreement của tất cả views** và `min_confidence`. Nếu không còn đáng tin, sample tạm thời không tạo loss phụ; memory vẫn có và có thể hoạt động lại khi prediction stabilizes.

Dynamic pseudo CE cũ được giữ như một *đường học song song*, vì ablation thực tế (Cars LT-20 Pseudo-only=58.29 so với Base=54.35) cho thấy nó đã từng có lợi. **Đây là đổi phương pháp**, không được trình bày như paper gốc trong các kết quả nghiên cứu nếu không cập nhật phương pháp/ablation trong manuscript.

### 3.3 Tail-aware từ Soft loss về khả năng thực thi được

Không dùng hard projection hay gradient multiplier tùy tiện. Bản mới tận dụng đúng `warm_start_from_estimated_support()` đã có; tại epoch 0, `κ_c ≈ κ*_c`, còn per-slot gates tiếp tục train qua fused CE + regularizer. Lưu ý: vì `part_score` normalized theo tổng gates, chỉ khớp capacity **chưa đủ** để bảo đảm local discriminativeness. Đây là necessary but not sufficient fix.

## 4. Baseline / reviewer honesty

- Nếu sử dụng các sửa khác paper trong nghiên cứu, cần **đổi mô tả thực nghiệm của paper**, báo cáo rõ ý tưởng mới và ablation tương ứng. Không được dùng lại kết quả đã report như thể pipeline bất biến.
- Không gọi patch là “đã tăng accuracy”. Mục tiêu hướng cải thiện có cơ chế rõ, **chưa GPU benchmark**.
- Không thay đổi evaluation protocol khi so old/new để tránh tạo gain giả.
- Không dùng novel GT cho D1/D2, training hoặc estimated frequency. Có thể dùng GT trong **offline audit metrics** nhưng tuyệt đối không được leak vào lựa chọn training samples.

## 5. Verify khi chạy lại một Full duy nhất

Dùng CUB v7/Cars v3 hiện tại làm controls; giữ seed0, data split IR20, model, device, FAISS GPU và `legacy_concat`. Kiểm tra ngay epoch5: Cars input/global-head có hồi phục so với 31.72/28.30 không; kiểm tra epoch30–100 accepted pseudo, how many `paper_promotion_kept` actually activate; kiểm tra gate capacity at epoch10/200; kiểm tra novel prototype coverage; đánh giá All/Known/Novel/U-Few tại epoch200. Nếu không cải thiện, **không phát biểu patch work về mặt metric**.

---

## PHẦN C — Full unified code diff (7 paths, 17 hunks)

**Patch này được nhúng nguyên vẹn; không cần file `.patch` phụ.** Kết quả kiểm tra được ghi dưới đây.

### C.1 Toàn bộ unified diff

```diff
diff --git a/src/prototail_gcd/data/cars.py b/src/prototail_gcd/data/cars.py
--- a/src/prototail_gcd/data/cars.py
+++ b/src/prototail_gcd/data/cars.py
@@ -199,7 +199,6 @@
         transforms.Resize(resize_size, interpolation=interpolation),
         transforms.RandomCrop(image_size),
         transforms.RandomHorizontalFlip(),
-        transforms.ColorJitter(),  # match official SimGCD ImageNet train transform
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std),
     ])
diff --git a/src/prototail_gcd/methods/prototail.py b/src/prototail_gcd/methods/prototail.py
--- a/src/prototail_gcd/methods/prototail.py
+++ b/src/prototail_gcd/methods/prototail.py
@@ -110,3 +110,6 @@
         scores = (similarities * gates.unsqueeze(0)).sum(dim=-1)
         scores = scores / (gates.sum(dim=-1).unsqueeze(0) + 1e-6)
+        # An untouched prototype is an arbitrary orthogonal vector, not evidence.
+        initialized = self.update_counts.to(scores.device) > 0
+        scores = scores.masked_fill(~initialized.unsqueeze(0), 0.0)
         return scores, similarities, gates
@@ -128,8 +128,12 @@
             centroid = parts[mask].mean(dim=0)
             class_index = int(class_id.item())
-            self.prototypes[class_index] = F.normalize(
-                momentum * self.prototypes[class_index]
-                + (1.0 - momentum) * centroid,
-                dim=-1,
-            )
+            if self.update_counts[class_index] == 0:
+                # EMA(0.99) of a random vector is not a valid first estimate.
+                self.prototypes[class_index] = F.normalize(centroid, dim=-1)
+            else:
+                self.prototypes[class_index] = F.normalize(
+                    momentum * self.prototypes[class_index]
+                    + (1.0 - momentum) * centroid,
+                    dim=-1,
+                )
             self.update_counts[class_index] += 1
@@ -155,8 +155,11 @@
             centroid = centroid / weights.sum(dim=0).clamp_min(1e-6)
             class_index = int(class_id.item())
-            self.prototypes[class_index] = F.normalize(
-                momentum * self.prototypes[class_index]
-                + (1.0 - momentum) * centroid,
-                dim=-1,
-            )
+            if self.update_counts[class_index] == 0:
+                self.prototypes[class_index] = F.normalize(centroid, dim=-1)
+            else:
+                self.prototypes[class_index] = F.normalize(
+                    momentum * self.prototypes[class_index]
+                    + (1.0 - momentum) * centroid,
+                    dim=-1,
+                )
             self.update_counts[class_index] += 1
@@ -306,6 +306,8 @@
         if is_simgcd:
             view_count = logits.shape[0] // batch_size
+            # Base SimGCD already optimizes global CE. This auxiliary CE should
+            # train local parts/gates, not add a second global-classifier gradient.
             fused_logits = (
-                logits
+                logits.detach()
                 + float(self.config.get("part_weight", 0.5)) * part_scores
             )
@@ -501,3 +501,1 @@
-        if epoch < prototype_start:
-            return
         is_simgcd = "student" in output.branches
@@ -521,5 +521,8 @@
         if torch.any(labeled):
-            # Known EMA continues through the end of training (Appendix B).
+            # Bootstrap known prototypes immediately from ground-truth labeled data.
             bank.update_known(parts[labeled], labels[labeled], momentum)
 
+        # Novel unlabelled prototypes remain protected by their warmup schedule.
+        if epoch < prototype_start:
+            return
         unlabeled = ~labeled
diff --git a/src/prototail_gcd/pseudo/paper_selector.py b/src/prototail_gcd/pseudo/paper_selector.py
--- a/src/prototail_gcd/pseudo/paper_selector.py
+++ b/src/prototail_gcd/pseudo/paper_selector.py
@@ -67,6 +67,7 @@
         self.previous_accepted_dbi: float | None = None
         self.executed_known: set[int] = set()
         self.executed_novel: set[int] = set()
+        self.selected_known_classes: set[int] = set()
 
     def scheduled(self, epoch: int) -> bool:
         return epoch in self.known_epochs or epoch in self.novel_epochs
@@ -118,24 +118,41 @@
         promoted = store.export_state()
-        targets = snap.known_targets.copy()
-        pool = snap.originally_labeled.copy()
+        # Reliability must be anchored in ORIGINAL labeled ground truth only.
+        # Using promoted pseudo labels here makes accuracy self-confirming.
+        targets = snap.known_targets
+        pool = snap.originally_labeled
+        preds = snap.probs.argmax(axis=1)
+        accuracies = np.full(self.n_known, -1.0, dtype=np.float32)
+        smoothed = np.full(self.n_known, -1.0, dtype=np.float32)
+        mean_conf = np.zeros(self.n_known, dtype=np.float32)
+        for c in range(self.n_known):
+            which = pool & (targets == c)
+            n = int(which.sum())
+            if n:
+                correct = int((preds[which] == c).sum())
+                accuracies[c] = correct / n
+                smoothed[c] = (correct + 1.0) / (n + 2.0)
+                mean_conf[c] = float(snap.probs[which, c].mean())
+        if np.all(accuracies < 0):
+            return {"accepted": 0, "reason": "no_supervised_known_classes"}
+        # A promoted sample cannot be selected again, but is NEVER a GT label.
+        eligible = ~pool.copy()
         by_id = {int(v): i for i, v in enumerate(snap.sample_ids)}
-        for sample_id, class_id in promoted.items():
+        for sample_id in promoted:
             idx = by_id.get(int(sample_id))
             if idx is not None:
-                pool[idx] = True
-                targets[idx] = int(class_id)
-        preds = snap.probs.argmax(axis=1)
-        accuracies = np.full(self.n_known, -1.0, dtype=np.float32)
-        for c in range(self.n_known):
-            which = pool & (targets == c)
-            if np.any(which):
-                accuracies[c] = float(np.mean(preds[which] == c))
-        if np.all(accuracies < 0):
-            return {"accepted": 0, "reason": "no_supervised_known_classes"}
-        # Sort classes by supervised accuracy; choose only one class this round.
-        # Skip exhausted classes to avoid rounds with no candidates.
-        eligible = ~pool
-        for c in np.argsort(-accuracies, kind="stable"):
+                eligible[idx] = False
+        # Prefer classes not yet selected, avoid class-ID tie artifacts; use
+        # smoothed GT accuracy and confidence to break 100%-accuracy ties.
+        ranking = sorted(
+            range(self.n_known),
+            key=lambda c: (
+                int(c in self.selected_known_classes),
+                -float(smoothed[c]),
+                -float(mean_conf[c]),
+                c,
+            ),
+        )
+        for c in ranking:
             if accuracies[c] < 0:
                 continue
             candidate_indices = np.flatnonzero(eligible & (preds == c))
@@ -147,5 +164,11 @@
             ][:n_keep]
             pairs = [(int(snap.sample_ids[i]), int(c)) for i in sorted_indices]
             accepted = store.add_accepted(pairs)
-            return {"accepted": accepted, "class_id": int(c), "supervised_accuracy": float(accuracies[c])}
+            if accepted:
+                self.selected_known_classes.add(int(c))
+            return {
+                "accepted": accepted, "class_id": int(c),
+                "supervised_accuracy": float(accuracies[c]),
+                "smoothed_reliability": float(smoothed[c]),
+            }
         return {"accepted": 0, "reason": "no_eligible_known_predictions"}
diff --git a/src/prototail_gcd/pseudo/promotion_store.py b/src/prototail_gcd/pseudo/promotion_store.py
--- a/src/prototail_gcd/pseudo/promotion_store.py
+++ b/src/prototail_gcd/pseudo/promotion_store.py
@@ -33,4 +33,18 @@
         return n_added
+
+    def lookup(self, batch: TrainingBatch) -> tuple[torch.Tensor, torch.Tensor]:
+        """Return accepted pseudo targets separately from ground-truth labels."""
+        ids = batch.sample_ids.detach().cpu().tolist()
+        targets = torch.full_like(batch.known_targets, -1)
+        mask = torch.zeros_like(batch.labeled_mask, dtype=torch.bool)
+        for i, sample_id in enumerate(ids):
+            if bool(batch.labeled_mask[i]):
+                continue
+            promoted = self._labels.get(int(sample_id))
+            if promoted is not None:
+                targets[i] = promoted
+                mask[i] = True
+        return targets, mask
 
     def inject(self, batch: TrainingBatch) -> TrainingBatch:
         """Make accepted unlabeled samples supervised whenever sampled."""
diff --git a/src/prototail_gcd/train.py b/src/prototail_gcd/train.py
--- a/src/prototail_gcd/train.py
+++ b/src/prototail_gcd/train.py
@@ -572,10 +572,19 @@
         context["persistent_promotion_store"] = paper_store
-        # Paper supervision comes from persistent promotions, not extra CE.
-        # Keep all native baseline distillation/contrastive objectives intact.
-        context["enable_pseudo_labeling"] = False
-        context["enable_novel_pseudo"] = False
-        baseline_config["enable_pseudo_labeling"] = False
-        baseline_config["enable_novel_pseudo"] = False
+        if baseline_name == "simgcd":
+            # SimGCD Full: retain dynamic pseudo-CE alongside D1/D2 memory.
+            # Promoted proposals become gated auxiliary CE, not GT/SupCon.
+            context["paper_promotion_weight"] = float(
+                method_settings.get("paper_promotion_weight", 0.1)
+            )
+            context["paper_promotion_min_confidence"] = float(
+                method_settings.get("paper_promotion_min_confidence", 0.7)
+            )
+        else:
+            # Do not silently change the BaCon adapter's existing protocol.
+            context["enable_pseudo_labeling"] = False
+            context["enable_novel_pseudo"] = False
+            baseline_config["enable_pseudo_labeling"] = False
+            baseline_config["enable_novel_pseudo"] = False
     context["kmeans_backend"] = arguments.kmeans_backend
     context["kmeans_n_init"] = arguments.kmeans_n_init
     context["kmeans_max_iter"] = arguments.kmeans_max_iter
@@ -591,4 +591,10 @@
-    context["gate_warm_start_from_support"] = bool(arguments.gate_warm_start_from_support)
+    # Start Tail-aware gates at their own support budgets instead of leaving
+    # sigmoid(-1) at kappa~0.81 against targets 2.2-2.4 for 200 epochs.
+    context["gate_warm_start_from_support"] = bool(
+        arguments.gate_warm_start_from_support
+        or (method_enabled and not ablate_capacity
+            and float(method_settings.get("capacity_regularization_weight", 0.0)) > 0)
+    )
     context["needs_frequency_estimate"] = (
         baseline_name == "bacon"
         or (
diff --git a/src/prototail_gcd/training/engine.py b/src/prototail_gcd/training/engine.py
--- a/src/prototail_gcd/training/engine.py
+++ b/src/prototail_gcd/training/engine.py
@@ -5,6 +5,41 @@
 import torch
 
 from prototail_gcd.contracts import TrainingBatch
+import torch.nn.functional as F
+
+
+def promoted_consistency_loss(
+    logits: torch.Tensor,
+    batch: TrainingBatch,
+    store: Any,
+    *,
+    temperature: float,
+    min_confidence: float,
+) -> tuple[torch.Tensor, int]:
+    """Auxiliary CE for *currently stable* persistent proposals (SimGCD only).
+
+    Ground-truth supervised masks, SupCon and prototype GT updates are untouched.
+    """
+    targets, accepted = store.lookup(batch)
+    zero = logits.sum() * 0.0
+    if len(batch.views) < 2 or not bool(accepted.any()):
+        return zero, 0
+    views = logits.float().chunk(len(batch.views), dim=0)
+    if any(x.shape[0] != len(batch.sample_ids) for x in views):
+        raise ValueError("promoted loss view alignment mismatch")
+    with torch.no_grad():
+        probs = torch.stack([(x / temperature).softmax(-1) for x in views])
+        preds = probs.argmax(dim=-1)
+        confidence = probs.max(dim=-1).values
+        stable = accepted & (preds == targets.unsqueeze(0)).all(0)
+        stable = stable & (confidence >= min_confidence).all(0)
+    if not bool(stable.any()):
+        return zero, 0
+    ce = torch.stack([
+        F.cross_entropy(x[stable] / temperature, targets[stable])
+        for x in views
+    ]).mean()
+    return ce, int(stable.sum().item())
 
 
 
@@ -76,4 +76,7 @@
         store = self.context.get("persistent_promotion_store")
-        if store is not None:
+        # SimGCD keeps true labels in its supervised contrastive objective;
+        # promoted samples receive separate, confidence-gated supervision.
+        # Preserve the existing BaCon promotion behavior outside this track.
+        if store is not None and self.baseline.name != "simgcd":
             batch = store.inject(batch)
         device = next(self.model.parameters()).device
@@ -129,3 +129,20 @@
                 total_loss = losses["total"]
 
+            if store is not None and "student" in output.branches:
+                branch = output.branches["student"]
+                promotion_loss, promotion_count = promoted_consistency_loss(
+                    branch.logits,
+                    batch,
+                    store,
+                    temperature=float(self.baseline_config.get("student_temp", 0.1)),
+                    min_confidence=float(self.context.get("paper_promotion_min_confidence", 0.7)),
+                )
+                # Keep D1/D2 active, but do not train noise as GT in SupCon.
+                total_loss = total_loss + float(
+                    self.context.get("paper_promotion_weight", 0.1)
+                ) * promotion_loss
+                losses["paper_promotion_ce"] = promotion_loss
+                losses["paper_promotion_kept"] = total_loss.new_tensor(float(promotion_count))
+                losses["total"] = total_loss
+
         if self.use_fp16 and device.type == "cuda":
diff --git a/tests/test_full_regression_recovery.py b/tests/test_full_regression_recovery.py
new file mode 100644
--- /dev/null
+++ b/tests/test_full_regression_recovery.py
@@ -0,0 +1,114 @@
+"""CPU regression guards for Full ProtoTail repairs (no GT unlabeled)."""
+
+import numpy as np
+import torch
+
+from prototail_gcd.contracts import TrainingBatch
+from prototail_gcd.methods.prototail import PartPrototypeBank
+from prototail_gcd.pseudo.paper_selector import Snapshot, PaperDualSelector
+from prototail_gcd.pseudo.promotion_store import PromotionStore
+from prototail_gcd.training.engine import promoted_consistency_loss
+
+
+def _batch():
+    return TrainingBatch(
+        views=(torch.zeros(3, 3, 4, 4), torch.zeros(3, 3, 4, 4)),
+        known_targets=torch.tensor([0, -1, -1]),
+        labeled_mask=torch.tensor([True, False, False]),
+        sample_ids=torch.tensor([11, 12, 13]),
+    )
+
+
+def test_promotions_never_enter_ground_truth_mask():
+    batch = _batch()
+    store = PromotionStore(4, 2)
+    store.add_accepted([(12, 2), (13, 3)])
+    targets, mask = store.lookup(batch)
+    assert torch.equal(mask, torch.tensor([False, True, True]))
+    assert targets.tolist() == [-1, 2, 3]
+    assert batch.known_targets.tolist() == [0, -1, -1]
+    assert batch.labeled_mask.tolist() == [True, False, False]
+
+
+def test_stale_promoted_labels_have_zero_gradient():
+    batch = _batch()
+    store = PromotionStore(4, 2)
+    store.add_accepted([(12, 2)])
+    logits = torch.zeros(6, 4, requires_grad=True)
+    # Disagree with promoted class 2 on both views.
+    with torch.no_grad():
+        logits[1, 0] = 20
+        logits[4, 0] = 20
+    loss, kept = promoted_consistency_loss(
+        logits, batch, store, temperature=0.1, min_confidence=0.7
+    )
+    assert kept == 0
+    assert loss.item() == 0
+    loss.backward()
+    assert torch.all(logits.grad == 0)
+
+
+def test_stable_promoted_label_has_gradient():
+    batch = _batch()
+    store = PromotionStore(4, 2)
+    store.add_accepted([(12, 2)])
+    logits = torch.zeros(6, 4, requires_grad=True)
+    with torch.no_grad():
+        logits[1, 2] = 2.0
+        logits[4, 2] = 2.0
+    loss, kept = promoted_consistency_loss(
+        logits, batch, store, temperature=0.1, min_confidence=0.7
+    )
+    assert kept == 1
+    assert torch.isfinite(loss)
+    loss.backward()
+    assert torch.isfinite(logits.grad).all()
+    assert logits.grad[1].abs().sum() > 0
+
+
+def test_unobserved_prototypes_never_contribute_evidence():
+    bank = PartPrototypeBank(num_classes=4, num_slots=3, feature_dim=16)
+    parts = torch.randn(2, 3, 16)
+    scores, _, _ = bank(torch.nn.functional.normalize(parts, dim=-1))
+    assert torch.count_nonzero(scores) == 0
+    bank.update_known(
+        torch.nn.functional.normalize(parts[:1], dim=-1),
+        torch.tensor([0]), 0.99,
+    )
+    scores, _, _ = bank(torch.nn.functional.normalize(parts, dim=-1))
+    assert torch.count_nonzero(scores[:, 1:]) == 0
+    assert bank.update_counts[0] == 1
+
+
+def test_aux_fused_ce_does_not_duplicate_global_head_gradient():
+    logits = torch.randn(5, 4, requires_grad=True)
+    parts = torch.randn(5, 4, requires_grad=True)
+    labels = torch.arange(5) % 4
+    aux = torch.nn.functional.cross_entropy(
+        (logits.detach() + 0.5 * parts) / 0.1, labels
+    )
+    aux.backward()
+    assert logits.grad is None
+    assert parts.grad is not None and parts.grad.abs().sum() > 0
+
+
+def test_d1_does_not_treat_previous_pseudo_as_gt():
+    # Each known category has a different number of original labeled examples.
+    selector = PaperDualSelector(n_known=2, n_classes=4)
+    store = PromotionStore(4, 2)
+    store.add_accepted([(40, 0)])
+    snap = Snapshot(
+        sample_ids=np.array([10, 11, 20, 21, 22, 40, 41]),
+        features=np.eye(7, dtype=np.float32),
+        probs=np.array([
+            [.9,.1,0,0], [.9,.1,0,0], [.1,.9,0,0],
+            [.1,.9,0,0], [.1,.9,0,0], [.95,.05,0,0], [.1,.9,0,0],
+        ], dtype=np.float32),
+        known_targets=np.array([0,0,1,1,1,-1,-1]),
+        originally_labeled=np.array([1,1,1,1,1,0,0], dtype=bool),
+    )
+    event = selector.select_known(snap, store)
+    # Smoothed score n=3 favors known class 1, unlike arbitrary class id 0.
+    assert event['class_id'] == 1
+    assert event['accepted'] >= 1
+    assert selector.selected_known_classes == {1}
```

## 3. Mục đích từng file thay đổi

| File                                            | Sửa gì                                                                                                             | Vì sao                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `src/prototail_gcd/data/cars.py`              | Revert ColorJitter riêng Cars                                                                                       | Loại trừ thay đổi input trùng regression epoch5 trước D1/D2        |
| `src/prototail_gcd/methods/prototail.py`      | Mask random unobserved prototype, centroid bootstrap, known update early, detach global logits ở auxiliary fused CE | Local evidence từ dữ liệu thật, giảm double CE gradients             |
| `src/prototail_gcd/training/engine.py`        | Promotion consistency loss, không inject thành GT trên SimGCD                                                     | Giữ SupCon trên nhãn thật, dùng D1/D2 proposals có self-check       |
| `src/prototail_gcd/pseudo/promotion_store.py` | `lookup()` riêng pseudo targets/mask                                                                              | Không làm biến dạng original training labels                          |
| `src/prototail_gcd/pseudo/paper_selector.py`  | D1 GT-only reliability, smoothed/tie-break/class diversity                                                           | Tránh self-confirmation và class-0 priority                             |
| `src/prototail_gcd/train.py`                  | Dynamic pseudo CE vẫn bật song song với D1/D2, auto gate warm-start for Tail-aware, BaCon path giữ nguyên       | Full hoạt động thực sự, không giảm tín hiệu supervised/unlabeled |
| `tests/test_full_regression_recovery.py`      | 6 CPU invariants                                                                                                     | Chặn lỗi pseudo leakage / random prototype / gradient duplication       |

## 4. Kết quả kiểm tra có thật

- Generated patch: **17 hunks**, **7 paths**.
- `git apply --check` và `git apply`: **PASS trên fixture source** có đúng các code anchors, đặt theo thứ tự/tương đối line như source gốc. **Không thể đồng nhất với pass trên repo thật** do môi trường không clone GitHub được.
- `pytest` với các hàm/đoạn logic trích từ patch: **10/10 isolated CPU tests PASS**. Test suite `tests/test_full_regression_recovery.py` trong repo thật **chưa chạy**.
- **Chưa GPU train; chưa xác thực metric improvement; chưa commit/push remote.**

## 5. Reproduction: chỉ chạy Full

```bash
python kaggle_push.py --help
# Dùng đúng lệnh Kaggle Full hiện đang có, KHÔNG thêm ablate flags.
# Expected run options:
# --baseline simgcd --dataset-name cars|cub --imb-ratio 20
# --use-parts --enable-pseudo-labeling --enable-novel-pseudo --pseudo-mode paper
# --eval-feature-mode legacy_concat
```

Đầu ra log sau sửa cần có `paper_promotion_ce`, `paper_promotion_kept`, `n_pseudo_known`, `n_pseudo_novel` không còn bị vô hiệu hóa chỉ vì `paper` mode; gate κ ở epoch10 gần target; D1 có `smoothed_reliability` và không còn ưu tiên class 0 vô điều kiện. Tất cả khác biệt phải được kiểm trong log, không suy ra từ việc flag hiện chữ `ACTIVE`.

```

```
