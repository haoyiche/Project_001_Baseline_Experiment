# EXP021 - Feature Layer Ablation

## 1. Goal

本实验比较 ResNet18 不同中间层作为 Patch Feature 时，
对 MVTec AD bottle image-level anomaly detection 的影响。

比较对象：

```text
layer2
vs
layer3
```

核心问题：

1. 更高空间分辨率的 layer2 是否更适合 small defect？
2. 更深、上下文范围更大的 layer3 是否更适合结构异常？
3. contamination 对 feature stage 是否敏感？
4. 不同 feature stage 对 stochastic Coreset stability 有什么影响？
5. 固定绝对 Coreset budget 时，哪个 layer 更适合作为当前工程方案？

---

## 2. Experimental Setup

### Dataset

```text
Dataset:
MVTec AD

Category:
bottle
```

### Data Split

严格复用 EXP018：

```text
Reference normal:
167 images

Validation normal:
42 images

Test:
83 images
```

因此数据 split 保持不变。

---

## 3. Fixed Variables

以下变量保持一致：

```text
Backbone:
ResNet18 pretrained

Candidate size:
5000

Coreset size:
100

Seeds:
42, 43, 44, 45, 46

Distance:
Euclidean distance

Image aggregation:
MAX patch anomaly score

Threshold:
Each layer × seed uses its own
validation-normal P95 threshold
```

主动修改变量：

```text
Feature Layer
```

即：

```text
layer2
vs
layer3
```

---

# 4. Feature Geometry

## Layer2

```text
Patch tensor:
[784, 128]

Patch count/image:
784

Feature dimension:
128

Reference full memory:
[130928, 128]
```

其中：

\[
167\times784=130928
\]

---

## Layer3

```text
Patch tensor:
[196, 256]

Patch count/image:
196

Feature dimension:
256

Reference full memory:
[32732, 256]
```

其中：

\[
167\times196=32732
\]

---

## Shape Sanity Check

Expected:

```text
layer2 = [784,128]
layer3 = [196,256]
```

Actual:

```text
layer2 = [784,128]
layer3 = [196,256]
```

Result:

```text
PASS
```

---

# 5. Prediction Before Experiment

## Prediction 1

layer2 具有更高空间分辨率。

因此原预测：

```text
broken_small
可能在 layer2 上更有优势
```

原因是假设 small localized defects 更依赖细粒度 spatial representation。

---

## Prediction 2

layer3 特征更深，具有更大的上下文范围。

因此预测：

```text
broken_large
可能从 layer3 获益
```

但该结果必须由实验验证。

---

## Prediction 3

contamination 的趋势不预设。

可能存在两种情况：

```text
如果细粒度 texture 更重要：
layer2 可能更好

如果更大上下文更重要：
layer3 可能更好
```

---

## Prediction 4

不比较 layer2 / layer3 的 raw score、
P95 threshold 或 coverage radius 的绝对大小。

因为：

```text
feature dimension 不同
feature distribution 不同
distance scale 不同
```

跨 layer 的 raw Euclidean score 不在同一个数值尺度。

---

## Prediction 5

layer2 seed42 应精确复现 EXP018：

```text
AUROC = 0.984921
```

否则本次实验不具有可靠可比性。

---

## Prediction 6

两个 layer 都必须进行 repeated-seed evaluation。

不能使用单个最佳 seed 作为稳定性能。

---

# 6. Layer2 Seed42 Reproduction Check

EXP018：

```text
Overall AUROC:
0.984921
```

EXP021 layer2 seed42：

```text
Overall AUROC:
0.984921
```

Absolute Delta：

```text
0.00000000
```

Result：

```text
PASS = True
```

因此本次 layer2 pipeline 与 EXP018 成功对齐。

---

# 7. Layer2 Results

## Seed 42

```text
Overall AUROC:
0.984921

AP:
0.995441

P95 Threshold:
4.214628
```

Per-defect AUROC：

```text
broken_large     1.000000
broken_small     1.000000
contamination    0.954762
```

P95：

```text
TP = 56
FP = 0
TN = 20
FN = 7

Recall = 0.888889
FPR    = 0.000000
```

---

## Seed 43

```text
Overall AUROC:
0.985714

AP:
0.995356

P95 Threshold:
4.267017
```

Per-defect AUROC：

```text
broken_large     0.995000
broken_small     0.990909
contamination    0.971429
```

P95：

```text
TP = 51
FP = 0
TN = 20
FN = 12

Recall = 0.809524
FPR    = 0.000000
```

---

## Seed 44

```text
Overall AUROC:
0.969048

AP:
0.990958

P95 Threshold:
4.180562
```

Per-defect AUROC：

```text
broken_large     1.000000
broken_small     0.997727
contamination    0.909524
```

P95：

```text
TP = 56
FP = 0
TN = 20
FN = 7

Recall = 0.888889
FPR    = 0.000000
```

---

## Seed 45

```text
Overall AUROC:
0.948413

AP:
0.981786

P95 Threshold:
4.119151
```

Per-defect AUROC：

```text
broken_large     0.957500
broken_small     0.956818
contamination    0.930952
```

P95：

```text
TP = 58
FP = 3
TN = 17
FN = 5

Recall = 0.920635
FPR    = 0.150000
```

---

## Seed 46

```text
Overall AUROC:
0.957143

AP:
0.984011

P95 Threshold:
4.257982
```

Per-defect AUROC：

```text
broken_large     0.960000
broken_small     0.968182
contamination    0.942857
```

P95：

```text
TP = 54
FP = 3
TN = 17
FN = 9

Recall = 0.857143
FPR    = 0.150000
```

---

# 8. Layer2 Repeated-Seed Summary

## Overall AUROC

```text
Mean = 0.969048
Std  = 0.016562
Min  = 0.948413
Max  = 0.985714
```

## Overall AP

```text
Mean = 0.989511
Std  = 0.006351
Min  = 0.981786
Max  = 0.995441
```

## broken_large AUROC

```text
Mean = 0.982500
Std  = 0.021794
Min  = 0.957500
Max  = 1.000000
```

## broken_small AUROC

```text
Mean = 0.982727
Std  = 0.019191
Min  = 0.956818
Max  = 1.000000
```

## contamination AUROC

```text
Mean = 0.941905
Std  = 0.023486
Min  = 0.909524
Max  = 0.971429
```

## P95 Recall

```text
Mean = 0.873016
Std  = 0.041996
Min  = 0.809524
Max  = 0.920635
```

## P95 FPR

```text
Mean = 0.060000
Std  = 0.082158
Min  = 0.000000
Max  = 0.150000
```

---

# 9. Layer3 Results

## Seed 42

```text
Overall AUROC:
1.000000

AP:
1.000000

P95 Threshold:
2.893785
```

Per-defect AUROC：

```text
broken_large     1.000000
broken_small     1.000000
contamination    1.000000
```

P95：

```text
TP = 63
FP = 0
TN = 20
FN = 0

Recall = 1.000000
FPR    = 0.000000
```

---

## Seed 43

```text
Overall AUROC:
0.990476

AP:
0.997283

P95 Threshold:
3.028658
```

Per-defect AUROC：

```text
broken_large     1.000000
broken_small     1.000000
contamination    0.971429
```

P95：

```text
TP = 61
FP = 1
TN = 19
FN = 2

Recall = 0.968254
FPR    = 0.050000
```

---

## Seed 44

```text
Overall AUROC:
0.996825

AP:
0.998984

P95 Threshold:
3.052204
```

Per-defect AUROC：

```text
broken_large     1.000000
broken_small     1.000000
contamination    0.990476
```

P95：

```text
TP = 63
FP = 1
TN = 19
FN = 0

Recall = 1.000000
FPR    = 0.050000
```

---

## Seed 45

```text
Overall AUROC:
1.000000

AP:
1.000000

P95 Threshold:
3.130998
```

Per-defect AUROC：

```text
broken_large     1.000000
broken_small     1.000000
contamination    1.000000
```

P95：

```text
TP = 62
FP = 0
TN = 20
FN = 1

Recall = 0.984127
FPR    = 0.000000
```

---

## Seed 46

```text
Overall AUROC:
0.999206

AP:
0.999752

P95 Threshold:
2.962546
```

Per-defect AUROC：

```text
broken_large     1.000000
broken_small     1.000000
contamination    0.997619
```

P95：

```text
TP = 62
FP = 0
TN = 20
FN = 1

Recall = 0.984127
FPR    = 0.000000
```

---

# 10. Layer3 Repeated-Seed Summary

## Overall AUROC

```text
Mean = 0.997302
Std  = 0.004031
Min  = 0.990476
Max  = 1.000000
```

## Overall AP

```text
Mean = 0.999204
Std  = 0.001152
Min  = 0.997283
Max  = 1.000000
```

## broken_large AUROC

```text
Mean = 1.000000
Std  = 0.000000
Min  = 1.000000
Max  = 1.000000
```

## broken_small AUROC

```text
Mean = 1.000000
Std  = 0.000000
Min  = 1.000000
Max  = 1.000000
```

## contamination AUROC

```text
Mean = 0.991905
Std  = 0.012094
Min  = 0.971429
Max  = 1.000000
```

## P95 Recall

```text
Mean = 0.987302
Std  = 0.013280
Min  = 0.968254
Max  = 1.000000
```

## P95 FPR

```text
Mean = 0.020000
Std  = 0.027386
Min  = 0.000000
Max  = 0.050000
```

---

# 11. Overall Comparison

| Metric | layer2 | layer3 |
|---|---:|---:|
| Overall AUROC mean | 0.969048 | 0.997302 |
| Overall AUROC std | 0.016562 | 0.004031 |
| Overall AP mean | 0.989511 | 0.999204 |
| broken_large AUROC | 0.982500 | 1.000000 |
| broken_small AUROC | 0.982727 | 1.000000 |
| contamination AUROC | 0.941905 | 0.991905 |
| P95 Recall | 0.873016 | 0.987302 |
| P95 FPR | 0.060000 | 0.020000 |

Mean AUROC difference：

\[
0.997302-0.969048
=
+0.028254
\]

即 layer3 在当前实验协议下平均高：

```text
2.8254 AUROC percentage points
```

---

# 12. Prediction vs Result

## Prediction 1 - broken_small 更偏向 layer2

Prediction：

```text
layer2 可能因为更高空间分辨率而更好
```

Actual：

```text
layer2:
0.982727 ± 0.019191

layer3:
1.000000 ± 0.000000
```

Result：

```text
FAILED
```

这说明：

> 更高 spatial resolution 不自动意味着更好的 image-level small-defect ranking。

---

## Important Boundary

该结果不能说明：

```text
layer3 localization 更准确
```

因为本实验的指标：

```text
image-level AUROC
```

只判断：

```text
defect image score
vs
normal image score
```

是否排序正确。

它不判断：

```text
anomaly location
是否和 GT mask 对齐
```

因此：

```text
Higher image-level AUROC
!=
Better localization
```

---

# 13. broken_large Analysis

layer2：

```text
Mean AUROC:
0.982500
```

layer3：

```text
Mean AUROC:
1.000000
```

五个 layer3 seeds：

```text
全部 = 1.000000
```

结果与下面的 hypothesis 一致：

> 更深层、具有更大上下文范围的 feature stage 可能更有利于识别较大的结构变化。

但是：

```text
该 hypothesis 尚未被纯因果隔离
```

因为 feature stage 改变时还同时改变了：

```text
feature dimension
patch count
memory density
representation depth
```

---

# 14. contamination Analysis

这是本实验最明显的变化之一。

layer2：

```text
Mean AUROC:
0.941905

Std:
0.023486
```

layer3：

```text
Mean AUROC:
0.991905

Std:
0.012094
```

Mean improvement：

\[
0.991905-0.941905
=
0.050000
\]

即：

```text
+5.0 AUROC percentage points
```

因此：

> contamination weakness 对 feature stage 高度敏感。

结合 EXP020：

```text
Aggregation 也能够显著改变 contamination performance
```

因此当前不能把 contamination failure 简化为：

```text
只有 representation 问题
```

更合理的理解是：

```text
Failure behavior
=
feature representation
+
memory construction
+
aggregation
+
threshold calibration
```

各部分都有可能贡献。

---

# 15. Seed Stability

layer2 Overall AUROC std：

```text
0.016562
```

layer3：

```text
0.004031
```

layer3 的随机 seed sensitivity 明显更低。

Std ratio：

\[
\frac{0.004031}{0.016562}
\approx0.243
\]

因此 layer3 AUROC std 大约只有 layer2 的：

```text
24.3%
```

或可表述为：

```text
约降低 75.7%
```

因此在当前实验协议下：

> layer3 不仅 mean performance 更高，同时 repeated-seed stability 也更好。

---

# 16. Critical Confound - Relative Memory Density

本实验两层都固定：

```text
Candidate size = 5000
Coreset size   = 100
```

但是 Full Memory 不相同。

layer2：

```text
130928 patches
```

layer3：

```text
32732 patches
```

---

## Candidate Sampling Ratio

layer2：

\[
5000/130928
\approx3.82\%
\]

layer3：

\[
5000/32732
\approx15.28\%
\]

因此 layer3 candidate pool 占 full memory 的比例约是 layer2 的 4 倍。

---

## Coreset Retention Ratio

layer2：

\[
100/130928
\approx0.076\%
\]

layer3：

\[
100/32732
\approx0.306\%
\]

layer3 的 relative retention ratio 同样约为 layer2 的 4 倍。

---

# 17. Why This Is a Confound

如果实验问题是：

```text
纯粹比较 layer2 representation
和 layer3 representation
```

那么性能差异可能同时来自：

```text
Feature Stage
+
Relative Memory Density
```

因此不能把当前：

```text
layer3 > layer2
```

完全解释为：

```text
layer3 representation 本身更优秀
```

需要进一步控制：

```text
Candidate sampling ratio
Coreset retention ratio
```

才能做更纯的 representation comparison。

---

# 18. Scientific Fairness vs Engineering Fairness

这里存在两种不同但都合理的实验问题。

## Scientific / Causal Question

问题：

> 单纯比较 representation quality 时谁更好？

此时应该尽量保持：

```text
relative sampling ratio
relative coreset retention ratio
```

一致。

这种设计更适合：

```text
causal attribution
```

---

## Engineering / Deployment Question

问题：

> 系统只能保存 100 个 memory vectors，哪个方案效果更好？

此时：

```text
layer2 Coreset100
layer3 Coreset100
```

恰好是公平比较。

因为工程资源限制就是：

```text
100 vectors
```

layer3 full memory 更小、能够用相同绝对预算覆盖更大的相对 feature space，

属于：

```text
layer3 自身的工程优势
```

不应该人为消除。

---

# 19. Correct Interpretation of EXP021

当前可以支持的结论：

> 在相同绝对 Candidate5000 / Coreset100 memory budget 下，
> layer3 在 MVTec bottle image-level anomaly detection 上
> 比 layer2 获得更高、更稳定的性能。

---

当前不能支持：

> layer3 representation 本身已被纯因果证明一定优于 layer2。

因为：

```text
relative memory density
```

尚未控制一致。

---

# 20. Image-level Detection vs Localization

layer3：

```text
broken_small AUROC = 1.000000
```

只能说明：

```text
broken_small defect images
被正确排在 normal images 前面
```

不能说明：

```text
layer3 anomaly map
和 GT mask 对齐更精确
```

如果要回答 localization：

需要额外评估：

```text
Pixel AUROC
PRO
IoU
GT overlap
Top anomaly location vs GT
```

当前 EXP021 没有测这些指标。

---

# 21. Raw Distance Scale Warning

本实验中 layer2 和 layer3：

```text
feature dimension:
128 vs 256
```

feature distribution 也不同。

因此不能直接使用：

```text
layer2 threshold = ...
layer3 threshold = ...
```

的数值大小判断：

```text
哪个 layer 更异常
```

跨 embedding space 的 raw Euclidean distance 不具有直接可比性。

应该比较：

```text
AUROC
AP
Recall
FPR
per-defect metrics
```

而不是 raw threshold magnitude。

---

# 22. What Was Truly Learned

## 1. Spatial Resolution != Image-level Detection Performance

更细 spatial grid：

```text
不保证
```

更高 image-level small-defect AUROC。

---

## 2. Image-level Detection != Localization

AUROC 高：

```text
只能说明 ranking 更好
```

不能说明：

```text
GT localization 更准
```

---

## 3. Feature Stage Strongly Affects Failure Modes

尤其 contamination：

```text
layer2:
0.941905

layer3:
0.991905
```

说明 failure mode 与 representation stage 明显相关。

---

## 4. Mean Performance and Stability Must Both Be Checked

layer3：

```text
Mean 更高
Std 更低
```

比只观察单个最佳 seed 更有说服力。

---

## 5. Fair Comparison Depends on the Question

如果研究：

```text
representation causality
```

需要控制 relative memory ratio。

如果研究：

```text
fixed deployment budget
```

相同 absolute Coreset size 是合理公平条件。

---

# 23. Limitations

### Limitation 1

只测试：

```text
MVTec bottle
```

不能直接推广至所有工业异常类别。

---

### Limitation 2

只比较：

```text
layer2
layer3
```

没有测试：

```text
layer1
layer4
multi-layer fusion
```

---

### Limitation 3

Candidate5000 / Coreset100 是固定绝对 budget。

没有控制：

```text
relative compression ratio
```

---

### Limitation 4

只评价：

```text
image-level anomaly detection
```

没有评价 pixel-level localization。

---

### Limitation 5

当前 aggregation 固定为：

```text
MAX
```

尚未测试：

```text
layer3 + Top-K aggregation
```

因此 EXP020 和 EXP021 的 improvement 不能直接相加解释。

---

### Limitation 6

仅 5 个 seeds。

能够观察稳定性趋势，但仍不是非常精确的 stochastic distribution estimate。

---

# 24. Main Conclusions

### Conclusion 1

在固定 absolute Candidate5000 / Coreset100 budget 下：

```text
layer3
```

明显优于 layer2。

---

### Conclusion 2

Overall AUROC：

```text
layer2:
0.969048 ± 0.016562

layer3:
0.997302 ± 0.004031
```

Difference：

```text
+0.028254
```

---

### Conclusion 3

原本预测：

```text
layer2 更适合 broken_small
```

被实验否定。

layer3：

```text
broken_small AUROC
= 1.000000 ± 0.000000
```

---

### Conclusion 4

该结果只说明：

```text
image-level ranking
```

不代表：

```text
localization quality
```

---

### Conclusion 5

contamination 获得最大改善之一：

```text
0.941905
→
0.991905
```

说明 feature stage 是 contamination failure 的重要影响因素。

---

### Conclusion 6

layer3 seed stability 明显更强：

```text
AUROC std:
0.016562
→
0.004031
```

---

### Conclusion 7

EXP021 存在 relative-memory-density confound。

因此当前最准确结论为：

> 在固定绝对 memory budget 下，layer3 是当前 Bottle anomaly detection pipeline 中更强、更稳定的 feature stage。

而不是：

> layer3 representation 已被纯因果证明优于 layer2。

---

# 25. Pass Test

能够解释：

```text
为什么 broken_small AUROC 高
不意味着 localization 好？
```

PASS。

能够解释：

```text
为什么固定 Coreset100
不是相同 relative compression ratio？
```

PASS。

能够区分：

```text
scientific comparison
vs
engineering fixed-budget comparison
```

PASS。

能够解释：

```text
为什么不能比较两个 layer
raw threshold 的绝对大小？
```

PASS。

---

# 26. Status

```text
Feature Shape Check                  PASS

Layer2 Seed42 Reproduction           PASS

Repeated Seeds                       PASS

Overall Metrics                      PASS

Per-defect Metrics                   PASS

Prediction vs Verification           PASS

Image-level vs Localization          PASS

Relative Memory Confound             PASS

Scientific vs Engineering Fairness   PASS

EXP021                               PASS

LEVEL 1                              CONTINUE

Git                                  NOT YET COMMITTED
```

---

# 27. Next Step

在进入下一实验前：

```text
1. 保存本实验记录
2. 检查 EXP021 source / CSV / note
3. Git selective staging
4. Commit
5. GitHub push
```

之后再决定：

```text
A. 补 equal-relative-memory control
```

还是：

```text
B. 结束当前 anomaly ablation branch，
   转入 Level 1 剩余模块
```

当前不应因为 layer3 得到接近 1.0 AUROC 就继续无止境优化 Bottle。

实验目标已经从：

```text
追更高数字
```

转为：

```text
建立可解释、可复现、可迁移的实验能力
```