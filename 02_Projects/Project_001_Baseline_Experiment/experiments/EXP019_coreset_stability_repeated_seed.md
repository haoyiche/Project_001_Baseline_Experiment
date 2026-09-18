# EXP019 - Coreset Stability / Repeated Seed Ablation

## 1. Goal

本实验目标是验证：

```text
Patch Coreset100 的性能
是否对 stochastic coreset construction 敏感？
```

前置现象：

```text
EXP017 Patch AUROC = 0.947619
EXP018 Patch AUROC = 0.984921
```

两次实验相差：

\[
0.984921-0.947619
=
0.037302
\]

该差异已经不能简单视为微小波动。

因此 EXP019 固定：

```text
Reference Split
Validation Split
Test Set
Backbone
Feature Layer
Candidate Size
Coreset Size
Image Scoring
```

仅修改 stochastic construction seed：

```text
42
43
44
45
46
```

并统计：

```text
Overall AUROC / AP
Per-defect AUROC
P95 Threshold
P95 Recall / FPR / F1
Coverage Radius
Mean / Std / Min / Max
```

核心问题：

1. 单次 Coreset100 是否能够代表稳定性能？
2. seed=42 能否精确复现 EXP018？
3. 不同 defect type 对 Coreset composition 的敏感性是否不同？
4. K-center geometric coverage 是否能够预测 downstream AUROC？
5. Ranking metric 与 calibrated operating point 是否具有相同稳定性？

---

## 2. Experimental Setup

### Fixed Variables

```text
Dataset:
MVTec AD

Category:
bottle

Reference Normal:
167 images

Validation Normal:
42 images

Test:
83 images

Backbone:
ResNet18

Feature Layer:
layer2

Patch Feature Dimension:
128

Feature Map:
28 × 28

Patches Per Image:
784

Candidate Size:
5000

Coreset Size:
100

Image Score:
max patch NN distance

Device:
CPU
```

EXP018 的 exact split 被直接复用。

因此：

```text
Reference Split 固定
Validation Split 固定
Test Set 固定
```

---

## 3. Independent Variable

本实验主动修改：

```text
seed = 42, 43, 44, 45, 46
```

需要注意：

该 seed 同时影响：

```text
Candidate Sampling
+
K-center Initial Center
```

所以本实验测试的是：

> 整个 stochastic coreset construction pipeline 的稳定性。

它不是：

```text
仅 Candidate Sampling Ablation
```

也不是：

```text
仅 First-center Ablation
```

若以后需要区分两种随机来源，应进一步做独立控制实验。

---

## 4. Feature Extraction Strategy

为了避免 repeated experiment 中重复运行 Backbone：

```text
167 Reference Images
42 Validation Images
83 Test Images
```

全部 Patch Features 只提取一次。

之后不同 seed 仅重新执行：

```text
Candidate Sampling
↓
K-center Selection
↓
Coreset100
↓
NN Scoring
```

这样确保不同 runs 的 Feature Representation 完全相同。

---

## 5. Reference Memory Sanity Check

Reference：

```text
167 images
```

每张：

```text
784 patches
```

因此：

\[
167\times784
=
130928
\]

实际：

```text
torch.Size([130928,128])
```

Expected：

```text
130928
```

完全一致。

Sanity Check：

```text
PASS
```

---

## 6. K-center Objective

对于 Candidate Set：

\[
X
\]

以及已选择 Coreset：

\[
C
\]

Coverage Radius：

\[
R(C)
=
\max_{x\in X}
\min_{c\in C}
\|x-c\|_2
\]

K-center Greedy 每一步选择当前离已有 Centers 最远的 Candidate。

目标是降低：

```text
Worst-case Normal Feature Coverage Radius
```

但该目标与：

```text
Anomaly Detection AUROC
```

并不是同一个 Objective。

EXP019 将验证二者是否具有简单对应关系。

---

## 7. Prediction

### Prediction 1

seed=42 应复现 EXP018：

```text
Patch AUROC ≈ 0.984921
```

若明显不一致，需要停止实验并检查：

```text
split
preprocessing
feature extraction
candidate sampling
K-center implementation
scoring
```

---

### Prediction 2

不同 seed 的 AUROC 不会完全相同。

---

### Prediction 3

如果 AUROC std 很小：

```text
Coreset100 较稳定
```

如果 std 较大：

```text
单次 run 不足以代表方法性能
```

---

### Prediction 4

每个 seed 必须根据自己的 Validation Normal Scores 重新计算：

```text
P95 Threshold
```

不能跨 seed 共用一个 threshold。

---

### Prediction 5

根据 EXP016～EXP018，预计：

```text
contamination
```

可能继续是较弱 defect type。

是否同时具有更大的 seed variance，由实际结果验证。

---

### Prediction 6

更小 Coverage Radius 不一定对应更高 AUROC。

因为：

```text
Feature-space coverage
```

与：

```text
Good / Defect Ranking
```

不是相同的优化目标。

---

# 8. Run Results

## Seed 42

```text
Coverage Radius:
3.511911

P95 Threshold:
4.214628

Overall AUROC:
0.984921

Overall AP:
0.995441
```

Per-defect：

```text
broken_large    1.000000
broken_small    1.000000
contamination   0.954762
```

P95：

```text
TP = 56
FP = 0
FN = 7

Recall = 0.888889
FPR    = 0.000000
F1     = 0.941176
```

---

## Seed 43

```text
Coverage Radius:
3.462558

P95 Threshold:
4.267017

Overall AUROC:
0.985714

Overall AP:
0.995356
```

Per-defect：

```text
broken_large    0.995000
broken_small    0.990909
contamination   0.971429
```

P95：

```text
TP = 51
FP = 0
FN = 12

Recall = 0.809524
FPR    = 0.000000
F1     = 0.894737
```

---

## Seed 44

```text
Coverage Radius:
3.515168

P95 Threshold:
4.180562

Overall AUROC:
0.969048

Overall AP:
0.990958
```

Per-defect：

```text
broken_large    1.000000
broken_small    0.997727
contamination   0.909524
```

P95：

```text
TP = 56
FP = 0
FN = 7

Recall = 0.888889
FPR    = 0.000000
F1     = 0.941176
```

---

## Seed 45

```text
Coverage Radius:
3.524106

P95 Threshold:
4.119151

Overall AUROC:
0.948413

Overall AP:
0.981786
```

Per-defect：

```text
broken_large    0.957500
broken_small    0.956818
contamination   0.930952
```

P95：

```text
TP = 58
FP = 3
FN = 5

Recall = 0.920635
FPR    = 0.150000
F1     = 0.935484
```

---

## Seed 46

```text
Coverage Radius:
3.461070

P95 Threshold:
4.257982

Overall AUROC:
0.957143

Overall AP:
0.984011
```

Per-defect：

```text
broken_large    0.960000
broken_small    0.968182
contamination   0.942857
```

P95：

```text
TP = 54
FP = 3
FN = 9

Recall = 0.857143
FPR    = 0.150000
F1     = 0.900000
```

---

# 9. Seed-42 Reproduction Check

EXP018：

```text
AUROC = 0.984921
```

EXP019 seed42：

```text
AUROC = 0.984921
```

Absolute Delta：

```text
0.00000000
```

Tolerance：

```text
1e-6
```

Result：

```text
Exact Reproduction = True
```

因此 EXP019 与 EXP018 的：

```text
split
feature extraction
candidate construction
K-center
scoring
```

成功对齐。

这使得 seed43～46 的差异可以合理解释为 stochastic pipeline variation。

---

# 10. Overall Stability

五个 seeds：

| Seed | AUROC | AP |
|---:|---:|---:|
| 42 | 0.984921 | 0.995441 |
| 43 | 0.985714 | 0.995356 |
| 44 | 0.969048 | 0.990958 |
| 45 | 0.948413 | 0.981786 |
| 46 | 0.957143 | 0.984011 |

AUROC：

```text
Mean = 0.969048
Std  = 0.016562
Min  = 0.948413
Max  = 0.985714
```

Range：

\[
0.985714-0.948413
=
0.037301
\]

即：

```text
3.73 AUROC percentage points
```

AP：

```text
Mean = 0.989511
Std  = 0.006351
Min  = 0.981786
Max  = 0.995441
```

---

# 11. Main Stability Conclusion

当前：

```text
Coreset100
```

存在不可忽略的 stochastic variance。

因此不能使用：

```text
单次最佳 AUROC
```

作为方法稳定性能。

更合理的报告方式：

```text
Overall AUROC:
0.9690 ± 0.0166

Range:
0.9484 – 0.9857
```

而不是只报告：

```text
0.985714
```

---

# 12. Relationship to EXP017 / EXP018

之前：

```text
EXP017 Patch:
0.947619

EXP018 Patch:
0.984921
```

差：

\[
0.037302
\]

EXP019 固定同一个 Reference Split，仅修改 stochastic coreset construction：

```text
Best:
0.985714

Worst:
0.948413
```

差：

\[
0.037301
\]

两者几乎完全处于同一量级。

因此可以得出：

> Coreset stochasticity 本身已经足以产生与 EXP017→EXP018 性能差异同量级的波动。

但不能进一步声称：

```text
EXP017 与 EXP018 的全部差异
100% 来自 seed
```

因为 EXP017 与 EXP018 仍存在：

```text
Reference Memory
```

等实验协议差异。

---

# 13. Per-defect Stability

## broken_large

```text
Mean = 0.982500
Std  = 0.021794
Min  = 0.957500
Max  = 1.000000
```

Range：

```text
0.042500
```

---

## broken_small

```text
Mean = 0.982727
Std  = 0.019191
Min  = 0.956818
Max  = 1.000000
```

Range：

```text
0.043182
```

---

## contamination

```text
Mean = 0.941905
Std  = 0.023486
Min  = 0.909524
Max  = 0.971429
```

Range：

```text
0.061905
```

---

# 14. Contamination Remains the Main Weakness

三类 defect 的平均 AUROC：

```text
broken_large:
0.982500

broken_small:
0.982727

contamination:
0.941905
```

contamination：

```text
Mean 最低
Std 最高
Range 最大
```

因此当前证据支持：

> contamination 不仅平均检测性能较弱，而且对 Coreset composition 更敏感。

这加强了之前建立的：

```text
Patch Failure Mode #1:
Low-response / unstable contamination
```

---

# 15. Seed44 as a Diagnostic Example

seed44：

```text
broken_large = 1.000000
broken_small = 0.997727
contamination = 0.909524
```

但 Overall：

```text
0.969048
```

如果只查看 Overall AUROC，会遗漏内部结构：

```text
Structural defects:
几乎完美

Contamination:
明显较差
```

因此：

> Overall Metric 必须和 per-defect metrics 联合分析。

---

# 16. Coverage Radius Stability

五次 Coverage Radius：

```text
Mean = 3.494963
Std  = 0.030593
Min  = 3.461070
Max  = 3.524106
```

几何 Coverage 的波动其实不大。

但 AUROC：

```text
Std = 0.016562
Range = 0.037301
```

明显具有 task-level variation。

---

# 17. Coverage Radius Does Not Predict AUROC

seed46：

```text
Coverage Radius:
3.461070
```

为五次最小。

但：

```text
AUROC:
0.957143
```

并不高。

seed43：

```text
Coverage Radius:
3.462558
```

与 seed46 几乎相同。

但：

```text
AUROC:
0.985714
```

两者 AUROC 差：

\[
0.028571
\]

因此：

\[
\boxed{
Better\ geometric\ coverage
\neq
Guaranteed\ better\ anomaly\ ranking
}
\]

原因：

K-center 优化：

\[
\max_x\min_c\|x-c\|
\]

AUROC 衡量：

```text
Good / Defect Relative Ranking
```

目标函数不同。

---

# 18. Possible Mechanism Hypothesis

K-center 倾向保留：

```text
sparse / tail normal patterns
```

这改善 Normal Feature Space 的 worst-case coverage。

但某些 rare normal representatives：

```text
可能与部分 anomaly feature 接近
```

如果被选入 Memory：

```text
anomaly nearest-neighbor distance
可能降低
```

从而降低 anomaly ranking。

该机制目前只是：

```text
Hypothesis
```

尚未通过 controlled experiment 验证。

不能作为最终原因。

---

# 19. P95 Threshold Stability

五次：

```text
Mean = 4.207868
Std  = 0.060564
Min  = 4.119151
Max  = 4.267017
```

说明：

```text
Coreset Composition
```

不仅改变 Test Scores，也改变：

```text
Validation Normal Score Distribution
```

因此不同 stochastic model 必须：

```text
分别 calibration
```

不能共享同一个 numeric threshold。

---

# 20. P95 Recall Stability

```text
Mean = 0.873016
Std  = 0.041996
Min  = 0.809524
Max  = 0.920635
```

说明实际 operating-point sensitivity 也不可忽略。

---

# 21. P95 FPR Stability

```text
Mean = 0.060000
Std  = 0.082158
Min  = 0.000000
Max  = 0.150000
```

该波动同时受到两个因素影响：

1. Coreset composition 不同；
2. Test Good 只有 20 张。

因此：

\[
1\ FP
=
5\%\ FPR
\]

测试集过小导致 FPR 粒度很粗。

---

# 22. AUROC and Operating Point Are Different

最有教育意义的对照：

## Seed43

Overall AUROC：

```text
0.985714
```

五次最高。

但 P95：

```text
Recall = 0.809524
FN = 12
```

五次 Recall 最低。

---

## Seed45

Overall AUROC：

```text
0.948413
```

五次最低。

但 P95：

```text
Recall = 0.920635
FN = 5
```

五次 Recall 最高。

---

因此：

> 高 AUROC 不保证某个具体 calibrated operating point 同样优秀。

因为：

```text
AUROC
=
overall ranking

P95 operating point
=
normal calibration distribution
+
specific threshold
+
test score placement
```

二者评价的是不同问题。

---

# 23. Why Each Seed Needs Its Own Threshold

不同 Coreset：

```text
Memory Composition 不同
```

因此 Validation Normal Scores 发生变化。

如果把 seed42：

```text
P95 = 4.214628
```

直接用于 seed43：

```text
错误
```

因为 seed43 自己的 Normal Distribution 得到：

```text
P95 = 4.267017
```

所以正确流程必须是：

```text
Model / Memory changes
↓
Re-calibrate validation threshold
↓
Test operating point
```

---

# 24. Prediction vs Result

## Prediction 1

seed42 复现 EXP018。

实际：

```text
0.984921
vs
0.984921
```

Verdict：

```text
PASS - Exact Reproduction
```

---

## Prediction 2

不同 seed AUROC 不完全相同。

实际：

```text
0.948413 ~ 0.985714
```

Verdict：

```text
STRONGLY PASS
```

---

## Prediction 3

Repeated runs 用于判断 stability。

实际：

```text
Mean = 0.969048
Std  = 0.016562
Range = 0.037301
```

说明 variance 不可忽略。

Verdict：

```text
PASS
```

---

## Prediction 4

每个 seed 独立校准 P95。

实际：

```text
4.119151 ~ 4.267017
```

不同 seed threshold 确实不同。

Verdict：

```text
PASS
```

---

## Prediction 5

contamination 可能较弱。

实际：

```text
Mean AUROC:
0.941905

Std:
0.023486
```

三类 defect 中平均最低，同时 std 最大。

Verdict：

```text
STRONGLY SUPPORTED
```

---

## Prediction 6

Coverage Radius 不一定预测 AUROC。

实际：

```text
seed43 radius ≈ seed46 radius

AUROC:
0.985714
vs
0.957143
```

Verdict：

```text
STRONGLY PASS
```

---

# 25. What Was Truly Learned

本实验完成后，应能够解释：

### 1. Why Repeated Experiments Matter

单次 stochastic run 可能受到：

```text
sampling luck
```

明显影响。

所以应该报告：

```text
mean
std
range
```

而不是只挑最佳 seed。

---

### 2. Why Exact Reproduction Comes First

如果 seed42 都不能复现 EXP018：

```text
不同 seed comparison
没有可信基础
```

所以：

```text
Reproduction
```

是 repeated experiment 的前提。

---

### 3. Geometry vs Task Metric

```text
Better Normal Coverage
```

不是：

```text
Better Defect Ranking
```

的充分条件。

---

### 4. Ranking vs Operating Point

```text
High AUROC
```

不保证：

```text
High Recall at P95
```

---

### 5. Calibration Belongs to the Model

Memory / Model 改变后：

```text
Threshold 也必须重新 calibration
```

---

# 26. Limitations

### Limitation 1 - Only 5 Seeds

当前：

```text
N = 5
```

能够暴露明显 variance，但还不足以非常精确估计性能分布。

因此：

```text
mean ± std
```

应视为初步稳定性估计。

---

### Limitation 2 - One Fixed Reference Split

本实验固定：

```text
EXP018 Reference Split
```

因此测量的是：

```text
Conditional on this reference split
```

的 Coreset stochasticity。

尚未测量：

```text
Reference Split Variance
```

---

### Limitation 3 - One Seed Controls Two Random Sources

seed 同时改变：

```text
Candidate Sampling
+
K-center Initial Center
```

因此无法判断两者各自贡献多少 variance。

---

### Limitation 4 - Bottle Only

仍只有：

```text
MVTec bottle
```

结果不能直接推广到全部 anomaly categories。

---

### Limitation 5 - Coreset Size Fixed

当前：

```text
Coreset Size = 100
```

尚不知道：

```text
更大 Coreset
```

是否能降低 stochastic variance。

---

# 27. Updated Interpretation of EXP017

EXP017 单次结果：

```text
Global = 0.974603
Patch  = 0.947619
```

在该具体 run 上：

```text
Global > Patch
```

仍然成立。

但 EXP019 表明 Patch：

```text
0.969048 ± 0.016562
```

且单次范围：

```text
0.948413 ~ 0.985714
```

因此不能把 EXP017 单次结果升级成：

```text
Global 方法稳定优于 Patch 方法
```

正确结论应是：

> EXP017 的具体 Patch run 低于 Global，但 Patch Coreset100 存在显著 stochastic variance，因此需要 repeated statistics 才能进行更稳定的方法比较。

---

# 28. Main Conclusions

### Conclusion 1

Coreset100 存在明显 stochastic performance variance：

```text
AUROC:
0.9690 ± 0.0166
```

---

### Conclusion 2

单次最佳：

```text
0.985714
```

不能代表稳定方法性能。

---

### Conclusion 3

EXP018 seed42 被精确复现：

```text
0.984921
```

说明 repeated experiment pipeline 正确。

---

### Conclusion 4

contamination：

```text
平均 AUROC 最低
seed variance 最大
range 最大
```

仍为当前 Patch Pipeline 主要 weakness。

---

### Conclusion 5

Coverage Radius 不能可靠预测 anomaly detection AUROC。

---

### Conclusion 6

Overall AUROC 最好的 run，不一定拥有最好的 calibrated P95 operating point。

---

### Conclusion 7

Stochastic model / memory 改变后，需要重新进行 threshold calibration。

---

# 29. New Questions

1. `max` aggregation 是否导致对极端单 Patch score 过度敏感？
2. Top-K Mean 是否能提高 score robustness？
3. Mean Aggregation 是否会再次稀释 small local defect？
4. contamination 的 weakness 是否与 aggregation strategy 有关？
5. Coreset Size 增大是否能够降低 seed variance？
6. Candidate Sampling 与 first-center randomness 各自贡献多少 variance？

---

# 30. Next Step

下一正式实验：

```text
EXP020
Image Score Aggregation Ablation
```

固定：

```text
Reference Split
Coreset
Patch Features
Test Set
Distance Metric
```

只改变：

```text
MAX
TOP-K MEAN
MEAN
```

目标：

> 分离 Image-level Aggregation Strategy 对不同 anomaly type 的影响。

重点验证：

```text
MAX
是否最适合 localized defects

MEAN
是否再次出现 anomaly dilution

TOP-K MEAN
是否能在 robustness 与 localization sensitivity
之间取得折中
```

---

# 31. Status

```text
Exact Seed42 Reproduction             PASS

Repeated Seeds >= 5                  PASS

Overall AUROC/AP                     PASS

Per-defect Metrics                   PASS

Mean / Std / Min / Max               PASS

Per-seed Calibration                 PASS

Operating-point Stability            PASS

Coverage-vs-AUROC Analysis           PASS

Repeated Experiment Understanding    PASS

Coreset Stability Issue              CONFIRMED

Low-response / unstable contamination CONFIRMED

EXP019                               PASS

LEVEL 1                              CONTINUE

Git                                  暂未提交
```