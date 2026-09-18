# EXP016 - Full Bottle Patch-Level Evaluation

## 1. Goal

本实验的目标是：

将 EXP015 中基于 40 张图像的 mini evaluation 扩展到完整 MVTec AD `bottle/test` 数据集，验证 Random Sampling 与 K-center Coreset 的结论能否推广到全部 defect type。

本实验重点比较：

```text
Candidate5000
Random100
Coreset100
```

在完整 Bottle Test Set 上的 image-level anomaly detection performance。

核心问题：

1. EXP015 中 Random100 性能退化的现象，是否会在完整测试集上继续出现？
2. Coreset100 是否仍然能够在只保留 100 个 Patch 的情况下接近 Candidate5000？
3. `broken_large`、`broken_small`、`contamination` 三种 defect type 的检测难度是否相同？
4. 哪些正常样本持续获得高 anomaly score？
5. 哪些异常样本持续获得低 anomaly score？
6. 当前 Patch Baseline 的主要 failure mode 是什么？

---

## 2. Environment

```text
Project:
Project_001_Baseline_Experiment

Conda Environment:
ai_base

Framework:
PyTorch

Backbone:
ResNet18

Pretrained Weights:
ImageNet

Feature Layer:
layer2

Patch Feature Dimension:
128

Input Size:
224 × 224

Feature Map:
28 × 28

Patch Count Per Image:
784

Device:
CPU

Dataset:
MVTec AD

Category:
bottle
```

---

## 3. Experimental Setup

完整 Normal Memory Bank：

```text
[163856,128]
```

来源：

```text
209 train/good images
×
784 patches/image
=
163856 normal patches
```

EXP014 已从 Full Memory Bank 中构造：

```text
Candidate5000:
[5000,128]

Random100:
[100,128]

Coreset100:
[100,128]
```

三种 Memory Strategy：

### Candidate5000

```text
5000 Normal Patches
```

作为本实验中的较大 Reference Bank。

### Random100

从 Candidate5000 中随机选择：

```text
100 patches
```

### Coreset100

从相同 Candidate5000 中使用 K-center Greedy 选择：

```text
100 representative patches
```

因此：

```text
Random100
和
Coreset100
```

具有相同：

```text
Candidate Pool
Memory Size
Feature Dimension
Backbone
Feature Layer
Distance Metric
Image Score Aggregation
```

主要变量只有：

```text
Memory Selection Strategy
```

---

## 4. Full Bottle Test Set

完整测试集：

```text
good           20
broken_large   20
broken_small   22
contamination  21

Total          83
```

Label 定义：

```text
good     = 0
defect   = 1
```

所有三种 defect type 均属于 anomaly。

---

## 5. Patch-Level Anomaly Score

对于一个 Query Patch：

\[
q_i\in\mathbb{R}^{128}
\]

以及 Normal Memory Bank：

\[
M=\{m_1,m_2,\ldots,m_N\}
\]

Patch anomaly score：

\[
s(q_i)
=
\min_{m\in M}
\|q_i-m\|_2
\]

含义：

```text
Query Patch
↓
与 Normal Memory 中所有 Patch 比较
↓
寻找最近 Normal Feature
↓
Nearest Neighbor Distance
↓
Patch Anomaly Score
```

如果 Query Patch 与某种正常 Pattern 相似：

```text
NN distance 小
→
anomaly score 低
```

如果无法被 Normal Memory 很好解释：

```text
NN distance 大
→
anomaly score 高
```

---

## 6. Image-Level Anomaly Score

每张 224×224 图像在 layer2 得到：

```text
28 × 28
=
784 patches
```

因此：

```text
784 patch anomaly scores
```

当前 minimal Patch Baseline 使用：

\[
S(x)
=
\max_i s(q_i)
\]

即：

```text
Image
↓
784 Patch Scores
↓
Maximum
↓
Image Anomaly Score
```

该设计延续 EXP011 的实验结论：

```text
Small Local Defect
可能只影响少数 Patch

Mean Aggregation
可能发生局部异常稀释

Max Aggregation
能够保留最强局部异常响应
```

注意：

当前仍属于：

```text
Minimal PatchCore-style Baseline
```

并非完整 PatchCore 复现。

---

## 7. Prediction

### Prediction 1

Random100 因 Normal Feature Coverage 丢失，预计会：

```text
提高正常样本 anomaly score
```

因此 Overall AUROC / AP 预计明显弱于 Coreset100。

---

### Prediction 2

EXP015 中：

```text
Candidate5000 AUROC = 0.9575
Coreset100 AUROC    = 0.9550
```

两者在 broken_small mini evaluation 上非常接近。

但完整 Bottle 还包含：

```text
broken_large
contamination
```

因此不假设 Coreset100 在所有 defect type 上都一定接近 Candidate5000。

---

### Prediction 3

由于 EXP014 已验证 Coreset 改善 Normal Feature Space 的 Tail / Worst-case Coverage，因此预计：

```text
Coreset100 Good P95
<
Random100 Good P95
```

以及：

```text
Coreset100 Good Max
<
Random100 Good Max
```

---

### Prediction 4

不同 defect type 的检测难度预计不同。

因此需要分别计算：

```text
good vs broken_large
good vs broken_small
good vs contamination
```

而不能只依赖 Overall AUROC。

---

### Prediction 5

当前没有正式 Threshold。

因此：

```text
highest-scoring good
```

仍应称为：

```text
High-scoring Normal Candidate
FP-risk Candidate
```

而：

```text
lowest-scoring defect
```

应称为：

```text
Low-scoring Defect Candidate
Missed-risk Candidate
```

尚不能正式称为 FP / FN。

---

## 8. Core Pipeline

实验 Pipeline：

```text
Full Normal Memory
[163856,128]
        ↓
读取 EXP014 indices
        ↓
重建
Candidate5000
Random100
Coreset100
        ↓
扫描完整 bottle/test
        ↓
83 Test Images
        ↓
每张图提取一次 layer2 feature
        ↓
[784,128]
        ↓
分别与三个 Memory Bank 做 NN Search
        ↓
[784] Patch Scores
        ↓
max()
        ↓
Image-level Anomaly Score
        ↓
Overall Metrics
        ↓
Per-defect Metrics
        ↓
Failure Candidate Analysis
```

---

## 9. Why Query Feature Is Extracted Only Once

每张图：

```python
query_patches = extract_image_patches(...)
```

只进行一次 Feature Extraction。

得到：

```text
[784,128]
```

然后同一份 Query Feature 分别送入：

```text
Candidate5000
Random100
Coreset100
```

这样保证：

```text
Feature Representation 完全相同
```

实验只比较：

```text
Memory Selection Strategy
```

同时避免重复 Backbone Forward。

---

## 10. Evaluation Metrics

本实验包含：

```text
Overall AUROC

Average Precision

Good Mean
Good P95
Good Max

Defect Min
Defect Mean
Defect Max

Separation Margin

Per-defect AUROC
Per-defect AP

Top-5 highest-scoring good
Top-5 lowest-scoring defect
```

---

## 11. Overall Result - Candidate5000

```text
Overall AUROC:
0.944444

Average Precision:
0.978732
```

GOOD：

```text
Mean:
3.320070

P95:
4.214130

Max:
4.331443
```

ALL DEFECTS：

```text
Min:
3.460963

Mean:
4.378356

Max:
5.956269
```

Separation Margin：

\[
3.460963-4.331443
=
-0.870480
\]

结果：

```text
Margin < 0
```

说明 Good / Defect Score Distribution 存在 overlap。

---

## 12. Overall Result - Random100

```text
Overall AUROC:
0.701587

Average Precision:
0.895593
```

GOOD：

```text
Mean:
4.940541

P95:
5.095362

Max:
5.512643
```

ALL DEFECTS：

```text
Min:
4.699180

Mean:
5.190648

Max:
6.244522
```

Separation Margin：

```text
-0.813463
```

Random100 相比 Candidate5000：

```text
Good Mean:
3.320070
→
4.940541
```

正常样本 anomaly score 被明显抬高。

---

## 13. Overall Result - Coreset100

```text
Overall AUROC:
0.947619

Average Precision:
0.978718
```

GOOD：

```text
Mean:
3.777570

P95:
4.361700

Max:
4.635723
```

ALL DEFECTS：

```text
Min:
3.836864

Mean:
4.641121

Max:
6.448766
```

Separation Margin：

```text
-0.798858
```

---

## 14. Overall Comparison

| Method | Memory Size | AUROC | AP | Good Mean | Good P95 | Good Max |
|---|---:|---:|---:|---:|---:|---:|
| Candidate5000 | 5000 | 0.944444 | 0.978732 | 3.320070 | 4.214130 | 4.331443 |
| Random100 | 100 | 0.701587 | 0.895593 | 4.940541 | 5.095362 | 5.512643 |
| Coreset100 | 100 | 0.947619 | 0.978718 | 3.777570 | 4.361700 | 4.635723 |

核心结果：

```text
Candidate5000 AUROC:
0.944444

Coreset100 AUROC:
0.947619
```

两者非常接近。

Random100：

```text
0.701587
```

明显更差。

---

## 15. Memory Compression

Candidate5000：

```text
5000 patches
```

Coreset100：

```text
100 patches
```

Memory Reduction：

\[
5000/100=50
\]

即：

```text
50× reduction
```

或者：

```text
仅保留 2% Candidate Memory
```

而 AUROC：

```text
Candidate5000:
0.944444

Coreset100:
0.947619
```

在当前完整 Bottle Test Set 上几乎没有性能损失。

因此可以描述为：

> Coreset100 在当前实验中使用 Candidate5000 仅 2% 的 Memory，仍基本保持其 image-level ranking performance。

不能描述为：

```text
无损压缩
```

因为 Good Score Distribution 仍有变化。

---

## 16. Random100 Failure Mechanism

Random100：

```text
Good Mean:
4.940541
```

Coreset100：

```text
Good Mean:
3.777570
```

Candidate5000：

```text
Good Mean:
3.320070
```

因此 Random100 的核心问题不是：

```text
Defect Score 太低
```

而是：

```text
Normal Score 被显著抬高
```

完整机制：

```text
Random Reduction
↓
部分 Normal Pattern 未被保留
↓
Normal Query 找不到合适最近邻
↓
Nearest Neighbor Distance 增大
↓
Normal Patch Score 增大
↓
Good Image Max Score 增大
↓
Good / Defect Ranking 被破坏
↓
AUROC 下降
```

这与 EXP013～EXP015 的结论一致。

---

## 17. Per-Defect Result - Candidate5000

### broken_large

```text
AUROC = 0.947500
AP    = 0.935220
```

### broken_small

```text
AUROC = 0.961364
AP    = 0.961010
```

### contamination

```text
AUROC = 0.923810
AP    = 0.926726
```

Candidate5000 最困难的 defect type：

```text
contamination
```

---

## 18. Per-Defect Result - Random100

### broken_large

```text
AUROC = 0.600000
AP    = 0.661999
```

### broken_small

```text
AUROC = 0.597727
AP    = 0.703168
```

### contamination

```text
AUROC = 0.907143
AP    = 0.908446
```

Random100 出现非常明显的 defect-dependent degradation：

```text
broken_large:
严重退化

broken_small:
严重退化

contamination:
仍保持相对较高 ranking performance
```

因此：

> Memory Reduction 并不会对所有 defect type 产生相同影响。

---

## 19. Per-Defect Result - Coreset100

### broken_large

```text
AUROC = 0.952500
AP    = 0.942552
```

### broken_small

```text
AUROC = 0.959091
AP    = 0.954006
```

### contamination

```text
AUROC = 0.930952
AP    = 0.926809
```

Coreset100 的三个 defect type 都与 Candidate5000 很接近。

---

## 20. Per-Defect Comparison

| Defect Type | Candidate5000 AUROC | Random100 AUROC | Coreset100 AUROC |
|---|---:|---:|---:|
| broken_large | 0.947500 | 0.600000 | 0.952500 |
| broken_small | 0.961364 | 0.597727 | 0.959091 |
| contamination | 0.923810 | 0.907143 | 0.930952 |

---

## 21. Important Result - Broken Small Is Not the Hardest Class

在 Candidate5000：

```text
broken_small AUROC:
0.961364
```

是三种 defect 中最高。

Coreset100：

```text
broken_small AUROC:
0.959091
```

也保持较高水平。

因此当前：

```text
layer2 patch feature
+
nearest-neighbor
+
max aggregation
```

对 `broken_small` 并不弱。

这与 EXP011 / EXP012 的观察一致：

```text
small local defect
→
局部 Patch Score 明显升高
→
Max Aggregation 能保留异常响应
```

因此不能简单认为：

```text
Small Defect
=
当前 Bottle 中最难类别
```

---

## 22. Important Result - Contamination Is the Main Weakness

Candidate5000：

```text
contamination AUROC:
0.923810
```

是三种 defect 中最低。

Coreset100：

```text
contamination AUROC:
0.930952
```

同样最低。

因此当前 Patch Baseline 的主要 defect-level weakness 是：

```text
contamination
```

---

## 23. Candidate5000 Failure Candidates

### Top-5 Highest-Scoring GOOD

```text
1. good/010.png    4.331443
2. good/006.png    4.207956
3. good/009.png    3.717607
4. good/019.png    3.662836
5. good/004.png    3.590076
```

这些样本应记录为：

```text
High-scoring Normal Candidates
```

尚不能正式称 FP。

---

### Top-5 Lowest-Scoring DEFECT

```text
1. contamination/003.png    3.460963
2. contamination/020.png    3.556686
3. contamination/000.png    3.604330
4. contamination/012.png    3.647284
5. contamination/004.png    3.668973
```

值得注意：

```text
5 / 5
```

全部来自：

```text
contamination
```

---

## 24. Coreset100 Failure Candidates

### Top-5 Highest-Scoring GOOD

```text
1. good/010.png    4.635723
2. good/006.png    4.347278
3. good/001.png    4.173130
4. good/016.png    3.885491
5. good/009.png    3.880455
```

---

### Top-5 Lowest-Scoring DEFECT

```text
1. contamination/004.png    3.836864
2. contamination/003.png    3.926649
3. contamination/012.png    3.971097
4. contamination/001.png    3.985341
5. contamination/020.png    3.990207
```

再次：

```text
5 / 5
```

全部来自 contamination。

---

## 25. Stable High-Scoring Normal Samples

Candidate5000：

```text
good/010
good/006
```

位于 Top-2。

Coreset100：

```text
good/010
good/006
```

仍位于 Top-2。

因此：

```text
good/010
good/006
```

是跨 Memory Strategy 稳定出现的：

```text
High-scoring Normal Candidates
```

说明问题不太可能只是单次 Random Sampling 导致。

后续需要研究：

```text
normal visual variation
reflection
edge structure
local texture
position variation
```

但当前尚不能把这些假设当成结论。

---

## 26. Stable Low-Scoring Defect Samples

Candidate5000 与 Coreset100 中均反复出现：

```text
contamination/003
contamination/004
contamination/012
contamination/020
```

因此这些样本是稳定的：

```text
Low-scoring Defect Candidates
```

后续 Failure Analysis 应优先研究。

---

## 27. Failure Mode #1

根据三组证据：

### Evidence A - Per-defect AUROC

```text
contamination
```

为 Candidate5000 / Coreset100 中最低 AUROC 类别。

### Evidence B - Candidate Failure Ranking

Top-5 lowest-scoring defect：

```text
5 / 5 contamination
```

### Evidence C - Coreset Failure Ranking

同样：

```text
5 / 5 contamination
```

因此可以建立当前正式 Failure Mode：

```text
Failure Mode #1:
Low-response contamination
```

注意：

目前只证明：

```text
contamination score 偏低
```

还没有证明原因。

---

## 28. Prediction vs Result

### Prediction 1

预测：

```text
Random100 Overall Performance
明显弱于 Coreset100
```

实际：

```text
Random100 AUROC:
0.701587

Coreset100 AUROC:
0.947619
```

Verdict：

```text
PASS
```

---

### Prediction 2

预测：

```text
Coreset100 不一定在所有 defect type
都接近 Candidate5000
```

实际：

```text
broken_large:
0.947500 vs 0.952500

broken_small:
0.961364 vs 0.959091

contamination:
0.923810 vs 0.930952
```

结果比预期更稳定。

Coreset100 在所有 defect type 上都与 Candidate5000 很接近。

---

### Prediction 3

预测：

```text
Coreset100 Good P95
<
Random100 Good P95
```

实际：

```text
4.361700
<
5.095362
```

同时：

```text
Coreset Good Max:
4.635723

Random Good Max:
5.512643
```

Verdict：

```text
PASS
```

---

### Prediction 4

预测：

```text
不同 defect type 难度不同
```

实际：

```text
broken_small:
较容易

broken_large:
中等

contamination:
最困难
```

Verdict：

```text
PASS
```

---

### Prediction 5

预测：

```text
无 threshold 时
Failure Samples 只能作为 Candidate
```

当前实验仍没有正式 threshold。

因此继续使用：

```text
High-scoring Normal Candidate
Low-scoring Defect Candidate
```

Verdict：

```text
PASS
```

---

## 29. Relationship to EXP015

EXP015：

```text
20 Good
+
20 Broken Small
```

得到：

```text
Candidate5000:
AUROC = 0.9575

Random100:
AUROC = 0.5600

Coreset100:
AUROC = 0.9550
```

EXP016 完整 Bottle：

```text
Candidate5000:
0.944444

Random100:
0.701587

Coreset100:
0.947619
```

因此 EXP015 的核心结论成功推广：

```text
Random100
显著损失 detection performance

Coreset100
在相同 100 Patch Capacity 下
基本保持 Candidate5000 performance
```

---

## 30. Why Coreset Can Slightly Outperform Candidate5000 in AUROC

Candidate5000 的 NN Search Space 更大。

对于任意单个 Query Patch：

\[
s(q,Candidate5000)
\le
s(q,Coreset100)
\]

因为 Candidate5000 包含更多 NN 候选。

但：

```text
NN Distance 更小
```

不等于：

```text
AUROC 必然更高
```

AUROC 衡量：

```text
Good / Defect Relative Ranking
```

而不是单个 Feature Reconstruction Distance。

因此 Coreset Compression 可能让：

```text
Good Score ↑ 一定程度
Defect Score ↑ 不同程度
```

最终 Ranking 偶然略有改善。

所以：

```text
0.947619
>
0.944444
```

不能解释成：

```text
100 Patch 理论上比 5000 Patch 更好
```

更准确是：

> 当前测试集上，两种方法的 ranking performance 基本相当。

---

## 31. Separation Margin Limitation

Candidate：

```text
-0.870480
```

Random：

```text
-0.813463
```

Coreset：

```text
-0.798858
```

如果只看 Margin，似乎：

```text
Coreset > Random > Candidate
```

但 AUROC：

```text
Coreset ≈ Candidate >> Random
```

说明 Separation Margin：

```text
只由最高 Good
和最低 Defect
两个极端样本决定
```

对 outlier 非常敏感。

而 AUROC：

```text
考虑所有 Good / Defect Pair Ranking
```

因此：

> Separation Margin 适合判断是否完全可分，不适合作为唯一整体性能指标。

---

## 32. Engineering Observation

本实验仍然每次重新：

```text
build full normal memory bank
```

后续实验建议逐步缓存：

```text
memory_bank.pt
```

并同时保存 metadata：

```text
dataset
category
backbone
weights
feature layer
input size
preprocess
feature dimension
patch count
```

避免未来修改模型配置后错误复用旧 Feature。

---

## 33. Limitations

### Limitation 1 - Candidate5000 Is Not Full Memory

本实验没有直接比较：

```text
Full Memory:
163856 patches
```

因此不能说：

```text
Coreset100
等价于 Full Patch Memory
```

只能说：

```text
Coreset100
基本保持 Candidate5000 performance
```

---

### Limitation 2 - Only Image-Level Metrics

当前主要评价：

```text
Image-level AUROC
Image-level AP
```

没有计算：

```text
Pixel AUROC
PRO
IoU
Dice
```

因此无法评价完整 pixel localization performance。

---

### Limitation 3 - No Formal Threshold

当前尚未定义正式 decision threshold。

因此：

```text
high-score good
low-score defect
```

仍属于风险候选。

尚不能正式统计：

```text
FP
FN
Precision
Recall
F1
```

---

### Limitation 4 - Max Aggregation Only

当前 image score 固定：

\[
S(x)=\max_i s(q_i)
\]

尚未系统比较：

```text
Top-K Mean
Mean
Alternative Aggregation
```

因此 aggregation strategy 仍可能影响不同 defect type。

---

## 34. Main Conclusions

EXP016 在完整 83 张 Bottle Test Set 上验证：

### Conclusion 1

Random Sampling 在强 Memory Compression 下会严重损失 anomaly detection performance。

```text
Random100 AUROC:
0.701587
```

---

### Conclusion 2

K-center Coreset 能够在相同 100 Patch 容量下显著优于 Random Sampling。

```text
Random100:
0.701587

Coreset100:
0.947619
```

说明：

```text
Memory Selection Strategy
```

非常重要。

---

### Conclusion 3

Coreset100 使用 Candidate5000 仅 2% 的 Memory：

```text
5000
→
100
```

但仍基本保持 Candidate5000 的 detection performance。

---

### Conclusion 4

`broken_small` 并不是当前 Patch Baseline 最困难的 defect type。

```text
broken_small AUROC:
0.959091
```

---

### Conclusion 5

当前 Patch Baseline 的主要弱点是：

```text
contamination
```

因为：

```text
Per-defect AUROC 最低
+
Top-5 low-score defects 全部 contamination
```

---

### Conclusion 6

以下样本是后续 Failure Analysis 的优先对象：

```text
High-scoring Normal:

good/010
good/006
```

以及：

```text
Low-scoring Defect:

contamination/003
contamination/004
contamination/012
contamination/020
```

---

## 35. New Questions

1. Global Feature Baseline 与 Patch Coreset100 在同一个 83-image Test Set 上谁更强？
2. Patch 的 `broken_small` 优势是否真的存在？
3. Global 是否也会在 contamination 上表现较弱？
4. `good/010` 与 `good/006` 为什么持续获得高 Patch Score？
5. `contamination/003` 等样本为什么持续获得低 anomaly score？
6. layer2 是否是当前最佳 Feature Layer？
7. Max Aggregation 是否适合所有 defect type？
8. Coreset100 的性能是否对 Random Seed / Candidate Sampling 敏感？

---

## 36. Next Step

下一实验：

```text
EXP017 - Global vs Patch Bottle Evaluation
```

将在完全相同的：

```text
83 Bottle Test Images
```

上比较：

```text
Global Baseline
vs
Patch Coreset100
```

重点评价：

```text
Overall AUROC / AP

Per-defect AUROC / AP

High-score Normal Candidates

Low-score Defect Candidates

Global vs Local Representation Behavior
```

---

## 37. Status

```text
Full Bottle Test Set                   PASS

Candidate5000 Evaluation               PASS
Random100 Evaluation                   PASS
Coreset100 Evaluation                  PASS

Overall AUROC / AP                     PASS

Per-defect Evaluation                  PASS

Failure Candidate Extraction           PASS

EXP015 Generalization Check            PASS

Memory Selection Understanding         PASS

Failure Mode:
Low-response Contamination             IDENTIFIED

EXP016                                 PASS

Patch-level Anomaly Detection Unit     CONTINUE

Git                                    暂不提交
```