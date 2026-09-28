# EXP024 — Embedding Geometry, Visual Retrieval & DINOv3 Representation

## 0. Learning Unit 信息

- **Level：** Level 1 — Visual Foundations & Experiment Capability
- **EXP：** EXP024
- **主题：** Embedding Geometry / Visual Retrieval / ViT / DINOv3 / Spatial Representation
- **状态：** `CONTINUE`

本 EXP 的目标不是“会从模型里拿 feature”，而是建立以下完整能力：

- 能解释 feature、embedding、global representation、spatial representation 的区别
- 能解释 cosine、L2、normalization 对检索空间的影响
- 能预测修改 embedding / readout / resolution 后 nearest-neighbor ranking 的变化
- 能实现真实图像 retrieval
- 能理解 ViT 中 CLS / register / patch token 的作用
- 能把 DINOv3 patch representation 迁移到工业 anomaly localization
- 能设计 controlled ablation
- 能分析 failure case，而不是只报一个平均指标
- 能区分 descriptive comparison 与 causal / controlled comparison

---

# 1. Knowledge Map

EXP024 的整体知识链：

```text
Feature
  ↓
Embedding
  ↓
Embedding Geometry
  ├── L2 distance
  ├── cosine similarity
  └── L2 normalization
        ↓
Nearest-neighbor retrieval
        ↓
Global representation
  ├── CNN pooled embedding
  ├── ViT CLS
  └── mean-patch
        ↓
Spatial representation
  └── patch tokens
        ↓
DINOv3
  ├── CLS
  ├── register tokens
  └── patch tokens
        ↓
Industrial transfer
  ├── visual retrieval
  └── patch-memory anomaly detection
        ↓
Resolution ablation
        ↓
Failure analysis
        ↓
Controlled experiment / protocol design
```

---

# 2. Feature 与 Embedding

## 2.1 Feature

`feature` 是模型中间得到的表示。

它可以是：

```text
vector
feature map
token sequence
patch descriptors
```

例如 CNN：

```text
[B, C, H, W]
```

ViT：

```text
[B, N, D]
```

因此：

> feature 不一定是一个单独向量。

---

## 2.2 Embedding

Embedding 通常指：

> 为了在某个向量空间中进行距离或相似度比较，而使用的向量表示。

例如：

```text
image
↓
backbone
↓
embedding ∈ R^D
↓
cosine / L2
↓
nearest neighbor
```

一个 feature 可以被进一步转换成 embedding。

例如：

```text
CNN feature map
[B,C,H,W]

↓ global average pooling

image embedding
[B,C]
```

---

# 3. Raw L2、Cosine 与 Normalization

设两个 embedding：

\[
x,y\in \mathbb{R}^{D}
\]

## 3.1 Raw L2

\[
d(x,y)=\|x-y\|_2
\]

Raw L2 同时受到：

```text
direction
+
magnitude
```

影响。

因此 embedding norm 的变化可能改变 nearest-neighbor ranking。

---

## 3.2 Cosine Similarity

\[
\cos(x,y)=
\frac{x^\top y}
{\|x\|_2\|y\|_2}
\]

它主要比较两个向量的方向。

如果：

```text
B = [2,0]
```

改成：

```text
B = [10,0]
```

B 的方向没变。

所以 cosine：

```text
cos(A,B)
```

不变。

但 raw L2：

```text
||A-B||
```

会明显变化。

---

# 4. L2 Normalization

定义：

\[
x'=\frac{x}{\|x\|_2}
\]

归一化以后：

\[
\|x'\|_2=1
\]

这意味着 magnitude 不再参与 distance。

## 4.1 Normalized L2 与 Cosine 的关系

如果：

\[
\|x\|=\|y\|=1
\]

则：

\[
\|x-y\|_2^2
=
(x-y)^T(x-y)
\]

展开：

\[
= x^Tx+y^Ty-2x^Ty
\]

因为：

\[
x^Tx=y^Ty=1
\]

所以：

\[
\boxed{
\|x-y\|_2^2
=
2-2\cos(x,y)
}
\]

因此：

> 对 L2-normalized embedding，cosine similarity 与 normalized L2 distance 的 nearest-neighbor ranking 理论上等价。

一个增加时，另一个单调变化。

因此：

```text
cosine NN ranking
==
normalized-L2 NN ranking
```

数值误差和 tie 除外。

---

# 5. EXP024 Phase A — Minimal Embedding Geometry

脚本：

```text
src/experiments/exp024_embedding_geometry_minimal.py
```

Toy vectors：

```text
A = [1,0]
B = [2,0]
C = [1,1]
```

主动修改：

```text
B:
[2,0]
→
[10,0]
```

观察：

```text
Raw L2 ranking
会变化

Cosine ranking
不变化

Normalized L2 ranking
不变化
```

进一步令：

\[
B_s=[s,0]
\]

发现 raw L2 ranking boundary：

```text
0 < s < 2    → B 更近
s = 2        → tie
s > 2        → C 更近
```

而 cosine / normalized-L2 不受这个 magnitude 修改影响。

### 核心结论

Normalization 不是“任何 embedding 都必须执行的固定预处理”。

它真正决定的是：

> magnitude 是否参与 similarity / distance。

因此应该根据 representation geometry 和任务设计决定。

**Phase A：PASS**

---

# 6. ResNet18 Real Retrieval

脚本：

```text
exp024_image_retrieval_minimal.py
```

模型：

```text
ResNet18 ImageNet pretrained
fc = Identity
embedding dim = 512
```

数据：

```text
COCO
80 gallery
5 query
```

观察到 embedding norm：

```text
min   22.387
mean  28.537
max   38.539
```

说明真实 CNN embedding 存在明显 magnitude variation。

## 6.1 Cosine vs normalized L2

5 个 query：

```text
cosine Top5
==
normalized-L2 Top5
```

全部一致。

这验证了前面的理论。

## 6.2 Raw L2 vs Cosine

5 个 query 中：

```text
raw L2 Top5
vs
cosine Top5
```

全部出现差异。

总 Top5 set overlap：

```text
18 / 25
= 72%
```

但必须注意：

> embedding norm 有变化 + ranking 有差异，并不能证明某一个具体 ranking difference 单独由 magnitude 导致。

因为真实网络 embedding 的 direction 与 magnitude 都可能同时变化。

---

# 7. Qualitative Retrieval Observation

脚本：

```text
exp024_visualize_retrieval.py
```

观察 ResNet global representation 会混合：

```text
object identity
scene
shape
texture
composition
```

因此 global embedding 并不是纯粹的“类别向量”。

对于部分 query：

```text
cosine retrieval
```

看起来比 raw L2 更集中于语义类似图像。

但样本只有 5 张，而且 gallery coverage 有限，因此：

> 不能从少量可视化案例推出 cosine 普遍优于 raw L2。

---

# 8. Retrieval Evaluation Metrics

在 COCO retrieval 中使用 category annotation 构造 relevance。

## 8.1 Binary relevance

若：

\[
Q\cap G\neq \emptyset
\]

则 gallery image 对 query 算 relevant。

Binary P@5：

\[
P@5=
\frac{
\#\text{Top5 relevant results}
}{
5
}
\]

## 8.2 Hit@5

只关心 Top5 中是否至少存在一个 relevant：

\[
Hit@5=
\mathbb{1}
(
\exists\ relevant
)
\]

因此：

```text
P@5
```

和：

```text
Hit@5
```

不是同一个指标。

## 8.3 Category Coverage

\[
Coverage(Q,G)=
\frac{|Q\cap G|}
{|Q|}
\]

回答：

> gallery image 覆盖了多少 query category。

## 8.4 Jaccard

\[
J(Q,G)=
\frac{|Q\cap G|}
{|Q\cup G|}
\]

同时考虑：

```text
missing query categories
+
extra gallery categories
```

## 8.5 Strict relevance

定义：

\[
Q\subseteq G
\]

即 gallery image 必须包含 query 的完整 category set。

Strict Hit@5 只应在：

```text
strict eligible query
```

上统计。

如果 gallery 根本不存在可以满足 query 全 category set 的图像，那么这个 query 对 strict retrieval 不可评估。

---

# 9. Retrieval Quantitative Experiment

脚本：

```text
exp024_retrieval_quantitative_v2.py
```

配置：

```text
gallery = 400
query   = 100
labeled query = 99
strict eligible = 62
```

ResNet：

```text
Raw L2 Binary P@5 = 0.7434
Cosine Binary P@5 = 0.7091

Raw Coverage       = 0.4192
Cosine Coverage    = 0.4203

Raw Jaccard        = 0.2949
Cosine Jaccard     = 0.3072

Strict Hit@5
Raw = 0.5323
Cos = 0.5323
```

paired：

```text
Coverage cosine W/T/L
27 / 47 / 25

Jaccard cosine W/T/L
38 / 26 / 35
```

Cosine vs normalized-L2 Top5 match：

```text
1.0
```

### 核心结论

同一个 retrieval system：

```text
Binary P@5
Coverage
Jaccard
Strict
```

可能给出不同的“谁更好”。

因此：

> metric definition 本身属于实验设计的一部分。

工业任务中，如果业务要求：

```text
产品类型 + 缺陷类型
```

必须同时匹配，那么 Strict metric 应成为主要 KPI，而 Coverage/Jaccard 可以作为 diagnostic metric。

---

# 10. Supervised ViT Token Anatomy

模型：

```text
torchvision ViT-B/16
ImageNet pretrained
```

输入：

```text
224 × 224
patch size = 16
```

因此：

\[
224/16=14
\]

patch 数：

\[
14\times14=196
\]

再加 CLS：

```text
sequence length = 197
```

实际：

```text
patch before CLS   [1,196,768]
with CLS           [1,197,768]
encoded            [1,197,768]

CLS                [1,768]
patch tokens       [1,196,768]
```

---

# 11. CNN Spatial Feature vs ViT Patch Token

CNN spatial feature：

```text
[B,C,H,W]
```

ViT patch token：

```text
[B,N,D]
```

两者都保留一定空间粒度，但产生机制不同。

CNN：

```text
local convolution
weight sharing
strong local inductive bias
```

ViT：

```text
patch embedding
+
self-attention
+
global token interaction
```

因此不能简单说：

```text
CNN feature == ViT patch token
```

更合理的是：

> 它们都可以作为 local descriptor，但 representation mechanism 不同。

---

# 12. CLS vs Mean-Patch

定义：

### CLS

```python
cls = tokens[:, 0]
```

### Mean-patch

\[
m=
\frac{1}{N}
\sum_{i=1}^{N}
p_i
\]

## 12.1 Mean of norms != Norm of mean

一般：

\[
\frac{1}{N}
\sum_i\|p_i\|
\neq
\left\|
\frac{1}{N}
\sum_i p_i
\right\|
\]

mean-patch norm 可以比单个 patch norm 小很多。

原因：

> 不同 patch vector 的方向会发生 cancellation。

因此：

```text
smaller norm
```

不能解释为：

```text
less important
```

---

# 13. Supervised ViT CLS vs Mean-Patch Geometry

单图实验：

```text
CLS norm        = 19.5975
mean-patch norm = 9.5328

cosine          = -0.2080
raw L2          = 23.509
normalized L2   = 1.55438
```

并验证：

\[
d_{norm}^2
=
2-2\cos
\]

数值误差：

```text
0
```

---

# 14. Supervised ViT Fixed-Split Retrieval

为了消除不同 query/gallery split 的混杂，最终使用 DINOv3 的：

```text
results/exp024/
dinov3_cls_vs_mean_retrieval/
split.csv
```

作为统一 protocol。

固定：

```text
Query           = 100
Gallery         = 400
Labeled         = 99
Strict eligible = 59
```

结果：

| Metric | CLS | Mean-patch |
|---|---:|---:|
| Binary P@5 | 0.8404 | 0.7091 |
| Coverage | 0.5275 | 0.4048 |
| Jaccard | 0.4000 | 0.2883 |
| Strict Hit@5 | 0.7458 | 0.6102 |

paired：

```text
Coverage CLS W/T/L
63 / 16 / 20

Jaccard CLS W/T/L
72 / 5 / 22
```

因此在当前 supervised ViT-B/16 中：

> CLS readout 在当前 COCO global semantic retrieval protocol 下明显优于 mean-patch。

---

# 15. Supervised ViT CLS vs Mean Geometry

500 张图：

```text
min     -0.3207
mean    -0.1354
median  -0.1333
max      0.0040
std      0.0513
```

也就是说：

> supervised ViT 中 CLS 与 mean-patch 的方向对齐非常低，整体甚至为负 cosine。

但不能把：

```text
negative cosine
```

直接解释成：

```text
CLS 和 patch “关注完全相反的东西”
```

cosine 只能描述 vector geometry。

如果要讨论“关注哪里”，还需要：

```text
attention analysis
token attribution
spatial visualization
```

等额外证据。

---

# 16. DINO / Self-Supervised Representation 基本概念

核心理解：

Self-supervised：

```text
≠ 没有 target
```

而是 target 并非人工类别标签。

DINO 类方法的基本思想可以理解为：

```text
same source image
→ different views / crops
→ encourage representation consistency
```

在原始 DINO 的 teacher-student 概念中：

```text
teacher output
→ student target

teacher
→ EMA update
```

因此：

> self-supervised learning 不需要 dog/cat 人工标签，但仍然存在学习目标。

---

# 17. 为什么 Patch Token 对工业视觉重要

Global representation：

```text
整张图 → 一个 vector
```

容易把：

```text
small defect
```

淹没在：

```text
large normal region
```

中。

Patch token 保留空间粒度：

```text
image
→ patch descriptors
→ local matching
```

因此天然适合：

```text
small defect detection
anomaly localization
local visual retrieval
```

这与之前 PatchCore 中：

```text
local CNN descriptor
→ normal memory
→ nearest neighbor
```

是同一种工业视觉思路。

---

# 18. DINOv3 Setup

模型：

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

本地路径：

```text
D:\AI_Lab\models\dinov3\
dinov3-vits16-pretrain-lvd1689m
```

环境：

```text
Torch        2.14.0+cu126
Transformers 5.17.0
CUDA         True
```

模型：

```text
DINOv3ViTModel
```

参数：

```text
patch size          = 16
hidden size         = 384
register tokens     = 4
default image size  = 224
```

---

# 19. DINOv3 Token Anatomy

输入：

```text
224 × 224
```

patch grid：

```text
14 × 14
```

patch token：

```text
196
```

sequence：

```text
1 CLS
+ 4 register
+ 196 patch
= 201
```

实际：

```text
last_hidden_state [1,201,384]

CLS               [1,384]
register          [1,4,384]
patch             [1,196,384]
patch grid        [1,14,14,384]
```

---

# 20. Register Tokens

当前 Transformers 接口称：

```text
register tokens
```

它们：

```text
不是 spatial patch
```

因此不能把：

```text
4 register tokens
```

reshape 成：

```text
2×2
```

然后解释成图像四个区域。

它们属于 non-spatial prefix tokens。

因此做 spatial anomaly map 时：

```python
patch_tokens = hidden[
    :,
    1 + num_register_tokens:,
    :
]
```

而不能直接：

```python
hidden[:, 1:, :]
```

否则 register tokens 会被错误当成 spatial descriptors。

---

# 21. DINOv3 CLS vs Mean-Patch Geometry

单图：

```text
CLS vs mean-patch cosine
≈ 0.578
```

dataset-level 500 张：

```text
min     0.1636
mean    0.4791
median  0.4844
max     0.6895
std     0.0891
```

因此：

> DINOv3 CLS 与 mean-patch 存在中等程度的方向对齐，但远不是同一种 representation。

不能把 cosine similarity 叫做统计意义上的 correlation。

---

# 22. DINOv3 CLS vs Mean-Patch Retrieval

固定 protocol：

```text
Query           = 100
Gallery         = 400
Labeled         = 99
Strict eligible = 59
```

结果：

| Metric | CLS | Mean-patch |
|---|---:|---:|
| Binary P@5 | 0.8263 | 0.8000 |
| Coverage | 0.4978 | 0.4656 |
| Jaccard | 0.3732 | 0.3304 |
| Strict Hit@5 | 0.6102 | 0.6271 |

paired：

```text
Coverage CLS W/T/L
34 / 32 / 33

Jaccard CLS W/T/L
46 / 15 / 38
```

Binary P@5：

```text
25 / 54 / 20
```

Strict Hit@5：

```text
4 / 50 / 5
```

---

# 23. Readout 会改变 NN Neighborhood

DINOv3：

```text
Exact Top5 ranking match rate
≈ 1%

Mean Top5 set overlap
≈ 49%
```

Supervised ViT：

```text
Exact Top5 ranking match rate
= 0%

Mean Top5 set overlap
= 33.8%
```

因此：

> 即使 backbone 和 token 完全一样，仅改变 CLS / mean-patch readout，也可能明显改变 nearest-neighbor neighborhood。

Pooling 不是无关紧要的实现细节。

它本身就是 representation design。

---

# 24. Supervised ViT vs DINOv3

统一 fixed split 后：

| Model / Readout | Binary P@5 | Coverage | Jaccard | Strict |
|---|---:|---:|---:|---:|
| Supervised ViT CLS | 0.8404 | 0.5275 | 0.4000 | 0.7458 |
| Supervised ViT Mean | 0.7091 | 0.4048 | 0.2883 | 0.6102 |
| DINOv3 CLS | 0.8263 | 0.4978 | 0.3732 | 0.6102 |
| DINOv3 Mean | 0.8000 | 0.4656 | 0.3304 | 0.6271 |

观察：

```text
Supervised CLS
整体略高

DINOv3 mean-patch
明显比 supervised mean-patch 更可用
```

但这是：

```text
descriptive comparison
```

不是 causal conclusion。

不能写：

```text
SSL 导致 mean-patch 更好
```

因为：

```text
training objective
training data
architecture
hidden dimension
register tokens
optimization
augmentation
pretraining recipe
```

等都不同。

---

# 25. Fixed Split ≠ Controlled Experiment

这是本 EXP 一个非常重要的方法论结论。

固定：

```text
query
gallery
annotation
metric
```

只能保证：

> evaluation protocol 一致。

它不能自动保证：

> model comparison 是 controlled experiment。

## 25.1 Controlled comparison

例如同一个 DINOv3：

```text
CLS
vs
mean-patch
```

如果：

```text
backbone fixed
data fixed
similarity fixed
evaluation fixed
```

只改变 readout，那么这是：

```text
controlled readout comparison
```

## 25.2 Descriptive comparison

例如：

```text
DINOv3 ViT-S/16
vs
supervised ViT-B/16
```

即使 fixed split 完全一致，也仍然存在大量模型差异。

因此只能说：

> 在当前 benchmark/protocol 下，模型 A 的某指标高于模型 B。

不能说：

> 某个训练因素导致了差异。

---

# 26. DINOv3 Patch-Memory Anomaly Detection

工业迁移：

```text
MVTec AD
Bottle
```

normal：

```text
train/good
209 images
```

query：

```text
test/broken_small/000.png
```

## 26.1 Pipeline

```text
normal images
↓
DINOv3 patch tokens
↓
L2 normalization
↓
normal patch memory
↓
query patch tokens
↓
cosine nearest neighbor
↓
max cosine
↓
anomaly score
```

定义：

\[
s_i=
1-
\max_j
\cos(q_i,m_j)
\]

其中：

```text
q_i = query patch
m_j = normal memory patch
```

如果 query patch 与所有 normal patch 都不相似：

```text
max cosine ↓
```

因此：

```text
anomaly score ↑
```

---

# 27. DINOv3 224 Resolution

输入：

```text
224×224
```

grid：

```text
14×14
```

normal memory：

\[
209\times196=40964
\]

所以：

```text
normal tensor [209,196,384]
memory bank   [40964,384]

query         [196,384]
similarity    [196,40964]

anomaly map   [14,14]
```

单图 `broken_small/000`：

```text
anomaly min   0.00223
mean          0.00635
max           0.05172

GT-positive patches = 4

max anomaly patch overlaps GT = True

GT+ mean = 0.03557
GT- mean = 0.00574

Patch AUROC = 1.0
```

注意：

> anomaly score 的绝对值很小并不等于模型失败。

因为：

```text
score = 1 - cosine
```

如果正常和异常 patch 都整体比较相似，则所有 score 都可能偏小。

真正重要的是：

```text
relative ranking
+
GT separation
```

---

# 28. GT Alignment Validation

DINOv3 processor：

```text
do_resize       = True
size            = 224×224
do_center_crop  = None
crop_size       = None
```

原始 image：

```text
900×900
```

GT：

```text
900×900
```

因此：

```text
image
900×900
→ resize 224×224

GT
900×900
→ resize 224×224
```

geometry 一致。

image 使用 bilinear。

mask 使用 nearest-neighbor。

这是合理的，因为 mask 是 discrete label。

因此：

```text
Image preprocessing geometry    PASS
GT preprocessing geometry       PASS
Image/GT alignment              PASS
```

---

# 29. Why Higher Resolution May Help

DINOv3 patch size 固定：

```text
16
```

224：

```text
224 / 16 = 14
14×14 = 196 patches
```

448：

```text
448 / 16 = 28
28×28 = 784 patches
```

因此提高输入 resolution：

```text
14×14
→
28×28
```

使每个 patch 对应原图更小的空间区域。

理论上更适合：

```text
small defect localization
```

---

# 30. 224 vs 448 Active Modification

448：

```text
query patches = 784

memory =
209 × 784
= 163856

memory shape =
[163856,384]
```

由于 similarity matrix 很大，因此使用：

```text
chunked nearest neighbor
```

避免一次构造过大的 GPU tensor。

---

# 31. Single-Image Resolution Ablation

`broken_small/000`：

| Metric | 224 | 448 |
|---|---:|---:|
| Grid | 14×14 | 28×28 |
| Memory patches | 40,964 | 163,856 |
| Pixel AUROC | 0.99593 | 0.99979 |
| Pixel AP | 0.52013 | 0.97562 |
| GT+ mean | 0.03180 | 0.05707 |
| GT− mean | 0.00613 | 0.00668 |
| Runtime | 8.41s | 19.39s |
| Peak allocated VRAM | 156 MiB | 351 MiB |

AP：

\[
0.5201
\rightarrow
0.9756
\]

非常明显。

AUROC：

\[
0.9959
\rightarrow
0.9998
\]

变化较小。

---

# 32. Why AUROC and AP Behave Differently

## AUROC

AUROC 本质上看：

> 随机选一个正类和一个负类时，正类 score 排在负类前的概率。

它主要反映：

```text
overall ranking
```

当 AUROC 已接近 1 时：

```text
ceiling effect
```

非常明显。

## AP

Average Precision 基于：

```text
precision-recall
```

对：

```text
high-ranking false positives
```

更加敏感。

对于工业 small defect：

```text
positive pixels 少
negative pixels 多
```

Pixel AP 往往更容易暴露：

```text
anomaly map spatial diffusion
boundary false positives
```

等问题。

因此：

> Pixel AUROC 接近饱和时，Pixel AP 对 localization quality 更敏感。

---

# 33. 22-Image Dataset-Level Paired Evaluation

Bottle：

```text
broken_small
22 images
```

结果：

```text
mean AP 224 = 0.85463
mean AP 448 = 0.95177

mean ΔAP   = +0.09713
median ΔAP = +0.08228
std ΔAP    = 0.09093
```

paired：

```text
448 wins = 21
ties     = 0
losses   = 1
```

bootstrap：

```text
95% CI of mean ΔAP

[0.06718, 0.14035]
```

完全高于 0。

AUROC：

```text
224 = 0.99530
448 = 0.99834

Δ = +0.00305
```

因此可以支持：

> 在当前 Bottle/broken_small 子集、当前 DINOv3 ViT-S/16、当前 patch-memory cosine-NN 方法下，提高输入 resolution 从 224 到 448 对 pixel localization 有稳定改善。

不能扩大成：

```text
448 对整个 MVTec 更好
```

也不能说：

```text
所有工业 defect 都应该用 448
```

---

# 34. Engineering Cost of 448

224 → 448：

```text
patch number
196 → 784
= 4×

memory patches
40964 → 163856
= 4×
```

single-run pipeline：

```text
runtime
8.41s → 19.39s
≈ 2.31×

peak allocated VRAM
156 → 351 MiB
≈ 2.25×
```

但这个 `2.31×` 不能解释成：

> resolution 单独使 inference 慢了 2.31 倍。

因为实验中：

```text
224 batch = 8
448 batch = 2
```

而 runtime 还包括：

```text
normal feature extraction
memory building
query extraction
NN
evaluation
```

因此它只能描述：

> 当前 pipeline configuration 下的端到端实验时间差异。

如果要做真实 inference latency benchmark，需要：

```text
same batch
same measurement boundary
warmup
CUDA synchronize
multiple repeats
mean / P95
```

---

# 35. Failure Case — 013.png

22 张中只有：

```text
013.png
```

出现 448 AP 下降。

结果：

```text
AP:
224 = 0.87740
448 = 0.86617

ΔAP = -0.01123
```

但：

```text
AUROC:
0.99538
→
0.99584
```

反而提高。

同时：

```text
GT+ mean
0.07750
→
0.12775

GT- mean
0.01511
→
0.01292
```

也就是说：

```text
positive mean ↑
negative mean ↓
```

但 AP ↓。

---

# 36. Mean Separation != Ranking Quality

这说明：

> 两组 score 的均值 separation 改善，并不保证 AP 改善。

因为 AP 看的是完整 score ranking。

可能发生：

```text
大部分 positive
得分变好

大部分 negative
得分下降

但少量 negative
得到极高 score
```

这些 high-score negative 会：

```text
进入 ranking top region
→ precision ↓
→ AP ↓
```

---

# 37. Tail Analysis

013.png：

## 224

```text
GT+ median      0.0781
GT− q99.9       0.0822
GT− max         0.0962

negative pixels above positive median:
1228
```

## 448

```text
GT+ median      0.1013
GT− q99.9       0.1449
GT− max         0.2065

negative pixels above positive median:
2408
```

448 的：

```text
negative high-score tail
```

明显更重。

Precision@GT-size：

```text
224 = 0.7818
448 = 0.7670
```

因此：

> 448 改善了主体分布 separation，但 high-score negative tail 增加，使 precision 下降。

---

# 38. sklearn Metric Sanity Check

自写 Pixel AUROC / AP 与 sklearn 比较：

224：

```text
custom AP   = 0.8773997997
sklearn AP  = 0.8773997699
```

448：

```text
custom AP   = 0.8661687983
sklearn AP  = 0.8661687962
```

AUROC 也一致。

差异仅为浮点误差量级。

因此：

```text
custom Pixel AUROC    PASS
custom Pixel AP       PASS
```

正式 benchmark 最好优先使用经过验证的 library metric；自己实现适合学习公式，但需要 sanity check。

---

# 39. Spatial Diagnosis of False Positives

进一步测：

> high-score negative pixel 到最近 GT-positive pixel 的欧氏距离。

Top 0.1% negatives：

## 224

```text
median distance = 3.61 px

<=5 px    66.6%
<=10 px   98.9%
<=20 px   100%

>100 px   0%
```

## 448

```text
median distance = 4.00 px

<=5 px    67.1%
<=10 px   100%
<=20 px   100%

>100 px   0%
```

Top 1%：

```text
             224      448

<=10 px      44.4%    49.2%
<=20 px      77.3%    88.0%
>100 px       0%       0%
```

因此：

> 013.png 的高分 false positives 几乎全部集中在 GT boundary 附近。

这支持：

```text
boundary / annotation sensitivity
```

而不是：

```text
far-away representation failure
normal-memory coverage failure
```

---

# 40. Pixel Metric 对 Boundary 的敏感性

Pixel AP 不知道：

```text
FP 离 GT 只有 1 pixel
```

还是：

```text
FP 离 GT 300 pixels
```

只要：

```text
GT = 0
```

就算 false positive。

因此，一个 spatially reasonable prediction：

```text
刚好落在人工 mask 外侧
```

仍然会被严格 pixel metric 惩罚。

这也是为什么工业 defect segmentation 中：

```text
annotation boundary uncertainty
```

需要认真考虑。

---

# 41. Global Retrieval vs Local Defect Retrieval

如果目标是：

```text
整图语义检索
```

可以优先使用：

```text
CLS
global pooled embedding
mean-patch
```

如果目标变成：

```text
局部缺陷形态相似
```

优先保留：

```text
patch-level descriptors
```

原因：

CLS：

```text
全局信息强
small defect 容易被稀释
```

mean-patch：

```text
所有区域等权平均
normal/background 会稀释 defect
```

patch matching：

```text
保留 local shape
texture
position granularity
```

之后再将 local similarity 聚合成：

```text
max
top-k mean
weighted aggregation
```

得到 image-level retrieval score。

---

# 42. Connection to Previous EXPs

EXP024 并不是孤立知识。

## 42.1 PatchCore

以前：

```text
CNN local patch descriptor
→ normal memory
→ nearest neighbor
→ anomaly map
```

现在：

```text
DINOv3 patch token
→ normal memory
→ cosine nearest neighbor
→ anomaly map
```

representation 换了，但 anomaly detection 的基本结构没有变。

## 42.2 EXP020 Aggregation

以前比较：

```text
MAX
Top5
Top20
Mean
```

现在局部 retrieval 同样可以使用：

```text
patch matching
→ aggregation
→ image score
```

因此 aggregation strategy 仍然属于核心参数。

## 42.3 EXP021 Layer Choice

以前发现：

```text
layer2 vs layer3
```

会改变 anomaly performance。

现在同理：

```text
DINOv3 different layer tokens
```

也可能具有不同：

```text
semantic level
localization ability
texture sensitivity
```

因此 layer selection 也是未来可消融变量。

## 42.4 EXP022 Engineering Benchmark

更高 resolution：

```text
accuracy ↑
```

并不意味着工程上自动更好。

还需要比较：

```text
latency
VRAM
throughput
memory bank size
startup
```

这是 accuracy / engineering trade-off。

---

# 43. Important Parameter Understanding

## `patch_size`

当前：

```text
16
```

输入 resolution 固定时：

```text
patch size ↓
→ grid 更密
→ spatial localization 更细
→ token 数增加
```

## `input_size`

224 → 448：

```text
grid
14×14 → 28×28

patch count
196 → 784
```

提高 localization granularity，但增加 compute/memory。

## `readout`

```text
CLS
vs
mean-patch
```

会改变：

```text
embedding direction
retrieval neighborhood
downstream metric
```

不是无关紧要的实现细节。

## `normalization`

决定：

```text
embedding magnitude
```

是否参与 similarity。

## `similarity`

对 normalized vectors：

```text
cosine
```

和：

```text
normalized L2
```

NN ranking 等价。

## `memory coverage`

如果正常变化：

```text
lighting
view
texture
background
```

没有被 normal memory 覆盖，正常 patch 也可能：

```text
nearest-neighbor similarity ↓
→ anomaly score ↑
```

产生 false positive。

---

# 44. Common Failure Modes / Debug Checklist

## 44.1 数据路径缺失

表现：

```text
No training images
Missing image dir
```

不能立刻解释为：

```text
model failure
```

首先检查 filesystem。

## 44.2 GT 与 image geometry 不一致

如果：

```text
image processor
```

和：

```text
GT resize/crop
```

不同，pixel metric 会失效。

必须验证：

```text
resize
crop
flip
padding
```

完全一致。

## 44.3 Register token 被当成 spatial token

错误：

```python
hidden[:, 1:, :]
```

正确：

```python
hidden[
    :,
    1 + num_register_tokens:,
    :
]
```

## 44.4 单图结论泛化

例如：

```text
000.png AP +0.455
```

不能写：

```text
448 对整个 dataset 更好
```

必须做 dataset-level paired evaluation。

## 44.5 只看 mean metric

必须同时检查：

```text
median
std
W/T/L
confidence interval
failure cases
```

## 44.6 AUROC 接近 1 后继续只盯 AUROC

对于 small-defect localization：

```text
AP
```

可能更敏感。

## 44.7 Fixed split 被误认为 causal control

Fixed split：

```text
controls evaluation data
```

并不：

```text
controls model training differences
```

## 44.8 用 embedding norm 判断重要性

错误：

```text
norm 大
→ token 更重要
```

没有这种一般结论。

## 44.9 把 cosine 当 correlation

Cosine：

```text
vector directional similarity
```

Correlation：

```text
statistical relationship
```

不是同一个概念。

---

# 45. What I Can Now Explain

完成 EXP024 主实验后，我应该能够不用代码解释：

## Embedding

为什么需要 embedding，以及它和普通 feature 的区别。

## Normalization

为什么 normalization 会改变 raw-L2 retrieval，但不会改变 vector direction。

## Cosine / normalized L2

为什么它们对 unit vectors 给出相同 NN ranking。

## CLS / mean-patch

为什么同一个 Transformer 内仅改变 readout 就可能改变 retrieval neighborhood。

## Register tokens

为什么它们不能直接用于 spatial map。

## Patch representation

为什么 patch token 适合 industrial defect localization。

## Resolution

为什么更高输入 resolution 会增加 spatial granularity，同时增加 memory / compute。

## Metric

为什么：

```text
AUROC ↑
AP ↓
```

可以同时发生。

## Failure analysis

为什么：

```text
mean positive ↑
mean negative ↓
```

也不能保证 AP ↑。

## Experimental design

为什么：

```text
same split
```

不等于：

```text
controlled model comparison
```

---

# 46. Key Experimental Results Summary

```text
Embedding geometry
------------------
Normalized L2 ranking
==
Cosine ranking


ResNet retrieval
----------------
real embedding norm varies
raw L2 ranking != cosine ranking


Supervised ViT
--------------
CLS vs mean cosine mean = -0.1354

CLS:
P@5      0.8404
Coverage 0.5275
Jaccard  0.4000
Strict   0.7458

Mean:
P@5      0.7091
Coverage 0.4048
Jaccard  0.2883
Strict   0.6102


DINOv3
------
CLS vs mean cosine mean = 0.4791

CLS:
P@5      0.8263
Coverage 0.4978
Jaccard  0.3732
Strict   0.6102

Mean:
P@5      0.8000
Coverage 0.4656
Jaccard  0.3304
Strict   0.6271


DINOv3 Bottle broken_small
--------------------------
224 mean Pixel AP = 0.8546
448 mean Pixel AP = 0.9518

mean ΔAP   = +0.0971
median ΔAP = +0.0823

W/T/L
21 / 0 / 1

bootstrap 95% CI:
[+0.0672, +0.1403]


013 failure case
----------------
AP:
0.8774 → 0.8662

negative pixels above positive median:
1228 → 2408

high-score FP:
almost entirely near GT boundary
```

---

# 47. Core Conclusions

本 EXP 最重要的不是“DINOv3 很强”，而是建立了一整套 representation reasoning：

```text
1. Embedding space 的 geometry 会直接决定 retrieval。

2. Normalization 决定 magnitude 是否参与比较。

3. Cosine 和 normalized L2 对 unit vectors 的 NN ranking 等价。

4. Global readout 不是无关紧要的一行 pooling。

5. CLS 和 mean-patch 可以产生明显不同的 representation space。

6. Patch token 保留 spatial granularity，
   因此适合工业 anomaly / local retrieval。

7. 更高 resolution 可以改善 small-defect localization，
   但必须付出 memory / compute / latency 成本。

8. AUROC、AP、Coverage、Jaccard、Strict
   回答的是不同问题。

9. 一个平均指标变好不代表所有样本变好。

10. Failure case 应进一步分析 score distribution
    和 spatial location。

11. Fixed evaluation protocol 只控制评估数据，
    不自动带来 causal comparison。

12. 跨模型结果可以描述，
    但不能随意归因于 SSL、architecture 或某个训练因素。
```

---

# 48. Industrial Transfer

工业系统中，representation 设计可以按任务拆分：

```text
整图产品检索
→ global embedding / CLS

产品 + 属性检索
→ global + structured relevance metric

局部缺陷检索
→ patch descriptor + local matching

异常检测
→ normal patch memory + NN distance

异常定位
→ patch score → spatial anomaly map

small defect
→ increase spatial granularity
→ evaluate quality vs engineering cost
```

因此：

> 不是寻找一个“最好的 embedding”，而是为任务选择合适的 representation granularity、similarity、aggregation 和 evaluation metric。

---
