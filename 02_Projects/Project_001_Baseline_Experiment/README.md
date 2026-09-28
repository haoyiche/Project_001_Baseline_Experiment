# Project_001_Baseline_Experiment

## 1. 项目目标

本项目用于建立一套**可复现、可分析、可比较的工业视觉实验流程**，并完成 AI Lab Level 1：

> 视觉基础与实验能力（Visual Foundations & Experiment Capability）

本项目的目标不是单纯做到“模型能运行”，也不是只追求某一个指标，而是形成完整的视觉实验能力：

```text
问题定义
→ 建立 Baseline
→ 选择指标
→ 设计消融实验
→ Failure Analysis
→ 原因验证
→ 工程评估
→ 可复现
```

当前 Level 1 主项目：

> Industrial Vision Benchmark Lab
> 工业视觉 Benchmark 实验室

---

# 2. 当前阶段

```text
当前 Level：
Level 1 — 视觉基础与实验能力

当前任务：
EXP025 — Level 1 BOSS
Industrial Vision Benchmark Lab

当前状态：
CONTINUE
```

目前已经完成的主要学习与实验包括：

- 异常检测
- PatchCore-style 特征记忆库
- 目标检测评估
- Embedding 几何
- 图像检索
- Supervised ViT
- DINOv3
- Patch-level 空间异常定位
- 输入分辨率消融
- Failure Analysis
- Autoencoder 重建式异常检测
- Autoencoder 容量消融
- 高梯度区域 False Positive 分析

---

# 3. 实验环境

## 操作系统

```text
Windows
```

## Python

```text
Python 3.10
```

## Conda 环境

```text
D:\conda_envs\ai_base
```

## GPU

```text
NVIDIA RTX 2060 Max-Q
6 GB VRAM
```

## PyTorch

```text
torch 2.14.0+cu126
torchvision 0.29.0+cu126
CUDA available: True
```

## Transformers

```text
transformers 5.17.0
```

---

# 4. 数据集

## MVTec AD

Level 1 异常检测实验主要使用：

```text
data/raw/mvtec_anomaly_detection/
```

EXP025 BOSS 对比实验使用：

```text
data/raw/mvtec_anomaly_detection/bottle/
```

正常训练数据：

```text
train/good

209 张正常图像
```

主要异常测试子集：

```text
test/broken_small

22 张异常图像
```

Ground Truth：

```text
ground_truth/broken_small
```

原始数据集不作为普通 Git 文件提交。

---

# 5. DINOv3 模型

DINOv3 实验使用模型：

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

本地模型路径：

```text
D:\AI_Lab\models\dinov3\dinov3-vits16-pretrain-lvd1689m
```

主要配置：

```text
Patch Size        = 16
Hidden Size       = 384
Register Tokens   = 4
```

当输入大小为：

```text
224 × 224
```

Patch Grid 为：

```text
14 × 14
```

因此：

```text
Patch Tokens      = 196
CLS Token         = 1
Register Tokens   = 4

Sequence Length   = 201
```

---

# 6. Level 1 BOSS：两类异常检测方法对比

EXP025 在相同的：

```text
MVTec AD
Bottle
broken_small
```

数据上比较两种不同的异常检测方法范式。

---

## 6.1 方法 A：特征表示 + 最近邻

使用：

```text
DINOv3 Patch Representation
+
Cosine Nearest Neighbor
```

基本流程：

```text
正常图像
→ 提取 Patch Embedding
→ 建立正常特征 Memory
→ 查询 Patch 与正常 Memory 做最近邻匹配
→ 得到 Patch Anomaly Score
→ 恢复为空间 Anomaly Map
```

核心异常假设：

```text
正常 Patch
→ 在正常特征空间中能够找到相似邻居

异常 Patch
→ 与正常特征 Memory 距离更远
```

---

## 6.2 方法 B：重建式异常检测

使用：

```text
Convolutional Autoencoder
```

基本流程：

```text
输入图像
→ Encoder
→ Bottleneck
→ Decoder
→ 重建图像
→ 计算 Reconstruction Error
```

Pixel-level 异常分数：

```text
RGB 三通道绝对重建误差的均值
```

即：

```text
score = mean(|x - x_hat|)
```

核心异常假设：

```text
正常区域
→ 模型能够较好重建

异常区域
→ 模型难以准确重建

因此：
异常区域 Reconstruction Error 应更高
```

---

# 7. 同协议方法对比结果

测试数据：

```text
MVTec AD
Bottle
broken_small
22 张图像
```

两种方法统一在**原始 Ground Truth 空间**进行 Pixel-level 评估。

| 方法 | Mean Pixel AUROC | Mean Pixel AP |
|---|---:|---:|
| DINOv3 Patch-NN，224 输入 | 0.995295 | 0.854635 |
| Convolutional Autoencoder，224 输入 | 0.803955 | 0.196218 |

差值：

```text
Pixel AUROC：

DINOv3 +0.191340


Pixel AP：

DINOv3 +0.658416
```

在当前实验协议下：

```text
DINOv3 Patch-NN
在 22 / 22 张 broken_small 图像上
均优于当前 Autoencoder Baseline。
```

---

## 7.1 结论边界

当前实验能够支持：

> 在当前 MVTec Bottle / broken_small / 224 输入实验条件下，DINOv3 Patch-NN 对小型局部异常的 Pixel-level 定位能力明显优于当前基础 Autoencoder。

当前实验**不能证明**：

> “因为 DINOv3 使用 Self-Supervised Learning，所以性能一定更高。”

因为两个方法还同时存在以下差异：

```text
模型结构
预训练数据
训练目标
Representation
异常分数定义
异常检测机制
```

因此这里属于：

```text
方法类别 Benchmark 对比
```

而不是：

```text
严格的 Self-Supervised Learning 因果消融实验
```

---

# 8. Autoencoder Canonical Baseline

固定配置：

```text
Seed          = 42
Input         = 224 × 224
Batch Size    = 8
Epochs        = 20
Optimizer     = Adam
Learning Rate = 1e-3
Loss          = L1
```

Encoder Channel：

```text
3
→ 32
→ 64
→ 128
→ 256
```

空间 Bottleneck：

```text
14 × 14
```

Checkpoint：

```text
results/exp025/autoencoder_bottle_minimal/autoencoder_seed42.pt
```

最终固定结果：

```text
Mean Pixel AUROC   = 0.803955
Mean Pixel AP      = 0.196218

Global Pixel AUROC = 0.797181
Global Pixel AP    = 0.190786
```

重新加载同一个 Checkpoint 后重新评估，指标只有浮点数精度级差异。

因此：

```text
Checkpoint Reproducibility
PASS
```

---

# 9. DINOv3 输入分辨率消融

保持：

```text
相同 DINOv3 ViT-S/16
相同正常 Memory
相同 broken_small 22 张测试数据
相同异常分数计算方式
```

只修改输入分辨率：

```text
224
→
448
```

结果：

| 输入分辨率 | Patch Grid | Mean Pixel AUROC | Mean Pixel AP |
|---|---:|---:|---:|
| 224 | 14 × 14 | 0.995295 | 0.854635 |
| 448 | 28 × 28 | 0.998341 | 0.951768 |

Mean Pixel AP：

```text
0.854635
→
0.951768
```

提升：

```text
+0.097133
```

22 张图像：

```text
448 Wins / Ties / Losses

21 / 0 / 1
```

Bootstrap 95% CI：

```text
[+0.067179, +0.140345]
```

结论：

> 在当前 Bottle / broken_small / DINOv3 ViT-S/16 实验条件下，提高空间粒度能够明显改善小缺陷 Pixel-level 定位。

---

# 10. Autoencoder 容量消融

本实验只主动修改：

```text
Channel Capacity
```

其余条件保持不变：

```text
数据集
Seed
输入尺寸
Epoch
Batch Size
Optimizer
Learning Rate
Loss
Evaluation Protocol
```

实验配置：

| 模型 | Channel | 参数量 | Normal Test L1 | Mean Pixel AP |
|---|---|---:|---:|---:|
| Small | 16-32-64-128 | 270,355 | 0.03028 | 0.16262 |
| Base | 32-64-128-256 | 1,078,307 | 0.02506 | 0.19252 |
| Large | 64-128-256-512 | 4,307,011 | 0.02228 | 0.19280 |

---

## 10.1 Small → Base

Normal Test L1：

```text
0.03028
→
0.02506
```

Mean Pixel AP：

```text
0.16262
→
0.19252
```

说明：

> 当模型容量过小时，正常结构本身就难以重建，从而产生大量正常区域 Reconstruction Error。

增加到 Base 容量后：

```text
正常重建能力提高
+
异常定位能力提高
```

因此：

```text
Under-Capacity
是当前 Autoencoder Failure 的原因之一。
```

---

## 10.2 Base → Large

Normal Test L1：

```text
0.02506
→
0.02228
```

继续明显改善。

但 Mean Pixel AP：

```text
0.19252
→
0.19280
```

几乎没有继续提高。

与此同时：

```text
GT+ Reconstruction Error
也在下降
```

也就是说，模型变大以后：

```text
正常区域重建更好
异常区域也重建得更好
```

因此：

> Reconstruction Loss 更低，不代表 Anomaly Detection 一定更强。

---

# 11. Top-5 Failure Modes

## Failure Mode 1：正常高梯度结构产生 False Positive

Autoencoder 输出的 Reconstruction 相比原图更平滑。

因此以下正常结构容易产生高 Reconstruction Error：

```text
瓶口环纹
物体轮廓
高光
反射
高对比纹理
```

对 22 张 broken_small 图像进行 Sobel Gradient 分析：

```text
Top-1% 高异常分正常像素
平均 Gradient

≈

所有正常像素平均 Gradient 的 2.894 倍
```

另外：

```text
Top-1% False Positive 中

33.2%

位于正常像素 Top-10% 高梯度区域中
```

如果不存在关联，随机基线约为：

```text
10%
```

实际 enrichment：

```text
3.32 ×
```

并且：

```text
22 / 22 张图像
enrichment > 1
```

结论：

> Autoencoder 的高分正常像素显著富集在高梯度正常结构中。

---

# 12. Failure Mode 2：Autoencoder 容量不足

Small AE：

```text
Normal Test L1 = 0.03028
Mean Pixel AP  = 0.16262
```

Base AE：

```text
Normal Test L1 = 0.02506
Mean Pixel AP  = 0.19252
```

Small → Base 后：

```text
正常重建误差下降
异常定位指标提高
```

因此：

> 模型容量不足会导致正常结构 Reconstruction Mismatch，并产生额外 False Positive。

---

# 13. Failure Mode 3：容量继续增加后出现收益饱和

Base → Large：

```text
Normal Test L1：

0.02506
→
0.02228
```

但：

```text
Mean Pixel AP：

0.19252
→
0.19280
```

几乎不变。

说明：

> 当模型容量达到一定水平以后，继续增加容量会同时改善正常区域和异常区域的 Reconstruction，因此 anomaly contrast 不再明显增强。

---

# 14. Failure Mode 4：空间粒度不足影响小缺陷定位

DINOv3：

```text
224 输入
→ 14 × 14 Patch Grid

448 输入
→ 28 × 28 Patch Grid
```

Mean Pixel AP：

```text
0.85463
→
0.95177
```

说明：

> 对 small defect，过低的空间粒度可能限制异常边界和小区域的定位能力。

---

# 15. Failure Mode 5：Boundary / Annotation Sensitivity

DINOv3 `broken_small / 013.png`：

当输入：

```text
224
→
448
```

时：

```text
Pixel AUROC
略微提高

Pixel AP
略微下降
```

进一步分析发现：

```text
高分 Negative Pixel
主要集中在 Ground Truth Boundary 附近
```

因此：

> 整体 score separation 变好，不代表 Pixel AP 一定同步提高。

如果高分 False Positive 恰好进入 GT 边界附近，AP 仍然可能下降。

---

# 16. 原因验证

## 原因验证 A：Autoencoder Capacity

主动修改：

```text
Small
→
Base
→
Large
```

其余实验条件固定。

结果：

```text
Small → Base

Normal Reconstruction 改善
Pixel AP 改善
```

但：

```text
Base → Large

Normal Reconstruction 继续改善
Pixel AP 基本饱和
```

结论：

> Autoencoder 容量不足确实是 Normal Reconstruction Mismatch 的原因之一，但单纯扩大模型容量不能彻底解决 anomaly localization 问题。

状态：

```text
PASS
```

---

## 原因验证 B：DINOv3 Spatial Resolution

主动修改：

```text
224
→
448
```

结果：

```text
Mean Pixel AP
+0.09713

22 张中
21 张提高
```

Bootstrap 95% CI：

```text
[+0.06718, +0.14035]
```

结论：

> 当前 small-defect localization 的一个重要影响因素是 Representation 的空间粒度。

状态：

```text
PASS
```

---

# 17. 指标理解

## 17.1 AUROC 与 AP

AUROC 更关注：

```text
整体 Positive / Negative 排序能力
```

Pixel AP 更容易受到以下因素影响：

```text
类别极不平衡
高分 False Positive
Top Ranking 纯度
异常区域过小
边界误差
```

因此可能出现：

```text
AUROC 看起来不错
但 AP 很差
```

Autoencoder 实验就是一个明显案例。

---

## 17.2 Mean Per-Image 与 Global Metric

Mean Per-Image：

```text
先对每张图计算指标
再对 22 张图取平均

每张图权重相同
```

Global：

```text
把 22 张图所有 Pixel 合并
再统一计算指标

每个 Pixel 权重相同
```

两者回答的问题不同，不能随意互换。

---

# 18. 关键工程结论

对于当前测试的 Bottle small localized defect：

```text
DINOv3 Patch Representation
+
Local Nearest-Neighbor Matching
```

明显优于当前基础：

```text
Convolutional Autoencoder
+
Pixel Reconstruction Error
```

Autoencoder 的主要问题并不是：

```text
完全无法感知异常
```

而是：

```text
正常结构本身
也容易产生较高 Reconstruction Error
```

最终导致：

```text
High-score False Positive
污染异常排序
→
Pixel AP 降低
```

---

## 18.1 Reconstruction Loss 不是最终目标

实验显示：

```text
模型越大
→ Normal Reconstruction 越好
```

但：

```text
Anomaly Localization
并不会无限提高
```

因为异常本身也可能随着模型容量增加而被更好地重建。

因此：

> 训练 Reconstruction Loss 下降，不等价于异常检测能力提高。

---

## 18.2 Spatial Granularity 很重要

DINOv3：

```text
224 → 448
```

显著改善 small defect localization。

但更高空间分辨率也会带来：

```text
更多 Patch
更大的 Feature Memory
更高计算成本
更高显存需求
更多 Boundary Sensitivity
```

因此工程中需要同时考虑：

```text
质量
延迟
显存
Memory Size
吞吐
维护成本
```

---

# 19. 实验复现

项目目录：

```text
D:\AI_Lab\02_Projects\Project_001_Baseline_Experiment
```

激活环境：

```powershell
conda activate D:\conda_envs\ai_base
```

---

## 19.1 Autoencoder Canonical Baseline

训练并评估：

```powershell
python src/experiments/exp025_autoencoder_bottle_minimal.py
```

如果已有 Checkpoint，只进行 Evaluation：

```python
EVAL_ONLY = True
```

Checkpoint：

```text
results/exp025/autoencoder_bottle_minimal/autoencoder_seed42.pt
```

---

## 19.2 Autoencoder Failure Diagnosis

```powershell
python src/experiments/exp025_autoencoder_failure_diagnosis.py
```

输出目录：

```text
results/exp025/autoencoder_failure_diagnosis/
```

---

## 19.3 Autoencoder Capacity Ablation

```powershell
python src/experiments/exp025_autoencoder_capacity_ablation.py
```

输出目录：

```text
results/exp025/autoencoder_capacity_ablation/
```

---

## 19.4 Autoencoder Edge False Positive Analysis

```powershell
python src/experiments/exp025_autoencoder_edge_fp_analysis.py
```

输出目录：

```text
results/exp025/autoencoder_edge_fp_analysis/
```

---

## 19.5 DINOv3 Resolution Dataset Evaluation

```powershell
python src/experiments/exp024_dinov3_resolution_dataset_eval.py
```

输出目录：

```text
results/exp024/dinov3_resolution_dataset_eval/
```

---

## 19.6 DINOv3 Failure Analysis

```powershell
python src/experiments/exp024_dinov3_failure_tail_analysis.py
python src/experiments/exp024_dinov3_fp_spatial_diagnosis.py
```

---

# 20. 关键结果文件

```text
results/
│
├── exp024/
│   │
│   ├── dinov3_resolution_dataset_eval/
│   │   ├── per_image.csv
│   │   └── summary.csv
│   │
│   ├── dinov3_fp_spatial_diagnosis/
│   │
│   └── dinov3_cls_vs_mean_retrieval/
│
└── exp025/
    │
    ├── autoencoder_bottle_minimal/
    │   ├── per_image.csv
    │   ├── summary.csv
    │   └── autoencoder_seed42.pt
    │
    ├── autoencoder_failure_diagnosis/
    │   └── diagnostics.csv
    │
    ├── autoencoder_capacity_ablation/
    │   ├── summary.csv
    │   ├── per_image.csv
    │   └── failure_diagnostics.csv
    │
    └── autoencoder_edge_fp_analysis/
        ├── per_image.csv
        └── summary.csv
```

模型 Checkpoint 和大型原始数据不一定进入 Git 仓库。

必要时通过实验脚本重新生成。

---

# 21. Level 1 BOSS 最终验收状态

```text
两类不同方法对比                   PASS
同数据集比较                       PASS
统一 Evaluation Protocol          PASS
至少 3 个消融实验                  PASS
Top-5 Failure Modes               PASS
Validate >=2 Causes               PASS
Failure Analysis                  PASS
Checkpoint Reproducibility        PASS
可复现实验命令                     PASS
Metric Table                      PASS

10 分钟技术答辩                     PASS
最终知识验收                        PASS
README                            PASS
EXP025 实验笔记                    PASS
核心脚本语法检查                    PASS
Selective Git Staging             PASS

Git Commit      PASS
GitHub Push     PASS

Level 1         LEVEL COMPLETE