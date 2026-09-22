# EXP023 Real Detector Evaluation & Failure Analysis

## 1. 实验定位

- **Level**：Level 1 — Visual Foundations & Experiment Capability
- **Learning Unit**：真实目标检测评估与失败分析
- **模型**：TorchVision Faster R-CNN ResNet50-FPN V2（COCO pretrained）
- **数据集**：COCO 2017 val
- **评估子集**：固定 500 张，`seed=42`
- **设备**：RTX 2060 Max-Q
- **训练**：无训练，仅推理与评估
- **当前状态**：EXP023 实验阶段完成；待通关测试、Git commit、GitHub push

本实验的目标不是“跑出一个 mAP 数字”，而是把已有的 TP / FP / FN、Precision / Recall / IoU、NMS、AP / mAP 理论连接到真实 detector 评估流程，并通过主动修改参数验证其行为。

---

## 2. 知识地图

完整评估链路：

```text
真实图像 + GT
    ↓
Faster R-CNN
    ↓
confidence filtering
    ↓
NMS
    ↓
detections
    ↓
COCO matching / evaluation
    ↓
AP50 / AP50-95 / AR100
    ↓
small / medium / large slice
    ↓
class-support slice
    ↓
failure analysis
```

核心理解：

```text
Confidence threshold
→ 控制保留哪些低分预测
→ 直接影响固定 operating point 的 Precision / Recall
→ 过高时会截断 PR 曲线高-recall部分

NMS threshold
→ 控制框之间的 suppression 强度
→ 太低可能误删相邻真实目标
→ 太高可能保留更多 duplicate

Input resolution
→ 改变进入 backbone / FPN 的有效像素信息
→ 对 small object 通常更敏感

COCO AP
≠
某一个固定 confidence threshold 下的 Precision / Recall / F1
```

---

## 3. 固定实验协议

### 3.1 数据

- COCO 2017 validation set
- 固定 500 张图像
- 抽样随机种子：`42`
- manifest：

```text
configs/exp023_coco_subset500_ids.json
```

图像目录：

```text
data/raw/coco2017/val2017_subset500
```

GT：

```text
data/raw/coco2017/annotations/instances_val2017.json
```

### 3.2 模型

```text
torchvision.models.detection.fasterrcnn_resnet50_fpn_v2
```

使用 COCO pretrained weights。

### 3.3 Baseline 参数

```text
score_thresh = 0.05
nms_thresh   = 0.50
maxDets      = 100
min_size     = 800
max_size     = 1333
```

### 3.4 主要指标

官方 COCOeval：

- AP50-95
- AP50
- AP75
- AP small
- AP medium
- AP large
- AR100

另外在 Failure Analysis 中使用自定义简化匹配：

```text
simplified Precision / Recall / F1 @ IoU=0.50
```

该简化指标只用于错误解释和固定 operating-point 分析，不替代官方 COCOeval。

---

# 4. Baseline

500 张固定子集结果：

| Metric | Value |
|---|---:|
| Predictions | 16,949 |
| AP50-95 | 0.4885 |
| AP50 | 0.7046 |
| AP75 | 0.5314 |
| AP small | 0.3470 |
| AP medium | 0.5224 |
| AP large | 0.6605 |
| AR100 | 0.6107 |
| Mean model latency | 907.6 ms |

> latency 仅为本脚本中的粗略 runtime observation，不是 EXP022 那种严格工程 benchmark。

### Baseline 观察

```text
AP50 > AP50-95
```

说明随着 IoU 要求变严格，检测性能下降。

并且：

```text
AP_small < AP_medium < AP_large
```

说明在当前固定 COCO500 子集上，小目标明显更困难。

---

# 5. Active Modification 1 — Confidence Threshold Ablation

## 5.1 先预测

预测：

```text
confidence threshold ↑
→ prediction 数量 ↓
→ 单点 Precision 可能 ↑
→ Recall 只能不变或 ↓
→ AP 可能保持或下降
```

原因：

提高 threshold 只会删除已有 detection，不会产生新的 GT 匹配。

---

## 5.2 实验结果

固定：

```text
NMS = 0.50
resolution = 800 / 1333
```

只修改 score threshold。

| score threshold | Predictions | AP50-95 | AP50 | AR100 |
|---:|---:|---:|---:|---:|
| 0.05 | 16,949 | 0.4885 | 0.7046 | 0.6107 |
| 0.20 | 8,758 | 0.4810 | 0.6904 | 0.5784 |
| 0.50 | 5,020 | 0.4655 | 0.6603 | 0.5358 |
| 0.80 | 3,078 | 0.4320 | 0.5999 | 0.4775 |

从 `0.05 → 0.80`：

```text
Predictions:
16949 → 3078
约 -81.8%

AP50-95:
0.4885 → 0.4320
约 -11.6%

AR100:
0.6107 → 0.4775
约 -21.8%
```

---

## 5.3 Object-size sensitivity

| score threshold | AP small | AP medium | AP large |
|---:|---:|---:|---:|
| 0.05 | 0.3470 | 0.5224 | 0.6605 |
| 0.20 | 0.3340 | 0.5140 | 0.6550 |
| 0.50 | 0.3140 | 0.4960 | 0.6380 |
| 0.80 | 0.2670 | 0.4600 | 0.6080 |

从 `0.05 → 0.80`：

```text
AP_small:
0.347 → 0.267
约 -23.1%

AP_medium:
0.522 → 0.460
约 -11.9%

AP_large:
0.661 → 0.608
约 -8.0%
```

AR 也表现出类似趋势：

```text
AR_small:
0.452 → 0.283
约 -37.4%

AR_medium:
0.633 → 0.494
约 -22.0%

AR_large:
0.763 → 0.655
约 -14.2%
```

### 结论

在当前模型、固定 COCO500 子集和当前 threshold sweep 下，小目标对高 confidence threshold 更敏感。

但不能据此断言：

```text
“小目标低分完全是因为尺寸小”
```

因为类别、遮挡、实例密度、纹理、场景等因素没有被控制。

---

# 6. Confidence Threshold：AP vs 固定 Operating Point

Failure Analysis 使用简化 IoU=0.50 matching 后：

| Variant | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---:|---:|---:|
| baseline | 0.1948 | 0.8692 | 0.3183 |
| conf0.80 | 0.7726 | 0.6260 | 0.6916 |

但官方 COCO AP50-95：

```text
0.4885 → 0.4320
```

所以出现：

```text
固定 operating-point F1 ↑
但整体 AP ↓
```

并不矛盾。

原因：

```text
confidence threshold ↑
→ 大量低分 FP 被删除
→ 单点 Precision ↑

同时
→ 一部分低分 TP 被删除
→ Recall ↓
→ PR 曲线高-recall区域被截断
→ AP ↓
```

### 核心理解

```text
F1 = 某一个 operating point 的表现
AP = 整条 score ranking / PR 曲线上的整体表现
```

两者回答的问题不同。

---

# 7. Active Modification 2 — NMS IoU Threshold Ablation

## 7.1 先预测

NMS 的规则：

```text
如果两个框 IoU > NMS threshold
→ 低分框被 suppression
```

因此：

```text
NMS threshold ↓
→ suppression 更激进
→ 保留框数量更少
→ duplicate 倾向减少
→ 相邻真实目标也可能被误删

NMS threshold ↑
→ suppression 更宽松
→ 保留框数量更多
→ duplicate 倾向增加
→ 可能保留更多真实目标候选框
```

AP / AR 不保证单调变化。

---

## 7.2 实验结果

固定：

```text
score_thresh = 0.05
resolution = 800 / 1333
```

| NMS IoU | Predictions | AP50-95 | AP50 | AP75 | AR100 |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 13,153 | 0.4819 | 0.6950 | 0.5258 | 0.5896 |
| 0.50 | 16,949 | 0.4885 | 0.7046 | 0.5314 | 0.6107 |
| 0.70 | 24,646 | 0.4890 | 0.6824 | 0.5438 | 0.6390 |

框数量验证：

```text
NMS 0.30 < NMS 0.50 < NMS 0.70
```

符合预测。

---

## 7.3 结果解释

`NMS=0.30`：

```text
suppression 更激进
→ AR100 0.5896 < baseline 0.6107
→ AP50-95 0.4819 < baseline 0.4885
```

说明过于激进的 NMS 可能删除真实目标的有效候选框。

`NMS=0.70`：

```text
AR100:
0.6107 → 0.6390

AP75:
0.5314 → 0.5438

AP50:
0.7046 → 0.6824
```

合理解释：

更宽松的 NMS 保留更多高度重叠框，提高了找到高 IoU 匹配框的机会，因此 AR / AP75 可能获益；但同时重复框更多，可能损伤 AP50。

注意：

```text
AP50-95:
0.4885 → 0.4890
```

差异只有约 `+0.0005`，不能宣称 NMS=0.70 明显优于 0.50。

---

# 8. NMS Cause Validation

定义：

```text
overlapping_same_class_gt
```

作为密集 / 相邻同类目标的 proxy。

Baseline：

```text
overlapping_same_class_gt
miss rate = 0.1167

other_gt
miss rate = 0.1348
```

NMS=0.30：

```text
overlapping_same_class_gt
miss rate = 0.1913

other_gt
miss rate = 0.1621
```

增量：

```text
overlapping same-class GT:
+0.0746

other GT:
+0.0273
```

### 结论

不能写：

```text
“重叠目标天生更容易漏检”
```

因为 baseline 中它的 miss rate 反而更低。

正确结论是：

> 在当前 overlapping_same_class_gt proxy 下，NMS 从 0.50 降至 0.30 后，重叠同类目标的 miss rate 增幅明显大于其他 GT，支持“激进 NMS 对重叠同类目标造成更大的额外漏检损失”。

---

# 9. Active Modification 3 — Input Resolution Ablation

## 9.1 先预测

```text
resolution ↓
→ inference 更快

resolution ↓
→ overall AP 预期下降

resolution ↓
→ small-object AP 预期比 large-object 更容易下降
```

---

## 9.2 实验结果

固定：

```text
score_thresh = 0.05
NMS = 0.50
```

| Resolution | AP50-95 | AP50 | AP75 | AP small | AP medium | AP large | AR100 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 / 1333 | 0.4885 | 0.7046 | 0.5314 | 0.3470 | 0.5224 | 0.6605 | 0.6107 | 907.6 ms |
| 600 / 1000 | 0.4891 | 0.6880 | 0.5328 | 0.3045 | 0.5010 | 0.6749 | 0.6085 | 539.1 ms |
| 400 / 667 | 0.4577 | 0.6552 | 0.5014 | 0.2593 | 0.4691 | 0.6776 | 0.5716 | 392.6 ms |

---

## 9.3 Speed

粗略 mean latency：

```text
800 → 600:
907.6 ms → 539.1 ms
约 -40.6%

800 → 400:
907.6 ms → 392.6 ms
约 -56.7%
```

这只说明明显的速度趋势。

不能把这里当作严格 deployment benchmark，因为：

- 无完整 warmup protocol
- 无重复 trial
- 无 balanced order
- 无系统噪声控制

严格工程 latency benchmark 已在 EXP022 中单独处理。

---

## 9.4 Accuracy

`800 → 600`：

```text
AP50-95:
0.4885 → 0.4891
```

基本持平。

不能因为 `+0.0006` 就宣称 600 更优。

`800 → 400`：

```text
AP50-95:
0.4885 → 0.4577

约 -6.3%
```

说明进一步降低输入分辨率后，整体检测性能明显下降。

---

## 9.5 Small-object sensitivity

`800 → 400`：

```text
AP_small:
0.3470 → 0.2593
约 -25.3%

AP_medium:
0.5224 → 0.4691
约 -10.2%

AP_large:
0.6605 → 0.6776
约 +2.6%
```

### 原因链

```text
resolution ↓
→ 小目标有效像素减少
→ 边缘 / 纹理 / 形状信息更容易损失
→ backbone / FPN 中有效表示进一步变弱
→ small-object detection 更困难
```

不能因此推出：

```text
“降低分辨率能提升 large-object AP”
```

当前 large AP 上升只能作为本子集上的观察。

---

# 10. Resolution Cause Validation

简化 IoU=0.50 GT matching：

Baseline：

```text
small  miss rate = 0.2210
medium miss rate = 0.0892
large  miss rate = 0.0359
```

Res400：

```text
small  miss rate = 0.3369
medium miss rate = 0.1014
large  miss rate = 0.0294
```

变化：

```text
small:
+0.1159

medium:
+0.0122

large:
-0.0065
```

### 结论

> 在当前 Faster R-CNN、固定 COCO500 和本实验协议下，将输入从 800/1333 降至 400/667 显著增加了 small-object miss rate，而 large-object miss rate 未出现同方向恶化。

不能证明：

- 低分辨率一定伤害所有数据集的小目标
- 低分辨率一定有利于 large object

---

# 11. Class-frequency / Evaluation-support Analysis

## 11.1 问题定义

这里统计的是：

```text
固定 COCO500 验证子集中
每个 category 的 GT instance count
```

它表示：

```text
evaluation support
```

不是：

```text
训练集 class frequency
```

因此不能直接用它证明训练 class imbalance 的因果影响。

---

## 11.2 Support 分布

```text
COCO categories total : 80
Categories represented: 79

Min GT count : 1
Max GT count : 1186
```

类别支持度极不均衡。

Top support 示例：

| Category | GT | AP50-95 |
|---|---:|---:|
| person | 1186 | 0.6100 |
| car | 214 | 0.5472 |
| chair | 161 | 0.3476 |
| bottle | 124 | 0.4153 |
| cup | 124 | 0.5613 |
| book | 88 | 0.2381 |

低 support 示例：

| Category | GT | AP50-95 |
|---|---:|---:|
| toaster | 1 | 0.9000 |
| scissors | 2 | 0.3262 |
| toothbrush | 2 | 0.4040 |
| bear | 3 | 0.7990 |
| stop sign | 5 | 0.5875 |
| microwave | 5 | 0.5297 |

---

## 11.3 反例

```text
toaster
GT=1
AP=0.9000

bear
GT=3
AP=0.7990

fire hydrant
GT=7
AP=0.7743
```

但：

```text
book
GT=88
AP=0.2381

handbag
GT=60
AP=0.2480
```

因此：

```text
GT 少 ≠ AP 必然低
GT 多 ≠ AP 必然高
```

---

## 11.4 Correlation

```text
Pearson GT count vs AP50-95  = +0.0116
Spearman GT count vs AP50-95 = -0.4291
```

Pearson 接近 0：

```text
当前样本中没有明显线性关系
```

Spearman 为负：

```text
当前类别排名中存在一定负向单调趋势
```

但不能解释为：

```text
“GT 越多导致 AP 越低”
```

因为：

- 当前统计的是验证子集 support
- 不是训练 frequency
- 类别难度不同
- object size 不同
- 遮挡不同
- 场景不同
- 极低 support 类别的 AP 很不稳定

例如：

```text
toaster GT=1 AP=0.90
```

不能据此说模型对 toaster 普遍特别强。

正确说法：

> 样本支持极少时，per-class AP 的统计稳定性很差，只能描述当前极小样本，不能代表模型对该类别的普遍能力。

---

# 12. Failure Analysis

Failure Analysis 使用：

```text
simplified greedy GT-pred matching
IoU threshold = 0.50
```

目的：

```text
解释错误模式
```

不是替代官方 COCOeval。

---

## 12.1 Baseline Top-5 Failure Modes

| Failure mode | Count |
|---|---:|
| Background FP | 8820 |
| Localization error | 3416 |
| Classification error | 1069 |
| Missed GT | 497 |
| Duplicate detection | 342 |

注意：

- `background_fp`
- `localization_error`
- `classification_error`
- `duplicate_detection`

是 prediction-side error。

`missed_gt` 是 GT-side error。

因此它们不是同一分母下的百分比，不应该直接画成五类占比饼图。

---

## 12.2 Failure definition

### Background FP

预测框与正确类别 GT 没有足够重叠，同时也没有以 `IoU >= 0.50` 空间匹配到其他类别 GT。

### Classification error

预测框与某个 GT：

```text
IoU >= 0.50
```

但 category 错误。

### Localization error

类别正确，但：

```text
0.10 <= IoU < 0.50
```

说明识别到大致位置，但框的位置或尺度不够准确。

### Duplicate detection

某个 GT 已经被更高 score prediction 匹配，又出现同类别且：

```text
IoU >= 0.50
```

的额外预测。

### Missed GT

GT 最终没有被任何有效同类别预测匹配。

---

# 13. Cause Validation 1 — Confidence Threshold

Baseline：

```text
Background FP = 8820
Missed GT     = 497
```

Conf0.80：

```text
Background FP = 392
Missed GT     = 1421
```

变化：

```text
Background FP:
8820 → 392
-8428
约 -95.6%

Missed GT:
497 → 1421
+924
约 +185.9%
```

### 结论

强支持：

```text
confidence threshold ↑
→ 大量低分 FP 被删除
→ FP 显著减少

同时
→ 一部分低分 TP 被删除
→ Missed GT 大幅增加
→ Recall 下降
```

---

# 14. Cause Validation 2 — Aggressive NMS

已在第 8 节给出。

核心结果：

```text
overlapping same-class GT
miss-rate increase = +0.0746

other GT
miss-rate increase = +0.0273
```

支持：

> NMS=0.30 的额外漏检损失在重叠同类目标 slice 上更明显。

---

# 15. Cause Validation 3 — Resolution / Object Size

已在第 10 节给出。

核心结果：

```text
small miss rate:
0.2210 → 0.3369

medium:
0.0892 → 0.1014

large:
0.0359 → 0.0294
```

支持：

> 降低输入分辨率对 small object 的额外漏检影响明显大于 medium / large。

---

# 16. Simplified Matching 的边界

当前 simplified Precision / Recall / F1 只使用：

```text
固定 IoU=0.50
自定义 greedy matching
```

没有完整复现 COCOeval 的：

- IoU 0.50:0.95
- crowd / ignore
- area range
- maxDets
- precision-recall interpolation

因此：

```text
simplified metric
→ 用于错误解释和 operating-point 分析

official COCOeval
→ 用于正式 AP / AR 报告
```

两者不能互相替代。

---

# 17. 本实验真正掌握的内容

## 17.1 Confidence threshold

我现在能够解释：

```text
threshold ↑
→ prediction ↓
→ fixed-point Precision 可能 ↑
→ Recall ↓ 或不变
→ AP 可能下降
```

并理解：

```text
F1 ↑
不代表
AP ↑
```

---

## 17.2 NMS

我现在能够解释：

```text
NMS threshold ↓
→ suppression 更激进

NMS threshold ↑
→ suppression 更宽松
```

并理解 NMS 不是简单单调参数：

- 太低会误删真实相邻目标
- 太高会保留更多 duplicate
- AP / AR 需要实验验证

---

## 17.3 Resolution

我现在能够解释：

```text
resolution ↓
→ compute ↓
→ latency 通常 ↓
→ small target 信息损失更明显
```

并且不会只看速度就宣布部署方案。

---

## 17.4 AP / Precision / Recall / F1

我现在能够区分：

```text
Precision / Recall / F1
→ 一个固定 operating point

AP
→ 整条 PR curve / ranking behavior
```

---

## 17.5 Evaluation support vs class imbalance

我现在能够区分：

```text
validation GT count
→ evaluation support

training class frequency
→ training distribution
```

不能把两者直接等同。

---

## 17.6 Failure analysis

我已经能够从：

```text
“AP 不高”
```

继续拆成：

```text
Background FP
Localization
Classification
Duplicate
Missed GT
```

并通过参数 ablation 验证部分失败原因。

---

# 18. 实验限制

1. 只使用固定 500 张 COCO validation 子集，不是官方完整 COCO benchmark。
2. per-class AP 对极低 support 类别非常不稳定。
3. class-support 分析不能证明训练 class imbalance 的因果作用。
4. Failure Analysis 使用 simplified IoU=0.50 matching，不完全等价于 COCOeval。
5. overlapping_same_class_gt 只是 dense-target proxy，不是 pre-NMS proposal 级因果证据。
6. Resolution ablation 的 latency 只是粗略运行时间，不是严格工程 benchmark。
7. NMS=0.70 与 baseline 的 AP50-95 差异极小，不能宣称显著更优。
8. 当前结果只适用于本模型、本固定子集和本实验协议。

---

# 19. 可复现实验命令

## Baseline

```powershell
python src/experiments/exp023_detector_baseline.py `
  --limit 500 `
  --score-thresh 0.05 `
  --nms-thresh 0.50 `
  --min-size 800 `
  --max-size 1333 `
  --tag baseline500
```

## Confidence Ablation

```powershell
python src/experiments/exp023_confidence_ablation.py
```

## NMS=0.30

```powershell
python src/experiments/exp023_detector_baseline.py `
  --limit 500 `
  --score-thresh 0.05 `
  --nms-thresh 0.30 `
  --min-size 800 `
  --max-size 1333 `
  --tag nms030
```

## NMS=0.70

```powershell
python src/experiments/exp023_detector_baseline.py `
  --limit 500 `
  --score-thresh 0.05 `
  --nms-thresh 0.70 `
  --min-size 800 `
  --max-size 1333 `
  --tag nms070
```

## Resolution=600

```powershell
python src/experiments/exp023_detector_baseline.py `
  --limit 500 `
  --score-thresh 0.05 `
  --nms-thresh 0.50 `
  --min-size 600 `
  --max-size 1000 `
  --tag res600
```

## Resolution=400

```powershell
python src/experiments/exp023_detector_baseline.py `
  --limit 500 `
  --score-thresh 0.05 `
  --nms-thresh 0.50 `
  --min-size 400 `
  --max-size 667 `
  --tag res400
```

## Class Support

```powershell
python src/experiments/exp023_class_support_analysis.py
```

## Failure Analysis

```powershell
python src/experiments/exp023_failure_analysis.py
```

---

# 20. 主要输出文件

```text
configs/
└── exp023_coco_subset500_ids.json

results/exp023/
├── baseline500_predictions.json
├── baseline500_summary.csv
├── nms030_predictions.json
├── nms030_summary.csv
├── nms070_predictions.json
├── nms070_summary.csv
├── res600_predictions.json
├── res600_summary.csv
├── res400_predictions.json
├── res400_summary.csv
├── confidence_ablation/
│   └── confidence_ablation_summary.csv
├── class_support/
│   └── class_support_per_class.csv
└── failure_analysis/
    ├── failure_summary_by_variant.csv
    ├── baseline_top5_failure_modes.csv
    ├── cause_validation_confidence.csv
    ├── cause_validation_nms.csv
    ├── cause_validation_resolution.csv
    └── examples/
```

---

# 21. EXP023 当前验收状态

```text
固定数据协议                         PASS
真实 detector                        PASS
Baseline500                          PASS

Confidence threshold ablation        PASS
NMS threshold ablation               PASS
Resolution ablation                  PASS
Class-support analysis               PASS
Failure analysis                     PASS

主动修改 >= 3                        PASS
先预测后验证                         PASS
Top-5 failure modes                  PASS
原因验证 >= 2                        PASS
正式实验笔记                         PASS

独立通关测试                         PENDING
Git commit                           PENDING
GitHub push                          PENDING
```

因此：

```text
EXP023 = CONTINUE
```

还不能标记 COMPLETE，剩余：

```text
独立通关测试
→ Git commit
→ GitHub push
```

---

# 22. 最终实验结论

1. **Confidence threshold 是 deployment operating-point 参数，不是 AP 的替代品。**  
   提高 threshold 可以显著减少 FP、提高单点 Precision / F1，但同时损失 Recall，并可能降低整体 AP。

2. **NMS threshold 不存在简单的单调最优关系。**  
   激进 NMS 会减少 duplicate，但也可能额外伤害重叠同类目标；宽松 NMS 可以提高 Recall，但会保留更多重叠框。

3. **降低输入分辨率能显著降低计算成本，但 small object 是最明显的受害者。**  
   在本实验中，400/667 分辨率导致 small-object AP 和 miss rate 明显恶化。

4. **类别 evaluation support 与 AP 之间不存在简单一一对应关系。**  
   极低 support 类别的 AP 非常不稳定，也不能从验证子集 support 推断训练 class imbalance 因果关系。

5. **平均 AP 不能解释模型为什么失败。**  
   将错误拆成 background FP、localization、classification、duplicate 和 missed GT 后，才能进一步通过 ablation 验证具体失败机制。

6. **真实检测评估的核心不是“得到一个 mAP”，而是理解参数、数据和失败模式之间的关系，并明确每个结论的实验边界。**
