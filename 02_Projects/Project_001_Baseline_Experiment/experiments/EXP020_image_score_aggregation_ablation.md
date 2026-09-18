# EXP020 - Image Score Aggregation Ablation

## Goal

固定相同的 patch features、Coreset100、NN patch scores，
仅改变 image-level aggregation：

- max
- top5_mean
- top20_mean
- mean

验证 aggregation 对 anomaly ranking 和 operating point 的影响。

## Controlled Variables

- MVTec AD bottle
- Reference normal: 167
- Validation normal: 42
- Test: 83
- ResNet18
- layer2
- Candidate size: 5000
- Coreset size: 100
- Coreset seed: 42
- Euclidean NN distance

唯一主动变量：

image-score aggregation。

## Prediction

1. MAX 应精确复现 EXP018。
2. Mean 可能稀释 localized anomaly，尤其 broken_small。
3. Top-K 可能降低单个极端 patch 的支配作用。
4. 不预设 Top-K 一定改善 contamination。
5. 每种 aggregation 必须独立进行 threshold calibration。
6. 对同一图片：
   MAX >= Top5 >= Top20 >= Mean。
   但这不代表 AUROC 也按该顺序。

## Sanity Checks

Aggregation ordering:

- Validation violations = 0
- Test violations = 0

MAX reproduction:

- EXP018 AUROC = 0.984921
- EXP020 MAX AUROC = 0.984921
- delta = 0
- exact reproduction = True

## Results

| Aggregation | Overall AUROC | broken_large | broken_small | contamination | P95 Recall | P95 FPR |
|---|---:|---:|---:|---:|---:|---:|
| max | 0.984921 | 1.000000 | 1.000000 | 0.954762 | 0.888889 | 0.000000 |
| top5_mean | 0.993651 | 1.000000 | 1.000000 | 0.980952 | 0.952381 | 0.000000 |
| top20_mean | 0.996825 | 1.000000 | 1.000000 | 0.990476 | 0.984127 | 0.100000 |
| mean | 0.946825 | 0.997500 | 0.888636 | 0.959524 | 0.936508 | 0.450000 |

## P95 False Negatives

MAX:
- broken_large: 2
- broken_small: 0
- contamination: 5

Top5:
- broken_large: 0
- broken_small: 0
- contamination: 3

Top20:
- broken_large: 0
- broken_small: 0
- contamination: 1

Mean:
- broken_large: 0
- broken_small: 3
- contamination: 1

## Key Findings

### 1. Mean causes localized anomaly dilution

broken_small AUROC:

- MAX = 1.000000
- Top5 = 1.000000
- Top20 = 1.000000
- Mean = 0.888636

Mean averages a small number of abnormal patches with hundreds
of normal patches, weakening localized anomaly signals.

### 2. Aggregation contributes to contamination weakness

The patch features and patch scores were unchanged.

Only aggregation changed:

- MAX contamination AUROC = 0.954762
- Top20 contamination AUROC = 0.990476

Therefore contamination weakness cannot be explained by
feature representation alone.

Aggregation strategy is also an important factor.

This does NOT prove representation has no problem.

### 3. Top-K provides a middle ground

MAX can be dominated by one extreme patch.

Mean can dilute localized anomalies.

Top-K uses several high anomaly responses while avoiding
averaging all 784 patches.

### 4. Ranking metric != operating point

Top20 has the highest AUROC:

0.996825

but compared with Top5:

- FP: 0 -> 2, more false positives
- FN: 3 -> 1, fewer false negatives

Therefore Top20 trades increased false alarms for fewer missed defects.

The appropriate operating point depends on business costs.

### 5. Raw score magnitude != ranking quality

For every image:

MAX >= Top5 >= Top20 >= Mean

But AUROC measures ordering between normal and defective images,
not absolute score magnitude.

Therefore Top20 can have lower raw scores than MAX while producing
better normal-vs-defect ranking.

## FP / FN Reminder

FP = False Positive = normal product classified as defect = false alarm.

FN = False Negative = defective product classified as normal = missed defect.

Top20 vs Top5:

- FP: 0 -> 2 = false alarms increase
- FN: 3 -> 1 = missed defects decrease

## Conclusions

1. Image-level aggregation is a meaningful model-design variable.
2. Mean strongly hurts localized broken_small anomalies.
3. Top-K improves ranking for the current Bottle setup.
4. Contamination weakness is partly aggregation-related.
5. Best AUROC does not automatically mean best industrial operating point.
6. Threshold calibration must be repeated after changing aggregation.

## Limitations

- Bottle only.
- Coreset seed fixed at 42.
- Top-K only tested at k=5 and k=20.
- Test normal set has only 20 images, so FPR has coarse granularity.
- Current experiment does not isolate the exact spatial mechanism
  behind the contamination improvement.

## Status

EXP020 = PASS
LEVEL 1 = CONTINUE