# EXP007 Failure Analysis


## 1. Experiment Overview


### Goal

分析ResNet18 Feature Distance Baseline在MVTec AD bottle任务中的异常检测行为。


### Dataset

Dataset:

MVTec AD


Category:

bottle


Train:

good samples


Test:

good + defect samples



---

# 2. Baseline Method


## Feature Extractor

Model:

ResNet18 pretrained on ImageNet


Feature Dimension:

512



## Normal Modeling


训练阶段：

train/good


计算所有正常样本feature均值：

center = mean(features)



## Anomaly Score


使用：

Euclidean Distance


score = distance(feature, center)



## Evaluation Metric

AUROC



---

# 3. Experiment Result


Category:

bottle


Test Images:

83


AUROC:

0.9730158730158731



---

# 4. Top Anomaly Analysis


Highest anomaly score samples:


1.

contamination/006.png

Score:

14.455582



2.

broken_large/003.png

Score:

14.234188



3.

contamination/010.png

Score:

13.863091



Observation:


模型能够有效检测明显视觉变化。


主要异常类型：

- contamination
- broken_large



---

# 5. Hard Defect Analysis


Lowest anomaly score defect samples:


主要类型：

- broken_small
- contamination



Example:


broken_small/007.png


Score:

4.028722



Analysis:


当前方法使用Global Feature。


对于小区域缺陷：

- 缺陷面积占比小
- Resize导致细节损失
- 全局feature变化有限


因此容易漏检。



---

# 6. False Positive Analysis


Highest score normal sample:


good/006.png


Score:

7.1333604



Analysis:


正常样本存在：

- 光照变化
- 姿态变化
- 外观差异


导致feature距离normal center较远。



---

# 7. Baseline Limitation


Current method:


Image-level Feature

+

Global Distance



Limitations:


1. 无法利用局部patch信息。


2. 对微小缺陷敏感度不足。


3. 无法输出缺陷位置。



---

# 8. Future Improvement


下一阶段方向:


1. Patch-level Feature


2. Memory Bank


3. Local Similarity


4. PatchCore方法



---

# 9. Conclusion


完成第一个工业异常检测Baseline实验闭环。


Baseline证明：

ImageNet pretrained feature具有一定工业异常检测能力。


同时发现：

细粒度局部缺陷是主要挑战。


后续需要研究更强视觉表征和局部异常检测方法。