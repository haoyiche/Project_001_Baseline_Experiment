# EXP006 Baseline Evaluation


## 实验目标

验证基于预训练ResNet18特征距离的工业异常检测Baseline。


## 实验假设

正常样本在特征空间中形成稳定分布。

异常样本距离正常中心更远。


## Dataset

MVTec AD

Category:

bottle


Train:

train/good


Test:

test/all


Train images:

209


Test images:

83


## Method


Feature Extractor:

ResNet18 pretrained


Feature Dimension:

512


Normal Model:

Feature Mean Center


Anomaly Score:

Euclidean Distance


Evaluation:

AUROC



## 实验结果


AUROC:

0.9730158730158731



## 分析


Baseline能够有效区分正常和异常样本。

说明ImageNet预训练特征具有一定工业视觉迁移能力。


## 局限


1. Feature来自通用分类模型。

2. 没有利用局部patch信息。

3. 对细微缺陷可能不足。



## 结论


完成第一个工业异常检测Baseline实验闭环。

后续可以研究更适合异常检测的方法。