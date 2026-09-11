# EXP004 MVTec AD Dataset Audit


## 实验目标

验证Dataset Audit工具在真实工业视觉数据上的适用性。


## 数据集

MVTec AD


## 实验结果

Total images:

6612


Image mode:

RGB:
4141

L:
2471


Image size:

主要为900×900


## 发现问题

1. 数据存在RGB和灰度两种格式。

2. 原始文件夹结构包含类别、划分、缺陷类型多个层级。

3. 当前Audit v1无法正确表达工业数据语义。


## 结论

基础扫描流程验证成功。

下一步需要升级工业数据结构解析。

# EXP004 MVTec AD Dataset Audit


## 实验目标

验证工业数据审计工具是否能够正确解析MVTec AD数据结构。


## 数据集

MVTec AD


## 实验假设

程序能够区分：

- 产品类别
- train/test划分
- 正常/缺陷类型


## 我的预测

输出：

Category

Train数量

Test缺陷类型


## 实际结果

成功解析15个工业类别。

例如：

bottle:

Train:
209

Test:

broken_large:
20

broken_small:
22

contamination:
21

good:
20


## 分析

Audit v2解决了Audit v1的问题。

v1将缺陷类型错误认为类别。

v2正确理解：

Category → Split → Defect Type


## 结论

完成MVTec AD数据结构解析。

具备进入Baseline实验阶段的条件。