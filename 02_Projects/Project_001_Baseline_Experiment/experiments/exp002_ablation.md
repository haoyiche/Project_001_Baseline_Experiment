# EXP002 Ablation Experiment


## 实验目标

验证单个参数变化对实验结果的影响。


## Baseline

配置：

epochs=10


结果：

score=20



## Ablation

修改：

epochs:

10 → 50


保持：

learning_rate=0.001

batch_size=32



## 我的预测

增加epochs后，

score提升。



## 实际结果

EXP001:

score=20


EXP002:

score=100



## 分析

只修改epochs参数，

score发生变化。

说明epochs影响实验结果。



## 结论

完成第一次Ablation实验。

验证控制变量实验流程。

## Result保存

实验结果自动保存：

EXP001:

results/EXP001_Baseline_result.txt


EXP002:

results/EXP002_Ablation_epochs50_result.txt


避免不同实验结果互相覆盖。