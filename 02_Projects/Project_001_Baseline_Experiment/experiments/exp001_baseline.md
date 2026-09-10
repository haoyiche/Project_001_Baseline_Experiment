# EXP001 Baseline


## 实验目标

建立第一个参数可控实验流程。


## 实验假设

修改配置文件参数，
可以影响实验行为。


## 初始参数

epochs=10


## 修改参数

epochs=50


## 我的预测

程序输出epochs变化。

## 实际结果

程序成功读取修改后的配置。

输出：

experiment:
EXP001_Baseline

learning_rate:
0.001

batch_size:
32

epochs:
50


## 分析

修改config文件中的epochs参数后，

main.py输出结果发生变化。

说明：

实验参数已经与代码逻辑解耦，

可以通过配置文件控制实验。


## 结论

EXP001完成第一次参数控制实验。

验证了：

config.yaml → main.py → 实验输出

的实验流程。

## Debug记录

### 问题1

错误：

AttributeError:
'NoneType' object has no attribute 'items'


原因：

run_experiment没有return结果。


解决：

增加return results。


---

### 问题2

错误：

datetime未定义。


原因：

缺少datetime导入。


解决：

from datetime import datetime


---

### 问题3

错误：

函数名称拼写不一致。


原因：

save_resulrts 和 save_result不同。


解决：

统一函数名称。