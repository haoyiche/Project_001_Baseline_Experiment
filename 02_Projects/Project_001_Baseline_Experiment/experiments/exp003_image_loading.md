## 1. 实验目标

验证图像从文件格式到深度学习模型输入格式的转换流程。

本实验目标：

1. 使用Python读取图片文件。
2. 查看图片基本属性。
3. 将图片转换为numpy数组。
4. 将numpy数组转换为PyTorch Tensor。
5. 理解H×W×C到C×H×W的维度转换。


---

# 2. 实验环境


Python环境：

ai_base


主要工具：

- Pillow
- NumPy
- PyTorch


实验路径：

Project_001_Baseline_Experiment


---

# 3. 实验输入


输入文件：

test.jpg


图片类型：

RGB图片


图片尺寸：

400 × 400


---

# 4. 实验流程


完整流程：


test.jpg

↓

PIL Image读取

↓

numpy array

↓

torch Tensor

↓

permute维度转换

↓

模型输入格式



---

# 5. 实验代码


主要代码流程：


```python
image = Image.open(image_path)


image_array = np.array(image)


tensor_image = torch.tensor(
    image_array
)


tensor_image = tensor_image.permute(
    2,0,1
)
````

---

# 6. 实验预测

## 预测1：图片读取

预计：

PIL成功读取RGB图片。

## 预测2：numpy转换

图片shape应该为：

H × W × C

即：

400 × 400 × 3

## 预测3：Tensor转换

Tensor初始维度保持：

400 × 400 × 3

## 预测4：permute转换

转换后：

3 × 400 × 400

符合深度学习模型输入格式。

---

# 7. 实际实验结果

运行命令：

```powershell
python .\src\image_test.py
```

输出：

```text
Image:
<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=400x400>


Image size:
(400,400)


Image array shape:
(400,400,3)


Image array data type:
uint8


Tensor image shape:
torch.Size([400,400,3])


Before permute:
torch.Size([400,400,3])


After permute:
torch.Size([3,400,400])
```

---

# 8. 结果分析

## 8.1 PIL Image

PIL读取图片后：

保存的是图像对象。

显示：

```
mode=RGB
```

说明图片包含三个颜色通道。

---

## 8.2 numpy array

转换后：

shape:

```
(400,400,3)
```

表示：

```
Height × Width × Channel
```

其中：

Height:

400

Width:

400

Channel:

3(RGB)

---

## 8.3 uint8数据类型

图片像素通常范围：

```
0-255
```

因此使用：

```
uint8
```

表示每个像素值。

---

## 8.4 Tensor转换

numpy转换为Tensor后：

维度仍保持：

```
H × W × C
```

即：

```
400 × 400 × 3
```

---

## 8.5 HWC到CHW转换

深度学习框架通常要求：

```
Channel × Height × Width
```

因此：

通过：

```python
permute(2,0,1)
```

完成：

```
400×400×3

↓

3×400×400
```

其中：

原来的Channel维度移动到第一维。

---

# 9. 我的理解

图片本质上是一个数字矩阵。

RGB图片由三个Channel组成。

计算机视觉模型不能直接读取jpg文件，

需要经过：

图片文件

↓

像素矩阵

↓

Tensor

同时为了符合深度学习框架输入要求，

需要将：

H×W×C

转换为：

C×H×W。

Dataset和数据预处理是视觉任务的重要组成部分。

如果输入数据格式错误，

模型无法正确学习。

---

# 10. 实验结论

本实验成功完成：

图片文件读取

↓

numpy转换

↓

Tensor转换

↓

维度调整

验证了视觉任务中最基础的数据处理流程。

EXP003 Image Loading Experiment 完成。

````

---

