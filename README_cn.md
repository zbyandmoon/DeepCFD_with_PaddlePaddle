# 【论文复现赛】DeepCFD

[English](./README.md) | 简体中文

- [【论文复现赛】DeepCFD](#论文复现赛deepcfd)
	- [一、论文简介](#一论文简介)
		- [1.1 论文信息](#11-论文信息)
		- [1.2 实现方法](#12-实现方法)
		- [1.3 预测结果](#13-预测结果)
	- [二、复现精度](#二复现精度)
		- [2.1 验收标准](#21-验收标准)
		- [2.2 指标实现情况](#22-指标实现情况)
		- [2.3 复现地址](#23-复现地址)
	- [三、数据集](#三数据集)
	- [四、环境依赖](#四环境依赖)
	- [五、快速开始](#五快速开始)
	- [六、代码结构与参数说明](#六代码结构与参数说明)
		- [6.1 代码结构](#61-代码结构)
		- [6.2 参数说明](#62-参数说明)
	- [七、模型信息](#七模型信息)

本项目基于PaddlePaddle框架复现DeepCFD网络。

## 一、论文简介

计算流体力学（Computational fluid dynamics, CFD）通过求解Navier-Stokes方程（N-S方程），可以获得流体的各种物理量的分布，如密度、压力和速度等，在微电子系统、土木工程和航空航天等领域应用广泛。然而，在某些复杂的应用场景中，如机翼优化和流体与结构相互作用方面，需要使用千万级甚至上亿的网格对问题进行建模（如下图所示，下图展示了F-18战斗机的全机内外流一体结构化网格模型），导致CFD的计算量非常巨大。因此，目前亟需发展出一种相比于传统CFD方法更高效，且可以保持计算精度的方法。这篇文章的作者提到，可以使用深度学习的方法，通过训练少量传统CFD仿真的数据，构建一种数据驱动（data-driven）的CFD计算模型，来解决上述的问题。

<img src="http://www.cannews.com.cn/files/Resource/attachement/2017/0511/1494489582596.jpg" alt="img" style="zoom:80%;" />

### 1.1 论文信息

* 引用：[1] Ribeiro M D, Rehman A, Ahmed S, et al. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks[J]. arXiv preprint arXiv:2004.08826, 2020.
* 论文地址：https://arxiv.org/abs/2004.08826
* 参考项目：https://github.com/mdribeiro/DeepCFD

### 1.2 实现方法

作者提出了一个基于卷积神经网络（Convolutional neural networks, CNN）的CFD计算模型，称作DeepCFD，该模型可以同时计算流体流过任意障碍物的流场。该方法有以下几个特点：

1. DeepCFD本质上是一种基于CNN的代理模型，可以用于快速计算二维非均匀稳态层流流动，相比于传统的CFD方法，该方法可以在保证计算精度的情况下达到至少三个数量级的加速。

2. DeepCFD可以同时计算流体在x方向和y方向的流体速度，同时还能计算流体压强。

3. 训练该模型的数据由OpenFOAM（一种开源CFD计算软件）计算得到。

下面两张图分别为该方法的计算示意图和网络结构图。文中使用的DeepCFD网络基本结构为有3个输入和3个输出的U-net型网络。该模型输入为计算域中障碍物的符号距离函数（Signed distance function, SDF）、计算域边界的SDF和流动区域的标签；输出为流体的x方向速度、y方向速度以及流体压强。该模型的基本原理就是利用编码器部分的卷积层将3个输入下采样，变为中间量，然后使用相同结构的解码器中的反卷积层将中间量上采样为3个流体物理量输出。

![compute_process.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/compute_process.png?raw=true)

![DeepCFD_Net.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/DeepCFD_Net.png?raw=true)

### 1.3 预测结果

下图展示了原文的预测结果，文中评价模型的优劣共包含四个指标：Total MSE、Ux MSE、Uy MSE、p MSE（MSE的意思是均方根误差）。

![metrics.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/metrics.png?raw=true)

下图展示了某种形状障碍物的CFD（注：simpleFOAM是OpenFOAM求解器的一种）和DeepCFD流场计算结果对比。

![pytorch_contour.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/pytorch_contour.png?raw=true)

## 二、复现精度

### 2.1 验收标准

复现的验收标准如下：

![standard.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/standard.png?raw=true)

### 2.2 指标实现情况

复现的实现指标如下：

```python
Total MSE = 1.8955801725387573
Ux MSE = 0.6953578591346741
Uy MSE = 0.21001338958740234
p MSE = 0.9902092218399048
```

其中，Total MSE、Ux MSE和Uy MSE在验收标准范围内，p MSE略小于验收标准的最小值。

### 2.3 复现地址

论文复现地址：

AI Studio: https://aistudio.baidu.com/aistudio/projectdetail/4400677?contributionType=1

github: https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle/tree/main/paddle

## 三、数据集

数据集使用原作者利用OpenFOAM计算的CFD算例，共981组，分为两个文件（dataX.pkl, dataY.pkl），两个文件大小都是152 MB，形状均为[981, 3, 172, 79]。dataX.pkl包括三种输入：障碍物的SDF、计算域边界的SDF和流动区域的标签；dataY.pkl包括三种输出：流体的x方向速度、y方向速度和流体压强。数据获取使用的计算网格为172×79。

数据集地址：https://aistudio.baidu.com/aistudio/datasetdetail/162674

或https://www.dropbox.com/s/kg0uxjnbhv390jv/Data_DeepCFD.7z?dl=0

## 四、环境依赖

* 硬件：GPU、CPU
* 框架：PaddlePaddle >= 2.0.0

## 五、快速开始

**step1：克隆本项目**

```python
git clone https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle.git
```

**step2：安装依赖项**

```python
cd DeepCFD_with_PaddlePaddle
pip install -r requirements.txt
```

**step3：配置数据集**

从https://www.dropbox.com/s/kg0uxjnbhv390jv/Data_DeepCFD.7z?dl=0 下载得到 Data_DeepCFD.7z，将dataX.pkl和dataY.pkl解压到DeepCFD_with_PaddlePaddle/data目录，如下所示。

![data.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/data.png?raw=true)

**step4：训练模型**

**单卡训练**

```python
python train.py
```

**多卡训练**

以四卡为例，

```python
python -m paddle.distributed.launch --gpus=0,1,2,3 train.py
```

结果保存在result文件中（注：result文件夹中已经包含了一个完整的训练过程，可在训练前将其清空）。多卡训练会额外生成一个./log/文件夹，存放训练日志

```python
.
├── log
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
└── train.py
```

部分训练日志如下所示

```python
Epoch #1
	Train Loss = 2709583278.0
	Train Total MSE = 63787.10677842566
	Train Ux MSE = 18700.295280612245
	Train Uy MSE = 3150.4019494237427
	Train p MSE = 41936.41287809767
	Validation Loss = 172989540.0
	Validation Total MSE = 5005.818114406779
	Validation Ux MSE = 2492.9440413135594
	Validation Uy MSE = 136.1532971398305
	Validation p MSE = 2376.7206832627116
Epoch #2
	Train Loss = 235236727.0
	Train Total MSE = 2468.315415451895
	Train Ux MSE = 1449.831023369169
	Train Uy MSE = 60.7796064994078
	Train p MSE = 957.7047905657798
	Validation Loss = 48469500.0
	Validation Total MSE = 894.0722722457627
	Validation Ux MSE = 704.5147047139831
	Validation Uy MSE = 19.184117617849576
	Validation p MSE = 170.37347887976694
```

**step5：评估模型**

```python
python eval.py
```

此时的输出为：

```python
Total MSE is 1.895322561264038, Ux MSE is 0.6951090097427368, Uy MSE is 0.21001490950584412, p MSE is 0.9901986718177795
```

**step6：使用预训练模型预测**

考虑到需要展示流场图像对比结果，单独写了一个predict.ipynb来进行模型的验证，需要在Jupyter notebook环境中运行。

```python
jupyter notebook predict.ipynb
```

某个障碍物的流场预测结果展示如下：


![paddle_contour.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/paddle_contour.png?raw=true)

## 六、代码结构与参数说明

### 6.1 代码结构

```python
DeepCFD_with_PaddlePaddle
├─ config
│    └─ config.ini
├─ data
│    └─ README.md
├─ model
│    └─ UNetEx.py
├─ result
│    ├─ DeepCFD_965.pdparams
│    ├─ results.json
│    └─ train_log.txt
└─ utils
       ├─ functions.py
       └─ train_functions.py        
├─ README.md
├─ README_cn.md
├─ eval.py
├─ train.py
├─ predict.ipynb
├─ requirements.txt
```

### 6.2 参数说明

可以在/DeepCFD_with_PaddlePaddle/config/config.ini中设置训练的参数，包括以下内容：

| 参数             | 推荐值               | 额外说明                                      |
| ---------------- | -------------------- | --------------------------------------------- |
| batch_size       | 64                   |                                               |
| train_test_ratio | 0.7                  | 训练集占数据集的比例，0.7即训练集70%测试集30% |
| learning_rate    | 0.001                |                                               |
| weight_decay     | 0.005                | AdamW专用，若修改优化算法需要修改train.py     |
| epochs           | 1000                 |                                               |
| kernel_size      | 5                    | 卷积核大小                                    |
| filters          | 8, 16, 32, 32        | 卷积层channel数目                             |
| batch_norm       | 0                    | 批量正则化，0为False，1为True                 |
| weight_norm      | 0                    | 权重正则化，0为False，1为True                 |
| data_path        | ./data               | 数据集路径，视具体情况设置                    |
| save_path        | ./result             | 模型和训练记录的保存路径，视具体情况设置      |
| model_name       | DeepCFD_965.pdparams | 具体加载的模型名称，后缀不能省略              |

## 七、模型信息

| 信息     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| 发布者   | zbyandmoon                                                   |
| 时间     | 2022.08                                                      |
| 框架版本 | Paddle 2.3.1                                                 |
| 应用场景 | 任意障碍物绕流的二维快速计算                                 |
| 支持硬件 | GPU、CPU                                                     |
| 下载链接 | [预训练模型](https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle/blob/main/result/DeepCFD_965.pdparams) |
| 在线运行 | [AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4400677AI) |



