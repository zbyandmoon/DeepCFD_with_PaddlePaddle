### 【论文复现赛】DeepCFD: Efficient Steady-State Laminar Flow Approximation with Deep Convolutional Neural Networks

* 队伍名称：zbyandmoon

#### 目录

* 一、论文简介

* 二、复现指标符合情况

* 三、数据集

* 四、环境依赖

* 五、快速开始

* 六、参数设置

* 七、代码结构

本项目基于PaddlePaddle框架复现DeepCFD网络。



#### 一、论文简介

##### 1.1 背景

计算流体力学（Computational fluid dynamics, CFD）通过求解Navier-Stokes方程（N-S方程），可以获得流体的各种物理量的分布，如密度、压力和速度等，在微电子系统、土木工程和航空航天等领域应用广泛。然而，在某些复杂的应用场景中，如机翼优化和流体与结构相互作用方面，需要使用千万级甚至上亿的网格对问题进行建模（如下图所示，下图展示了F-18战斗机的全机内外流一体结构化网格模型），导致CFD的计算量非常巨大。因此，目前亟需发展出一种相比于传统CFD方法更高效，且可以保持计算精度的方法。这篇文章的作者提到，可以使用深度学习的方法，通过训练少量传统CFD仿真的数据，构建一种数据驱动（data-driven）的CFD计算模型，来解决上述的问题。

<img src="http://www.cannews.com.cn/files/Resource/attachement/2017/0511/1494489582596.jpg" alt="img" style="zoom:80%;" />

##### 1.2 方法

作者提出了一个基于卷积神经网络（Convolutional neural networks, CNN）的CFD计算模型，称作DeepCFD，该模型可以同时计算流体流过任意障碍物的流场。该方法有以下几个特点：

1. DeepCFD本质上是一种基于CNN的代理模型，可以用于快速计算二维非均匀稳态层流流动，相比于传统的CFD方法，该方法可以在保证计算精度的情况下达到至少三个数量级的加速。

2. DeepCFD可以同时计算流体在x方向和y方向的流体速度，同时还能计算流体压强。

3. 训练该模型的数据由OpenFOAM（一种开源CFD计算软件）计算得到。

下面两张图分别为该方法的计算示意图和网络结构图。文中使用的DeepCFD网络基本结构为有3个输入和3个输出的U-net型网络。该模型输入为计算域中障碍物的符号距离函数（Signed distance function, SDF）、计算域边界的SDF和流动区域的标签；输出为流体的x方向速度、y方向速度以及流体压强。该模型的基本原理就是利用编码器部分的卷积层将3个输入下采样，变为中间量，然后使用相同结构的解码器中的反卷积层将中间量上采样为3个流体物理量输出。

![compute_process.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/compute_process.png?raw=true)

![DeepCFD_Net.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/DeepCFD_Net.png?raw=true)

##### 1.3 结果

下图展示了原文的预测结果，文中评价模型的优劣共包含四个指标：Total MSE、Ux MSE、Uy MSE、p MSE（MSE的意思是均方根误差）。

![metrics.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/metrics.png?raw=true)

下图展示了某种形状障碍物的CFD（注：simpleFOAM是OpenFOAM求解器的一种）和DeepCFD流场计算结果对比。

![pytorch_contour.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/pytorch_contour.png?raw=true)

##### 1.4 论文信息

* 引用：[1] Ribeiro M D, Rehman A, Ahmed S, et al. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks[J]. arXiv preprint arXiv:2004.08826, 2020.
* 论文地址：https://arxiv.org/abs/2004.08826
* 项目地址：https://github.com/mdribeiro/DeepCFD

#### 二、复现指标符合情况

复现的验收标准如下：

![standard.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/standard.png?raw=true)

复现的实现指标如下：

```python
Total MSE = 1.8955801725387573
Ux MSE = 0.6953578591346741
Uy MSE = 0.21001338958740234
p MSE = 0.9902092218399048
```

其中，Total MSE、Ux MSE和Uy MSE在验收标准范围内，p MSE略小于验收标准的最小值。

论文复现地址：

AI Studio: https://aistudio.baidu.com/aistudio/projectdetail/4400677?contributionType=1

github: https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle/tree/main/paddle

#### 三、数据集

数据集使用原作者利用OpenFOAM计算的CFD算例，共981组，分为两个文件（dataX.pkl, dataY.pkl），两个文件大小都是152 MB，形状均为[981, 3, 172, 79]。dataX.pkl包括三种输入：障碍物的SDF、计算域边界的SDF和流动区域的标签；dataY.pkl包括三种输出：流体的x方向速度、y方向速度和流体压强。数据获取使用的计算网格为172×79。

数据集地址：

https://aistudio.baidu.com/aistudio/datasetdetail/162674

或https://www.dropbox.com/s/kg0uxjnbhv390jv/Data_DeepCFD.7z?dl=0

#### 四、环境依赖

* 硬件：GPU\CPU
* 框架：PaddlePaddle >= 2.0.0

#### 五、快速开始

此处分github和AI Studio两个方面进行介绍。

##### 5.1 github

**step1：克隆本项目**

```python
git clone https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle.git
```

**step2：配置数据集**

从https://www.dropbox.com/s/kg0uxjnbhv390jv/Data_DeepCFD.7z?dl=0下载得到Data_DeepCFD.7z，将dataX.pkl和dataY.pkl解压到DeepCFD_with_PaddlePaddle/data目录，如下所示。

![data.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/data.png?raw=true)

**step3：训练模型**

```python
cd DeepCFD_with_PaddlePaddle/paddle
python DeepCFD.py
```

结果保存在Result文件中（注：Result文件夹中已经包含了一个完整的训练过程，可在训练前将其清空）。

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

**step4：验证模型**

考虑到需要展示流场图像对比结果，单独写了一个main.ipynb来进行模型的验证，需要在Jupyter notebook环境中运行。

部分代码以及结果展示如下：

```python
# 测试训练模型
out = net(test_x)
# 计算残差
error = paddle.abs(out.cpu() - test_y.cpu())
# 作出CFD和CNN的计算结果对比图以及对应的残差图(s可修改)
s = 0
visualize(test_y.detach().numpy(), out.detach().numpy(), error.detach().numpy(), s)
```

![paddle_contour.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/paddle_contour.png?raw=true)

```python
# 指标符合性分析
# Total MSE
Total_MSE = paddle.sum((out-test_y)**2)/len(test_x)
# Ux MSE
Ux_MSE = paddle.sum((out[:,0,:,:]-test_y[:,0,:,:])**2)/len(test_x)
# Uy MSE
Uy_MSE = paddle.sum((out[:,1,:,:]-test_y[:,1,:,:])**2)/len(test_x)
# p MSE
p_MSE = paddle.sum((out[:,2,:,:]-test_y[:,2,:,:])**2)/len(test_x)
print("Total MSE is {}, Ux MSE is {}, Uy MSE is {}, p MSE is {}".format(Total_MSE.detach().numpy()[0], Ux_MSE.detach().numpy()[0], Uy_MSE.detach().numpy()[0], p_MSE.detach().numpy()[0]))
```

```python
Total MSE is 1.8955801725387573, Ux MSE is 0.6953578591346741, Uy MSE is 0.21001338958740234, p MSE is 0.9902092218399048
```

##### 5.2 AI Studio

**step1：克隆本项目**

搜索DeepCFD_with_PaddlePaddle，选择对应的版本，Fork。

![fork.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/fork.png?raw=true)

**step2：进入项目**

进入后，文件如下所示，data文件夹存储了数据集，Result存储训练模型和日志，tool包括了一些训练中使用的网络和函数文件，DeepCFD.py为训练主程序，main.ipynb为测试程序，README.md为复现结果展示，requirements.txt为程序所需库。

![project_files.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/project_files.png?raw=true)

**step3：开始训练**

点击左上角加号。

![click_lefttop.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/click_lefttop.png?raw=true)

选择进入终端。

![click_terminal.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/click_terminal.png?raw=true)

输入

```python
python DeepCFD.py
```

部分输出结果如下

```python
aistudio@jupyter-128577-4402661:~$ python DeepCFD.py 
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  from collections import MutableMapping
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  from collections import Iterable, Mapping
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  from collections import Sized
W0803 17:13:28.556344   269 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0803 17:13:28.560719   269 gpu_resources.cc:91] device: 0, cuDNN Version: 7.6.
Previous train log deleted successfully
Epoch #1
        Train Loss = 884808909.0
        Train Total MSE = 10197.3000353043
        Train Ux MSE = 3405.3426083044824
        Train Uy MSE = 4334.0962839376825
        Train p MSE = 2457.8616943359375
        Validation Loss = 53205074.5
        Validation Total MSE = 1027.7523040254237
        Validation Ux MSE = 419.7688029661017
        Validation Uy MSE = 543.9674920550848
        Validation p MSE = 64.01604872881356
Epoch #2
        Train Loss = 75408434.25
        Train Total MSE = 603.198411591199
        Train Ux MSE = 277.9321616481414
        Train Uy MSE = 303.4222437021684
        Train p MSE = 21.843986488987337
        Validation Loss = 17892356.5
        Validation Total MSE = 312.7194186970339
        Validation Ux MSE = 169.64230501853814
        Validation Uy MSE = 140.46789757680085
        Validation p MSE = 2.6092084981627384
```

**step4：验证模型**

此处类似上述github中的操作，点击左侧main.ipynb

![click_main.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/click_main.png?raw=true)

进入后可直接点击

![run_all.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/run_all.png?raw=true)

重启并运行整个代码块，验证结果同上，不再展示。

#### 六、参数设置

复现使用的参数和原文完全一致，如下

| 参数                 | 值              |
| -------------------- | --------------- |
| 学习率               | 0.001           |
| Batch size           | 64              |
| Epochs               | 1000            |
| 卷积核大小           | 5               |
| 卷积层channel数目    | [8, 16, 32, 32] |
| Batch normalization  | False           |
| Weight normalization | False           |
| AdamW的权重衰减因子  | 0.005           |

#### 七、代码结构

##### 7.1 github

```python
|-- DeepCFD_with_PaddlePaddle
    |-- Models	#原论文网络模型（pytorch版本）
        |-- AutoEncoder.py 
        |-- AutoEncoderEx.py    
        |-- UNet.py             
        |-- UNetEx.py   # DeepCFD网络，其余为文中对比网络
        |-- UNetExAvg.py   		
        |-- UNetExMod.py   		
    |-- pytorch		#原论文训练代码（pytorch版本）
        |-- Run
    	    |-- CFD和CNN结果对比图.png
            |-- 原文结果.png
            |-- 训练截图.png
            |-- results.json
        |-- DeepCFD.py
        |-- functions.py
        |-- pytorchtools.py
        |-- train_functions.py
        |-- README.md	#原论文参考代码复现说明
    |-- data     #数据集
    	|-- README.md	#数据集下载地址
    |-- paddle     #复现代码（paddle版本）
        |-- Result
    	    |-- DeepCFD_*.pdparams	#训练中保存的模型
            |-- results.json	#训练中保存的loss及各项指标变化         
            |-- train_log.txt	#训练过程记录
        |-- tool
    		|-- UNetEx.py   # DeepCFD网络的paddle版本
            |-- functions.py	#包括张量分割和图像显示函数         
            |-- train_functions.py	#训练代码
        |-- DeepCFD.py	#复现主程序
        |-- main.ipynb	#模型验证程序
    |-- README.md	#复现结果展示      
    |-- requirements.txt	#所需python库
```

##### 7.2 AI Studio

```python
|-- DeepCFD_with_PaddlePaddle
    |-- data	#数据集
        |-- data162674
            |-- dataX.pkl
            |-- dataY.pkl
    |-- Result
    	|-- DeepCFD_*.pdparams	#训练中保存的模型
        |-- results.json	#训练中保存的loss及各项指标变化         
        |-- train_log.txt	#训练过程记录 
    |-- tool
    	|-- UNetEx.py   # DeepCFD网络的paddle版本
        |-- functions.py	#包括张量分割和图像显示函数         
        |-- train_functions.py	#训练代码
    |-- work
    |-- DeepCFD.py	#复现主程序
    |-- main.ipynb	#模型验证程序
    |-- README.md	#复现结果展示      
    |-- requirements.txt	#所需python库
```

