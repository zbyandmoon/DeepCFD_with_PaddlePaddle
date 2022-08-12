# 【Paper Reproduction Contest】DeepCFD

English | [简体中文](./README_cn.md)

- [【Paper Reproduction Contest】DeepCFD](#paper-reproduction-contestdeepcfd)
	- [1 Introduction](#1-introduction)
		- [1.1 Paper information](#11-paper-information)
		- [1.2 Method](#12-method)
		- [1.3 Results](#13-results)
	- [2 Accuracy](#2-accuracy)
		- [2.1 Acceptance criteria](#21-acceptance-criteria)
		- [2.2 Model accuracy](#22-model-accuracy)
		- [2.3 Reproduction link](#23-reproduction-link)
	- [3 Dataset](#3-dataset)
	- [4 Enviroment](#4-enviroment)
	- [5 Quick start](#5-quick-start)
	- [6 Code structure and parameter clarification](#6-code-structure-and-parameter-clarification)
		- [6.1 Code structure](#61-code-structure)
		- [6.2 parameter clarification](#62-parameter-clarification)
	- [7 Model information](#7-model-information)

This project is based on the PaddlePaddle framework to replicate the DeepCFD network.

## 1 Introduction

Computational fluid dynamics (CFD) can obtain the distribution of various physical quantities of a fluid, such as density, pressure and velocity, by solving the Navier-Stokes equations (N-S equations), and is widely used in fields such as MEMS, civil engineering and aerospace. However, in some complex application scenarios, such as wing optimization and fluid-structure interaction, the problem needs to be modeled using tens of millions or even hundreds of millions of meshes (as shown in the figure below, which illustrates a fully integrated structured mesh model of the internal and external flow of an F-18 fighter jet), resulting in a very large computational effort for CFD. Therefore, there is an urgent need to develop a method that is more efficient than traditional CFD methods and can maintain computational accuracy. The authors of this paper suggest that a data-driven CFD computational model can be built using deep learning methods by training a small amount of data from traditional CFD simulations to solve the above problem.

<img src="http://www.cannews.com.cn/files/Resource/attachement/2017/0511/1494489582596.jpg" alt="img" style="zoom:80%;" />

### 1.1 Paper information

* Reference：[1] Ribeiro M D, Rehman A, Ahmed S, et al. DeepCFD: Efficient steady-state laminar flow approximation with deep convolutional neural networks[J]. arXiv preprint arXiv:2004.08826, 2020.
* Paper：https://arxiv.org/abs/2004.08826
* Github project：https://github.com/mdribeiro/DeepCFD

### 1.2 Method

The authors propose a convolutional neural networks-based CFD computational model, called DeepCFD, which can simultaneously compute the flow field of a fluid flowing over an arbitrary obstacle. The method has the following features:

1. DeepCFD is essentially a CNN-based model that can be used to rapidly compute two-dimensional non-uniform steady-state laminar flow, which can achieve at least three orders of magnitude speedups compared to traditional CFD methods while maintaining computational accuracy.

2. DeepCFD can calculate fluid velocities in both x- and y-directions, as well as fluid pressures.

3. The data for training the model were calculated by OpenFOAM (an open source CFD calculation software).

The following two figures show the calculation schematic and the network structure of the method, respectively. The basic structure of the DeepCFD network used in this paper is a U-net type network with three inputs and three outputs. The inputs of the model are the signed distance function (SDF) of the obstacle in the computational domain, the SDF of the computational domain boundary and the label of the flow region; the outputs are the x-direction velocity, y-direction velocity and fluid pressure of the fluid. The basic principle of the model is to use the convolution layer in the encoder part to downsample the three inputs into intermediate quantities, and then use the same structure of the decoder in the deconvolution layer to upsample the intermediate quantities into three fluid physical quantities output.

![compute_process.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/compute_process.png?raw=true)

![DeepCFD_Net.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/DeepCFD_Net.png?raw=true)

### 1.3 Results

The following figure shows the prediction results of the original article, which contains a total of four indicators to evaluate the merits of the model: Total MSE, Ux MSE, Uy MSE, and p MSE (MSE means root mean square error).

![metrics.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/metrics.png?raw=true)

The following figure shows the comparison of CFD (note: simpleFOAM is a kind of OpenFOAM solver) and DeepCFD flow field calculation results for a certain shape of obstacle.

![pytorch_contour.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/pytorch_contour.png?raw=true)

## 2 Accuracy

### 2.1 Acceptance criteria

The acceptance criteria for reproduction are as follows.

![standard.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/standard.png?raw=true)

### 2.2 Model accuracy

The realization metrics for reproduction are as follows.

```python
Total MSE = 1.8955801725387573
Ux MSE = 0.6953578591346741
Uy MSE = 0.21001338958740234
p MSE = 0.9902092218399048
```

Total MSE, Ux MSE and Uy MSE are within the acceptance criteria and p MSE is slightly less than the minimum value of the acceptance criteria.

### 2.3 Reproduction link

The paper is reproduced at

AI Studio: https://aistudio.baidu.com/aistudio/projectdetail/4400677?contributionType=1

github: https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle/tree/main/paddle

## 3 Dataset

The data set uses CFD examples computed by the original authors using OpenFOAM, with a total of 981 sets, divided into two files (dataX.pkl, dataY.pkl), both of 152 MB in size and [981, 3, 172, 79] in shape. dataX.pkl includes three inputs: the SDF of the obstacle, the SDF of the computational domain boundary, and the flow labels of the region; dataY.pkl includes three outputs: x-directional velocity of the fluid, y-directional velocity, and fluid pressure. The computational grid used for data acquisition is 172 × 79.

dataset link：https://aistudio.baidu.com/aistudio/datasetdetail/162674

or https://www.dropbox.com/s/kg0uxjnbhv390jv/Data_DeepCFD.7z?dl=0

## 4 Enviroment

* hardware：GPU、CPU
* framework：PaddlePaddle >= 2.0.0

## 5 Quick start

**step1：clone**

```python
git clone https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle.git
```

**step2：install dependency**

```python
cd DeepCFD_with_PaddlePaddle
pip install -r requirements.txt
```

**step3：download dataset**

download Data_DeepCFD.7z from https://www.dropbox.com/s/kg0uxjnbhv390jv/Data_DeepCFD.7z?dl=0

Extract dataX.pkl and dataY.pkl to DeepCFD_with_PaddlePaddle/data directory as follows.

![data.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/data.png?raw=true)

**step4：train**

**Single GPU**

```python
python train.py
```

**Multi GPU**

Take four GPUs as an example.

```python
python -m paddle.distributed.launch --gpus=0,1,2,3 train.py
```

The results are saved in the result file (note: the result folder already contains a complete training process, which can be emptied before training). Multi GPU training generates an additional . /log/ folder to store the training logs.

```python
.
├── log
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
└── train.py
```

Some of the training logs are shown below.

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

**step5：evaluation**

```python
python eval.py
```

The output is:

```python
Total MSE is 1.895322561264038, Ux MSE is 0.6951090097427368, Uy MSE is 0.21001490950584412, p MSE is 0.9901986718177795
```

**step6：prediction**

Considering the need to show the flow field image comparison results, a separate predict.ipynb was written for model validation, which needs to be run in the Jupyter notebook environment.

```python
jupyter notebook predict.ipynb
```

The flow field prediction results for a particular obstacle are shown below.


![paddle_contour.png](https://github.com/zbyandmoon/Picture/blob/main/picture_DeepCFD/paddle_contour.png?raw=true)

## 6 Code structure and parameter clarification

### 6.1 Code structure

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

### 6.2 parameter clarification

The parameters for training can be set in /DeepCFD_with_PaddlePaddle/config/config.ini, including:

| Parameter        | Recommended value    | Additional notes                                             |
| ---------------- | -------------------- | ------------------------------------------------------------ |
| batch_size       | 64                   |                                                              |
| train_test_ratio | 0.7                  | The ratio of training set to data set, 0.7 i.e. training set 70% test set 30% |
| learning_rate    | 0.001                |                                                              |
| weight_decay     | 0.005                | for AdamW, if modify the optimization algorithm need to modify train.py |
| epochs           | 1000                 |                                                              |
| kernel_size      | 5                    | Convolution kernel size                                      |
| filters          | 8, 16, 32, 32        | Number of convolutional layer channels                       |
| batch_norm       |                      | Batch normalization, 0 is False, 1 is True                   |
| weight_norm      | 0                    | Weight normalization，0 is False，1 is True                  |
| data_path        | ./data               | Dataset path, set as appropriate                             |
| save_path        | ./result             | Save path for models and training records, set as appropriate |
| model_name       | DeepCFD_965.pdparams | The name of the specific model to be loaded, the suffix cannot be omitted |

## 7 Model information

| Information           | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| Author                | zbyandmoon                                                   |
| Date                  | 2022.08                                                      |
| Framework version     | Paddle 2.3.1                                                 |
| Application scenarios | Fast 2D calculation of arbitrary obstacle bypass flow        |
| Support hardware      | GPU、CPU                                                     |
| Download link         | [Pre-trained model](https://github.com/zbyandmoon/DeepCFD_with_PaddlePaddle/blob/main/result/DeepCFD_965.pdparams) |
| Online operation      | [AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4400677AI) |



