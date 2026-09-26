# PyTorch MNIST 手写数字识别

## 项目介绍

本项目基于 PyTorch 实现 MNIST 手写数字分类任务。

通过构建全连接神经网络，实现对 0~9 十类手写数字图片的自动识别。

## 数据集

使用 MNIST 手写数字数据集：

数据集来源：
http://yann.lecun.com/exdb/mnist/

数据包含：

- 训练集：60000张图片
- 测试集：10000张图片
- 图片尺寸：28×28 像素


## 模型结构

本实验采用全连接神经网络：
Flatten->Linear(784,128)->ReLU->Linear(128,10)


## 实验结果

训练轮数：5 epochs

测试准确率：

97%左右


## 环境

Python 3.x

PyTorch

torchvision

matplotlib


## 运行方法

安装依赖：
pip install -r requirements.txt


运行：
python mnist_experiment.py
