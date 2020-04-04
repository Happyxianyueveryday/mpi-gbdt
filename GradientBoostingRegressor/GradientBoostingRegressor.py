# GradientBoostingRegressor.py: GDBT回归器模块

import os, sys, time
import numpy as np
import math
from mpi4py import MPI
from collections import Counter
from scipy import stats 
from DecisionTreeCartRegressionMpi import DecisionTreeCartRegressionMpi 

class GradientBoostingRegressor:
    '''
    : GradientBoostingRegressor: GDBT并行回归器
    '''
    def __init__(self, comm, main, n_estimators=10, subsample=1.0, learning_rate=0.1, loss='ls', min_impurity_decrease=None, max_depth=3, min_samples_leaf=1, min_samples_split=2, max_features=None):
        '''
        : __init__: GDBT回归器构造函数
        : param comm: mpi4py.MPI.Intracomm, MPI并行通信子
        : param main: int, 选定主进程的编号
        : param n_estimators: int, Boosting框架参数，弱学习器也即回归树的数量，等价于弱学习器的学习迭代次数，默认值为10
        : param subsample: float, Boosting框架参数，正则化参数，指定每次训练回归树时的无放回抽样比例，默认值为1.0，即不进行子采样，使用子采样时，并行训练效率可以进一步提升
        : param learning_rate: float, Boosting框架参数，正则化参数，该参数用于控制单个弱学习器的权重以减少过拟合现象，默认值为0.1
        : param loss: str, Boosting框架参数，指定回归树所使用的损失函数，若为'ls'则使用均方差损失作为损失函数，若为'abs'则使用绝对损失作为损失函数，默认值为'ls'
        : param min_impurity_decrease: float, 回归树剪枝参数，选择特征进行分枝过程时的最小均方差，均方差小于输入值的结点分枝操作不会发生，默认值为None
        : param max_depth: int, 回归树剪枝参数, 决策树的最大深度，默认值为3
        : param min_samples_leaf: int, 回归树剪枝参数，限制一个结点分枝后的子结点中均至少含有的样本数量，不满足条件则分枝不会发生，默认值为1
        : param min_samples_split: int, 回归树剪枝参数，限制一个分枝的结点所至少需要含有的样本数量，默认值为2
        : param max_features: int/float/str, 回归树剪枝参数，决策树每次分裂时需要考虑的特征数量，默认值为None，即考虑所有的特征，若输入为int，则表示每次分裂考虑的特征具体值；若输入为float，则表示每次分裂考虑的特征比例；若输入为str，'sqrt'表示每次分裂考虑的特征数量为总特征数量的平方根，'log'表示每次分裂考虑的特征数量为总特征数量的以2作为底的对数
        '''
        # 1. 回归树剪枝参数
        self.max_features = max_features                     # 决策树每次分裂时需要考虑的特征数量
        self.min_impurity_decrease = min_impurity_decrease   # 决策树每次分裂时的样本评估参数（信息增益比，信息增益值，或者基尼系数）的阈值
        self.max_depth = max_depth                           # 决策树的最大深度
        self.min_samples_leaf = min_samples_leaf             # 决策树分枝后的子结点中至少含有的样本数量
        self.min_samples_split = min_samples_split           # 决策树分枝所至少含有的样本数量
        # 2. Boosting框架参数
        self.n_estimators = n_estimators                     # 回归树的数量/弱分类器的学习迭代次数
        self.subsample = subsample                           # 每次训练回归树的样本不放回抽样比例
        self.loss = loss                                     # 回归树所使用的损失函数
        self.learning_rate = learning_rate                   # 单个弱学习器的权重
        # 3. 基础参数和数据
        self.comm = comm         # MPI通信子
        self.main = main         # 主进程的编号
        self.size = comm.size    # 总进程数量
        self.rank = comm.rank    # 当前进程的编号
        self.treeroot = []       # 回归树根结点列表
    
    def fit(self, data=None, label=None):
        '''
        : fit: 使用训练数据集训练GDBT回归器
        : param data: np.array, 二维训练数据集，其中列表示特征，行表示样本，辅进程只需要传入None即可
        : param label: np.array, 一维训练标签集，辅进程只需要传入None即可
        '''
        # 1. 主进程将训练数据集和训练标签集发送给其他进程
        data = self.comm.bcast(data, root=self.main)
        label = self.comm.bcast(label, root=self.main)
        n_sample = label.size
        # 2. 如果用户指定了需要使用子采样，则进行子采样，子采样使用无放回抽样法
        if self.subsample<1:
            if self.rank==self.main:
                inbag_index = np.random.permutation(n_sample)[:math.floor(n_sample*self.subsample)]
            else:
                inbag_index = None
            inbag_index = self.comm.bcast(inbag_index, root=self.main)
            data, label = data[inbag_index], label[inbag_index]
        # 3. 主进程和辅进程并行迭代进行训练
        if self.loss == 'ls': 
            self.init_predict = np.average(label)      # 使用均方差损失，则初始回归树为f0(x) = np.average(label)             
            gradient = label - self.init_predict
            for i in range(self.n_estimators):
                tree = DecisionTreeCartRegressionMpi(self.comm, self.main, self.loss)   
                tree.fit(data, gradient, self.min_impurity_decrease, self.max_depth, self.min_samples_leaf, self.min_samples_split, self.max_features)
                gradient -= self.learning_rate*tree.predict(data)
                self.treeroot.append(tree)
        elif self.loss == 'abs':
            self.init_predict = np.median(label)       # 使用绝对损失，则初始回归树为f0(x) = np.median(label)
            gradient = np.sign(label - self.init_predict)
            for i in range(self.n_estimators):
                tree = DecisionTreeCartRegressionMpi(self.comm, self.main, self.loss)   
                tree.fit(data, gradient, self.min_impurity_decrease, self.max_depth, self.min_samples_leaf, self.min_samples_split, self.max_features)
                gradient = self.learning_rate*np.sign(gradient - tree.predict(data))
                self.treeroot.append(tree)
        else:
            pass

    def predict(self, data=None):
        '''
        : predict: 预测测试数据集的回归结果
        : param data: np.array, 预测二维数据集，其中行代表样本，其中列代表特征，主进程需要传入测试集数据，辅进程则直接传入None
        : return: np.array, 预测一维标签集结果
        '''
        # 1. 主进程将待测试数据进行划分，发送给辅进程
        if self.rank==self.main:
            n_sample, n_feature = np.shape(data)
            divide_index = [(i+1)*math.floor(n_sample/self.size) for i in range(self.size-1)]
            data = np.split(data, divide_index, axis=0)
        data = self.comm.scatter(data, root=self.main)
        # 2. 主进程和辅进程并行进行预测
        local_res = self.init_predict
        for tree in self.treeroot:
            label = tree.predict(data)
            local_res += self.learning_rate*label
        local_res = local_res.tolist()
        # 3. 主进程从辅进程汇总结果
        local_res = self.comm.gather(local_res, root=self.main)
        if self.rank==self.main:
            res = []
            for x in local_res:
                res += x
            res = np.array(res)
            return res
        else:
            return None
        
    def score(self, data=None, label=None):
        '''
        : score: 使用测试数据集并行化预测，并返回R^2值作为评估
        : param data: np.array, 二维测试数据集，其中行代表样本，列代表特征，主进程需要提供该参数，辅进程则直接使用默认值None
        : param label: np.array, 一维测试标签集，主进程需要提供该参数，辅进程则直接使用默认值None
        : return: float, 主进程返回本次测试的R^2值，辅进程则返回None
        '''
        predict = self.predict(data)
        if self.rank==self.main:
            r2 = 1 - np.sum((label - predict)**2)/np.sum((label - np.average(label))**2)
            return r2
        else:
            return None

