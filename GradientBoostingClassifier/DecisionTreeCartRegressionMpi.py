# DecisionTreeCartRegressionMpi.py: 并行化的CART回归树

import os, sys, time
import numpy as np
import math
from mpi4py import MPI
from collections import Counter
from scipy import stats  

class TreeNode:
    '''
    : TreeNode: CART决策树结点
    '''
    def __init__(self, _sign, _sample_mask=None, _depth=None, _feature=None, _split=None, _out=None):
        '''
        : __init__: 初始构造函数
        : param _sign: bool, 结点属性标志，值为True表示当前结点为中间结点，值为False表示当前结点为叶子结点
        : param _feature: int, 中间结点划分选取的最佳划分特征的列下标，默认值为None
        : param _split: int, 中间结点划分选取的最佳划分特征的特征值，默认值为None
        : param _out: int, 叶子结点的最终输出值
        '''
        # 1. 初始化结点属性标记
        self.sign = _sign                   # 结点类型标记: True表明结点为中间结点，False表明结点为叶子结点
        self.sample_mask = _sample_mask     # 结点样本标记: 当前结点所具有的样本
        self.depth = _depth                 # 结点深度标记: 当前结点在回归树中的深度
        self.left, self.right = None, None  # 中间结点参数: 左子结点和右子结点
        self.feature = _feature             # 中间结点参数: 选择分枝的属性的列下标
        self.split = _split                 # 中间结点参数: 选择分枝属性的取值
        self.out = _out                     # 叶子结点参数: 结点的最终输出值


class DecisionTreeCartRegressionMpi:
    '''
    : DecisionTreeCartRegressionMpi: CART回归树模块
    '''
    def __init__(self, comm, main, K):
        '''
        : __init__: 初始构造函数
        : param comm: mpi4py.MPI.Intracomm, MPI通信子
        : param main: int, 指定的主进程编号
        : param K: int, 本轮迭代次数
        '''
        # 1. 并行化参数
        self.comm = comm            # 并行化MPI通信子
        self.main = main            # 主进程编号
        self.rank = comm.rank       # 当前进程编号
        self.size = comm.size       # 进程的总数量
        self.K = K                  # 本轮迭代次数
        # 2. 回归树训练参数
        self.treeroot = None        # cart回归树的根结点

    def fit(self, data, label, min_impurity_decrease=None, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=None):
        '''
        : fit: 根据输入的训练点进行回归
        : param data: np.array, 二维训练集
        : param label: np.array, 一维标签集
        : param min_impurity_decrease: float, 预剪枝调整参数，选择特征进行分枝过程时的最小均方差，均方差小于输入值的结点分枝操作不会发生，默认值为None
        : param max_depth: int, 预剪枝调整参数, 决策树的最大深度，默认值为None
        : param min_samples_leaf: int, 预剪枝调整参数，限制一个结点分枝后的子结点中均至少含有的样本数量，不满足条件则分枝不会发生，默认值为1
        : param min_samples_split: int, 预剪枝调整参数，限制一个分枝的结点所至少需要含有的样本数量，默认值为2
        : param max_features: int/float/str, 决策树剪枝参数，决策树每次分裂时需要考虑的特征数量，默认值为None，即考虑所有的特征，若输入为int，则表示每次分裂考虑的特征具体值；若输入为float，则表示每次分裂考虑的特征比例；若输入为str，'sqrt'表示每次分裂考虑的特征数量为总特征数量的平方根，'log'表示每次分裂考虑的特征数量为总特征数量的以2作为底的对数
        : return: TreeNode, 构建的回归树的根结点，每个线程都能得到一个独立的回归树
        '''
        n_sample = np.shape(data)[0]     # 参与训练的样本数量和特征数量
        self.treeroot = self._fit_cart(data, label, np.full(n_sample, True), min_impurity_decrease, max_depth, min_samples_leaf, min_samples_split, max_features)
    
    def predict(self, data):
        '''
        : predict: 使用测试集进行并行测试
        : param data: np.array, 二维数据集
        : return: np.array, 测试集的回归输出结果值
        '''
        local_res = []
        for sample in data:
            now = self.treeroot
            while now and now.sign:
                if sample[now.feature]<=now.split:
                    now = now.left
                else:
                    now = now.right
            local_res.append(now.out)
        local_res = np.array(local_res)
        return local_res

    def score(self, data, label):
        '''
        : score: 使用测试数据集并行化预测，并返回R^2值作为评估
        : param data: np.array, 二维测试数据集，其中行代表样本，列代表特征
        : param label: np.array, 一维测试标签集
        : return: float, 主进程返回本次测试的R^2值
        '''
        predict = self.predict(data)
        r2 = 1 - np.sum((label - predict)**2)/np.sum((label - np.average(label))**2)
        return r2

    def print(self):
        '''
        : print: 输出当前训练得到的决策树的层次遍历序列
        '''
        res = self.level_traverse()
        for x in res:
            print("---------------------------------------")
            for y in x:
                print(y)
        print("")

    def level_traverse(self):
        '''
        : level_traverse: 层次遍历整个决策树并且返回遍历序列
        : return: list[list[dict]], 当前决策树的层次遍历序列
        '''
        que = []
        res = []
        if not self.treeroot:
            return res
        que.append((self.treeroot, 0))
        while que:
            now_node, now_level = que[0]
            que.pop(0)
            if not now_node:
                continue
            node_out = {"sign":now_node.sign, "feature":now_node.feature,  "split":now_node.split, "out":now_node.out}
            if now_level>=len(res):
                res.append([node_out])
            else:
                res[now_level].append(node_out)
            if now_node.left:
                que.append((now_node.left, now_level+1))
            if now_node.right:
                que.append((now_node.right, now_level+1))
        return res

    def _fit_cart(self, data, label, sample_mask, min_impurity_decrease, max_depth, min_samples_leaf, min_samples_split, max_features):
        '''
        : _fit_cart: 本方法为私有方法，使用cart算法并行训练生成回归树
        : param data: np.array, 二维训练集，其中行作为样本，列作为特征
        : param label: np.array, 一维标签集
        : param sample_mask: np.array, 样本掩码向量，用True表示参与本次训练的样本集，False表示不参与
        : param now_depth: int, 当前递归的深度
        : param min_impurity_decrease: float, 预剪枝调整参数，选择特征进行分枝过程时的信息增益阈值，信息增益小于输入值的结点分枝操作不会发生，默认值为None
        : param max_depth: int, 预剪枝调整参数, 决策树的最大深度，默认值为None
        : param min_samples_leaf: int, 预剪枝调整参数，限制一个结点分枝后的子结点中均至少含有的样本数量，不满足条件则分枝不会发生，默认值为1
        : param min_samples_split: int, 预剪枝调整参数，限制一个分枝的结点所至少需要含有的样本数量，默认值为2
        : param max_features: int/float/str, 决策树剪枝参数，决策树每次分裂时需要考虑的特征数量，若输入为int，则表示每次分裂考虑的特征具体值；若输入为float，则表示每次分裂考虑的特征比例；若输入为str，'sqrt'表示每次分裂考虑的特征数量为总特征数量的平方根，'log'表示每次分裂考虑的特征数量为总特征数量的以2作为底的对数
        : return: TreeNode, 训练生成的回归树根结点
        '''
        n_feature = np.shape(data)[1]
        best_feature, best_split = 0, 0
        if self.rank==self.main:
            if max_features=='sqrt':
                n_select = math.floor(math.sqrt(n_feature))   # 若参数max_features为'sqrt'，则floor(sqrt(n_feature))个特征参与计算基尼系数
            elif max_features=='log':
                n_select = math.floor(math.log(n_feature, 2)) # 若参数max_features为'log'，则floor(log(n_feature, 2))个特征参与计算基尼系数
            elif isinstance(max_features, int):
                n_select = max_features                       # 若参数max_features为整数，则max_features个特征参与计算基尼系数
            elif isinstance(max_features, float):
                n_select = math.floor(n_feature*max_features) # 若参数max_features为浮点数，则n_feature*max_feature个特征参与计算基尼系数    
        stack = []
        root = TreeNode(False, _sample_mask=sample_mask, _depth=0)
        stack.append(root)
        while stack:
            now = stack.pop()
            # 1. 出现如下的若干种情况，生成叶子结点
            # (1). 训练样本集的样本数量小于或者等于2
            # (2). 当前建树的深度大于或者等于最大深度
            sample_label = label[now.sample_mask]
            if (sample_label.size<=min_samples_split) or (max_depth and now.depth>=max_depth):
                abs = np.abs(sample_label)
                now.sign, now.out = False, (self.K-1/self.K)*np.sum(sample_label)/np.sum(abs*(1-abs))
                continue
            # 2. 其他一般情况，进行进一步计算
            else:
                if self.rank==self.main:
                    # 2.1. 根据max_feature参数生成参与计算最佳分割点的特征列下标
                    if not max_features:
                        feature_index = np.arange(n_feature)        # 若参数max_features为空，则所有的特征均参与计算基尼系数
                    else:
                        feature_index = np.random.permutation(n_feature)[:n_select]
                    feature_index = np.split(feature_index, [(i+1)*math.floor(feature_index.size/self.size) for i in range(self.size-1)])   
                else:
                    feature_index = 0
                feature_index = self.comm.scatter(feature_index, root=self.main)
                # 2.2 每个进程只固定计算一部分的feature_index
                min_mse, left_split_samples, right_split_samples = float('inf'), 0, 0  
                for i in feature_index:
                    feature = data[:, i]
                    feature_value = feature[now.sample_mask]
                    sorted_index = np.argsort(feature_value)
                    feature_value, sample_value = feature_value[sorted_index], sample_label[sorted_index]
                    feature_unique = np.unique(feature_value)
                    feature_unique = (feature_unique[1:]+feature_unique[:-1])/2
                    for x in feature_unique:
                        split_index = np.searchsorted(feature_value, x, 'right')   # 选定的切分值的下标
                        left_space_label = sample_value[:split_index]      # 切分得到的左空间，也即特征值小于或者等于x的样本
                        right_space_label = sample_value[split_index:]     # 切分得到的右空间，也即特征值大于x的样本
                        left_ave, right_ave = np.average(left_space_label), np.average(right_space_label)       # 左空间输出y的均值，右空间输出y的均值
                        mse = np.sum(np.square(left_space_label - left_ave)) + np.sum(np.square(right_space_label - right_ave)) # 在列下标为i的特征上按照特征值为x切分的最小均方差
                        if mse<min_mse:
                            min_mse, best_feature, best_split = mse, i, x  # 最优切分方案: 最小平方误差，取得最小平方误差的切分特征列下标，取得最小平方误差的切分特征值
                            left_split_samples, right_split_samples = left_space_label.size, right_space_label.size  # 最优切分方案下的左子树样本数，右子树样本数
                divide = (min_mse, best_feature, best_split, left_split_samples, right_split_samples)
                divide = self.comm.gather(divide, root=self.main)
                if self.rank==self.main:
                    divide = min(divide)
                divide = self.comm.bcast(divide, root=self.main)
                min_mse, best_feature, best_split, left_split_samples, right_split_samples = divide
                # 2.3 按照上述查找到的最优切分方案切分出左右子结点
                if (min_impurity_decrease and min_mse<min_impurity_decrease) or left_split_samples<min_samples_leaf or right_split_samples<min_samples_leaf:
                    abs = np.abs(sample_label)
                    now.sign, now.out = False, (self.K-1/self.K)*np.sum(sample_label)/np.sum(abs*(1-abs))
                    continue
                else:
                    now.sign, now.feature, now.split = True, best_feature, best_split    
                    feature = data[:, best_feature]
                    temp = np.where(feature<best_split, True, False)
                    sample_mask_left = np.bitwise_and(now.sample_mask, temp)    # 生成左子结点掩码
                    np.bitwise_not(temp, out=temp)
                    sample_mask_right = np.bitwise_and(now.sample_mask, temp)   # 生成右子结点掩码
                    now.left = TreeNode(True, _sample_mask=sample_mask_left, _depth=now.depth+1)
                    now.right = TreeNode(True, _sample_mask=sample_mask_right, _depth=now.depth+1)
                    stack.append(now.right)
                    stack.append(now.left)
        return root
                
