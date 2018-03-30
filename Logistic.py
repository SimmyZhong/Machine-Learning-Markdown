import numpy as np
from numpy import mat, exp


class LogisticSigmoidClassifier(object):
    """logistic回归分类器"""
    
    def __init__(self, sample_data, labels):
        """初始化"""
        
        self.sample_data = sample_data
        self.labels = labels
    
    def sigmoid(self, data_set):
        """sigmoid函数"""
        
        return 1.0 / 1 + exp * (-data_set))
    
    def gradAscent(self, sample_data, labels):
        """梯度上升优化算法得出最优权重系数"""
        
        traning_data = mat(sample_data) # 转化为矩阵
        num_data, num_feature = shape(sample_data) # 获取数据集样本数和特征数
        traning_labels = mat(labels).transpose() # 转置
        weights = np.ones((num_data, 1)) # 初始化权重系数 
        step = 0.001 # 步长
        for i in range(500):
            z = traning_data * weights # 对权重系数及样本集进行矩阵运算
            distance = traning_labels - h 
            weights = weights + traning_data.transpose() * distance * step # 梯度上升算法的迭代公式：参考机器学习实战P76【Logistic回归】
        return  weights


if __name__ == "__main__":
    pass
