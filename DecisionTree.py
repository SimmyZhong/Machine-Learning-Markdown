import numpy as np
from math import log


class DecisionTree(object):
    """ID3算法决策树简单实现"""
    
    def __init__(self, sample_data, labels):
        """初始化"""
        
        self.sample_data = sample_data
        self.labels = labels
    
    def calcShannonEnt(self, sample_data):
        """计算给定数据集的香农熵"""
        
        sample_num = len(sample_data) #获取数据集数据量
        label_counts = dict() # 保存数据集的标签及出现的次数
        for data in sample_data:
            label_counts[data[-1]] = 1 + label_counts.get(data[-1], 0) # 默认数据最后的特征为标签
        shannonEnt = 0.0
        # 计算香农熵
        for key in label_counts.keys():
            prob = float(label_counts[key]) / sample_num
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt
    
    def classifyDataSet(self, axis, value):
        """划分数据集"""
        
        data_set = self.sample_data
        result_data = []
        for data in data_set:
            if data[axis] == value:
                result = data[:axis] # 注意，避免修改原始数据集，要新建一个列表对象
                result.extend(data[axis + 1:])
                result_data.append(result)
        return result_data
    
    def getBestFeatureToSplit(self):
        """根据香农熵选择最好的数据划分方式"""
        
        sample_data = self.sample_data
        shannoEnt_base = self.calcShannonEnt(sample_data) # 计算原始数据集的香农熵
        labels = self.labels
        num_features = len(sample_data[0]) - 1
        shannoEnt_best = shannoEnt_base
        feature_best = -1
        for i in range(num_features): #对每一个特征值进行划分并计算香农熵，找出熵最小的特征值
            feat_list = [example[i] for example in sample_data] # 获取每个数据的第i个特征值
            unique_vals = set(feat_list) # 特征值集合
            shannoEnt_new = 0.0 
            for value in unique_vals:        
                # 计算每种划分方式的信息熵
                result_data = self.classifyDataSet(i, value) #按特征值划分
                prob = len(result_data) / float(len(sample_data)) # 计算子集的概率
                shannoEnt_new += prob * self.calcShannonEnt(result_data) #计算香农熵
           
            if shannoEnt_best > shannoEnt_new:
                shannoEnt_best = shannoEnt_new
                feature_best = i
        return feature_best
    
    def majorityCnt(self, classList):
        """多数表决的方法定义最后的叶子节点"""
        pass
    
                
if __name__ == '__main__':
    sample_data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    tree = DecisionTree(sample_data, labels)
#     print(tree.calcShannonEnt(sample_data))
#     print(tree.classifyDataSet(0, 0))
#     print(tree.getBestFeatureToSplit())

            
