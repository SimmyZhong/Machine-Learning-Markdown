import numpy as np
from numpy import tile


class Algorithm_kNN(object):
    """k近邻算法简单实现"""
    
    def __init__(self, sample_data, label, k):
        self.sample_data = sample_data
        self.label = label
        self.k = k
        
    def autoNorm(self, dataSet):
        """归一化特征值"""
        # newValue = (oldValue-min)/(max-min)
        dataMax = dataSet.max(0)
        dataMin = dataSet.min(0)
        m = dataSet.shape[0]
        dataSetMin = tile(dataMin, (m, 1))
        dataSetRange = tile(dataMax - dataMin, (m, 1))
        return (dataSet - dataSetMin)/dataSetRange
  
    def run(self, target_data):
        
        #样本数
        sample_count = self.sample_data.shape[0] 
        
        # 获取目标向量和样本之间的距离（欧式距离公式）d = ((xA0-xB0)**2 + (xA1-xB1)**2 + ...+ (xAn-xBn)**2)**0.5
        target_ = np.tile(target_data, (sample_count, 1)) - self.sample_data
        target_dis = ((target_**2).sum(axis=1))**0.5
        
        # 获取最近k个样本的标签索引
        tar_dis_order = target_dis.argsort()
        
        # 获取目标预测
        result = dict()
        for i in range(self.k):
            label = self.label[tar_dis_order[i]]
            result[label] = result.get(label, 0) + 1
        
        #对预测的结果进行排序，取出最优结果
        resultOrder = sorted(result.items(), key=lambda x: x[0], reverse=True)
        return resultOrder[0][0]
        

if __name__ == '__main__':
    sampleData = np.array([[0, 0], [0, 0.1], [1, 1], [1.1, 1]])
    label = ['A', 'A', 'B', 'B']
    textData = np.array([[0.1, 0.2], [1, 2], [100, 100], [-1, -1]])
    knn = Algorithm_kNN(sampleData, label, 2)
    for data in knn.autoNorm(textData):
        targetLable = knn.run(data)
        print(data, targetLable)
