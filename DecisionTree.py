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
    
    def classifyDataSet(self, data_set, axis, value):
        """划分数据集"""
        
        result_data = []
        for data in data_set:
            if data[axis] == value:
                result = data[:axis] # 注意，避免修改原始数据集，要新建一个列表对象
                result.extend(data[axis + 1:])
                result_data.append(result)
        return result_data
    
    def getBestFeatureToSplit(self, sample_data):
        """根据香农熵选择最好的数据划分方式"""
        
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
                result_data = self.classifyDataSet(sample_data, i, value) #按特征值划分
                prob = len(result_data) / float(len(sample_data)) # 计算子集的概率
                shannoEnt_new += prob * self.calcShannonEnt(result_data) #计算香农熵
           
            if shannoEnt_best > shannoEnt_new:
                shannoEnt_best = shannoEnt_new
                feature_best = i
        return feature_best
    
    def majorityCnt(self, classList):
        """多数表决的方法定义最后的叶子节点"""
        
        class_count = {}
        for value in classList:
            class_count[value] = 1 + class_count.get(value, 0)
        order_class_count = sorted(class_count.items(), key=lambda x: x[1],reverse=True )
        return order_class_count[0][0]
    
    def createTree(self, dataSet, labels):
        """创建决策树"""
        
        class_list = [example[-1] for example in dataSet]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0] # 类别完全相同，停止划分
        if len(dataSet[0]) == 1:
            return self.majorityCnt(dataSet) # 遍历完所有特征时，返回出现次数最多的，即用多数表决方法定义最后的叶子节点
        best_feature = self.getBestFeatureToSplit(dataSet)
        best_feature_label = labels[best_feature]
        mytree = {best_feature_label: dict()}
        del(labels[best_feature])
        feat_values = [example[best_feature] for example in dataSet]
        unique_vals = set(feat_values)
        for value in unique_vals:
            sub_labels = labels[:]
            mytree[best_feature_label][value] = self.createTree(self.classifyDataSet(dataSet, best_feature, value), sub_labels)
        return mytree
        
        
if __name__ == '__main__':
    sample_data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    tree = DecisionTree(sample_data, labels)
#     print(tree.calcShannonEnt(sample_data))
#     print(tree.classifyDataSet(sample_data, 0, 1))
#     print(tree.getBestFeatureToSplit(sample_data))
#     print(tree.majorityCnt([1, 1, 2, 3, 3, 3, 4, 4, 4, 4]))
#     print(tree.createTree(sample_data, labels))

