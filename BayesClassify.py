import numpy as np


class BayesClassifyTrainNB0(object):
    """朴素贝叶斯分类器简单实现"""
    
    def __init__(self, sample_data, data_category):
        """初始化，样本集和对应的分类标签"""
        
        self.sample_data = sample_data
        self.data_category = data_category
    
    def createVocabList(self, data_set):
        """创建词汇表"""
        
        vocab_list = set([])
        for sub_data_set in data_set:
            vocab_list = vocab_list | set(sub_data_set)
        return list(vocab_list)
    
    def createBagOfWords(self, vocab_list, data_set):
        """创建文档词带模型"""
        
        result_set = np.zeros(len(vocab_list))
        for word in data_set:
            try:
                result_set[vocab_list.index(word)] += 1
            except:
                pass
        return result_set
    
    def tranNB0(self, sample_data, data_category):
        """构建朴素贝叶斯分类器训练函数"""
        
        num_data_set = len(sample_data)
        p1 = sum(data_category) / num_data_set  # 计算类别为1的出现的概率
        vocab_list = self.createVocabList(sample_data)
        data_trains = list()
        for data in sample_data:   
            data_trains.append(self.createBagOfWords(vocab_list, data))
    
        # 初始化，默认每个词为1次（不设置为0是为了避免分子为0导致概率为0的情况出现）
        sub_data_0 = np.ones(len(vocab_list))
        num_data_0 = len(vocab_list)
        num_data_1 = num_data_0
        sub_data_1 = np.ones(len(vocab_list))
        
        for i in range(num_data_set):
            # 根据数据集的类别，分别统计各词出现的次数以及总词数
            
            if data_category[i] == 1:
                sub_data_1 = sub_data_1 + data_trains[i] 
                num_data_1 += sum(data_trains[i])
            else:
                sub_data_0 = sub_data_0 + data_trains[i]
                num_data_0 += sum(data_trains[i])
        # 计算词频 = 词出现的次数 / 总词数 为了避免很小的频率相乘导致为0，可对乘积取自然对数：即ln(pw1*pw2) = lnpw1 + lnpw2
        p0_words = sub_data_0 / num_data_0
        p1_words = sub_data_1 / num_data_1
        return p1, p0_words, p1_words
        


if __name__ == '__main__':
    sample_data = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks',  'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    data_category = [0, 1, 0, 1, 0, 1]
    bayes_classify = BayesClassifyTrainNB0(sample_data, data_category)
    vocab_list = bayes_classify.createVocabList(sample_data)
    for data in sample_data:
        bags_words = list()
        bags_words.append(bayes_classify.createBagOfWords(vocab_list, data))
    p1,p0_words, p1_words = bayes_classify.tranNB0(sample_data, data_category)
    print(p1,p0_words, p1_words)
        
