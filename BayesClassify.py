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
        data_trains = self.createBagOfWords(vocab_list, sample_data)
        p0_init = np.ones(len(vocab_list))
        p1_init = np.ones(len(vocab_list))
        for i in range(num_data_set):
            if data_category[i] == 1:


if __name__ == '__main__':
    sample_data = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    data_category = [0, 1, 0, 1, 0, 1]
    bayes_classify = BayesClassifyTrainNB0(sample_data, data_category)
    vocab_list = bayes_classify.createVocabList(sample_data)
    for data in sample_data:
        bags_words = list()
        bags_words.append(bayes_classify.createBagOfWords(vocab_list, data))
