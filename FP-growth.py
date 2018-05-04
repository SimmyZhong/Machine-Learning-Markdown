'''
Apriori算法关联分析
Created on 4th, May, 2018
Update  on 2018-05-05
@author: Simmy
https://github.com/SimmyZhong/Machine-Learning-Markdown/blob/master/FP-growth.py
'''

def createFeature(dataSet, supportConf=0.8):
    """数据预处理，返回处理过的数据集dict和频繁项集"""
    results = dict()
    supportData = dict()
    for data in dataSet:
        results[frozenset(data)] = results[frozenset(data)] + 1 if frozenset(data) in results.keys() else 1
    for k, v in results.items():
        for item in k:
            supportData[frozenset(item)] = results[k] + supportData[frozenset(item)] if frozenset(item) in supportData.keys() else 1
    FrequentSet = list(filter(lambda x: supportData[x] / len(results) > supportConf, supportData.keys()))
    return results, FrequentSet


if __name__ == "__main__":
    dataSet = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               # ['r', 'x', 'n', 'o', 's', 's', 'o'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    print(createFeature(dataSet))