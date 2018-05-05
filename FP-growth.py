'''
Apriori算法关联分析
Created on 4th, May, 2018
Update  on 2018-05-05
@author: Simmy
https://github.com/SimmyZhong/Machine-Learning-Markdown/blob/master/FP-growth.py
'''


class FPTree(object):
    """FP频繁项集树"""

    def __init__(self, name, count, parentnode):
        self.name = name
        self.count = count
        self.nodepath = None
        self.parentnode = parentnode
        self.children = {}

    def disp(self, ind=1):
        """disp(用于将树以文本形式显示)
        """
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createFeature(dataSet, supportConf):
    """构建FP树"""
    results = dict()
    supportData = dict()
    for data in dataSet:
        results[frozenset(data)] = results[frozenset(data)] + 1 if frozenset(data) in results.keys() else 1
    for k, v in results.items():
        for item in k:
            supportData[frozenset(item)] = results[k] + supportData[frozenset(item)] if frozenset(item) in supportData.keys() else 1
    supportData = dict(filter(lambda x: x[1] / len(results) > supportConf, supportData.items()))
    supportData = dict(map(lambda x: (x[0], x[1] / len(results)), supportData.items()))
    frequentSet = list(supportData.keys())
    if not len(frequentSet):
        return None, None

    # 初始化
    fpTree = FPTree('FP_tree', 1, None)

    # 对数据集按频繁项集排序
    for keys in results.keys():
        filterKeys = filter(lambda x: frozenset(x) in frequentSet, keys)
        sortKeys = sorted(filterKeys, key=lambda x: supportData[frozenset(x)], reverse=True)
        updateTree(sortKeys, fpTree, results[keys])
    return fpTree, frequentSet

def updateTree(sortKeys, inTree, count):
        if sortKeys[0] in inTree.children.keys():
            inTree.children[sortKeys[0]].count += count
        else:
            inTree.children[sortKeys[0]] = FPTree(sortKeys[0], count, inTree)
        if len(sortKeys) > 1:
            updateTree(sortKeys[1:], inTree.children[sortKeys[0]], count)


if __name__ == "__main__":
    dataSet = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               # ['r', 'x', 'n', 'o', 's', 's', 'o'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    fptree, frequentSet = createFeature(dataSet, supportConf=0.4)
    fptree.disp()
    print(frequentSet)
