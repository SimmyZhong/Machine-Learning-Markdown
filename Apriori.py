'''
Apriori算法关联分析
Created on April, 2018
Update  on 2018-05-01
@author: Simmy
https://github.com/SimmyZhong/Machine-Learning-Markdown/blob/master/apriori.py
'''


def createSets(dataSet):
    """构建集合"""
    results = set()
    for transaction in dataSet:
        for item in transaction:
            results.add(frozenset(item))
    results = list(results)
    results.sort()
    return results

def scanD(dataSet, supportSets, minSupport):
    dataSet = list(map(set, dataSet))
    totalSum = float(len(dataSet))
    results = dict()
    for data in dataSet:
        for j in supportSets:
            if j.issubset(data):
                results[j] = results[j] + 1 if j in results.keys() else 1
    results = dict(filter(lambda x: x[1]/totalSum >= minSupport, results.items()))
    resultsKeys = list(results.keys())
    results = dict(map(lambda x: (x[0], x[1]/totalSum), results.items()))
    return resultsKeys, results

def aprioriGen(supportSet, k):
    results = list()
    lenS = len(supportSet)
    for i in range(lenS):
        for j in range(i+1, lenS):
            L1, L2 = list(supportSet[i])[:k-1], list(supportSet[j])[:k-1]
            L1.sort()
            L2.sort()
            if L1 == L2:
                results.append(supportSet[i] | supportSet[j])
    return results

def apriori(dataSet, minSupport=0.6):
    supportSet = createSets(dataSet)
    dataSet = list(map(set, dataSet))
    resultList, supportData = scanD(dataSet, supportSet, minSupport)
    resultList = [resultList]
    k = 1
    while k < len(supportSet):
        support = aprioriGen(resultList[k-1], k)
        rL, sD = scanD(dataSet, support, minSupport)
        if not rL:
            break
        resultList.append(rL)
        supportData.update(sD)
        k += 1
    return resultList, supportData


def generateRules(resultList, supportData, minConf=0.7):
    """
    关联规则生成
    :param resultList: 频繁项集列表
    :param supportData: 支持数据列表
    :param minConf: 最小可信度阈值
    :return: 关联规则
    """
    RulesList = list()
    if not resultList:
        return
    for i in range(1, len(resultList)):
        print(i)
        for freqSet in resultList[i]:
            items = [frozenset([x]) for x in freqSet]
            if i > 1:
                print(freqSet)
                rulesFromConseq(freqSet, items, supportData, RulesList, minConf)
            else:
                calConf(freqSet, items, supportData, RulesList, minConf)
    return RulesList


def calConf(freqSet, items, supportData, RulesList, minConf):
    """
    寻找满足最小阈值的关联规则"""
    prunedH = list()
    for item in items:
        conf = supportData[freqSet] / supportData[freqSet - item]
        if conf > minConf:
            print(freqSet-item, '-->', item, 'conf: ', conf)
            RulesList.append((freqSet-item, item, conf))
            prunedH.append(item)
    return prunedH


def rulesFromConseq(freqSet, items, supportData, RulesList, minConf):
    """构建关联规则"""
    m = len(items[0])
    if len(freqSet) > (m + 1):
        Hmp1 = aprioriGen(items, m)
        Hmp1 = calConf(freqSet, items, supportData, RulesList, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, items, supportData, RulesList, minConf)



if __name__ == "__main__":
    dataSet = [['1', '3', '4'], ['2', '3', '5'], ['1', '2', '3', '5'], ['2', '5']]
    # print(scanD(dataSet, createSets(dataSet), 0.5))
    # print(createSets(dataSet))
    # print(aprioriGen(createSets(['3', '1', '2']), 1))
    resultList, supportData = (apriori(dataSet))
    print(resultList, supportData)
    generateRules(resultList, supportData)
