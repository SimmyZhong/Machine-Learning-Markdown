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
    results = list(map(lambda x: {x[0]: x[1]/totalSum}, results.items()))
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
        supportData.append(sD)
        k += 1
    return resultList, supportData

if __name__ == "__main__":
    dataSet = [['1', '2', '3', '3'], ['4', '6', '6', '7'], ['1', '3', '4'], ['5', '6', '1'], ['1', '5', '3']]
    # print(scanD(dataSet, createSets(dataSet), 0.5))
    # print(createSets(dataSet))
    # print(aprioriGen(createSets(dataSet), 1))
    print(apriori(dataSet))
