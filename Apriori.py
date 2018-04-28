'''
Apriori算法关联分析
Created on April, 2018
Update  on 2018-05-01
@author: Simmy
https://github.com/SimmyZhong/Machine-Learning-Markdown/blob/master/apriori.py
'''
import numpy as np


def createSets(dataSet):
    """构建集合"""
    results = dict()
    dataSet = map(set, dataSet)
    for transaction in dataSet:
        for item in transaction:
            results[item] = results[item] + 1 if item in results.keys() else 1
    return results


if __name__ == "__main__":
    print(createSets([['1', '2', '3', '3'], ['4', '5', '6', '6', '7'], ['1', '3', '4'], ['5', '6', '1']]))