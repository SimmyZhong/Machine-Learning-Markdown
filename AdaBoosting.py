from numpy import mat, multiply, array, zeros, ones, shape
import numpy as np


class AdaBoostingClassfier(object):
    """集成方法adaboosting"""

    def readFromTxt(self, file):
        """
        读取文件得到训练数据集
        :param file: 文件名
        :return: 样本集和对应分类标签
        """
        file = 'sample_data/' + file
        dataMatrix = []
        labels = []
        with open(file) as content:
            for line in content.readlines():
                data = line.strip().split('\t')
                dataMatrix.append([data[0], data[1]])
                labels.append(data[-1])
        return mat(dataMatrix), mat(labels)

    def stumpClassify(self, dataMatrix, i, threshVal, threshIneq):
        """
        决策树桩
        :param dataMatrix: 训练数据集
        :param i: 数据集的第i个特征
        :param threshVal: 决策树阈值
        :param threshIneq: 决策树分类方法
        :return: 决策树桩信息，误差率，分类结果
        """
        m = shape(dataMatrix)
        bestClasEst = ones((m, 1))
        if threshVal == 'lt':
            bestClasEst[dataMatrix[:, i] <= threshVal] = -1.0
        else:
            bestClasEst[dataMatrix[:, i] > threshVal] = 1.0