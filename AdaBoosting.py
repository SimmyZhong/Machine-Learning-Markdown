from numpy import mat, multiply, array, zeros, ones, shape
import numpy as np
from math import log, exp


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
                dataMatrix.append([float(data[0]), float(data[1])])
                labels.append(float(data[-1]))
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
        m, n = shape(dataMatrix)
        bestClasEst = ones((m, 1))
        if threshIneq == 'lt':
            bestClasEst[dataMatrix[:, i] <= threshVal] = -1.0
        else:
            bestClasEst[dataMatrix[:, i] > threshVal] = -1.0
        return bestClasEst

    def builtStump(self, dataMatrix, labels, D):
        """
        单层决策树生成函数
        :param dataMatrix: 样本数据集
        :param labels: 分类标签
        :param D: 各样本权重
        :return:最优分类结果，分类器信息，错误率
        """

        m, n = shape(dataMatrix)
        minError = np.inf
        bestClasEst = mat(zeros((m, 1)))
        stump = dict()
        step = 10.0
        for i in range(n):
            stepSize = (dataMatrix[:, i].max() - dataMatrix[:, i].min()) / step
            for j in range(-1, int(step)+1):
                for method in ['lt', 'gt']:
                    threshVal = dataMatrix[:, i].min() + float(j) * stepSize
                    predictVals = self.stumpClassify(dataMatrix, i, threshVal, method)
                    retarray = mat(ones((m, 1)))
                    retarray[predictVals == labels] = 0
                    errorRate = D.T * retarray
                    # print(i, threshVal, method, '%.3f' % errorRate)
                    if errorRate < minError:
                        bestClasEst = predictVals.copy()
                        minError = errorRate
                        stump['threshVal'] = threshVal
                        stump['method'] = method
                        stump['i'] = i
        return bestClasEst, stump, minError

    def adaBoostingClassfier(self, dataMatrix, labels):
        """
        基于单层决策树的训练函数
        :param dataMatrix: 数据集
        :param labels: 样本分类
        :return:弱分类器集
        """

        dataMatrix = mat(dataMatrix)
        labels = mat(labels).T
        m, n = shape(dataMatrix)
        D = mat(ones((m, 1)) / m)
        bestClassifier = list()
        aggClassEst = mat(zeros((m, 1)))
        while True:
            bestClasEst, stump, minError = self.builtStump(dataMatrix, labels, D)
            alpha = float(log((1.0 - minError) / max(minError, 1e-16)) / 2.0)
            stump['alpha'] = alpha
            result = multiply(labels, bestClasEst)
            result[result == 1] = exp(-alpha)
            result[result == -1] = exp(alpha)
            D = multiply(result, D) / D.sum()
            bestClassifier.append(stump)
            aggClassEst += alpha * bestClasEst
            error_rate = ones((m, 1))
            error_rate[np.sign(aggClassEst) == labels] = 0
            # print(error_rate.sum() / m)
            if error_rate.sum() == 0:
                break
        return bestClassifier

    def adaClassfierTest(self, dataMatrix, classfierArray):
        dataMatrix = mat(dataMatrix)
        m, n =shape(dataMatrix)
        result = zeros((m, 1))
        for classfier in classfierArray:
            threshVal = classfier['threshVal']
            method = classfier['method']
            i = classfier['i']
            alpha = classfier['alpha']
            result += self.stumpClassify(dataMatrix, i, threshVal, method) * alpha
            # print(result)
        return np.sign(result)


if __name__ == "__main__":
    adaboosting = AdaBoostingClassfier()
    dataMatrix, labels = adaboosting.readFromTxt('text.txt')
    classfierArray = adaboosting.adaBoostingClassfier(dataMatrix, labels)
    print(adaboosting.adaClassfierTest([0.6, 1.5], classfierArray))