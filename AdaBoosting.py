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
            bestClasEst[dataMatrix[:, i] > threshVal] = 1.0
        return bestClasEst

    def builtStump(self, dataMatrix, labels, D):
        """
        单层决策树生成函数
        :param dataMatrix: 样本数据集
        :param labels: 分类标签
        :param D: 各样本权重
        :return:最优分类结果，分类器信息，错误率
        """

        labels = mat(labels).T
        m, n = shape(dataMatrix)
        minError = np.inf
        retarray = mat(ones((m, 1)))
        bestClasEst = mat(zeros((m, 1)))
        stump = dict()
        for i in range(n):
            step = 10.0
            stepSize = (dataMatrix[:, i].max() - dataMatrix[:, i].min()) / step
            for j in range(-1, int(step)+1):
                for method in ['lt', 'gt']:
                    threshVal = dataMatrix[:, i].min() + j * stepSize
                    predictVals = self.stumpClassify(dataMatrix, i, threshVal, method)
                    retarray[predictVals == labels] = 0
                    errorRate = retarray.T * D
                    # print(bestClasEst, i, threshVal, method, '%.3f' % errorRate)
                    if errorRate < minError:
                        bestClasEst = predictVals
                        minError = errorRate
                        stump['threshVal'] = threshVal
                        stump['method'] = method
                        stump['i'] = i
        return bestClasEst, stump, minError


if __name__ == "__main__":
    adaboosting = AdaBoostingClassfier()
    dataMatrix, labels = adaboosting.readFromTxt('svm_simple_test.txt')
    D = mat(ones((shape(dataMatrix)[0], 1)) / 5)
    bestClasEst, stump, minError = adaboosting.builtStump(dataMatrix, labels, D)
    print(bestClasEst, stump, minError)