import numpy as np
from numpy import mat, shape, ones, zeros
import matplotlib.pyplot as plt
import copy
import math


class StandRegresion(object):
    """标准回归"""

    @staticmethod
    def readFromTxtFile(fileName):
        """
        读取数据
        :param fileName: 数据源文件
        :return: 返回数据集和输出值
        """
        file = 'sample_data/' +fileName
        dataMatrix, yAttr = [], []
        with open(file) as content:
            for line in content.readlines():
                data = line.strip().split('\t')
                num_figure = len(data)
                figures = list()
                for i in range(num_figure - 1):
                    figures.append(float(data[i]))
                dataMatrix.append(figures)
                yAttr.append(float(data[num_figure - 1]))
        return mat(dataMatrix), mat(yAttr).T

    @staticmethod
    def standRegres(dataMatrix, yAttr):
        """
        标准回归函数
        :param dataMatrix: 样本集
        :param yAttr: 输出值
        :return: 回归系数
        """
        xTx = dataMatrix.T * dataMatrix
        if np.linalg.det(xTx) == 0:
            print('逆矩阵不存在,无法求解')
        ws = xTx.I * (dataMatrix.T * yAttr)
        return ws

    @staticmethod
    def figureShow(dataMatrix, yAttr, ws):
        figure = plt.figure()

        # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
        ax = figure.add_subplot(111)

        ax.scatter(dataMatrix[:, 1].flatten().A[0], yAttr[:, 0].flatten().A[0], s=2, c='red')
        dataMatrixCopy = copy.copy(dataMatrix)
        dataMatrixCopy.sort(0)
        yHat = dataMatrixCopy * ws
        ax.plot(dataMatrixCopy[:, 1], yHat)
        plt.show()

    @staticmethod
    def weightdRegres(testPoint, datamatrix, yAttr, k=1.0):
        """
        局部加权回归
        :param testPoint: 测试实例
        :param datamatrix: 训练数据集
        :param yAttr: 训练集输出
        :return: 测试实例预测值
        """
        m, n = shape(datamatrix)
        weights = mat(np.eye(m))
        for j in range(m):
            diffMat = testPoint - dataMatrix[j, :]
            weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        xTwx = dataMatrix.T * (weights * dataMatrix)
        if np.linalg.det(xTwx) == 0:
            print('该矩阵不存在逆矩阵')
            return
        ws = xTwx.I * (dataMatrix.T * (weights * yAttr))
        return testPoint * ws

    @classmethod
    def weightRegresText(cls, testDataMatrix, dataMatrix, yAttr, k=1.0):
        """
        局部加权回归函数测试
        :param textDataMatrix: 测试集
        :param dataMatrix: 训练集
        :param yAttr: 训练集输出值
        :param k: 高斯核k参数
        :return: 测试集预测值
        """
        num = len(testDataMatrix)
        result = mat(zeros((num, 1)))
        for i in range(num):
            result[i] = cls.weightdRegres(testDataMatrix[i], dataMatrix, yAttr, k)
        return result


def pltShow(dataMatrix, yAttr, results):
    """
    局部加权函数可视化
    :param dataMatrix: 训练集
    :param yAttr: 训练集输出值
    :param results: 测试集预测值
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    srtInd = dataMatrix[:, 1].argsort(0)
    xSort = dataMatrix[srtInd][:, 0, :]
    ax.plot(xSort[:, 1], results[srtInd, 0])
    ax.scatter(dataMatrix[:, 1].flatten().A[0], yAttr[:, 0].flatten().A[0], s=2, c='red')
    plt.show()


if __name__ == "__main__":
    fileName = 'data8-01.txt'
    regression = StandRegresion()
    dataMatrix, yAttr = regression.readFromTxtFile(fileName)
    ws = regression.standRegres(dataMatrix, yAttr)
    print(ws)
    # regression.figureShow(dataMatrix, yAttr, ws)
    results = regression.weightRegresText(dataMatrix, dataMatrix, yAttr, k=0.01)
    pltShow(dataMatrix, yAttr, results)

