import numpy as np
from numpy import mat, shape, ones, zeros
import matplotlib.pyplot as plt


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

        ax.scatter(dataMatrix[:, 1].flatten().A[0], yAttr[:, 0].flatten().A[0])
        dataMatrixCopy = dataMatrix.copy()
        dataMatrixCopy.sort(0)
        yHat = dataMatrixCopy * ws
        ax.plot(dataMatrixCopy[:, 1], yHat)
        plt.show()


if __name__ == "__main__":
    fileName = 'data8-01.txt'
    regression = StandRegresion()
    dataMatrix, yAttr = regression.readFromTxtFile(fileName)
    ws = regression.standRegres(dataMatrix, yAttr)
    print(ws)
    regression.figureShow(dataMatrix, yAttr, ws)