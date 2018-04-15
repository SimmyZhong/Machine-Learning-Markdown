import numpy as np
from numpy import mat, shape, ones, zeros


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
                yAttr.append(float(data[num_figure]))
        return dataMatrix, yAttr

    @staticmethod
    def standRegres(dataMatrix, yAttr):
        """
        标准回归函数
        :param dataMatrix: 样本集
        :param yAttr: 输出值
        :return: 回归系数
        """
        dataMatrix = mat(dataMatrix)
        yAttr = mat(yAttr).T
        xTx = dataMatrix.T * dataMatrix
        if np.linalg.det(xTx) == 0:
            print('逆矩阵不存在,无法求解')
        ws = xTx.I * (dataMatrix.T * yAttr)
        return ws


if __name__ == "__main__":
    fileName = 'data8-01.txt'
    regression = StandRegresion()
    dataMatrix, yAttr = regression.readFromTxtFile(fileName)
    # ws = regression.standRegres(dataMatrix, yAttr)
    # print(ws)