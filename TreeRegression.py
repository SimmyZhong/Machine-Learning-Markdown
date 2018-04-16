import numpy as np
from numpy import nonzero, mat, shape


class TreeRegression(object):
    """树回归"""

    @staticmethod
    def readFromTxtFile(fileName):
        """读取数据"""
        file = 'sample_data/' + fileName
        results = list()
        with open(file) as content:
            for line in content.readlines():
                data = line.strip().split('\t')
                data = map(float, data)
                results.append(data)
        return mat(results)

    @staticmethod
    def splitDataSet(dataMatrix, figure, value):
        """
        划分数据
        :param dataMatrix:训练数据
        :param figure: 划分特征
        :param value: 划分特征值
        :return: 划分后的子集
        """
        leftDataSet = dataMatrix[nonzero(dataMatrix[:, figure] > value)[0], :]
        rightDataSet = dataMatrix[nonzero(dataMatrix[:, figure] <= value)[0], :]
        return leftDataSet, rightDataSet


if __name__ == "__main__":
    treeRegression = TreeRegression()
    test = mat(np.eye(4))
    left, right = treeRegression.splitDataSet(test, 1, 0.5)