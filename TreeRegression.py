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

    def splitDataSet(self, dataMatrix, feature, value):
        """
        划分数据
        :param dataMatrix:训练数据
        :param figure: 划分特征
        :param value: 划分特征值
        :return: 划分后的子集
        """
        leftDataSet = dataMatrix[nonzero(dataMatrix[:, feature] > value)[0], :]
        rightDataSet = dataMatrix[nonzero(dataMatrix[:, feature] <= value)[0], :]
        return leftDataSet, rightDataSet

    def chooseBestFeature(self, dataMatrix, leafFunc, errToler, ops=(1, 4)):
        """
        选取最优划分特征
        :param dataMatrix:
        :param leafFunc:
        :param errToler:
        :param ops:
        :return:
        """
        pass

    def createTreeRegression(self, dataMatrix, leafFunc, varFunc, ops=(1, 4)):
        """
        创建树回归
        :param dataMatrix: 训练集
        :param leafFunc: 创建叶节点方法
        :param errToler: 方差计算方法
        :param ops: 自定义参数组
        :return: 树回归
        """
        feature, value = self.chooseBestFeature(dataMatrix, leafFunc, varFunc, ops)
        if feature is None:
            return value
        treeRegress = dict()
        leftDataSet, rightDataSet = self.splitDataSet(dataMatrix, feature, value)
        if len(leftDataSet) < ops[1] or len(rightDataSet) < ops[1]:
            return value
        varDataMatrix = np.power(np.var(dataMatrix[:, feature]), 2) * shape(dataMatrix)[0]
        varleftDataSet = np.power(np.var(leftDataSet[:, feature]), 2)* shape(leftDataSet)[0]
        varRightDataSet = np.power(np.var(rightDataSet[:, feature]), 2)* shape(rightDataSet)[0]
        if varDataMatrix - varRightDataSet - varleftDataSet < ops[0]:
            return value
        treeRegress['left'] = leftDataSet
        treeRegress['right'] = rightDataSet
        treeRegress['feature'] = feature
        treeRegress['val'] = value
        return treeRegress


if __name__ == "__main__":
    treeRegression = TreeRegression()
    test = mat(np.eye(4))
    left, right = treeRegression.splitDataSet(test, 1, 0.5)