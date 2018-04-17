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

    @staticmethod
    def leafFunc(dataMatrix):
        return np.mean(dataMatrix[:, -1])

    @staticmethod
    def varFunc():
        pass

    def chooseBestFeature(self, dataMatrix, leafFunc, varFunc, ops=(1, 4)):
        """
        选取最优划分特征
        :param dataMatrix: 训练集
        :param leafFunc: 叶节点方法
        :param errToler: 计算方差的引用
        :param ops: 自定义参数
        :return: 最优特征，最优特征阈值
        """
        # 统计剩余特征值的数目，若唯一则返回叶节点
        if len(set(dataMatrix[:, -1].T.tolist()[0])) == 1:
            return None, self.leafFunc(dataMatrix)
        m, n = shape(dataMatrix)

        # 计算初始总方差
        varDataMatrix = np.var(dataMatrix[:, -1]) * shape(dataMatrix)[0]

        # 初始化
        bestS = np.inf
        bestFeature = 0
        bestVal = 0

        # 遍历特征，以及特征值
        for feature in range(n - 1):
            for val in set(dataMatrix[:, feature].T.tolist()[0]):

                # 划分左右子树
                leftDataSet, rightDataSet = self.splitDataSet(dataMatrix, feature, val)

                # 若 左右子树小于自定义参数，则不划分
                if len(leftDataSet) < ops[1] or len(rightDataSet) < ops[1]:
                    continue
                varleftDataSet = np.var(leftDataSet[:, -1]) * shape(leftDataSet)[0]
                varRightDataSet = np.var(rightDataSet[:, -1]) * shape(rightDataSet)[0]
                sumS = varleftDataSet + varRightDataSet

                # 若划分后左右子树方差和更小，则保存划分阈值和特征为最优
                if bestS > sumS:
                    bestS = sumS
                    bestFeature = feature
                    bestVal = val

        # 划分后最优方差和小于自定义参数，则不进行划分，返回叶节点，否则返回最优划分特征和特征阈值
        if bestS - varDataMatrix < ops[0]:
            return None, self.leafFunc(dataMatrix)
        return bestFeature, bestVal

    def createTreeRegression(self, dataMatrix, leafFunc, varFunc, ops=(1, 4)):
        """
        创建树回归
        :param dataMatrix: 训练集
        :param leafFunc: 创建叶节点方法
        :param varFunc: 方差计算方法
        :param ops: 自定义参数组
        :return: 树回归
        """
        feature, value = self.chooseBestFeature(dataMatrix, leafFunc, varFunc, ops)
        if feature is None:
            return value
        treeRegress = dict()
        leftDataSet, rightDataSet = self.splitDataSet(dataMatrix, feature, value)
        treeRegress['left'] = self.createTreeRegression(leftDataSet, leafFunc, varFunc, ops=(1, 4))
        treeRegress['right'] = self.createTreeRegression(rightDataSet, leafFunc, varFunc, ops=(1, 4))
        treeRegress['feature'] = feature
        treeRegress['val'] = value
        return treeRegress


if __name__ == "__main__":
    treeRegression = TreeRegression()
    test = mat(np.eye(4))
    left, right = treeRegression.splitDataSet(test, 1, 0.5)