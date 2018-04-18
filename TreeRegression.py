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
        """计算均值"""
        return np.mean(dataMatrix[:, -1])

    @staticmethod
    def varFunc(dataMatrix):
        """计算数据集总方差"""
        return np.var(dataMatrix[:, -1]) * shape(dataMatrix)[0]

    def chooseBestFeature(self, dataMatrix, ops=(1, 4)):
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
        varDataMatrix = self.varFunc(dataMatrix)

        # 初始化
        bestS, bestFeature, bestVal = np.inf, 0, 0

        # 遍历特征，以及特征值
        for feature in range(n - 1):
            for val in set(dataMatrix[:, feature].T.tolist()[0]):

                # 划分左右子树
                leftDataSet, rightDataSet = self.splitDataSet(dataMatrix, feature, val)

                # 若 左右子树小于自定义参数，则不划分
                if len(leftDataSet) < ops[1] or len(rightDataSet) < ops[1]:
                    continue
                sumS = self.varFunc(leftDataSet) + self.varFunc(rightDataSet)

                # 若划分后左右子树方差和变小，则保存划分阈值和特征为最优
                if bestS > sumS:
                    bestS = sumS
                    bestFeature = feature
                    bestVal = val

        # 划分后最优方差和小于自定义参数，则不进行划分，返回叶节点，否则返回最优划分特征和特征阈值
        if bestS - varDataMatrix < ops[0]:
            return None, self.leafFunc(dataMatrix)
        return bestFeature, bestVal

    def createTreeRegression(self, dataMatrix, ops=(1, 4)):
        """
        创建树回归
        :param dataMatrix: 训练集
        :param leafFunc: 创建叶节点方法
        :param varFunc: 方差计算方法
        :param ops: 自定义参数组
        :return: 树回归
        """
        # ops=(1,4)，非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
        # 之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。
        # 最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分

        feature, value = self.chooseBestFeature(dataMatrix, ops)
        if feature is None:
            return value
        treeRegress = dict()
        leftDataSet, rightDataSet = self.splitDataSet(dataMatrix, feature, value)
        treeRegress['left'] = self.createTreeRegression(leftDataSet, ops=(1, 4))
        treeRegress['right'] = self.createTreeRegression(rightDataSet, ops=(1, 4))
        treeRegress['feature'] = feature
        treeRegress['val'] = value
        return treeRegress

    @staticmethod
    def isTree(treeRegress):
        """判断子树是否叶节点"""
        return isinstance(treeRegress, dict)

    def getMean(self, treeRegress):
        """遍历树回归知道叶节点，然后计算均值"""
        if self.isTree(treeRegress['left']):
            treeRegress['left'] = self.getMean(treeRegress['left'])
        if self.isTree(treeRegress['right']):
            treeRegress['right'] = self.getMean(treeRegress['right'])
        return (treeRegress['left'] + treeRegress['right']) / 2.0

    def treePrune(self, treeRegress, testData):
        """
        树回归后剪枝
        :param treeRegress: 训练好的树回归模型
        :param testData: 测试数据集
        :return: 剪枝后的树回归
        """
        if shape(testData)[0] == 0:
            return self.getMean(treeRegress)
        if self.isTree(treeRegress['left']) or self.isTree(treeRegress['right']):
            leftDataSet, rightDataSet = self.splitDataSet(testData, treeRegress['feature'], treeRegress['val'])
        if self.isTree(treeRegress['left']):
            treeRegress['left'] = self.treePrune(treeRegress['left'], leftDataSet)
        if self.isTree(treeRegress['right']):
            treeRegress['right'] = self.treePrune(treeRegress['right'], rightDataSet)
        if not self.isTree(treeRegress['left']) and self.isTree(treeRegress['right']):
            varS = self.varFunc(leftDataSet) + self.varFunc(rightDataSet)
            var_unsplit = self.varFunc(testData)
            if var_unsplit < varS:
                return np.mean(testData[:, -1])
            else:
                return treeRegress
        return treeRegress


class ModelTree(object):
    """模型树，和回归树区别在于叶节点不是用常数表示，而是分段线性函数"""

    def splitTree(self, dataSet, feature, val):
        """数据划分"""

        leftTree = dataSet[np.nonzero(dataSet[:, feature] > val)[0], :]
        rightTree = dataSet[np.nonzero(dataSet[:, feature] <= val)[0], :]
        return leftTree, rightTree

    @staticmethod
    def transfer(dataSet):
        """转换函数"""
        m, n = shape(dataSet)
        xMatrix, yMatrix = mat(np.ones(m, n)), mat(np.ones(m, 1))
        xMatrix[:, 1:n], yMatrix = dataSet[:, :n - 1], dataSet[:, -1]
        return xMatrix, yMatrix

    def leafFunc(self, dataSet):
        """叶节点"""

        xMatrix, yMatrix = self.transfer(dataSet)
        xTx = xMatrix.T * xMatrix
        if np.linalg.det(xTx) == 0:
            print('不存在逆矩阵')
        ws = xTx.I * xMatrix.T * yMatrix
        return ws

    def errFunc(self, dataSet):
        """求取回归的总方差"""

        xMatrix, yMatrix = self.transfer(dataSet)
        ws = self.leafFunc(dataSet)
        yHat = xMatrix * mat(ws).T
        return np.sum(np.power((yMatrix - yHat), 2))

    def chooseBestFeature(self, dataSet, leafFunc, errFunc, ops):
        """选取最优划分特征"""

        m ,n = shape(dataSet)
        if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
            return None, leafFunc(dataSet)
        bestS, bestFeature, bestVal = np.inf, 0, 0
        for i in range(n - 1):
            for val in set(dataSet[:, i].T.tolist()[0]):
                leftTree, rightTree = self.splitTree(dataSet, i, val)
                if shape(leftTree)[1] < ops[1] or shape(rightTree)[1] < ops[1]:
                    continue
                errS = errFunc(leftTree) + errFunc(rightTree)
                if bestS > errS:
                    bestFeature = i
                    bestVal = val
                    bestS = errS
        if errFunc(dataSet) - bestS < ops[0]:
            return None, leafFunc(dataSet)
        return bestFeature, bestVal

    def createModelTree(self, dataSet, leafFunc, errFunc, ops=(1, 4)):
        """构建模型树"""

        feature, val = self.chooseBestFeature(dataSet, leafFunc, errFunc, ops)
        if feature is None:
            return val
        modelTree = dict(feature=feature, val=val)
        leftTree, rightTree = self.splitTree(dataSet, feature, val)
        modelTree['left'] = self.createModelTree(leftTree, leafFunc, errFunc, ops=(1, 4))
        modelTree['right'] = self.createModelTree(rightTree, leafFunc, errFunc, ops=(1, 4))
        return modelTree


if __name__ == "__main__":
    treeRegression = TreeRegression()
    test = mat(np.eye(4))
    left, right = treeRegression.splitDataSet(test, 1, 0.5)