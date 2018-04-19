# K均值聚类算法
import numpy as np
from numpy import mat, shape, zeros


def readFromTxtFile(fileName):
    """读取数据集"""

    file = 'sample_data/' + fileName
    results = list()
    with open(file) as content:
        for line in content.readlines():
            data = line.strip().split('\t')
            results.append(list(map(float, data)))
    return mat(results)


def meanFunc(dataSet):
    """均值计算方法"""

    return np.mean(dataSet, axis=0)


def errFunc(VectorA, VectorB):
    """欧氏距离计算方法"""

    return np.sqrt(np.sum(np.power(VectorA - VectorB, 2)))


def createCenterField(dataSet, k):
    """构建k个质心集合"""

    m, n = shape(dataSet)
    centerFiled = mat(zeros((k, n)))
    for i in range(n):
        iRange = np.max(dataSet[:, i]) - np.min(dataSet[:, i])
        centerFiled[:, i] = np.min(dataSet[:, i]) + iRange * mat(np.random.rand(k, 1))
    return centerFiled


def createKMeanClustering(dataSet, distanceFunc, k):
    """
    K均值聚类构建
    :param dataSet: 数据集
    :param distanceFunc: 距离方法
    :param k: 质心数
    :return: 质心集合， 方差
    """

    m, n = shape(dataSet)
    centroids = createCenterField(dataSet, k)
    clusterAssment = mat(zeros((m, 2)))
    changeFlag = True
    while changeFlag:
        changeFlag = False
        for i in range(m):
            minDis, bestIndex, bestErr = np.inf, 0, 0
            for j in range(k):
                distance = distanceFunc(dataSet[i, :], centroids[j, :])
                if distance < minDis:
                    minDis = distance
                    bestIndex = j
            if clusterAssment[i, :] != bestIndex:
                clusterAssment = True
            clusterAssment[i, 0],  clusterAssment[i, 0] = bestIndex, minDis ** 2
        for i in range(k):
            centroids[i, :] = meanFunc(dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0]])
        return centroids, clusterAssment



if __name__ == "__main__":
    # 测试案例
    data = np.random.randint(0, 4, 12).reshape(3, 4)
    print(data)
    print(createCenterField(data, 2))