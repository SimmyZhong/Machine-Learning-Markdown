# K均值聚类算法
import numpy as np
from numpy import mat, shape, zeros
import matplotlib.pyplot as plt


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
        min_ = np.min(dataSet[:, i]) if not np.min(dataSet[:, i])==np.nan else 0
        max_ = np.max(dataSet[:, i]) if not np.max(dataSet[:, i])==np.nan else 0
        iRange = float(max_ - min_)
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
            minDis, bestIndex = np.inf, -1
            for j in range(k):
                distance = distanceFunc(dataSet[i, :], centroids[j, :])
                if distance < minDis:
                    minDis = distance
                    bestIndex = j
            if clusterAssment[i, 0] != bestIndex:
                changeFlag = True
            clusterAssment[i, :] = bestIndex, minDis ** 2
        for i in range(k):
            centroids[i, :] = meanFunc(dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0]])
    return centroids, clusterAssment


def binKmeans(dataSet, distanceFunc, k):
    """
    二分K均值聚类算法
    :param dataSet: 数据集
    :param distanceFunc: 距离算法
    :param k: 聚类簇数
    :return: 质心，聚类结果
    """
    m, n = shape(dataSet)
    clusterAssment = mat(zeros((m, 2)))
    # 初始质心
    centroids = [list(np.mean(dataSet, axis=0).tolist()[0])]
    clusterAssment[:, 1] = np.mat(list(map(lambda x: (distanceFunc(x, mat(centroids[0])))**2, dataSet))).T
    while len(centroids) < k:
        bestSse, bestIndex = np.inf, -1
        for i in range(len(centroids)):
            cluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0]]
            centroidMat, ssesplit = createKMeanClustering(cluster, distanceFunc, 2)
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0]][:, 1])
            sse = sseNotSplit + sum(ssesplit[:, 1])

            # 选取划分后SSE最小的簇进行二分K均值聚类
            if sse < bestSse:
                bestSse, bestIndex = sse, i
        centroidMat, ssesplit = createKMeanClustering(dataSet[np.nonzero(clusterAssment[:, 0].A == bestIndex)[0]], distanceFunc, 2)

        # 因为是二分K聚类算法，所以聚类后索引为0和1,
        ssesplit[np.nonzero(ssesplit[:, 0].A == 0)[0], 0] = bestIndex
        ssesplit[np.nonzero(ssesplit[:, 0].A == 1)[0], 0] = len(centroids)

        # 更新簇的分配结果
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestIndex)[0], :] = ssesplit

        # 更新质心
        centroids[bestIndex] = centroidMat[0, :].tolist()[0]
        centroids.append(centroidMat[1, :].tolist()[0])
    return mat(centroids), clusterAssment

if __name__ == "__main__":
    # 测试案例
    data = np.random.randint(0, 4, 12).reshape(3, 4)
    # print(data)
    # print(createCenterField(data, 2))
    dataSet = [[3.15357605, -3.94962877],
               [3.3593825, 2.05965957],
               [2.41900657, 3.30513371],
               [-2.80505526, -3.73280289],
               [2.35622556, -3.02056425],
               [2.95373358, 2.32801413],
               [2.46154315, 2.78737555],
               [-3.38237045, -2.9473363],
               [2.65077367, -2.79019029],
               [2.6265299, 3.10868015],
               [-2.46154315, -2.78737555],
               [-3.53973889, -2.89384326]]
    # centroids, clusterAssment = createKMeanClustering(np.mat(dataSet), errFunc, 3)
    centroids, clusterAssment = binKmeans(np.mat(dataSet), errFunc, 3)
    # print(centroids)
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1.scatter(np.mat(dataSet).A[:, 0], np.mat(dataSet).A[:, 1], color='red')
    ax_1.scatter(centroids.A[:, 0], centroids.A[:, 1], color='black')
    plt.show()