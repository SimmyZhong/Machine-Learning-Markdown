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


if __name__ == "__main__":
    # 测试案例
    data = np.random.randint(0, 4, 12).reshape(3, 4)
    print(data)
    print(createCenterField(data, 2))