import numpy as np


def readFromTxt(fileName):
    """读取数据集"""
    file = 'sample_data/' + fileName
    results = list()
    with open(file) as content:
        for data in content.readlines():
            result = data.strip().split(',')
            result = list(map(lambda x: float(x), result))
            results.append(result)
    return np.mat(results)


def pca(dataSet, topNfeat=9999):
    """PCA算法降维"""
    # 求均值
    meanVals = np.mean(dataSet, axis=0)
    # 数据集减去均值
    meanRemoved = dataSet - meanVals
    # 求协方差矩阵
    covmat = np.cov(dataSet, rowvar=0)
    # 特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covmat))
    # 将特征值按大小排序，取出前topNfeat个特征值
    eigValsSorted = np.argsort(eigVals)[: -(topNfeat+1):-1]
    # 特征向量截取
    redEigVects = eigVects[:, eigValsSorted]
    # 将数据集映射到低维空间
    lowDdataMat = meanRemoved * redEigVects
    reconMat = (lowDdataMat * redEigVects.T) + meanVals
    return lowDdataMat, reconMat


if __name__ == "__main__":
    dataSet = readFromTxt('data_PCA.txt')
    #测试
    print(pca(dataSet))#12131
