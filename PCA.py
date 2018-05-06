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


def pca(dataSet, topNfeat=6):
    """PCA算法降维"""
    meanVals = np.mean(dataSet, axis=0)
    meanRemoved = dataSet - meanVals
    covmat = np.cov(dataSet, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covmat))
    eigValsSorted = np.argsort(eigVals)[: -(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValsSorted]
    lowDdataMat = meanRemoved * redEigVects
    reconMat = (lowDdataMat * redEigVects.T) + meanVals
    return lowDdataMat, reconMat


if __name__ == "__main__":
    dataSet = readFromTxt('data_PCA.txt')
    print(pca(dataSet))