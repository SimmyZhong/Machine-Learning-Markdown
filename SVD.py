from numpy import shape, nonzero, logical_and, mat, corrcoef
from numpy import linalg as la


def loadExData3():
    # 利用SVD提高推荐效果，菜肴矩阵
    """
    行：代表人
    列：代表菜肴名词
    值：代表人对菜肴的评分，0表示未评分
    """
    return mat(
        [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
         [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
         [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
         [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
         [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
         [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
         [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
         [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]])


def eulidSim(inA, inB):
    """欧氏距离相识度计算方法"""
    return 1.0/(1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    """皮尔逊相关系数计算方法"""
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    """余弦相似度计算方法"""
    num = float(inA.T*inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num/denom)


def standEst(dataMat, user, simMeans, product):
    """
    要对某一物品预估用户对其评价，首先找出此用户已评分的其他商品，根据这些商品与需评分商品的相似度进行评分计算，
    预估评分 = sum（某一商品与此商品的相似度 * 此用户评分） / sum（相似度）
    :param dataMat: 数据集，行对应用户，列对应商品
    :param user: 用户，即数据集中对应的行
    :param simMeans: 相似度计算方法，共列举了以上3种
    :param product: 需预估评分的商品，数据集中用户尚未评分的某一列
    :return: 商品的预估评分
    """
    m, n = shape(dataMat)
    similary, score = 0, 0
    for i in range(n):
        if dataMat[user, i] == 0:

            # 如果商品未被用户评分，不计入相似度计算
            continue

        # 变量 calVect 给出的是两个物品当中都已经被评分的那些元素的索引ID，如果为空则不计算相似度
        # logical_and 计算x1和x2元素的真值。
        calVect = nonzero(logical_and(dataMat[:, i].A > 0, dataMat[:, product].A > 0))[0]
        if len(calVect) == 0:
            continue

        # 计算2个商品的相似度
        similarity = simMeans(dataMat[calVect, product], dataMat[calVect, i])

        similary += similarity
        score += similarity * dataMat[user, i]
    return score / similary if similary > 0 else 0


def recommendSystem(dataMat, user, N=3, simMeans=eulidSim, estMethod=standEst):
    """
    推荐系统主体
    :param dataMat: 数据集
    :param user: 用户
    :param N: 需推荐的数目
    :param simMeans: 相似度计算方法
    :param estMethod: 商品评分计算方法
    :return: 预估分最高的前N个商品
    """

    # 找出所有用户未评分的商品
    productIndex = nonzero(dataMat[user, :].A == 0)[1]
    if len(productIndex) == 0:
        print('此用户已对所有商品评分')
        return
    elif len(productIndex) == len(dataMat):
        print('此用户未对任何商品评分，无法推荐')
        return
    scoreList = dict()

    # 计算评分
    for product in productIndex:
        score = estMethod(dataMat, user, simMeans, product)
        scoreList[product] = score
    # print(scoreList)

    # 排序，返回前N个推荐商品
    sortProductList = list(sorted(scoreList.items(), key=lambda x: x[1], reverse=True))
    return list(map(lambda x: x[0], sortProductList[:N]))


if __name__ == "__main__":
    dataMat = loadExData3()
    print(recommendSystem(dataMat, 3))