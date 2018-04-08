import numpy as np
import pandas as pd
from numpy import mat, ones, zeros, multiply, shape, arange, array
import random
import matplotlib.pylab as plt


class SVMClassifier(object):
    """SVM分类器实现"""

    # def __init__(self, sample_data, labels, constant, toler, max_iter):
    #     """
    #         args:
    #         sample_data 样本集
    #         labels 分类标签
    #         constant 松弛变量C
    #         toler 容错率
    #         max_iter 最大迭代次数
    #     """
    #
    #     self.sample_data = mat(sample_data)
    #     self.labels = mat(labels)
    #     self.constant = constant
    #     self.toler = toler
    #     self.max_iter = max_iter

    def readTxt(self, file):
        """读取数据源"""

        path = 'sample_data/' + file
        with open(path) as file:
            sample_data = list()
            labels = list()
            for line in file.readlines():
                data = line.strip().split('\t')
                sample_data.append([float(data[0]), float(data[1])])
                labels.append(float(data[2]))
        return sample_data, labels

    def selctJ(self, i, m):
        """随机选择一个不等于i的点"""

        j = i
        while (j == i):
            j = random.choice(range(m))
        return j

    def clipAlpha(self, value, L, H):
        """边界约束"""

        if value < L:
            return L
        elif value > H:
            return H
        else:
            return value

    def smoAlgorithm(self, sample_data, labels, constant, toler, max_iter):
        """SMO算法简单实现"""

        sample_data = mat(sample_data)
        labels = mat(labels).transpose()
        m, n = np.shape(sample_data)

        alpha = mat(zeros((m, 1)))  # 初始化α
        iter_times = 0  # 初始化迭代次数
        b = 0

        while (iter_times < max_iter):
            alpha_change_times = 0
            for i in range(m):
                fx_i = float(multiply(alpha, labels).T * (sample_data * sample_data[i, :].T)) + b
                Ei = fx_i - float(labels[i])
                # 此处解释不是很懂https://github.com/apachecn/MachineLearning/blob/master/src/py2.x/6.SVM/svm-simple.py
                # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。
                # 我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
                # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
                # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
                '''
                # 检验训练样本(xi, yi)是否满足KKT条件
                yi*f(i) >= 1 and alpha = 0 (outside the boundary)
                yi*f(i) == 1 and 0<alpha< C (on the boundary)
                yi*f(i) <= 1 and alpha = C (between the boundary)
                '''
                if ((Ei * labels[i] < -toler) and alpha[i] < constant) or ((Ei * labels[i] > toler) and alpha[i] > 0):
                    j = self.selctJ(i, m)
                    fx_j = float(multiply(alpha, labels).T * (sample_data * sample_data[j, :].T)) + b
                    Ej = fx_j - float(labels[j])
                    alpha_i_old = alpha[i].copy()
                    alpha_j_old = alpha[j].copy()

                    # 边界约束
                    if labels[i] == labels[j]:
                        L = max(0, alpha[i] + alpha[j] - constant)
                        H = min(constant, alpha[i] + alpha[j])
                    else:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(constant, alpha[j] - alpha[i] + constant)

                    if L == H:
                        print('L==H')
                        continue

                    eta = 2.0 * sample_data[i, :] * sample_data[j, :].T - sample_data[i, :] * sample_data[i, :].T - \
                          sample_data[j, :] * sample_data[j, :].T

                    if eta >= 0:
                        print('eta>=0')
                        continue
                    alpha[j] -= labels[j] * (Ei - Ej) / eta
                    alpha[j] = self.clipAlpha(alpha[j], L, H)
                    if (abs(alpha[j] - alpha_j_old) < 0.00001):
                        print('j 变化量太小')
                        continue
                    alpha[i] += (alpha_j_old - alpha[j]) * labels[j] / labels[i]
                    # 这个地方没看懂
                    # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                    # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                    # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
                    b1 = b - Ei - labels[i] * (alpha[i] - alpha_i_old) * sample_data[i, :] * sample_data[i, :].T - \
                         labels[j] * (alpha[j] - alpha_i_old) * sample_data[j, :] * sample_data[j, :].T
                    b2 = b - Ej - labels[i] * (alpha[i] - alpha_i_old) * sample_data[i, :] * sample_data[i, :].T - \
                         labels[j] * (alpha[j] - alpha_i_old) * sample_data[j, :] * sample_data[j, :].T
                    if (0 < alpha[i]) and (constant > alpha[i]):
                        b = b1
                    elif (0 < alpha[j]) and (constant > alpha[j]):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    # 优化了alpha,优化次数+1
                    alpha_change_times += 1
                    print("iter: %d i:%d, pairs changed %d" % (iter_times, i, alpha_change_times))
            # 若优化次数为0，迭代次数+1，否则重置迭代次数。若迭代iter_times后alpha仍无优化，则表示其已收敛
            if alpha_change_times == 0:
                iter_times += 1
            else:
                iter_times = 0
            print("iteration number: %d" % iter_times)
        return alpha, b

    def calcW(self, alphas, sample_data, labels):
        """求解W"""

        X = mat(sample_data)
        labelMat = mat(labels).transpose()
        m, n = np.shape(X)
        w = zeros((n, 1))
        for i in range(m):
            w += multiply(alphas[i] * labelMat[i], X[i, :].T)
        return w

    def plotfig_SVM(self, xMat, yMat, ws, b, alphas):
        """
        参考地址：
           http://blog.csdn.net/maoersong/article/details/24315633
           http://www.cnblogs.com/JustForCS/p/5283489.html
           http://blog.csdn.net/kkxgx/article/details/6951959
        """

        xMat = mat(xMat)
        yMat = mat(yMat)

        # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
        b = array(b)[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # 注意flatten的用法
        ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

        # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
        x = arange(-1.0, 10.0, 0.1)

        # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
        y = (-b - ws[0, 0] * x) / ws[1, 0]
        ax.plot(x, y)

        for i in range(shape(yMat[0, :])[1]):
            if yMat[0, i] > 0:
                ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
            else:
                ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

        # 找到支持向量，并在图中标红
        for i in range(len(alpha)):
            if alphas[i] > 0.0:
                ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
        plt.show()


if __name__ == "__main__":
    sample_data = ''
    labels = ''
    svmClassfier = SVMClassifier()
    sample_data, labels = svmClassfier.readTxt('svm_simple_test.txt')
    alpha, b = svmClassfier.smoAlgorithm(sample_data=sample_data, labels=labels, constant=0.01, toler=0.001, max_iter=500)
    w = svmClassfier.calcW(alpha, sample_data, labels)
    svmClassfier.plotfig_SVM(sample_data, labels, w, b, alpha)