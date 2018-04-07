import numpy as np
import pandas as pd
from numpy import mat, ones, zeros, multiply
import random


class SVMClassifire(object):
    """SVM分类器实现"""
    
    def __init__(self, sample_data, labels, constant, toler, max_iter):
        """
            args:
            sample_data 样本集
            labels 分类标签
            constant 松弛变量C
            toler 容错率
            max_iter 最大迭代次数
        """

        self.sample_data = mat(sample_data) 
        self.labels = mat(labels) 
        self.constant = constant
        self.toler = toler
        self.max_iter = max_iter 

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
        """SMO算法"""
        
        sample_data = mat(sample_data)
        labels = mat(labels)
        m, n = np.shape(sample_data) 

        alpha = zeros((m, 1)) # 初始化α
        iter_times = 0 # 初始化迭代次数
        b = 0

        while (iter_times < max_iter):
            alpha_change_times = 0
            for i in range(m):
                fx_i = float(multiply(alpha * labels.transpose()) * (sample_data * sample_data[i, :].T)) + b
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
                    fx_j = float(multiply(alpha * labels.transpose()) * (sample_data * sample_data[j, :].T)) + b
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

                    eta = 2.0 * sample_data[i, :] * sample_data[j, :].T - sample_data[i,: ] * sample_data[i,: ].T - \
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