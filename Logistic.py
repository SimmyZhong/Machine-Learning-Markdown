from numpy import mat, exp
import numpy as np
import matplotlib.pyplot as plt
import random


class LogisticSigmoidClassifier(object):
    """logistic回归分类器"""

    def read_txt(self, file):
        """读取数据"""

        sample_data = []
        labels = []
        file = open(file)
        for line in file.readlines():
            data = line.strip().split('\t')
            sample_data.append([1.0, float(data[0]), float(data[1])])  # 读取样本数据集，设置常数项为1.0
            labels.append(int(data[2]))  # 读取样本数据集标签
        file.close()
        return sample_data, labels

    def sigmoid(self, data_set):
        """sigmoid函数"""

        return 1.0 / (1 + exp(-data_set))

    def gradAscent(self, sample_data, labels):
        """梯度上升优化算法得出最优权重系数"""

        traning_data = mat(sample_data)  # 转化为矩阵
        num_data, num_feature = np.shape(sample_data)  # 获取数据集样本数和特征数
        traning_labels = mat(labels).transpose()  # 转置
        weights = np.ones((num_feature, 1))  # 初始化权重系数
        step = 0.001  # 步长
        for i in range(500):
            z = traning_data * weights  # 对权重系数及样本集进行矩阵运算
            distance = traning_labels - self.sigmoid(z)
            weights = weights + traning_data.transpose() * distance * step  # 梯度上升算法的迭代公式：参考机器学习实战P76【Logistic回归】
        return weights.getA()[:, 0]

    def stocGradAscent0(self, sample_data, labels):
        """训练算法：随机梯度上升"""
        # 梯度上升算法每次更新回归系数时需要遍历整个数据集，当数据源很大时，计算复杂度太高

        traning_data = mat(sample_data)
        num_data, num_figture = np.shape(traning_data)
        step = 0.1
        weights = np.ones((num_figture))
        for i in range(num_data):
            z = sum(weights * traning_data[i])
            distance = labels[i] - self.sigmoid(z)
            weights = weights + step * distance * traning_data[i]
        return weights

    def stocGradAscent1(self, sample_data, labels, num_iter=150):
        """
        改进的随机梯度上升算法
        :param sample_data: 数据集
        :param labels: 数据集分类
        :param num_iter: 迭代次数
        :return: 权重系数
        """
        traning_data = np.array(sample_data)
        num_data, num_figure = np.shape(traning_data)
        weights = np.ones(num_figure)
        for i in range(num_iter):
            for j in range(num_data):
                index = list(range(0, num_data))
                step = 4 / (1.0 + i + j) + 0.01
                data_index = random.choice(index)
                z = sum(weights * traning_data[data_index])
                distance = labels[data_index] - self.sigmoid(z)
                weights = weights + step * distance * traning_data[data_index]
                index.pop(data_index)
        return weights

    def classify(self, data, weights):
        """分类函数"""

        z = self.sigmoid(sum(weights * data))
        if z > 0.5:
            return 1.0
        else:
            return 0.0

    def classifierTest(self, algorithm, sample_data, sample_labels, test_data, test_labels, times=10):
        """测试分类器的错误率"""

        num_data = len(test_data)
        rate = 0
        for j in range(times):
            weights = algorithm(sample_data, sample_labels)
            for i in range(num_data):
                if self.classify(test_data[i], weights) == test_labels[i]:
                    rate += 1
        rate = float((times * num_data - rate) / num_data / times)
        msg = '测试样本数据集共{num}个，经过{times}次测试，错误率为{rate}'.format(num=num_data, rate=rate,times=times)
        print(msg)


def plotShowFit(sample_data, labels, weights):
    """利用matplotlib展示"""

    x1, y1 = [], []
    x2, y2 = [], []
    for i in range(len(sample_data)):
        if int(labels[i]) == 1:
            x1.append(sample_data[i][1]) #取出类别为1的数据集的特征1
            y1.append(sample_data[i][2]) #取出类别为1的数据集的特征2
        else:
            x2.append(sample_data[i][1])
            y2.append(sample_data[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # weights = weights.getA()
    # 此处设定Sigmoid函数的值为0.因为0是两个分类（0和1）的分界处。因此设定0 = W0X0 + W1X1 + W2X2
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    logisticClassify = LogisticSigmoidClassifier()

    # 读取测试数据集和标签
    sample_data, sample_labels = logisticClassify.read_txt('sample_data/logistic_data/logistic_test.txt')

    # 梯度上升优化算法得出最优权重系数
    weights_1 = logisticClassify.gradAscent(sample_data, sample_labels)

    # 改进的随机梯度上升算法
    weights_2 = logisticClassify.stocGradAscent1(sample_data, sample_labels)
    # 可视化
    plotShowFit(sample_data, sample_labels, weights_2)

    # 获取测试样本
    test_data, test_labels = logisticClassify.read_txt('sample_data/logistic_data/logistic_test.txt')

    # 测试算法正确率
    logisticClassify.classifierTest(
        algorithm=logisticClassify.gradAscent,
        sample_data=sample_data,
        sample_labels=sample_labels,
        test_data=test_data,
        test_labels=test_labels,
        times=10)