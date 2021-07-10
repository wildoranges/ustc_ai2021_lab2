import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc


class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''

    def __init__(self):
        self.Pc = {}
        self.Pxc = {}
        self.Labels = set()

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def fit(self, traindata, trainlabel, featuretype):
        '''
        需要你实现的部分
        '''
        size = len(traindata)
        self.size = size
        dim = len(featuretype)
        self.dim = dim
        total = {}
        avg = {}
        variance = {}
        cnt = {}
        single_kind = {}
        for i in range(size):
            cur_data = traindata[i]
            cur_label = trainlabel[i]
            self.Labels.add(cur_label[0])
            try:
                single_kind[cur_label[0]] += 1
            except KeyError:
                single_kind[cur_label[0]] = 1
            for j in range(dim):
                if featuretype[j] == 0:
                    label_map = None
                    index_map = None
                    x_sum = None
                    try:
                        label_map = total[cur_label[0]]
                    except KeyError:
                        total[cur_label[0]] = {}
                        total[cur_label[0]][j] = {}
                        total[cur_label[0]][j][cur_data[j]] = 1
                        continue
                    try:
                        index_map = label_map[j]
                    except KeyError:
                        total[cur_label[0]][j] = {}
                        total[cur_label[0]][j][cur_data[j]] = 1
                        continue
                    try:
                        x_sum = index_map[cur_data[j]]
                    except KeyError:
                        total[cur_label[0]][j][cur_data[j]] = 1
                        continue
                    total[cur_label[0]][j][cur_data[j]] = x_sum + 1

                else:
                    cnt_label = None
                    avg_label = None
                    cnt_index = None
                    avg_index = None
                    try:
                        avg_label = avg[cur_label[0]]
                        cnt_label = cnt[cur_label[0]]
                    except KeyError:
                        avg[cur_label[0]] = {}
                        avg[cur_label[0]][j] = cur_data[j]
                        cnt[cur_label[0]] = {}
                        cnt[cur_label[0]][j] = 1
                        variance[cur_label[0]] = {}
                        variance[cur_label[0]][j] = 0
                        continue
                    try:
                        avg_index = avg_label[j]
                        cnt_index = cnt_label[j]
                    except KeyError:
                        avg[cur_label[0]][j] = cur_data[j]
                        cnt[cur_label[0]][j] = 1
                        variance[cur_label[0]][j] = 0
                        continue
                    avg[cur_label[0]][j] = avg_index + cur_data[j]
                    cnt[cur_label[0]][j] = cnt_index + 1

        for k, v in avg.items():
            for index, s in v.items():
                avg[k][index] = s / cnt[k][index]

        for i in range(size):
            cur_data = traindata[i]
            cur_label = trainlabel[i]
            for j in range(dim):
                if featuretype[j] == 1:
                    try:
                        average = avg[cur_label[0]][j]
                        variance[cur_label[0]][j] += (cur_data[j] - average) ** 2
                    except KeyError:
                        average = avg[cur_label[0]][j]
                        variance[cur_label[0]][j] = (cur_data[j] - average) ** 2

        for label in self.Labels:
            self.Pxc[label] = {}
            self.Pc[label] = (single_kind[label] + 1) / (size + len(self.Labels))
            for i in range(dim):
                if featuretype[i] == 0:
                    self.Pxc[label][i] = {}

        for k, v in variance.items():
            for index, s in v.items():
                variance[k][index] = s / cnt[k][index]
                self.Pxc[k][index] = (avg[k][index], variance[k][index])

        for label, v in total.items():
            for index, all_x in v.items():
                for x, s in all_x.items():
                    self.Pxc[label][index][x] = (s + 1) / (single_kind[label] + len(all_x))

        return

    def gauss_prob(self, avg, var, x):
        sqrt2pi = 2.5066282746310002  # sqrt(2 * pi)
        coefficient = 1 / (sqrt2pi * math.sqrt(var))
        prob = coefficient * math.exp(-(((x - avg) ** 2) / (2 * var)))
        return prob

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        '''
        需要你实现的部分
        '''
        test_len = len(features)
        out = np.empty(shape=(test_len, 1), dtype=int)
        for index in range(test_len):
            cur_data = features[index]
            maxP = 0.0
            predict_label = 0
            for label in self.Labels:
                prob = 1.0
                for i in range(self.dim):
                    if featuretype[i] == 0:
                        prob = prob * self.Pxc[label][i][cur_data[i]]
                    else:
                        avg, var = self.Pxc[label][i]
                        prob = prob * self.gauss_prob(avg, var, cur_data[i])
                if prob >= maxP:
                    predict_label = label
                    maxP = prob
            out[index] = predict_label
            print("test data {},prob={},label={}".format(index, maxP, predict_label))
        return out


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    feature_type = [0, 1, 1, 1, 1, 1, 1, 1]  # 表示特征的数据类型，0表示离散型，1表示连续型

    Nayes = NaiveBayes()
    Nayes.fit(train_data, train_label, feature_type)  # 在训练集上计算先验概率和条件概率

    pred = Nayes.predict(test_data, feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
