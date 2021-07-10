from process_data import load_and_process_data
from evaluation import get_macro_F1, get_micro_F1, get_acc
import random as rd
import numpy as np
import torch


# 实现线性回归的类
class LinearClassification:
    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''

    def __init__(self, lr=0.05, Lambda=0.001, epochs=1000):
        self.lr = lr
        self.Lambda = Lambda
        self.epochs = epochs

    '''根据训练数据train_features,train_labels计算梯度更新参数W'''

    def targetmul(self, b):
        sum = 0.0
        lena = self.dim
        lenb = len(b)
        if lena != lenb:
            raise TypeError("imcompatable length")
        else:
            for i in range(self.dim):
                sum += self.param[i]*b[i]
            sum += self.param[self.dim]
        return sum

    def Dloss(self):
        gradient = []
        for index in range(self.dim+1):
            loss = 0.0
            if index == self.dim:
                for i in range(self.size):
                    loss += (self.train_labels[i] - self.targetmul(self.train_features[i]))
                loss -= self.Lambda * self.param[index]

            elif self.dim > index >= 0:
                for i in range(self.size):
                    loss += (self.train_labels[i] - self.targetmul(self.train_features[i])) * self.train_features[i][index]
                loss -= self.Lambda * self.param[index]

            else:
                raise TypeError("imcompatable length")

            gradient.append(loss)

        return gradient

    def save(self, path):
        with open(path, "w+") as f:
            f.write(str(self.param))

    def fit(self, train_features, train_labels):
        ''''
        需要你实现的部分
        '''
        self.train_features = train_features
        self.train_labels = train_labels
        self.size = len(train_features)
        self.dim = len(train_features[0])
        self.param = []
        for i in range(self.dim+1):
            self.param.append(rd.random())

        for i in range(self.epochs):
            grad = self.Dloss()
            ts = torch.tensor(grad)
            ts = ts / ts.norm(p=2)
            print("loop{}:\nbefore:{}".format(i, str(self.param)))
            for j in range(self.dim+1):
                self.param[j] = self.param[j] + self.lr * ts[j].item()
            print("after :{}".format(str(self.param)))

        self.save("./param.txt")

    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''

    def predict(self, test_features):
        ''''
        需要你实现的部分
        '''
        test_len = len(test_features)
        out = np.empty(shape=(test_len, 1), dtype=int)
        for i in range(test_len):
            pred_raw = round(self.targetmul(test_features[i]))
            out[i] = pred_raw

        return out

def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    lR = LinearClassification()
    lR.fit(train_data, train_label)  # 训练模型
    pred = lR.predict(test_data)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
