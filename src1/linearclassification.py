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

    def targetmul(self, b, cnt):#计算w*x +b,cnt代表第几类(cnt取值应为0,1,2，因为实验数据共三类，需要三个线性分类器)
        sum = 0.0
        lena = self.dim
        lenb = len(b)
        if lena != lenb:
            raise TypeError("imcompatable length")
        else:
            for i in range(self.dim):
                sum += self.param[cnt][i]*b[i]
            sum += self.param[cnt][self.dim]
        return sum

    def Dloss(self, cnt):#求梯度，cnt含义同上
        gradient = []
        for index in range(self.dim+1):
            loss = 0.0
            if index == self.dim:#b的梯度
                for i in range(self.size):
                    if int(self.train_labels[i][0]) == cnt + 1:#手动将label变为1或0
                        label = 1.0
                    else:
                        label = 0.0
                    loss += (label - self.targetmul(self.train_features[i], cnt))#样本梯度求和
                loss -= self.Lambda * self.param[cnt][index]#L2规范化

            elif self.dim > index >= 0:
                for i in range(self.size):#w的梯度
                    if int(self.train_labels[i][0]) == cnt + 1:#手动将label变为1或0
                        label = 1.0
                    else:
                        label = 0.0
                    loss += (label - self.targetmul(self.train_features[i], cnt)) * self.train_features[i][index]#样本梯度求和
                loss -= self.Lambda * self.param[cnt][index]#L2规范化

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
        
        for k in range(3):#要分三类，需要训练三个分类器
            param = [] 

            for i in range(self.dim+1):
                param.append(rd.random())#初值随机

            self.param.append(param)

            for i in range(self.epochs):
                grad = self.Dloss(k)#求梯度
                ts = torch.tensor(grad)#转成tensor，方便求范数用于归一化
                ts = ts / ts.norm(p=2)#梯度归一化
                #print("loop{}:\nbefore:{}".format(i, str(self.param[k])))
                for j in range(self.dim+1):
                    self.param[k][j] = self.param[k][j] + self.lr * ts[j].item()#梯度下降
                #print("after :{}".format(str(self.param[k])))

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
            pred = []
            for j in range(3):
                pred_raw = (self.targetmul(test_features[i], j))
                pred.append(pred_raw)
            pred_final = pred.index(max(pred)) + 1#取三个分类器中结果最大的作为预测label
            #print("test_case[{}]:res:{},pred:{}".format(i, str(pred), pred_final))
            out[i] = pred_final
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
