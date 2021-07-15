import numpy as np
import matplotlib.pyplot as plt
import random
from time import strftime
import torch


def sigmoid(z):#计算sigmoid
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(fz):#计算导数，输入的是sigmoid(z)的值
    return np.multiply(fz, 1.0 - fz)


def softmax(z):#最终输出变换到0-1区间
    return np.exp(z) / sum(np.exp(z))


class MLP(object):
    def __init__(self, lr=0.05, epoch=100, size=None):
        if size is None:
            size = [5, 4, 4, 3]
        self.lr = lr
        self.epochs = epoch
        self.size = size
        self.W = []
        self.b = []
        for i in range(len(self.size) - 1):#参数初始化
            self.W.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], self.size[i]))))
            self.b.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], 1))))

    def FP(self, mlp_in):#向前传播
        all_out = [mlp_in]
        for index in range(len(self.W)):
            all_out.append(sigmoid(self.W[index] * all_out[-1] + self.b[index]))
        all_out[-1] = softmax(all_out[-1])
        return all_out

    def BP(self, label, all_out):#反向传播
        delta = []
        label = label.reshape(all_out[-1].shape)
        one = np.ones(shape=all_out[-1].shape, dtype=float)
        out_delta = np.multiply((all_out[-1] - label), np.multiply(all_out[-1], (one - all_out[-1])))#最后一层的delta
        delta.append(out_delta)
        W_grad = []
        b_grad = []

        for i in range(len(self.W) - 1):#反向计算delta
            derivative = dsigmoid(all_out[-2 - i])
            cur_delta = np.multiply(self.W[-i - 1].T * delta[-1], derivative)
            delta.append(cur_delta)

        for i in range(len(delta)):
            cur_w_grad = (delta[i] * all_out[-2 - i].T)#w的梯度
            W_grad.insert(0, cur_w_grad)
            cur_b_grad = delta[i]#b的梯度
            b_grad.insert(0, cur_b_grad)
            self.W[-i - 1] = self.W[-i - 1] - self.lr * cur_w_grad
            self.b[-i - 1] = self.b[-i - 1] - self.lr * cur_b_grad

        ln = np.log(all_out[-1])
        loss = -np.dot(label.T, ln)#交叉熵
        return loss, W_grad, b_grad

    def torch_grad(self, x, y):#使用torch自动计算梯度
        x = torch.tensor(x)
        y = torch.tensor(y)
        # y = y.reshape((y.shape[0], 1))
        all_out = [x]
        W = []
        b = []
        for item in self.W:
            item_tensor = torch.tensor(item)
            W.append(item_tensor)
            W[-1].requires_grad = True
        for item in self.b:
            item_tensor = torch.tensor(item)
            b.append(item_tensor)
            b[-1].requires_grad = True
        for index in range(len(W)):
            res = torch.sigmoid(torch.matmul(W[index], all_out[-1]) + b[index])
            all_out.append(res)
        all_out[-1] = torch.softmax(all_out[-1], dim=0)
        ln = -torch.log(all_out[-1])
        ln = ln.reshape((ln.shape[0]))
        loss = torch.dot(y.T, ln)
        loss.backward()
        W_grad = []
        b_grad = []
        for i in range(len(W)):
            W_grad.append(W[i].grad)
            b_grad.append(b[i].grad)
        return W_grad, b_grad

    def fit(self, train_data, tran_label, show=False, compare=False):
        plt.ion()
        plt.figure(1)
        plt.xlabel('epochs')
        plt.ylabel('average loss')
        plt.title('loss-epochs')
        for i in range(self.epochs):
            all_loss = []
            for index in range(train_data.shape[0]):
                cur_data = train_data[index]
                cur_label = tran_label[index]
                tmp = self.FP(cur_data.T)#进行向前传播，获取各层输出
                torch_w_grad, torch_b_grad = self.torch_grad(cur_data.T, cur_label)
                cur_loss, cur_w_grad, cur_b_grad = self.BP(cur_label, tmp)#反向传播更新参数
                all_loss.append(float(cur_loss))
                cos = torch.nn.CosineSimilarity(dim=0)
                if compare:
                    for j in range(len(torch_w_grad)):#手动梯度与torch自动梯度对比
                        print(cos(torch.tensor(cur_w_grad[j]).reshape(-1,), torch_w_grad[j].reshape(-1,)))

            avg_loss = sum(all_loss) / len(all_loss)
            plt.scatter(i + 1, abs(avg_loss))
            plt.draw()

            if show:
                print("loop {},loss {}".format(i + 1, abs(avg_loss)))

        plt.show()
        plt.savefig("./MLP-{}.png".format(strftime("%Y-%m-%d-%H-%M-%S")))
        return

    def predict(self, test_data):
        out = np.zeros(shape=(test_data.shape[0], self.size[-1]))
        for index in range(test_data.shape[0]):
            res = self.FP(test_data[index])
            out[index] = softmax(res[-1])
        return out


def main():
    train_size = 100
    train_dim = 5
    label_dim = 3
    train_data = np.mat(np.random.uniform(-1.0, 1.0, size=(train_size, train_dim)))#随机生成数据
    train_label = np.zeros(shape=(train_size, label_dim))
    for index in range(train_data.shape[0]):#生成label
        label = random.randint(0, label_dim - 1)
        train_label[index][label] = 1.0

    mlp = MLP()
    mlp.fit(train_data, train_label, show=True, compare=False)


if __name__ == '__main__':
    main()
