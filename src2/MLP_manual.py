import numpy as np
import matplotlib.pyplot as plt
import random
from time import strftime
import torch


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(fz):
    return np.multiply(fz, 1.0 - fz)


def softmax(z):
    return np.exp(z) / sum(np.exp(z))


class MLP(object):
    def __init__(self, lr=0.05, epsilon=1e-3, epoch=10, size=None):
        if size is None:
            size = [5, 4, 4, 3]
        self.lr = lr
        self.epsilon = epsilon
        self.epochs = epoch
        self.size = size
        self.W = []
        self.b = []
        for i in range(len(self.size) - 1):
            self.W.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], self.size[i]))))
            self.b.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], 1))))

    def FP(self, item):
        a = [item]
        for index in range(len(self.W)):
            a.append(sigmoid(self.W[index] * a[-1] + self.b[index]))
        a[-1] = softmax(a[-1])
        return a

    def BP(self, label, a):
        delta = []
        label = label.reshape(a[-1].shape)
        one = np.ones(shape=a[-1].shape, dtype=float)
        out_delta = np.multiply((a[-1] - label), np.multiply(a[-1], (one - a[-1])))
        delta.append(out_delta)
        W_grad = []
        b_grad = []

        for i in range(len(self.W) - 1):
            derivative = dsigmoid(a[-2 - i])
            cur_delta = np.multiply(self.W[-i - 1].T * delta[-1], derivative)
            delta.append(cur_delta)

        for i in range(len(delta)):
            cur_w_grad = (delta[i] * a[-2 - i].T)
            W_grad.insert(0, cur_w_grad)
            cur_b_grad = delta[i]
            b_grad.insert(0, cur_b_grad)
            self.W[-i - 1] = self.W[-i - 1] - self.lr * cur_w_grad
            self.b[-i - 1] = self.b[-i - 1] - self.lr * cur_b_grad

        ln = np.log(a[-1])
        loss = np.dot(label.T, ln)
        return loss, W_grad, b_grad

    def torch_grad(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        # y = y.reshape((y.shape[0], 1))
        a = [x]
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
            res = torch.sigmoid(torch.matmul(W[index], a[-1]) + b[index])
            a.append(res)
        a[-1] = torch.softmax(a[-1], dim=0)
        ln = -torch.log(a[-1])
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
                tmp = self.FP(cur_data.T)
                torch_w_grad, torch_b_grad = self.torch_grad(cur_data.T, cur_label)
                cur_loss, cur_w_grad, cur_b_grad = self.BP(cur_label, tmp)
                all_loss.append(float(cur_loss))
                cos = torch.nn.CosineSimilarity(dim=0)
                if compare:
                    for j in range(len(torch_w_grad)):
                        print(cos(torch.tensor(cur_w_grad[j]).reshape(-1,), torch_w_grad[j].reshape(-1,)))

            avg_loss = sum(all_loss) / len(all_loss)
            plt.scatter(i + 1, abs(avg_loss))
            plt.draw()

            if show:
                print("loop {},loss {}".format(i + 1, abs(avg_loss)))

            if abs(avg_loss) < abs(self.epsilon):
                print("finish train at loop{}".format(i + 1))
                plt.show()
                plt.savefig("./MLP-{}.png".format(strftime("%Y-%m-%d-%H-%M-%S")))
                return
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
    train_data = np.mat(np.random.uniform(-1.0, 1.0, size=(train_size, train_dim)))
    train_label = np.zeros(shape=(train_size, label_dim))
    for index in range(train_data.shape[0]):
        label = random.randint(0, label_dim - 1)
        train_label[index][label] = 1.0

    mlp = MLP()
    mlp.fit(train_data, train_label, show=True, compare=True)


if __name__ == '__main__':
    main()
