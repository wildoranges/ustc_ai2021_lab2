import numpy as np
import matplotlib.pyplot as plt
import random
from time import strftime


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z):
    return np.multiply(z, 1.0 - z)


class MLP(object):
    def __init__(self, lr=0.1, epsilon=1e-3, epoch=10000, size=None):
        if size is None:
            size = [5, 4, 4, 3]
        self.lr = lr
        self.epsilon = epsilon
        self.epochs = epoch
        self.size = size
        self.W = []
        self.b = []
        self.init()

    def init(self):
        for i in range(len(self.size) - 1):
            self.W.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], self.size[i]))))
            self.b.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], 1))))

    def FP(self, item):
        a = [item]
        for index in range(len(self.W)):
            a.append(sigmoid(self.W[index] * a[-1] + self.b[index]))
        return a

    def BP(self, label, a):
        delta = []
        label = label.reshape(a[-1].shape)
        one = np.ones(shape=a[-1].shape, dtype=float)
        out_delta = np.multiply((a[-1] - label), np.multiply(a[-1], (one - a[-1])))
        delta.append(out_delta)

        for i in range(len(self.W) - 1):
            derivative = dsigmoid(a[-2 - i])
            cur_delta = np.multiply(self.W[-i - 1].T * delta[-1], derivative)
            delta.append(cur_delta)

        for i in range(len(delta)):
            self.W[-i - 1] = self.W[-i - 1] - self.lr * (delta[i] * a[-2 - i].T)
            self.b[-i - 1] = self.b[-i - 1] - self.lr * delta[i]

        ln = np.log(a[-1])
        loss = np.dot(label.T, ln)
        return loss

    def fit(self, train_data, tran_label, show=False):
        plt.ion()
        plt.figure(1)
        plt.xlabel('epochs')
        plt.ylabel('average loss')
        plt.title('loss-epochs')
        for i in range(self.epochs):
            all_loss = []
            for index in range(train_data.shape[0]):
                tmp = self.FP(train_data[index].T)
                cur_loss = self.BP(tran_label[index], tmp)
                all_loss.append(float(cur_loss))
            avg_loss = sum(all_loss) / len(all_loss)
            plt.scatter(i + 1, abs(avg_loss))
            plt.draw()

            if show:
                print("loop {},loss {}".format(i + 1, avg_loss))

            if abs(avg_loss) < abs(self.epsilon):
                print("finish train at loop{}".format(i + 1))
                plt.show()
                plt.savefig("./MLP-{}.png".format(strftime("%Y-%m-%d %H:%M:%S")))
                return
        plt.show()
        plt.savefig("./MLP-{}.png".format(strftime("%Y-%m-%d %H:%M:%S")))
        return

    def predict(self, test_data):
        out = np.zeros(shape=(test_data.shape[0], self.size[-1]))
        for index in range(test_data.shape[0]):
            res = self.FP(test_data[index])
            out[index] = res[-1]
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

    train_label.reshape((train_size, label_dim))
    mlp = MLP()
    mlp.fit(train_data, train_label, show=True)


if __name__ == '__main__':
    main()
