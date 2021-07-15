import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

# 禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Mixer_Layer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(Mixer_Layer, self).__init__()
        ########################################################################
        # 这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(#第一个mlp，在patch层mix
            nn.Linear((28 // patch_size) ** 2, (28 // patch_size) ** 2),
            nn.GELU(),
            nn.Linear((28 // patch_size) ** 2, (28 // patch_size) ** 2),
        )
        self.mlp2 = nn.Sequential(#第二个mlp,在channel层mix
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        ########################################################################

    def forward(self, x):
        ########################################################################
        x1 = self.norm1(x)
        mixed_token = self.mlp1(x1.transpose(-2, -1))
        x = x + mixed_token.transpose(-1, -2)#实现skip_connection
        ########################################################################
        x2 = self.norm2(x)
        mixed_channel = self.mlp2(x2)
        return x + mixed_channel#skip_connection


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        # 这里写Pre-patch Fully-connected, Global average pooling, fully connected
        self.patch_num = (28 // patch_size) ** 2
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.mix_model = nn.Sequential(
            nn.Linear(patch_size ** 2, hidden_dim),#Pre-patch Fully-connected
            *[Mixer_Layer(patch_size=patch_size, hidden_dim=hidden_dim) for _ in range(depth)],#mix layer
            nn.LayerNorm(hidden_dim),
        )
        #Global Average Pooling在下面的forward中
        self.out_model = nn.Linear(hidden_dim, 10)#Fully-connected

        ########################################################################

    def forward(self, data):
        rows = data.split(self.patch_size, dim=2)
        columns = [row.split(self.patch_size, dim=3) for row in rows]
        patch_data = [torch.stack(column, dim=2) for column in columns]
        input_data = torch.cat(patch_data, dim=2)
        input_data = input_data.reshape(
            (input_data.shape[0], input_data.shape[1], input_data.shape[2], input_data.shape[3] * input_data.shape[4]))
        '''
        以上为将数据分成patches
        '''
        out = self.mix_model(input_data)
        out = torch.mean(out, dim=-2)#均值,Global Average Pooling
        out = self.out_model(out)
        #out = torch.softmax(out, dim=-1)
        return out
        ########################################################################
        # 注意维度的变化

        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            out_label = model(data)#数据的分patch在forward时处理，不在这处理
            out_label = out_label.reshape(shape=(out_label.shape[0], 10))
            loss = criterion(out_label, target)#计算loss
            optimizer.zero_grad()#梯度清零
            loss.backward()
            optimizer.step()#train

            ########################################################################
            # 计算loss并进行优化

            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    num_correct = 0  # correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out_label = model(data)
            out_label = out_label.reshape(shape=(out_label.shape[0], 10))
            cur_loss = criterion(out_label, target)
            test_loss += cur_loss
            for i in range(len(out_label)):
                cur_batch = out_label[i]
                cur_batch = cur_batch.reshape(shape=(cur_batch.shape[-1],))
                pred_target = cur_batch.argmax()
                if int(target[i]) == (pred_target):#计算正确个数
                    num_correct += 1
                
        test_loss = test_loss / (len(test_loader))
        accuracy = num_correct / (len(test_loader.dataset))
        ########################################################################
        # 需要计算测试集的loss和accuracy

        ########################################################################
        print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss.item(), accuracy))


if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 1e-3

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    ########################################################################
    model = MLPMixer(patch_size=4, hidden_dim=100, depth=5).to(device)  # 参数自己设定，其中depth必须大于1
    # 这里需要调用optimizer，criterion(交叉熵)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate)


    ########################################################################

    train(model, train_loader, optimizer2, n_epochs, criterion)
    test(model, test_loader, criterion)
