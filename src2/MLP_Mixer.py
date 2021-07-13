import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

# 禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Mixer_Layer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(Mixer_Layer, self).__init__()
        ########################################################################
        # 这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear((28 // patch_size) ** 2, (28 // patch_size) ** 2),
            nn.GELU(),
            nn.Linear((28 // patch_size) ** 2, (28 // patch_size) ** 2),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        ########################################################################

    def forward(self, x):
        ########################################################################
        x = self.norm1(x)
        mixed_token = self.mlp1(x.transpose(1, 2))
        x = x + mixed_token.transpose(1, 2)
        ########################################################################
        x2 = self.norm2(x)
        mixed_channel = self.mlp2(x2)
        return x + mixed_channel


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
            nn.Linear(patch_size ** 2, hidden_dim),
            *[Mixer_Layer(patch_size=patch_size, hidden_dim=hidden_dim) for _ in range(depth)],
            nn.LayerNorm(hidden_dim),
        )
        self.out_model = nn.Linear(hidden_dim, 10)

        ########################################################################

    def forward(self, data):
        rows = data.split(self.patch_size, dim=2)
        columns = [row.split(self.patch_size, dim=3) for row in rows]
        patch_data = [torch.stack(column, dim=2) for column in columns]
        input_data = torch.cat(patch_data, dim=2)
        input_data = input_data.reshape(
            (input_data.shape[0], input_data.shape[1], input_data.shape[2], input_data.shape[3] * input_data.shape[4]))
        out = self.mix_model(input_data)
        out = torch.mean(out, dim=1)
        out = self.out_model(out)
        return out
        ########################################################################
        # 注意维度的变化

        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            #data, target = data.to(device), target.to(device)
            out_label = model(data)
            loss = criterion(out_label, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ########################################################################
            # 计算loss并进行优化

            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.
    num_correct = 0  # correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

        ########################################################################
        # 需要计算测试集的loss和accuracy

        ########################################################################
        #print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss.item(), accuracy))


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
    model = MLPMixer(patch_size=4, hidden_dim=4, depth=4).to(device)  # 参数自己设定，其中depth必须大于1
    # 这里需要调用optimizer，criterion(交叉熵)
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.8)


    ########################################################################

    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)
