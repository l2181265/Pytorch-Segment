import os
import torch
import torch.nn.functional as F
from utils.dataset import Data_Loader
from torch.utils.data.dataloader import DataLoader
from utils.net import Unet
from torch import optim
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs')


def train_net(data_path):
    # 加载训练集
    dataset = Data_Loader(data_path, resize=[512, 512], normal=True)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # 选择设备，有cuda用cuda，没有就用cpu
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(n_channels=1, n_classes=1)
    net.to(device=device)  # 将网络拷贝到deivce中

    optimizer = optim.Adam(net.parameters(), lr=0.000001)  # 优化器
    criterion = torch.nn.BCELoss()

    best_loss = float('inf')
    epochs = 100
    for epoch in range(epochs):
        net.train()  # 训练模式
        epoch_loss = 0
        step = 0
        for image, label in train_loader:
            step += 1
            image, label = image.to(device=device), label.to(device=device)

            optimizer.zero_grad()
            pred = net(image)
            loss = criterion(F.sigmoid(pred), label)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        writer.add_scalar('loss', epoch_loss / step, epoch + 1)
        print('Epoch %d Loss:' % (epoch + 1), epoch_loss / step)
        # 保存loss值最小的网络参数
        if epoch_loss / step < best_loss:
            best_loss = epoch_loss / step
            torch.save(net.state_dict(), os.path.join("models", 'best_model.pth'))


if __name__ == "__main__":
    data_path = "./data/train/"
    train_net(data_path)

