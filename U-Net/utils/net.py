import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv2_1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.conv3_1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv4_1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.conv3_3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv2_3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)

        self.outconv = nn.Conv2d(64, self.n_classes, 1)

    def forward(self, x):
        x1 = self.conv1_2(self.conv1_1(x))
        x2 = self.conv2_2(self.conv2_1(x1))
        x3 = self.conv3_2(self.conv3_1(x2))
        x4 = self.conv4_2(self.conv4_1(x3))

        x5 = self.conv3_4(self.conv3_3(torch.cat((x3, self.up1(x4)), 1)))
        x6 = self.conv2_4(self.conv2_3(torch.cat((x2, self.up2(x5)), 1)))
        x7 = self.conv1_4(self.conv1_3(torch.cat((x1, self.up3(x6)), 1)))

        out = self.outconv(x7)
        return out


if __name__ == '__main__':
    net = Unet(n_channels=1, n_classes=2)
    input = torch.rand([1, 1, 512, 512])
    output = net(input)
    print(output.shape)
