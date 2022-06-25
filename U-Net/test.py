import os
import cv2
import torch
from utils.dataset import Data_Loader
from torch.utils.data.dataloader import DataLoader
from utils.net import Unet
from utils.metrics import dice_coeff


def test_net(test_path):
    # 加载训练集
    dataset = Data_Loader(test_path, resize=[512, 512], normal=True)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(n_channels=1, n_classes=1)
    net.to(device=device)  # 将网络拷贝到deivce中

    net.load_state_dict(torch.load(os.path.join("models", 'best_model.pth'), map_location=device))
    net.eval()
    dice = 0
    step = 0
    for image, label in test_loader:
        step += 1
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        pred = net(image)
        pred = torch.sigmoid(pred)
        pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
        cv2.imwrite(os.path.join(test_path, "pred", "pred%2d.bmp" % step), 255. * pred[0][0].cpu().numpy())
        dice += dice_coeff(pred, label)
    dice_avg = dice / step
    print('Average Dice:', dice_avg)


if __name__ == "__main__":
    test_path = "./data/test/"
    test_net(test_path)


