import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, label):
        pred = torch.argmax(pred, dim=1)
        m1 = pred.view(-1)
        m2 = label.view(-1)
        intersection = (m1 * m2)

        loss = 1 - 2. * (torch.sum(intersection) + self.smooth) / (torch.sum(m1) + torch.sum(m2) + self.smooth)

        return loss

