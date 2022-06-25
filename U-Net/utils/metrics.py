import torch


def dice_coeff(pred, label):
    smooth = 1e-8
    m1 = pred.view(-1)
    m2 = label.view(-1)
    intersection = (m1 * m2)
    dice = 2. * (torch.sum(intersection) + smooth) / (torch.sum(m1) + torch.sum(m2) + smooth)
    return dice