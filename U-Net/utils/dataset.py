import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class Data_Loader(Dataset):
    def __init__(self, data_path, resize=None, normal=True):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.resize = resize
        self.normal = normal
        self.image_path = sorted(os.listdir(os.path.join(data_path, "image")))
        self.label_path = sorted(os.listdir(os.path.join(data_path, "label")))
        assert len(self.image_path) == len(self.label_path)

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = os.path.join(self.data_path, "image", self.image_path[index])
        label_path = os.path.join(self.data_path, "label", self.label_path[index])
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        if self.resize:
            image = cv2.resize(image, self.resize)
            label = cv2.resize(label, self.resize, interpolation=cv2.INTER_NEAREST)
        if self.normal:
            image = image / 255.
            if label.max() == 255.:
                label = label / 255.
            # plt.imshow(image)
            # plt.show()
            # plt.imshow(label)
            # plt.show()

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.image_path)



