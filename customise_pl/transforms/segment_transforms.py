import pytorch_lightning as pl
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import numpy as np


class SegmentRandomHorizontalFlip(pl.LightningModule):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image, mask):
        if random.random() > self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)

        return image, mask


class SegmentRandomVerticalFlip(pl.LightningModule):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image, mask):
        if random.random() > self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)

        return image, mask


class SegmentCenterCrop(pl.LightningModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, image, mask):
        image = F.center_crop(image, self.size)
        mask = F.center_crop(mask, self.size)

        return image, mask


class SegmentRandomCrop(pl.LightningModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, image, mask):
        i, j, h, w = transforms.RandomCrop.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return image, mask


class SegmentToTensor(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, image, mask):
        default_float_dtype = torch.get_default_dtype()
        mask = torch.from_numpy(np.array(mask, np.int16, copy=True))
        if isinstance(mask, torch.ByteTensor):
            if default_float_dtype == "torch.float32":
                mask = mask.to(dtype="torch.int32")
            elif default_float_dtype == "torch.float16":
                mask = mask.to(dtype="torch.int8")
        return image, mask


class SegmentResize(pl.LightningModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, image, mask):
        image = F.resize(image, self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)

        return image, mask
