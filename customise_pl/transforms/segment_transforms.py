import pytorch_lightning as pl
from torchvision import transforms
import torchvision.transforms.functional as F
import random


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
        i, j, h, w = transforms.RandomCrop.get_parameter(image, self.size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return image, mask
