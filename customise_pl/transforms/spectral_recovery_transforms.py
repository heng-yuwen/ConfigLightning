import pytorch_lightning as pl
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import numpy as np


class SpectralNorm(pl.LightningModule):
    def __init__(self):
        super(SpectralNorm, self).__init__()

    def forward(self, rgb, spectral):
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb = np.transpose(rgb, [2, 0, 1])  # [3,482,512]
        spectral = np.transpose(spectral, [0, 2, 1])

        return rgb, spectral


class SpectralRotateFlip(pl.LightningModule):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, rgb, spectral):
        for j in range(random.randint(0, 3)):
            rgb = np.rot90(rgb.copy(), axes=(1, 2))
            spectral = np.rot90(spectral.copy(), axes=(1, 2))
        # Random vertical Flip
        if random.random() > self.p:
            rgb = rgb[:, :, ::-1].copy()
            spectral = spectral[:, :, ::-1].copy()
        # Random horizontal Flip
        if random.random() > self.p:
            rgb = rgb[:, ::-1, :].copy()
            spectral = spectral[:, ::-1, :].copy()

        return rgb, spectral


class SpectralRandomCrop(pl.LightningModule):
    """Use this after the SpectralNorm class forward"""
    def __init__(self, patch_size):
        super().__init__(self)
        self.patch_size = patch_size

    def forward(self, rgb, spectral):
        i, j, h, w = transforms.RandomCrop.get_parameter(rgb, self.size)
        rgb = rgb[:, i:i + h, j: j+w]
        spectral = spectral[:, i:i + h, j: j+w]

        return rgb, spectral
