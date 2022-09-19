from torchvision.transforms import Compose
from .segment_transforms import *


class CommonCompose(Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
