from torchvision.transforms import *
from .segment_transforms import *


class CommonCompose(Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def init_class(transform):
    class_path = transform["class_path"]
    class_ = eval(class_path)

    if "init_args" in transform:
        init_args = transform["init_args"]
        return class_(**init_args)

    else:
        return class_()


def init_transforms(transforms):
    initialised = [init_class(transform) for transform in transforms]
    return initialised
