import os
from glob import glob
import random

import h5py
import numpy as np
import pytorch_lightning as pl
from PIL import Image

from customise_pl.transforms import CommonCompose
from customise_pl.transforms import init_transforms
from torch.utils.data import DataLoader, Dataset


class SpectralRecoveryDataModule(pl.LightningDataModule):
    def __init__(self, data_root, train_rgb_folder="Train_RGB", train_spectral_folder="Train_spectral",
                 valid_rgb_folder="Valid_RGB", valid_spectral_folder="Valid_spectral", test_rgb_folder="Test_RGB",
                 num_workers=8, batch_size=8, pin_memory=True, train_transform=None,
                 valid_transform=None, test_transform=None):
        super().__init__()
        self.test_files = None
        self.valid_files = None
        self.train_files = None
        self.data_root = data_root
        self.train_rgb_folder = train_rgb_folder
        self.train_spectral_folder = train_spectral_folder
        self.valid_rgb_folder = valid_rgb_folder
        self.valid_spectral_folder = valid_spectral_folder
        self.test_rgb_folder = test_rgb_folder
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.train_transform = CommonCompose(init_transforms(train_transform))
        self.test_transform = CommonCompose(init_transforms(test_transform))

        if valid_transform is None:
            self.valid_transform = CommonCompose(init_transforms(test_transform))
        else:
            self.valid_transform = CommonCompose(init_transforms(valid_transform))

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        self.train_files = [os.path.basename(file)[:-4] for file in
                            glob(os.path.join(self.data_root, self.train_rgb_folder, "*.jpg"))]
        random.shuffle(self.train_files)
        self.valid_files = [os.path.basename(file)[:-4] for file in
                            glob(os.path.join(self.data_root, self.valid_rgb_folder, "*.jpg"))]
        random.shuffle(self.valid_files)
        self.test_files = [os.path.basename(file)[:-4] for file in
                           glob(os.path.join(self.data_root, self.test_rgb_folder, "*.jpg"))]
        random.shuffle(self.test_files)

    def train_dataloader(self):
        train_split = ARAD1KDataset(self.train_files, self.data_root, self.train_rgb_folder, self.train_spectral_folder,
                                    image_transform=self.train_transform)
        return DataLoader(train_split, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        val_split = ARAD1KDataset(self.valid_files, self.data_root, self.valid_rgb_folder, self.valid_spectral_folder,
                                  image_transform=self.valid_transform)
        return DataLoader(val_split, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def test_dataloader(self):
        test_split = ARAD1KDataset(self.test_files, self.data_root, self.test_rgb_folder, None,
                                   image_transform=self.test_transform)
        return DataLoader(test_split, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass


class ARAD1KDataset(Dataset):
    def __init__(self, image_files, data_root, rgb_folder="Train_RGB", spectral_folder="Train_spectral", image_transform=None):
        super().__init__()
        self.image_files = image_files
        self.data_root = data_root
        self.rgb_folder = rgb_folder
        self.spectral_folder = spectral_folder
        self.image_transform = image_transform

    def __getitem__(self, index):
        rgb_path = os.path.join(self.data_root, self.rgb_folder, self.image_files[index] + ".jpg")
        if self.spectral_folder is not None:
            spectral_path = os.path.join(self.data_root, self.spectral_folder, self.image_files[index] + ".mat")
            try:
                with h5py.File(spectral_path, 'r') as mat:
                    spectral = np.array(mat['cube'], dtype="float32")
                mat.close()
            except:
                raise EOFError("{} is corrupted".format(spectral_path))
        else:
            spectral = None

        rgb = np.array(Image.open(rgb_path).convert('RGB'), dtype="float32")
        if self.image_transform is not None:
            rgb, spectral = self.image_transform(rgb, spectral)

        if spectral is not None:
            return np.ascontiguousarray(rgb), np.ascontiguousarray(spectral)
        else:
            return np.ascontiguousarray(rgb)

    def __len__(self):
        return len(self.image_files)
