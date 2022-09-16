import os.path

from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob
import torch


class XJ3SegmentDataModule(LightningDataModule):
    def __init__(self, data_root, image_folder="images", mask_folder="masks", batch_size=4, num_workers=8,
                 pin_memory=True, split_portion=(0.7, 0.15, 0.15)):
        super().__init__()
        self.train_files = None
        self.test_files = None
        self.valid_files = None
        self.train_names = None
        self.image_path = None
        self.mask_path = None
        self.data_root = data_root
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split_portion = split_portion

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``, include transform and dataset
        # called on every process in DDP
        self.image_path = os.path.join(self.data_root, self.image_folder)
        self.mask_path = os.path.join(self.data_root, self.mask_folder)
        files_all = glob(os.path.join(self.image_path, "*.jpg"))
        train_files, valid_files, test_files = random_split(files_all, self.split_portion,
                                                            generator=torch.Generator().manual_seed(42))
        self.train_files = [os.path.basename(file)[:-4] for file in train_files]
        self.valid_files = [os.path.basename(file)[:-4] for file in valid_files]
        self.test_files = [os.path.basename(file)[:-4] for file in test_files]

    def train_dataloader(self):
        train_split = XJ3SegmentDataset(self.train_files, self.data_root, self.image_folder, self.mask_folder)
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        val_split = XJ3SegmentDataset(self.valid_files, self.data_root, self.image_folder, self.mask_folder)
        return DataLoader(val_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        test_split = XJ3SegmentDataset(self.test_files, self.data_root, self.image_folder, self.mask_folder)
        return DataLoader(test_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass


class XJ3SegmentDataset(Dataset):
    CLASSES = ["车道线-鱼骨线",
               "车道线-普通车道线",
               "车道线-横向斑马线",
               "车道线-单线停止线",
               "车道线-停车让行线",
               "车道线-虚减速让行线",
               "车道线-分段停止线",
               "路沿-凸起马路牙子",
               "路沿-交通警戒标识",
               "路沿-道路围栏",
               "减速带-凸起减速带",
               "减速带-道路中画线"]

    def __init__(self, image_files, data_root, image_folder="images", mask_folder="masks"):
        super().__init__()
        self.image_files = image_files
        self.data_root = data_root
        self.image_folder = image_folder
        self.mask_folder = mask_folder

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.image_files[index] + ".jpg")
        mask_path = os.path.join(self.data_root, self.image_files[index] + ".png")
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()

    def __len__(self):
        return len(self.image_files)
