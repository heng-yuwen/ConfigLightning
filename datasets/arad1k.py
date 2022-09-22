import pytorch_lightning as pl


class SpectralRecovery(pl.LightningModule):
    def __init__(self, data_root, train_rgb_folder="Train_RGB", train_spectral_folder="Train_spectral",
                 valid_rgb_folder="Valid_RGB", valid_spectral_folder="Valid_spectral", patch_size=128, batch_size=4,
                 num_workers=8, pin_memory=True, train_rgb_transform=None, train_spectral_transform=None,
                 valid_rgb_transform=None, valid_spectral_transform=None, test_rgb_transform=None,
                 test_valid_transform=None):
        super().__init__()
        self.data_root = data_root
        self.train_rgb_folder = train_rgb_folder
        self.train_spectral_folder = train_spectral_folder
        self.valid_rgb_folder = valid_rgb_folder
        self.valid_spectral_folder = valid_spectral_folder
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_rgb_transform = train_rgb_transform
        self.train_spectral_transform = train_spectral_transform
        self.valid_rgb_transform = valid_rgb_transform
        self.valid_spectral_transform = valid_spectral_transform
        self.test_rgb_transform = test_rgb_transform
        self.test_valid_transform = test_valid_transform
