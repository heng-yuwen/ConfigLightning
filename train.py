import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    LightningCLI(pl.LightningModule, pl.LightningDataModule, subclass_mode_model=True, subclass_mode_data=True)
