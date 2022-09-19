import os
import pytorch_lightning as pl
import segmentation_models_pytorch.losses as losses
import models
import torch
import torchmetrics


class SemanticSegmentor(pl.LightningModule):
    def __init__(self, parameters: dict):
        super().__init__()
        ignore_index = parameters.pop("ignore_index")
        num_class = parameters["classes"]
        self.loss = losses.FocalLoss("multiclass", ignore_index=ignore_index)
        self.model = models.get_models(**parameters)
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes=num_class, ignore_index=ignore_index)
        self.valid_confmat = torchmetrics.ConfusionMatrix(num_classes=num_class, ignore_index=ignore_index)
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_class, ignore_index=ignore_index)

    def forward(self, data):
        # in lightning,
        # forward defines the prediction/inference actions
        return self.model(data)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # It is independent of forward
        data, target = batch
        preds = self.model(data)
        loss = self.loss(preds, target)

        return {'loss': loss, 'preds': preds, 'target': target}

    def training_step_end(self, outputs):
        self.train_confmat(outputs["preds"], outputs["target"])
        self.log("train_loss", outputs["loss"])
        # self.log("train_acc", self.train_confmat)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        data, target = batch
        preds = self.model(data)
        valid_loss = self.loss(preds, target)
        return {'loss': valid_loss, 'preds': preds, 'target': target}

    def validation_step_end(self, outputs):
        self.valid_confmat(outputs["preds"], outputs["target"])
        self.log("valid_loss", outputs["loss"])
        # self.log("valid_acc", self.valid_acc)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        data, target = batch
        preds = self.model(data)
        test_loss = self.loss(preds, target)
        return {'loss': test_loss, 'preds': preds, 'target': target}

    def test_step_end(self, outputs):
        self.test_confmat(outputs["preds"], outputs["target"])
        self.log("test_loss", outputs["loss"])
        # self.log("test_acc", self.test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
