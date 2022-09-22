import pytorch_lightning as pl
import segmentation_models_pytorch.losses as losses
import models
import torchmetrics
from customise_pl.metrics import SegmentEvaluator, pretty_print
from customise_pl.schedulers import build_scheduler


class SemanticSegmentor(pl.LightningModule):
    def __init__(self, parameters: dict, optimizer_dict: dict, scheduler_dict: dict):
        super().__init__()
        ignore_index = parameters.pop("ignore_index")
        num_class = parameters["classes"]
        is_sparse = parameters.pop("is_sparse")
        self.automatic_optimization = False
        self.loss = losses.FocalLoss("multiclass", ignore_index=ignore_index)
        self.model = models.get_models(**parameters)
        self.train_confmat = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_class,
                                                                                   ignore_index=ignore_index)
        self.valid_confmat = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_class,
                                                                                   ignore_index=ignore_index)
        self.test_confmat = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_class,
                                                                                  ignore_index=ignore_index)
        self.segment_evaluator = SegmentEvaluator(is_sparse=is_sparse)

        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict

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

        # train
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        # scheduler
        scheduler = self.lr_schedulers()
        if scheduler.stage == 0:
            self.lr_scheduler_step(scheduler, 0, self.trainer.current_epoch)

        return {'loss': loss, 'preds': preds, 'target': target}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def training_step_end(self, outputs):
        step_confmat = self.train_confmat(outputs["preds"], outputs["target"])
        _ = self.segment_evaluator(step_confmat, log_func=self.log, pre_fix="train")
        self.log("train_loss", outputs["loss"], prog_bar=True)
        # scheduler
        scheduler = self.lr_schedulers()
        if scheduler.stage == 1:
            self.lr_scheduler_step(scheduler, 0, self.trainer.current_epoch)

    def on_training_epoch_end(self):
        epoch_confmat = self.train_confmat.compute()
        self.train_confmat.reset()
        _ = self.segment_evaluator(epoch_confmat, log_func=self.log, pre_fix="ep_train")

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        data, target = batch
        preds = self.model(data)
        valid_loss = self.loss(preds, target)
        return {'loss': valid_loss, 'preds': preds, 'target': target}

    def validation_step_end(self, outputs):
        self.valid_confmat(outputs["preds"], outputs["target"])
        # print nothing during evaluation.
        # _ = self.segment_evaluator(step_confmat, log_func=self.log, pre_fix="valid")
        # self.log("valid_loss", outputs["loss"])

    def on_validation_epoch_end(self):
        epoch_confmat = self.valid_confmat.compute()
        self.valid_confmat.reset()
        _ = self.segment_evaluator(epoch_confmat, log_func=self.log, pre_fix="ep_valid")

    def test_step(self, batch, batch_idx):
        # this is the test loop
        data, target = batch
        preds = self.model(data)
        test_loss = self.loss(preds, target)
        return {'loss': test_loss, 'preds': preds, 'target': target}

    def test_step_end(self, outputs):
        self.test_confmat(outputs["preds"], outputs["target"])
        # no need to print during testing
        # accuracy, acc_per_cls, mean_acc, iou_per_cls, miou = self.segment_evaluator(step_confmat, log_func=self.log, pre_fix="test")
        self.log("test_loss", outputs["loss"])
        # self.log("test_acc", accuracy, prog_bar=True)
        # self.log("test_mean_acc", mean_acc, prog_bar=True)
        # self.log("test_miou", miou, prog_bar=True)

    def on_test_epoch_end(self):
        epoch_confmat = self.test_confmat.compute()
        self.test_confmat.reset()
        accuracy, acc_per_cls, mean_acc, iou_per_cls, miou = self.segment_evaluator(epoch_confmat, log_func=None)
        # TODO: prtty print per-class analysis, use a table or something, support latex table format.
        CLASSES = self.trainer.test_dataloaders[0].dataset.CLASSES
        assert len(iou_per_cls) == len(acc_per_cls) >= len(CLASSES), \
            "The number of classes does not match the evaluation, densely nor sparsely"
        if len(acc_per_cls) > len(CLASSES):
            is_sparse = True
        else:
            is_sparse = False
        pretty_print(CLASSES, accuracy, acc_per_cls, mean_acc, iou_per_cls, miou, is_sparse, self.log)

    def configure_optimizers(self):
        from mmcv.runner import build_optimizer
        optimizer = build_optimizer(model=self.model, cfg=self.optimizer_dict)
        scheduler = build_scheduler(optimizer=optimizer, cfg=self.scheduler_dict, num_epochs=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
