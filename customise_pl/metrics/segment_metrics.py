import torch
import math


def pretty_print(CLASSES, accuracy, acc_per_cls, mean_acc, iou_per_cls, miou, is_sparse=False, log_func=None):
    if log_func is None:
        pass
    num_classes = len(CLASSES)
    for i in range(num_classes):
        log_func(CLASSES[i] + "_acc", acc_per_cls[i])
    for i in range(num_classes):
        log_func(CLASSES[i] + "_iou", iou_per_cls[i])

    log_func("Pixel Acc", accuracy)
    log_func("Mean Acc", mean_acc)
    log_func("mIoU", miou)


def nanmean(v, *args, inplace=True, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


class SegmentEvaluator:
    def __init__(self, is_sparse=False):
        self.is_sparse = is_sparse

    def __call__(self, confmat, log_func, pre_fix="train"):
        if self.is_sparse:
            true_confmat = confmat[:-1, :-1]
        else:
            true_confmat = confmat

        # calculate pixel accuracy
        with torch.no_grad():
            correct = torch.diag(true_confmat).sum()
            total = true_confmat.sum()

            # pixel acc and mean class accuracy
            accuracy = correct / total
            acc_per_cls = (torch.diag(confmat) / confmat.sum(axis=1))
            mean_acc = nanmean(acc_per_cls)

            # iou
            intersection = torch.diag(confmat)
            union = confmat.sum(0) + confmat.sum(1) - intersection
            iou_per_cls = intersection.float() / union.float()
            iou_per_cls[torch.isinf(iou_per_cls)] = float('nan')
            miou = nanmean(iou_per_cls)

        if log_func is not None:
            log_func(pre_fix + "_acc", accuracy.tolist(), prog_bar=True)
            log_func(pre_fix + "_mean_acc", mean_acc.tolist(), prog_bar=True)
            log_func(pre_fix + "_miou", miou.tolist(), prog_bar=True)

        return accuracy.tolist(), acc_per_cls.tolist(), mean_acc.tolist(), iou_per_cls.tolist(), miou.tolist()
