import torch


def nanmean(v, *args, inplace=True, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


class SegmentEvaluator:
    def __init__(self, is_sparse=False):
        self.is_sparse = is_sparse

    def __call__(self, confmat):
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

        return accuracy.tolist(), acc_per_cls.tolist(), mean_acc.tolist(), iou_per_cls.tolist(), miou.tolist()
