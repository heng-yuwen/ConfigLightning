import torch


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
            correct = torch.diag(true_confmat).sum().item()
            total = true_confmat.sum().item()

        accuracy = correct / total
        return accuracy
