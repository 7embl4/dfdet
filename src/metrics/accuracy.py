import torch

from src.metrics import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, pred: torch.Tensor, target: torch.Tensor, **batch):
        correct = torch.sum(torch.argmax(pred.detach().cpu(), dim=1) == target.detach().cpu())
        total = len(target)
        return correct / total
