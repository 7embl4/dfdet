import torch

from src.metrics import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, pred: torch.Tensor, target: torch.Tensor, **batch):
        return torch.sum(torch.argmax(pred, dim=1) == target) / len(target)