import torch
from sklearn.metrics import roc_auc_score

from src.metrics import BaseMetric


class AUC(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_probs = []
        self.all_targets = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, **batch):
        self.all_probs.extend(pred.detach().cpu().numpy())
        self.all_targets.extend(target.detach().cpu().numpy())

    def avg(self):
        return roc_auc_score(self.all_targets, self.all_probs)

    def reset(self):
        self.all_probs = []
        self.all_targets = []
