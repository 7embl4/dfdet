import torch
import numpy as np
from sklearn.metrics import f1_score

from src.metrics import BaseMetric


class F1(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_probs = []
        self.all_targets = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, **batch):
        self.all_probs.extend(pred.detach().cpu().numpy())
        self.all_targets.extend(target.cpu().numpy())

    def result(self):
        all_preds = (np.array(self.all_probs) > 0.5).astype(int)
        return f1_score(self.all_targets, all_preds)

    def reset(self):
        self.all_probs = []
        self.all_targets = []
