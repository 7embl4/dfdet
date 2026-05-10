import torch
import numpy as np
from sklearn.metrics import f1_score, roc_curve

from src.metrics import BaseMetric


class F1(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_probs = []
        self.all_targets = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, **batch):
        self.all_probs.extend(pred.detach().cpu().softmax(dim=1).numpy()[:, 1])
        self.all_targets.extend(target.detach().cpu().numpy())

    def avg(self):
        fpr, tpr, treshholds = roc_curve(self.all_targets, self.all_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_th = treshholds[optimal_idx]
        print(f"    Optimal th: {optimal_th}")
        all_preds = (np.array(self.all_probs) > optimal_th).astype(int)
        return f1_score(self.all_targets, all_preds)

    def reset(self):
        self.all_probs.clear()
        self.all_targets.clear()
