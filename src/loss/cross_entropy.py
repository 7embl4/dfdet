import torch
import torch.nn as nn

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, train_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        y = [item["target"] for item in train_dataset._index]
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, **batch):
        return self.criterion(pred, target)
