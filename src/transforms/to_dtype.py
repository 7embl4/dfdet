import torch
import torch.nn as nn
import torchvision.transforms.v2 as T


class ToDtype(nn.Module):
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32
    }

    def __init__(self, dtype: str, scale: bool = True):
        super().__init__()
        self.transform = T.ToDtype(self.dtypes[dtype], scale=scale)
    
    def forward(self, x):
        return self.transform(x)
