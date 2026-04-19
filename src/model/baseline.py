import timm

import torch
import torch.nn as nn


class RGBModel(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = "resnet"):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = self._init_backbone(backbone)

    def forward(self, frame: torch.Tensor, **batch):
        output = self.backbone(frame)
        return {
            "pred": output
        }

    def _init_backbone(self, backbone: str):
        if backbone == "resnet":
            return timm.create_model("resnet50", pretrained=True, num_classes=self.num_classes)
        elif backbone == "mobilenet":
            return timm.create_model("mobilenetv2_100", pretrained=True, num_classes=self.num_classes)
        elif backbone == "vit":
            return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown backbone {backbone}")
