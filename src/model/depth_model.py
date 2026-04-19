import timm

import torch
import torch.nn as nn



class DepthFake(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.25, backbone: str = "resnet"):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.backbone = self._init_backbone(backbone)
        self._adapt_input_layer(backbone)
        self.classifier = self._build_classifier(backbone)

    def forward(self, frame: torch.Tensor, **batch):
        features = self.backbone(frame)
        return self.classifier(features)

    def _init_backbone(self, backbone: str):
        if backbone == "resnet":
            return timm.create_model("resnet50", pretrained=True, num_classes=self.num_classes)
        elif backbone == "mobilenet":
            return timm.create_model("mobilenetv2_100", pretrained=True, num_classes=self.num_classes)
        elif backbone == "vit":
            return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown backbone {backbone}")
    
    def _adapt_input_layer(self, backbone: str):
        if backbone == "resnet":
            old_conv = self.backbone.conv1
            rgb_w = old_conv.weight
            depth_w = torch.mean(rgb_w, dim=1, keepdim=True)
            new_w = torch.concat((rgb_w, depth_w), dim=1)
            new_conv = nn.Conv2d(
                old_conv.in_channels + 1,
                old_conv.out_channels,
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                bias=old_conv.bias
            )
            new_conv.weight.data = new_w
            self.backbone.conv1 = new_conv
        elif backbone == "mobilenet":
            old_conv = self.backbone.features[0][0]
            rgb_w = old_conv.weight
            depth_w = torch.mean(rgb_w, dim=1, keepdim=True)
            new_w = torch.concat((rgb_w, depth_w), dim=1)
            new_conv = nn.Conv2d(
                old_conv.in_channels + 1,
                old_conv.out_channels,
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                bias=old_conv.bias
            )
            new_conv.weight.data = new_w
            self.backbone.features[0][0] = new_conv
        elif backbone == "xception":
            old_conv = self.backbone.conv1
            rgb_w = old_conv.weight
            depth_w = torch.mean(rgb_w, dim=1, keepdim=True)
            new_w = torch.concat((rgb_w, depth_w), dim=1)
            new_conv = nn.Conv2d(
                old_conv.in_channels + 1,
                old_conv.out_channels,
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                bias=old_conv.bias
            )
            new_conv.weight.data = new_w
            self.backbone.conv1 = new_conv
        else:
            raise ValueError(f"Unknown backbone {backbone}")

    def _build_classifier(self, backbone: str):    
        if backbone == "resnet":
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "mobilenet":
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "xception":
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone {backbone}")

        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features // 2, in_features // 4),
            nn.BatchNorm1d(in_features // 4),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features // 4, self.num_classes),
        )
