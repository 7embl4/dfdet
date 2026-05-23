

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class FaceParser(nn.Module):
    """
    Face parsing model from jonathandinu/face-parsing.
    Wraps SegFormer to produce soft segmentation maps.

    Produces 19-class segmentation:
        0: background, 1: skin, 2: l_brow, 3: r_brow,
        4: l_eye, 5: r_eye, 6: eye_g, 7: l_ear, 8: r_ear,
        9: ear_r, 10: nose, 11: mouth, 12: u_lip, 13: l_lip,
        14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
    """
    MODEL_NAME = "jonathandinu/face-parsing"
    NUM_CLASSES = 19

    def __init__(self):
        super().__init__()
        self.processor = SegformerImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_NAME)

        # Frozen — not trained, used as feature extractor only
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, 3, H, W] — normalized images (ImageNet stats)

        Returns:
            mask: [B, 19, H, W] — soft segmentation map (after softmax)
        """
        B, C, H, W = frames.shape

        # SegFormer expects pixel values in [0, 1] or raw uint8
        # Denormalize from ImageNet normalization first
        mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
        frames_raw = (frames * std + mean).clamp(0, 1)

        # Forward through SegFormer
        # logits shape: [B, 19, H/4, W/4]
        outputs = self.model(pixel_values=frames_raw)
        logits = outputs.logits                          # [B, 19, H', W']

        # Upsample to original resolution
        mask = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )                                                # [B, 19, H, W]

        # Soft mask via softmax — values in [0, 1], sum to 1 per pixel
        mask = mask.softmax(dim=1)

        return mask


class MaskGuidedAttention(nn.Module):
    """
    Mask-Guided Attention module.

    For each of the num_classes segmentation regions:
      1. Mask the feature map with the region mask
      2. Global average pool → class-specific feature vector
    Concatenate all → MLP → sigmoid → channel attention weights.

    Args:
        in_channels (int): number of input feature channels
        num_classes (int): number of segmentation classes
    """
    def __init__(self, in_channels: int, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(in_channels * num_classes, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            mask:     [B, num_classes, H, W]

        Returns:
            attention: [B, C, H, W]
        """
        B, C, H, W = features.shape

        # Resize mask to match feature spatial size if needed
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        # Per-class masked pooling
        masked_pooled = []
        for k in range(self.num_classes):
            mask_k = mask[:, k:k+1, :, :]       # [B, 1, H, W]
            masked = features * mask_k            # [B, C, H, W]
            pooled = masked.mean(dim=(-2, -1))    # [B, C]
            masked_pooled.append(pooled)

        concat = torch.cat(masked_pooled, dim=-1) # [B, C * num_classes]

        attn_weights = self.mlp(concat)           # [B, C]
        attn_map = attn_weights[:, :, None, None] # [B, C, 1, 1]
        attn_map = attn_map.expand_as(features)   # [B, C, H, W]

        return attn_map


class MGAMBlock_B(nn.Module):
    """Config B: attention(input) re-weights resnet output."""
    def __init__(self, resnet_block: nn.Module, in_channels: int, num_classes: int = 19):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention = MaskGuidedAttention(in_channels, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        resnet_out = self.resnet_block(x)
        attn = self.attention(x, mask)
        return resnet_out * attn + resnet_out


class MGAMBlock_C(nn.Module):
    """Config C: attention(input) re-weights input, added to resnet output."""
    def __init__(self, resnet_block: nn.Module, in_channels: int, num_classes: int = 19):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention = MaskGuidedAttention(in_channels, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        resnet_out = self.resnet_block(x)
        attn = self.attention(x, mask)
        return resnet_out + x * attn


class MGAMBlock_D(nn.Module):
    """Config D: attention applied BEFORE resnet block."""
    def __init__(self, resnet_block: nn.Module, in_channels: int, num_classes: int = 19):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention = MaskGuidedAttention(in_channels, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn = self.attention(x, mask)
        return self.resnet_block(x * attn)


class MGAMBlock_E(nn.Module):
    """Config E (best): attention applied AFTER resnet block to refine features."""
    def __init__(self, resnet_block: nn.Module, out_channels: int, num_classes: int = 19):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention = MaskGuidedAttention(out_channels, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        resnet_out = self.resnet_block(x)
        attn = self.attention(resnet_out, mask)
        return resnet_out * attn + resnet_out


class MGAMNET(nn.Module):
    """
    MGAMNET: Mask-Guided Attention Network for Deepfake Detection.

    Integrates face parsing (jonathandinu/face-parsing) directly into
    the forward pass — no need to precompute masks externally.

    Args:
        num_seg_classes (int): segmentation classes (19 for face-parsing)
        config (str): block config — 'a' (baseline) | 'b' | 'c' | 'd' | 'e' (best)
        num_output_classes (int): 2 for binary real/fake
        pretrained (bool): ImageNet pretrained ResNet-18
    """

    _BLOCK_CLS = {'b': MGAMBlock_B, 'c': MGAMBlock_C, 'd': MGAMBlock_D, 'e': MGAMBlock_E}

    # ResNet-18: stem output = 64ch, then layer1-4 outputs: 64, 128, 256, 512
    _IN_CHANNELS  = [64, 64,  128, 256]   # input channels to each layer
    _OUT_CHANNELS = [64, 128, 256, 512]   # output channels of each layer

    def __init__(
        self,
        num_seg_classes: int = 19,
        config: str = 'e',
        num_output_classes: int = 2,
        pretrained: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert config in ('a', 'b', 'c', 'd', 'e'), \
            f"config must be one of 'a','b','c','d','e', got '{config}'"

        self.config = config
        self.num_seg_classes = num_seg_classes

        # Face parser — frozen, produces masks during forward
        self.face_parser = FaceParser()

        # ResNet-18 backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        resnet_layers = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]

        if config == 'a':
            # Vanilla ResNet — no attention
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet_layers
        else:
            block_cls = self._BLOCK_CLS[config]
            # Configs b/c/d use input channels, config e uses output channels
            channels = self._IN_CHANNELS if config in ('b', 'c', 'd') else self._OUT_CHANNELS
            ch_kwarg  = 'in_channels' if config in ('b', 'c', 'd') else 'out_channels'

            self.layer1, self.layer2, self.layer3, self.layer4 = [
                block_cls(layer, **{ch_kwarg: ch}, num_classes=num_seg_classes)
                for layer, ch in zip(resnet_layers, channels)
            ]

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_output_classes)

    def _get_mask(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Get face parsing mask from frames using frozen FaceParser.

        Args:
            frames: [B, 3, H, W] — normalized images

        Returns:
            mask: [B, 19, H, W] — soft segmentation map
        """
        self.face_parser.eval()
        with torch.no_grad():
            mask = self.face_parser(frames)
        return mask

    def forward(self, frames: torch.Tensor, **batch) -> dict:
        """
        Args:
            frames: [B, 3, H, W] — normalized face images (ImageNet stats)

        Returns:
            dict with:
              'pred': [B, num_output_classes] — classification logits
              'mask': [B, 19, H, W] — face parsing mask (for visualization/aux loss)
        """
        # Get segmentation mask from face parser
        mask = self._get_mask(frames)        # [B, 19, H, W]

        # ResNet stem
        x = self.stem(frames)                # [B, 64, H/4, W/4]

        # Layers
        if self.config == 'a':
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            x = self.layer1(x, mask)         # [B, 64,  H/4,  W/4]
            x = self.layer2(x, mask)         # [B, 128, H/8,  W/8]
            x = self.layer3(x, mask)         # [B, 256, H/16, W/16]
            x = self.layer4(x, mask)         # [B, 512, H/32, W/32]

        x = self.gap(x).flatten(1)           # [B, 512]
        pred = self.classifier(x)            # [B, num_output_classes]

        return {"pred": pred, "mask": mask}


if __name__ == "__main__":
    # Sanity check — config e, без загрузки face parser
    B, H, W = 2, 224, 224
    image = torch.randn(B, 3, H, W)
    mask  = torch.randn(B, 19, H, W).softmax(dim=1)

    # Тест только MGAM блоков без face parser
    for cfg in ('a', 'b', 'c', 'd', 'e'):
        resnet = models.resnet18(weights=None)
        stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        if cfg == 'a':
            layers = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            x = stem(image)
            for layer in layers:
                x = layer(x)
        else:
            block_cls = MGAMNET._BLOCK_CLS[cfg]
            ch_key = 'in_channels' if cfg in ('b', 'c', 'd') else 'out_channels'
            channels = MGAMNET._IN_CHANNELS if cfg in ('b', 'c', 'd') else MGAMNET._OUT_CHANNELS
            layers = [
                block_cls(layer, **{ch_key: ch}, num_classes=19)
                for layer, ch in zip(
                    [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4],
                    channels
                )
            ]
            x = stem(image)
            for layer in layers:
                x = layer(x, mask)

        gap = nn.AdaptiveAvgPool2d(1)
        clf = nn.Linear(512, 2)
        out = clf(gap(x).flatten(1))
        print(f"Config {cfg} | output: {out.shape}")

    print("\nAll configs OK.")
    print("\nFull model (with face parser) usage:")
    print("  model = MGAMNET(config='e')")
    print("  out = model(frames)  # frames: [B, 3, 224, 224]")
    print("  pred = out['pred']   # [B, 2]")
    print("  mask = out['mask']   # [B, 19, 224, 224] — для визуализации")

