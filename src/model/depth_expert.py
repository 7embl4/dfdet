import timm
import torch
import torch.nn as nn

from depth_model.depth_anything_v2.dpt import DepthAnythingV2


class DepthExpert(nn.Module):
    """
    Depth Expert based on https://arxiv.org/pdf/2411.18572
    TL;DR: Attention between rgb and depth features
    """
    depth_estimator_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]}
    }

    def __init__(
        self,
        # encoders
        depth_estimator="vits",
        rgb_backbone="vit_base_patch16_224",
        
        # model params
        hidden_dim=128,
        num_heads=4,
        fusion_dropout=0.1,
        
        # classifier
        cls_dropout=0.3,
        num_classes=2,

        # other
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # depth extimator
        self.depth_estimator = DepthAnythingV2(**self.depth_estimator_configs[depth_estimator]).pretrained

        # rgb backbone
        self.rgb_backbone = timm.create_model(rgb_backbone, pretrained=False)
        self.outputs = {} 
        self._register_hooks(["norm"])
        self._disable_grad()

        # projection of rgb encoder space
        # and depth estimator space to hidden space
        depth_dim = self.depth_estimator.embed_dim
        rgb_dim = self._get_rgb_dim(rgb_backbone)
        self.rgb_proj = self._build_projection_layer(rgb_dim, hidden_dim)
        self.depth_proj = self._build_projection_layer(depth_dim, hidden_dim)

        # spatial fusion 
        self.spatial_fusion = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=num_heads,
            dropout=fusion_dropout,
            batch_first=True
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # patch agregation
        self.patch_attention = nn.Linear(hidden_dim, 1)
        
        # classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cls_dropout),

            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(cls_dropout),

            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, frames: torch.Tensor, **batch):
        """
        Args:
            frames (torch.Tensor): tensor of frames with size [B, C, H, W]
        """
        self.rgb_backbone.eval()
        self.depth_estimator.eval()

        # extract depth features and rgb features
        depth_features = self._get_depth_features(frames)  # [B, N, depth_D]
        rgb_features = self._get_rgb_features(frames)  # [B, N, rgb_D]
        if depth_features.shape[1] != rgb_features.shape[1]:
            depth_features, rgb_features = self._interpolate_features(depth_features, rgb_features)
        
        # patch projection 
        depth_features = self.depth_proj(depth_features)  # [B, N, H]
        rgb_features = self.rgb_proj(rgb_features)  # [B, N, H]
   
        # spatial fusion
        fused_features = rgb_features + self.spatial_fusion(
            query=depth_features,
            key=rgb_features,
            value=rgb_features
        )[0]
        fused_features = rgb_features + self.fusion_mlp(fused_features)  # [B, N, H]

        # patch agregation
        weights = torch.softmax(self.patch_attention(fused_features), dim=1)
        fused_features = torch.sum(weights * fused_features, dim=1)

        # classifier
        out = self.classifier(fused_features)

        return {"pred": out}

    def _interpolate_features(
        self, 
        a_features: torch.Tensor, 
        b_features: torch.Tensor,
        mode: str = "nearest"
    ):
        """
        Interpolate to same number of embeddings

        Args:
            a_features (torch.Tensor): tensor of features
            b_features (torch.Tensor): tensor of features
            mode (str): interpolation mode
        """
        target_dim = min(a_features.shape[1], b_features.shape[1])
        if a_features.shape[1] != target_dim:
            a_features = a_features.permute(0, 2, 1)
            a_features = nn.functional.interpolate(
                a_features, 
                size=target_dim,
                mode=mode
            )
            a_features = a_features.permute(0, 2, 1)

        if b_features.shape[1] != target_dim:
            b_features = b_features.permute(0, 2, 1)
            b_features = nn.functional.interpolate(
                b_features, 
                size=target_dim,
                mode=mode
            )
            b_features = b_features.permute(0, 2, 1)

        return a_features, b_features 

    def _get_rgb_dim(self, rgb_backbone: str):
        """
        Get dimension size of a RGB backbone

        Args:
            rgb_backbone (str): name of a RGB backbone
        """
        num_features = None
        if "vit" in rgb_backbone:
            num_features = self.rgb_backbone.norm.normalized_shape[0]
        else:
            raise NotImplementedError(f"Unknown backbone {rgb_backbone}")

        return num_features

    def _get_hook_fn(self, name: str):
        """
        Get hook for a module

        Args: 
            name (str): name of a module to hook
        """
        def hook_fn(model, input, output):
            self.outputs[name] = input
        return hook_fn

    def _register_hooks(self, names: list[str]):
        """
        Hooks registration for rgb_backbone
        
        Args:
            names (list[str]): list of module names
        """
        target_modules = {}
        for name, module in self.rgb_backbone.named_modules():
            if name in names:
                target_modules[name] = module

        for name, module in target_modules.items():
            module.register_forward_hook(self._get_hook_fn(name))

    def _build_projection_layer(self, source_dim: int, target_dim: int):
        """
        Building a projection layer with norms and activations

        Args: 
            source_dim (int): num of dims of a source space
            target_dim (int): num of dims of a target space
        """
        return nn.Sequential(
            nn.LayerNorm(source_dim),
            nn.Linear(source_dim, source_dim // 2),
            nn.GELU(),

            nn.LayerNorm(source_dim // 2),
            nn.Linear(source_dim // 2, source_dim // 4),
            nn.GELU(),

            nn.LayerNorm(source_dim // 4),
            nn.Linear(source_dim // 4, target_dim)
        )

    @torch.no_grad()
    def _get_depth_features(self, frames: torch.Tensor):
        """
        Get features from depth extimator

        Args:
            frames (torch.Tensor): tensor of frames with size [B, C, H, W]
        """
        latent = self.depth_estimator.forward_features(frames)
        return latent["x_norm_patchtokens"]

    @torch.no_grad()
    def _get_rgb_features(self, frames: torch.Tensor):
        """
        Get features from rgb backbone

        Args:
            frames (torch.Tensor): tensor of frames with size [B, C, H, W]
        """
        self.rgb_backbone(frames)
        return self.outputs["norm"][0][:, 1:, :]

    def _disable_grad(self):
        """
        Disable gradients of backbone models
        """
        for p in self.depth_estimator.parameters():
            p.requires_grad = False

        for p in self.rgb_backbone.parameters():
            p.requires_grad = False
