import torch
import torch.nn as nn

from depth_anything_v2.dpt import DepthAnythingV2


class DepthExpert(nn.Module):
    depth_estimator_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]}
    }

    def __init__(
        self, 
        depth_encoder="vits",
        hidden_dim=128,
        n_heads=4,
        n_transformer_layers=2,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()
        self.depth_estimator = DepthAnythingV2(**self.depth_estimator_configs[depth_encoder]).pretrained
        self._disable_grad()

        # hidden space projection
        latent_dim = self.depth_estimator.embed_dim
        self.patch_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # patch aggregation: mean + learnable weights
        self.patch_aggregator = nn.Linear(hidden_dim, hidden_dim)
        
        # temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
            enable_nested_tensor=False
        )
        
        # classifier head
        hidden_dim = 3 * hidden_dim  # vel + acc + jerk
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, frames: torch.Tensor, **batch):
        """
        Args:
            frames (torch.Tensor): tensor of frames with size [B, T, C, H, W]
        """
        B, T, C, H, W = frames.shape
        
        # extract depth latents
        frames = frames.reshape(B * T, C, H, W)
        latents = self._get_depth_latents(frames)  # [B*T, N, D]
        latents = latents.reshape(B, T, *latents.shape[1:])  # [B, T, N, D]

        # patch projection 
        latents = self.patch_proj(latents)  # [B, T, N, H]
        
        # aggregate patches 
        latents = self.patch_aggregator(
            latents.mean(dim=2)
        )  # [B, T, H]
        
        # get temporal dynamics
        vel, acc, jerk = self._get_temporal_dynamics(latents)
        
        # temporal transformer 
        min_T = jerk.shape[1]
        vel  = vel[:, :min_T]
        acc  = acc[:, :min_T]
        jerk = jerk[:, :min_T]
        
        vel_enc  = self.temporal_transformer(vel)  # [B, T-3, H]
        acc_enc  = self.temporal_transformer(acc)  # [B, T-3, H]
        jerk_enc = self.temporal_transformer(jerk) # [B, T-3, H]
        
        # temporal aggregation
        vel_vec  = self._aggregate_sequence(vel_enc)  # [B, H]
        acc_vec  = self._aggregate_sequence(acc_enc)  # [B, H]
        jerk_vec = self._aggregate_sequence(jerk_enc) # [B, H]
        
        # classification
        combined = torch.cat([vel_vec, acc_vec, jerk_vec], dim=-1)  # [B, 3*H]
        logits = self.classifier(combined)
        
        return {"pred": logits}

    @torch.no_grad()
    def _get_depth_latents(self, frames: torch.Tensor):
        """
        Get latents from depth extimator

        Args:
            frames (torch.Tensor): tensor of frames with size [B, C, H, W]
        """
        latent = self.depth_estimator.forward_features(frames)
        return latent["x_norm_patchtokens"]

    def _get_temporal_dynamics(self, frames: torch.Tensor):
        """
        Calculates velocity, acceleration and jerk of frames

        Args:
            frames (torch.Tensor): temporal features of size [B, T, H]
        """
        vel = frames[:, 1:] - frames[:, :-1]  # [B, T-1, H]
        acc = vel[:, 1:] - vel[:, :-1]  # [B, T-2, H]
        jerk = acc[:, 1:] - acc[:, :-1]  # [B, T-3, H]
        
        return vel, acc, jerk

    def _aggregate_sequence(self, x: torch.Tensor):
        """
        Temporal sequence aggregation
        
        Args:
            x (torch.Tensor): tensor of size [B, T, H]
        """
        return x.mean(dim=1)

    def _disable_grad(self):
        """
        Disable gradients of depth estimator
        """
        for p in self.depth_estimator.parameters():
            p.requires_grad = False
