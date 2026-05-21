import torch
import torch.nn as nn

from src.model import FAUExpert, DepthExpert


class FAUDepthModel(nn.Module):
    def __init__(
        self, 
        backbone,
        fau_path, 
        depth_path, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.fau_model = FAUExpert(encoder_model=backbone)
        self.depth_model = DepthExpert(rgb_backbone=backbone)
        self._load_models(fau_path, depth_path)
        self._off_grad()

        fau_d_model = self.fau_model.temporal_attn.in_features
        depth_d_model = self.depth_model.patch_attention.in_features
        d_model = fau_d_model + depth_d_model
        dropout = self.fau_model.classifier[3].p
        num_classes = self.fau_model.classifier[-1].out_features

        self.fau_model.classifier = nn.Identity()
        self.depth_model.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_model // 4, num_classes),
        )

    def forward(self, frames: torch.Tensor, **batch):
        fau_embedding = self._get_fau_embedding(frames)
        depth_embedding = self._get_depth_embedding(frames)

        total_embedding = torch.concat((fau_embedding, depth_embedding), dim=1)
        out = self.classifier(total_embedding)
        return {"pred": out}

    @torch.no_grad()
    def _get_fau_embedding(self, frames: torch.Tensor):
        return self.fau_model(frames)["pred"]
    
    @torch.no_grad()
    def _get_depth_embedding(self, frames: torch.Tensor):
        return torch.stack([torch.mean(self.depth_model(x)["pred"], dim=0) for x in frames])

    def _load_models(self, fau_path: str, depth_path: str):
        fau_state_dict = torch.load(fau_path, map_location="cpu", weights_only=False)
        self.fau_model.load_state_dict(fau_state_dict["state_dict"])
        depth_state_dict = torch.load(depth_path, map_location="cpu", weights_only=False)
        self.depth_model.load_state_dict(depth_state_dict["state_dict"])

    def _off_grad(self):
        for p in self.fau_model.parameters():
            p.requires_grad = False
        
        for p in self.depth_model.parameters():
            p.requires_grad = False
