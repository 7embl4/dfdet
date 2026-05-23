import torch
import torch.nn as nn

import clip
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

        self.backbone, _ = clip.load(backbone)
        self.outputs = {}
        self._register_hooks(["transformer"])

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
        self.fau_model.image_encoder = None
        self.depth_model.classifier = nn.Identity()
        self.depth_model.rgb_backbone = None

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
        self.backbone.eval()
        self.fau_model.eval()
        self.depth_model.eval()

        fau_embeddings, depth_embeddings = self._get_clip_embeddings(frames)
        
        fau_embedding = self._get_fau_embedding(frames, fau_embeddings)
        depth_embedding = self._get_depth_embedding(frames, depth_embeddings)

        total_embedding = torch.concat((fau_embedding, depth_embedding), dim=1)
        out = self.classifier(total_embedding)
        return {"pred": out}

    @torch.no_grad()
    def _get_clip_embeddings(self, frames: torch.Tensor):
        """
        Args: 
            frames (torch.Tensor): size of [B, T, C, H, W]
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fau_embeddings, depth_embeddings = [], []
        for video in frames:
            with torch.autocast(device_type=device, dtype=torch.float16):
                fau_embed = self.backbone.encode_image(video).float()  # [T, D_sem]
            depth_embed = self.outputs["transformer"][1:, :, :].permute(1, 0, 2).float()  # [T, N, D_rgb]

            fau_embeddings.append(fau_embed)
            depth_embeddings.append(depth_embed)
        
        return torch.stack(fau_embeddings), torch.stack(depth_embeddings)

    @torch.no_grad()
    def _get_fau_embedding(self, frames: torch.Tensor, clip_embbedings: torch.Tensor):
        return self.fau_model(frames, clip_embbedings=clip_embbedings)["pred"]
    
    @torch.no_grad()
    def _get_depth_embedding(self, frames: torch.Tensor, clip_embbedings: torch.Tensor):
        return torch.stack([
            torch.mean(self.depth_model(x, clip_embbedings=embed)["pred"], dim=0) 
            for x, embed in zip(frames, clip_embbedings)
        ])

    def _get_hook_fn(self, name: str):
        """
        Get hook for a module

        Args: 
            name (str): name of a module to hook
        """
        def hook_fn(model, input, output):
            self.outputs[name] = output
        return hook_fn

    def _register_hooks(self, names: list[str]):
        """
        Hooks registration for rgb_backbone
        
        Args:
            names (list[str]): list of module names
        """
        target_modules = {}
        for name, module in self.backbone.visual.named_modules():
            if name in names:
                target_modules[name] = module

        for name, module in target_modules.items():
            module.register_forward_hook(self._get_hook_fn(name))

    def _load_models(self, fau_path: str, depth_path: str):
        fau_state_dict = torch.load(fau_path, map_location="cpu", weights_only=False)
        self.fau_model.load_state_dict(fau_state_dict["state_dict"])
        depth_state_dict = torch.load(depth_path, map_location="cpu", weights_only=False)
        self.depth_model.load_state_dict(depth_state_dict["state_dict"])

    def _off_grad(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.fau_model.parameters():
            p.requires_grad = False
        
        for p in self.depth_model.parameters():
            p.requires_grad = False
