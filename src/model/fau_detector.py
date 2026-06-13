import torch
import torch.nn as nn
import torchvision


class FAUDetector(nn.Module):
    """
    FAU detection model based on ResNet-18 
    from LibreFace framework (https://github.com/ihp-lab/LibreFace).
    """
    def __init__(
		self, 
		out_aus=12, 
		cls_dim=512, 
        cls_reduction=4,
    	cls_dropout=0.1,
        pretrained="models/fau/resnet.pt"
	):
        super().__init__()
        resnet18 = torchvision.models.resnet18(
			weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
		)
        resnet18_layers = list(resnet18.children())[:-1]
        self.encoder = nn.Sequential(*resnet18_layers)

        self.classifier = nn.Sequential(
            nn.Linear(cls_dim, cls_dim // cls_reduction),
            nn.ReLU(),
            nn.BatchNorm1d(cls_dim // cls_reduction),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_dim // cls_reduction, out_aus),
            nn.Sigmoid()
        )

        if pretrained:
            self.load_pretrained(pretrained)
   
    def forward(self, frames: torch.Tensor, **batch):
        """
        Args:
            frames (torch.Tensor): batch of frames of size [B, C, H, W]
        """
        batch_size = frames.shape[0]
        features = self.encoder(frames).reshape(batch_size, -1)
        labels = self.classifier(features)
        return labels

    def load_pretrained(self, pretrained_path: str):
        state_dict = torch.load(pretrained_path, map_location="cpu")
        self.load_state_dict(state_dict["model"])
