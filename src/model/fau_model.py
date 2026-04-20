import torch
import torch.nn as nn

import clip
from feat import Detector


class FAUModel(nn.Module):
    def __init__(
            self, 
            # fau lstm
            input_features=7,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            
            # vision encoder
            encoder_model="ViT-L/14",
            
            # attention fusion
            num_heads=4,
            fusion_dropout=0.1,

            # classification head
            reduction=2,
            cls_dropout=0.3,
            num_classes=2,

            # other
            device="cpu", 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.device = device

        # fau branch
        self.fau_extractor = self._build_fau_extractor()
        self.fau_lstm = nn.LSTM(
            input_size=input_features, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # image branch
        self.image_encoder = self._build_image_encoder(encoder_model)
        self.image_proj = nn.Linear()

        # fusion
        d_model = 2 * hidden_size if bidirectional else hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=fusion_dropout,
            batch_first=True
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.ReLU(),
            nn.Linear(d_model // reduction, num_classes)
        )

    def forward(self, frames: torch.Tensor, **batch):
        pass

    def _build_fau_extractor(self):
        fau_detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            emotion_model="resmasknet",
            device=self.device
        )
        return fau_detector

    def _build_image_encoder(self, encoder_model):
        image_encoder, _ = clip.load(encoder_model)
        for p in image_encoder.parameters():
            p.requires_grad = False

        return image_encoder
