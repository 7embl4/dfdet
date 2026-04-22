import torch
import torch.nn as nn

import clip
from src.model.fau_detector import FAUDetector


class FAUModel(nn.Module):
    """
    Facial Action Units (FAUs) Expert based on HAUTE (https://dl.acm.org/doi/10.1145/3774904.3792919).
    TL;DR: FAUs analysed with BiLSTM and image embeddings from CLIP are fused in multihead attention.
    """
    def __init__(
            self, 
            # fau lstm
            input_features=12,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            
            # vision encoder
            encoder_model="ViT-L/14",
            
            # attention fusion
            num_heads=4,
            fusion_dropout=0.1,

            # classification head
            cls_dropout=0.3,
            num_classes=2,

            # other
            *args, 
            **kwargs
        ):
        """
        Args:

        """
        super().__init__(*args, **kwargs)
        # fau branch
        self.fau_detector = FAUDetector()
        self.fau_detector.eval()
        self.fau_lstm = nn.LSTM(
            input_size=input_features, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        d_model = 2 * hidden_size if bidirectional else hidden_size

        # image branch
        self.image_encoder, _ = clip.load(encoder_model)
        self.image_encoder.eval()
        clip_emb_dim = self.image_encoder.token_embedding.embedding_dim
        self.enc_proj = nn.Linear(clip_emb_dim, d_model)

        # fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=fusion_dropout,
            batch_first=True
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.PReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.BatchNorm1d(d_model // 4),
            nn.PReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(d_model // 4, num_classes),
        )

    def forward(self, frames: torch.Tensor, **batch):
        """
        Args:
            frames (torch.Tensor): batch of videos with size [B, T, C, H, W]
        """
        # get aus and embeddings
        with torch.no_grad():
            aus = torch.stack([self.fau_detector(video) for video in frames])   # [B, T, D]
            embeddings = torch.stack(
                [self.image_encoder.encode_image(video) for video in frames]
            ).float() # [B, T, D_enc]

        # lstm and projection
        aus, (h_n, c_n) = self.fau_lstm(aus)  # [B, T, 2*D]
        print(embeddings.dtype)
        print(self.enc_proj.weight.dtype)
        embeddings = self.enc_proj(embeddings)  # [B, T, 2*D]

        # attention fusion and classification
        out, _ = self.attention(
            query=aus, 
            key=embeddings, 
            value=embeddings
        )  # [B, T, 2*D]
        out = torch.mean(out, dim=1)  # [B, 2*D]
        out = self.classifier(out)

        return out
