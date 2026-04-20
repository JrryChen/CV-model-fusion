"""Internal transformer fusion with four models (DEKR, OpenPose, MediaPipe, ViTPose).

ViTPose branch uses ``JointFeatureEmbedding`` on ``feats_vitpose`` (B, K, F_vit).
Default F_vit=3 from cached ``[x,y,conf]`` (same construction as DEKR/OpenPose).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .transformer_fusion import FusionTransformerDecoderLayer, InputEmbedding
from .transformer_internal_fusion import JointFeatureEmbedding


class TransformerInternalPoseFusion4(nn.Module):
    """Four-way internal fusion: DEKR + OpenPose + MediaPipe (coords) + ViTPose features."""

    def __init__(
        self,
        num_joints: int,
        dekr_feat_dim: int,
        openpose_feat_dim: int,
        vitpose_feat_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.num_joints = num_joints
        self.d_model = d_model
        self.num_models = 4

        self.embed_dekr = JointFeatureEmbedding(
            in_dim=dekr_feat_dim, d_model=d_model, hidden_dim=d_model // 2
        )
        self.embed_openpose = JointFeatureEmbedding(
            in_dim=openpose_feat_dim, d_model=d_model, hidden_dim=d_model // 2
        )
        self.embed_vitpose = JointFeatureEmbedding(
            in_dim=vitpose_feat_dim, d_model=d_model, hidden_dim=d_model // 2
        )
        self.embed_mediapipe = InputEmbedding(
            base_dim=3, d_model=d_model, hidden_dim=d_model // 2
        )

        self.joint_pos_embed = nn.Parameter(torch.randn(num_joints, d_model))
        self.model_type_embed = nn.Parameter(torch.randn(self.num_models, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.joint_queries = nn.Parameter(torch.randn(num_joints, d_model))
        self.num_decoder_layers = num_decoder_layers
        self.decoder_layers = nn.ModuleList(
            [
                FusionTransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        self.head_coords = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 2),
        )
        self.head_conf = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.joint_pos_embed)
        nn.init.xavier_uniform_(self.model_type_embed)
        nn.init.xavier_uniform_(self.joint_queries)

    def _encode_embedded(self, x: torch.Tensor, model_idx: int) -> torch.Tensor:
        x = x + self.joint_pos_embed.unsqueeze(0)
        x = x + self.model_type_embed[model_idx].view(1, 1, -1)
        return self.encoder(x)

    def forward(
        self,
        feats_dekr: torch.Tensor,
        feats_openpose: torch.Tensor,
        feats_vitpose: torch.Tensor,
        coords_mediapipe: torch.Tensor,
        conf_mediapipe: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, K, _ = feats_dekr.shape
        assert K == self.num_joints

        enc_dekr = self._encode_embedded(self.embed_dekr(feats_dekr), 0)
        enc_openpose = self._encode_embedded(self.embed_openpose(feats_openpose), 1)
        enc_vitpose = self._encode_embedded(self.embed_vitpose(feats_vitpose), 2)
        mp_emb = self.embed_mediapipe(coords_mediapipe, conf_mediapipe, None)
        enc_mediapipe = self._encode_embedded(mp_emb, 3)

        memory = torch.cat([enc_dekr, enc_openpose, enc_vitpose, enc_mediapipe], dim=1)

        queries = self.joint_queries.unsqueeze(0).expand(B, -1, -1)
        queries = queries + self.joint_pos_embed.unsqueeze(0)

        cross_attn_last: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.decoder_layers):
            need_w = return_attention_weights and (i == self.num_decoder_layers - 1)
            queries, attn_w = layer(queries, memory, need_weights=need_w)
            if attn_w is not None:
                cross_attn_last = attn_w

        fused = self.decoder_norm(queries)
        coords_fused = self.head_coords(fused)
        conf_fused_logits = self.head_conf(fused)

        model_attn: Optional[torch.Tensor] = None
        if return_attention_weights and cross_attn_last is not None:
            B_attn, H_attn, K_attn, fourK = cross_attn_last.shape
            assert fourK == self.num_models * self.num_joints
            attn_mean = cross_attn_last.mean(dim=1)
            attn_reshaped = attn_mean.view(B, K, self.num_models, self.num_joints)
            model_attn = attn_reshaped.sum(dim=-1)
            model_attn = model_attn / (model_attn.sum(dim=-1, keepdim=True) + 1e-8)

        return coords_fused, conf_fused_logits, model_attn


class LightweightTransformerInternalFusion4(nn.Module):
    """Smaller default hyperparameters."""

    def __init__(
        self,
        num_joints: int,
        dekr_feat_dim: int,
        openpose_feat_dim: int,
        vitpose_feat_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model = TransformerInternalPoseFusion4(
            num_joints=num_joints,
            dekr_feat_dim=dekr_feat_dim,
            openpose_feat_dim=openpose_feat_dim,
            vitpose_feat_dim=vitpose_feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
