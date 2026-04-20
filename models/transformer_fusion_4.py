"""Four-input transformer fusion (DEKR, OpenPose, MediaPipe, ViTPose)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .transformer_fusion import FusionTransformerDecoderLayer, InputEmbedding


class TransformerPoseFusion4(nn.Module):
    """Same as ``TransformerPoseFusion`` but with four encoder branches and memory ``4*K``."""

    def __init__(
        self,
        num_joints: int,
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

        self.embed_hrnet = InputEmbedding(base_dim=3, d_model=d_model, hidden_dim=d_model // 2)
        self.embed_openpose = InputEmbedding(base_dim=3, d_model=d_model, hidden_dim=d_model // 2)
        self.embed_mediapipe = InputEmbedding(base_dim=3, d_model=d_model, hidden_dim=d_model // 2)
        self.embed_vitpose = InputEmbedding(base_dim=3, d_model=d_model, hidden_dim=d_model // 2)

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

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.joint_pos_embed)
        nn.init.xavier_uniform_(self.model_type_embed)
        nn.init.xavier_uniform_(self.joint_queries)

    def _encode_model(
        self,
        coords: torch.Tensor,
        conf: torch.Tensor,
        extra_feats: Optional[torch.Tensor],
        embed_module: InputEmbedding,
        model_idx: int,
    ) -> torch.Tensor:
        B, K, _ = coords.shape
        x = embed_module(coords, conf, extra_feats)
        x = x + self.joint_pos_embed.unsqueeze(0)
        x = x + self.model_type_embed[model_idx].view(1, 1, -1)
        return self.encoder(x)

    def forward(
        self,
        coords_hrnet: torch.Tensor,
        conf_hrnet: torch.Tensor,
        coords_openpose: torch.Tensor,
        conf_openpose: torch.Tensor,
        coords_mediapipe: torch.Tensor,
        conf_mediapipe: torch.Tensor,
        coords_vitpose: torch.Tensor,
        conf_vitpose: torch.Tensor,
        extra_feats_hrnet: Optional[torch.Tensor] = None,
        extra_feats_openpose: Optional[torch.Tensor] = None,
        extra_feats_mediapipe: Optional[torch.Tensor] = None,
        extra_feats_vitpose: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, K, _ = coords_hrnet.shape
        assert K == self.num_joints

        enc_hr = self._encode_model(coords_hrnet, conf_hrnet, extra_feats_hrnet, self.embed_hrnet, 0)
        enc_op = self._encode_model(coords_openpose, conf_openpose, extra_feats_openpose, self.embed_openpose, 1)
        enc_mp = self._encode_model(coords_mediapipe, conf_mediapipe, extra_feats_mediapipe, self.embed_mediapipe, 2)
        enc_pb = self._encode_model(coords_vitpose, conf_vitpose, extra_feats_vitpose, self.embed_vitpose, 3)

        memory = torch.cat([enc_hr, enc_op, enc_mp, enc_pb], dim=1)

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
            assert B_attn == B and K_attn == K and fourK == self.num_models * self.num_joints
            attn_mean = cross_attn_last.mean(dim=1)
            attn_reshaped = attn_mean.view(B, K, self.num_models, self.num_joints)
            model_attn = attn_reshaped.sum(dim=-1)
            model_attn = model_attn / (model_attn.sum(dim=-1, keepdim=True) + 1e-8)

        return coords_fused, conf_fused_logits, model_attn


class LightweightTransformerFusion4(nn.Module):
    """Thin wrapper using smaller default dims (same pattern as LightweightTransformerFusion)."""

    def __init__(
        self,
        num_joints: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model = TransformerPoseFusion4(
            num_joints=num_joints,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
