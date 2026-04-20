"""Transformer-based fusion of *internal* per-joint embeddings from three pose models.

This module is analogous to ``transformer_fusion.py`` but is designed to work with
intermediate / internal embeddings from:
    - DEKR / HRNet-style backbone
    - OpenPose
    - MediaPipe

Instead of taking final joint coordinates + confidences, it expects that upstream
code has already:
    1) Run each backbone on the input image(s)
    2) Extracted one feature vector per joint, per model

The expected interface is:
    - feats_dekr:      (B, K, F_dekr)
    - feats_openpose:  (B, K, F_openpose)
    - feats_mediapipe: (B, K, F_mediapipe)

where:
    B = batch size
    K = number of joints (must match ``num_joints``)
    F_* = dimensionality of the internal per-joint embedding for that model

High-level architecture:
    - Per-model joint feature embedding into a shared ``d_model`` space
    - Add joint positional embeddings and model-type embeddings
    - Shared TransformerEncoder applied *independently* to each model's sequence
    - Concatenate all encoded tokens as encoder "memory" (length = 3 * K)
    - Learned joint queries passed through a stack of custom decoder layers that
      perform self-attention + cross-attention to the memory
    - Output heads map fused joint representations to final (x, y) coordinates
      and confidence logits, just like in ``TransformerPoseFusion``
    - Optional per-model attention weights derived from the last decoder layer
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .transformer_fusion import FusionTransformerDecoderLayer, InputEmbedding


class JointFeatureEmbedding(nn.Module):
    """Embeds per-joint backbone features into ``d_model`` dimension.

    This is a more generic version of ``InputEmbedding`` from ``transformer_fusion``
    that assumes the caller already provides a single feature vector per joint.
    """

    def __init__(self, in_dim: int, d_model: int = 256, hidden_dim: int = 128):
        """
        Args:
            in_dim:      Input feature dimension F_* for this model
            d_model:     Transformer hidden dimension
            hidden_dim:  Hidden dimension of the MLP used for projection
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, K, F_in) per-joint internal embeddings from a pose backbone.

        Returns:
            embeddings: (B, K, d_model)
        """
        x = self.proj(feats)
        x = self.norm(x)
        return x


class TransformerInternalPoseFusion(nn.Module):
    """Transformer-based fusion of per-joint embeddings, with mixed inputs:

    - DEKR:      internal per-joint embeddings (B, K, F_dekr)
    - OpenPose:  internal per-joint embeddings (B, K, F_openpose)
    - MediaPipe: final joint predictions: coords (B, K, 2), conf (B, K, 1)

    DEKR and OpenPose embeddings are projected with ``JointFeatureEmbedding``.
    MediaPipe joint predictions are embedded via the same ``InputEmbedding``
    used in ``TransformerPoseFusion`` (coords+conf -> d_model).

    Args:
        num_joints:          Number of body joints (K)
        dekr_feat_dim:       Feature dimension for DEKR per-joint embeddings (F_dekr)
        openpose_feat_dim:   Feature dimension for OpenPose per-joint embeddings (F_openpose)
        d_model:             Transformer hidden dimension
        nhead:               Number of attention heads
        num_encoder_layers:  Number of encoder layers per model
        num_decoder_layers:  Number of decoder layers for fusion
        dim_feedforward:     FFN hidden dimension
        dropout:             Dropout probability
    """

    def __init__(
        self,
        num_joints: int,
        dekr_feat_dim: int,
        openpose_feat_dim: int,
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
        self.num_models = 3  # [DEKR, OpenPose, MediaPipe]

        # Input embedding modules:
        #   - DEKR/OpenPose: generic per-joint embeddings
        #   - MediaPipe: coords+conf -> d_model via InputEmbedding
        self.embed_dekr = JointFeatureEmbedding(
            in_dim=dekr_feat_dim,
            d_model=d_model,
            hidden_dim=d_model // 2,
        )
        self.embed_openpose = JointFeatureEmbedding(
            in_dim=openpose_feat_dim,
            d_model=d_model,
            hidden_dim=d_model // 2,
        )
        # MediaPipe only exposes final joint predictions, not internal features.
        # We therefore reuse the coord+conf embedding from ``TransformerPoseFusion``.
        self.embed_mediapipe = InputEmbedding(
            base_dim=3, d_model=d_model, hidden_dim=d_model // 2
        )

        # Positional embeddings for joints
        self.joint_pos_embed = nn.Parameter(torch.randn(num_joints, d_model))

        # Model type embeddings (to distinguish between models)
        # 0: DEKR / HRNet-style, 1: OpenPose, 2: MediaPipe
        self.model_type_embed = nn.Parameter(torch.randn(self.num_models, d_model))

        # Encoder: process each model's features independently (shared weights)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Learned queries for each joint (decoder will use these to extract fused features)
        self.joint_queries = nn.Parameter(torch.randn(num_joints, d_model))

        # Decoder: fuse features from all models, exposing cross-attention weights
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

        # Output heads: same as in ``TransformerPoseFusion``
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

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.joint_pos_embed)
        nn.init.xavier_uniform_(self.model_type_embed)
        nn.init.xavier_uniform_(self.joint_queries)

    def _encode_embedded(self, x: torch.Tensor, model_idx: int) -> torch.Tensor:
        """Encode already-embedded per-joint features for a single model.

        Args:
            x:        (B, K, d_model) embedded per-joint features
            model_idx: Index of the model (0=DEKR, 1=OpenPose, 2=MediaPipe)

        Returns:
            encoded:  (B, K, d_model)
        """
        # Add positional encoding for joints
        x = x + self.joint_pos_embed.unsqueeze(0)  # (B, K, d_model)

        # Add model type embedding
        x = x + self.model_type_embed[model_idx].view(1, 1, -1)  # (B, K, d_model)

        # Pass through encoder
        encoded = self.encoder(x)  # (B, K, d_model)
        return encoded

    def forward(
        self,
        feats_dekr: torch.Tensor,
        feats_openpose: torch.Tensor,
        coords_mediapipe: torch.Tensor,
        conf_mediapipe: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            feats_dekr:      (B, K, F_dekr) per-joint embeddings from DEKR / HRNet
            feats_openpose:  (B, K, F_openpose) per-joint embeddings from OpenPose
            coords_mediapipe: (B, K, 2) joint coordinates from MediaPipe
            conf_mediapipe:   (B, K, 1) confidence scores from MediaPipe
            return_attention_weights: if True, also return per-model attention weights
                describing how much each fused joint attends to each input model.

        Returns:
            coords_fused:        (B, K, 2) fused joint coordinates
            conf_fused_logits:   (B, K, 1) fused confidence logits
            model_attn: Optional (B, K, 3) tensor of per-model attention weights for
                each joint (order: [DEKR, OpenPose, MediaPipe]) when
                ``return_attention_weights=True``; otherwise ``None``.
        """
        B, K, _ = feats_dekr.shape
        assert K == self.num_joints, f"Expected {self.num_joints} joints, got {K}"

        # Embed and encode DEKR and OpenPose internal features
        dekr_emb = self.embed_dekr(feats_dekr)  # (B, K, d_model)
        enc_dekr = self._encode_embedded(dekr_emb, model_idx=0)

        openpose_emb = self.embed_openpose(feats_openpose)  # (B, K, d_model)
        enc_openpose = self._encode_embedded(openpose_emb, model_idx=1)

        # MediaPipe: use final joint predictions (coords + conf) as inputs
        # coords_mediapipe: (B, K, 2), conf_mediapipe: (B, K, 1)
        mp_emb = self.embed_mediapipe(coords_mediapipe, conf_mediapipe, None)  # (B, K, d_model)
        enc_mediapipe = self._encode_embedded(mp_emb, model_idx=2)

        # Concatenate encoded features from all models as memory for decoder
        # Shape: (B, 3*K, d_model)
        memory = torch.cat([enc_dekr, enc_openpose, enc_mediapipe], dim=1)

        # Prepare initial decoder queries for each joint
        # Expand queries for batch: (B, K, d_model)
        queries = self.joint_queries.unsqueeze(0).expand(B, -1, -1)

        # Add positional encoding to queries
        queries = queries + self.joint_pos_embed.unsqueeze(0)

        # Decode: fuse information from all models.
        # We capture cross-attention weights from the *last* decoder layer.
        cross_attn_last: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.decoder_layers):
            need_w = return_attention_weights and (i == self.num_decoder_layers - 1)
            queries, attn_w = layer(queries, memory, need_weights=need_w)
            if attn_w is not None:
                cross_attn_last = attn_w  # (B, num_heads, K, 3*K)

        # Final layer norm on decoder output
        fused = self.decoder_norm(queries)  # (B, K, d_model)

        # Predict coordinates and confidence
        coords_fused = self.head_coords(fused)  # (B, K, 2)
        conf_fused_logits = self.head_conf(fused)  # (B, K, 1)

        model_attn: Optional[torch.Tensor] = None
        if return_attention_weights and cross_attn_last is not None:
            # cross_attn_last: (B, num_heads, K, 3*K)
            B_attn, H_attn, K_attn, threeK = cross_attn_last.shape
            assert (
                B_attn == B and K_attn == K and threeK == self.num_models * self.num_joints
            ), "Unexpected attention shape"

            # Average over heads -> (B, K, 3*K)
            attn_mean = cross_attn_last.mean(dim=1)

            # Reshape source dimension into (num_models, num_joints)
            attn_reshaped = attn_mean.view(
                B, K, self.num_models, self.num_joints
            )  # (B, K, 3, K)

            # Sum over source joints inside each model block so we get a single
            # importance per model for each fused joint.
            model_attn = attn_reshaped.sum(dim=-1)  # (B, K, 3)

            # Normalize across models so they form a probability distribution
            model_attn = model_attn / (model_attn.sum(dim=-1, keepdim=True) + 1e-8)

        return coords_fused, conf_fused_logits, model_attn


class LightweightTransformerInternalFusion(nn.Module):
    """Lightweight wrapper around ``TransformerInternalPoseFusion``.

    This mirrors ``LightweightTransformerFusion`` in ``transformer_fusion.py`` but
    is specialized for internal per-joint embeddings.
    """

    def __init__(
        self,
        num_joints: int,
        dekr_feat_dim: int,
        openpose_feat_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model = TransformerInternalPoseFusion(
            num_joints=num_joints,
            dekr_feat_dim=dekr_feat_dim,
            openpose_feat_dim=openpose_feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.model(*args, **kwargs)

