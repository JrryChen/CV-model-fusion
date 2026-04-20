"""Multi-model pose fusion using PyTorch Transformer Encoder and Decoder layers.

This module implements a transformer-based architecture for fusing predictions from
multiple pose estimation models (HRNet, OpenPose, MediaPipe).

Architecture:
- Encoder: Processes each model's predictions independently using TransformerEncoderLayer
- Decoder: Fuses the encoded features using TransformerDecoderLayer with learned queries
- Output heads: Predict fused coordinates and confidence scores

Key shapes (B=batch, K=num_joints, M=num_models):
- coords_*: (B, K, 2)
- conf_*:   (B, K, 1)
- extra_feats_* (optional): (B, K, F)
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    """Embeds input features (coords, confidence, extra) into d_model dimension."""
    
    def __init__(self, base_dim: int = 3, d_model: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, coords: torch.Tensor, conf: torch.Tensor, 
                extra_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            coords: (B, K, 2)
            conf: (B, K, 1)
            extra_feats: Optional (B, K, F)
        
        Returns:
            embeddings: (B, K, d_model)
        """
        # Concatenate base features
        x = torch.cat([coords, conf], dim=-1)  # (B, K, 3)
        
        if extra_feats is not None:
            x = torch.cat([x, extra_feats], dim=-1)  # (B, K, 3+F)
            
            # If extra features added, need dynamic projection
            if x.shape[-1] != self.proj[0].in_features:
                # Create dynamic projection layer
                device = x.device
                dynamic_proj = nn.Sequential(
                    nn.Linear(x.shape[-1], self.proj[0].out_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.proj[0].out_features, self.proj[2].out_features),
                ).to(device)
                x = dynamic_proj(x)
                return self.norm(x)
        
        x = self.proj(x)
        x = self.norm(x)
        return x


class FusionTransformerDecoderLayer(nn.Module):
    """Minimal Transformer decoder layer that also exposes cross-attention weights.

    This is a simplified variant of ``nn.TransformerDecoderLayer`` tailored to this
    project. It only implements the components we actually use: self-attention on
    the decoder queries, cross-attention to the encoder memory, and a feedforward
    block, with LayerNorm & residual connections.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Cross-attention: queries come from decoder, keys/values from encoder memory
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one decoder block.

        Args:
            tgt: (B, T, D) - current decoder states (queries)
            memory: (B, S, D) - encoder memory (keys/values)
            need_weights: if True, also return cross-attention weights

        Returns:
            tgt_out: (B, T, D) - updated decoder states
            attn_weights: Optional (B, num_heads, T, S) cross-attention weights
        """
        # Self-attention on decoder queries
        tgt2, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # Cross-attention from queries to encoder memory
        tgt2, attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Feedforward
        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff))

        return tgt, attn_weights if need_weights else None


class TransformerPoseFusion(nn.Module):
    """Transformer-based multi-model pose fusion.
    
    Uses TransformerEncoderLayer to process each model's features and
    TransformerDecoderLayer to fuse them with learned joint queries.
    
    Args:
        num_joints: Number of body joints (K)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers per model
        num_decoder_layers: Number of decoder layers for fusion
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
    """
    
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
        self.num_models = 3
        
        # Input embedding modules for each model
        self.embed_hrnet = InputEmbedding(base_dim=3, d_model=d_model, hidden_dim=d_model//2)
        self.embed_openpose = InputEmbedding(base_dim=3, d_model=d_model, hidden_dim=d_model//2)
        self.embed_mediapipe = InputEmbedding(base_dim=3, d_model=d_model, hidden_dim=d_model//2)
        
        # Positional embeddings for joints
        self.joint_pos_embed = nn.Parameter(torch.randn(num_joints, d_model))
        
        # Model type embeddings (to distinguish between models)
        self.model_type_embed = nn.Parameter(torch.randn(self.num_models, d_model))
        
        # Encoder: Process each model's features independently
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
        
        # Decoder: Fuse features from all models.
        # We use a custom decoder layer so we can expose cross-attention weights
        # for interpretability / visualization.
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
        
        # Output heads
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
    
    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
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
        """Encode a single model's predictions.
        
        Args:
            coords: (B, K, 2)
            conf: (B, K, 1)
            extra_feats: Optional (B, K, F)
            embed_module: Input embedding module
            model_idx: Index of the model (0, 1, or 2)
        
        Returns:
            encoded: (B, K, d_model)
        """
        B, K, _ = coords.shape
        
        # Embed input features
        x = embed_module(coords, conf, extra_feats)  # (B, K, d_model)
        
        # Add positional encoding for joints
        x = x + self.joint_pos_embed.unsqueeze(0)  # (B, K, d_model)
        
        # Add model type embedding
        x = x + self.model_type_embed[model_idx].view(1, 1, -1)  # (B, K, d_model)
        
        # Pass through encoder
        encoded = self.encoder(x)  # (B, K, d_model)
        
        return encoded
    
    def forward(
        self,
        coords_hrnet: torch.Tensor,
        conf_hrnet: torch.Tensor,
        coords_openpose: torch.Tensor,
        conf_openpose: torch.Tensor,
        coords_mediapipe: torch.Tensor,
        conf_mediapipe: torch.Tensor,
        extra_feats_hrnet: Optional[torch.Tensor] = None,
        extra_feats_openpose: Optional[torch.Tensor] = None,
        extra_feats_mediapipe: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            coords_*: (B, K, 2) - Joint coordinates from each model
            conf_*: (B, K, 1) - Confidence scores from each model
            extra_feats_*: Optional (B, K, F) - Extra features from each model
            return_attention_weights: if True, also return a tensor describing
                how much each fused joint attends to each input model.
        
        Returns:
            coords_fused: (B, K, 2) - Fused coordinates
            conf_fused_logits: (B, K, 1) - Fused confidence logits
            model_attn: Optional (B, K, 3) tensor of per-model attention weights
                for each joint (order: [HRNet/DEKR, OpenPose, MediaPipe]) when
                ``return_attention_weights=True``; otherwise ``None``.
        """
        B, K, _ = coords_hrnet.shape
        assert K == self.num_joints, f"Expected {self.num_joints} joints, got {K}"
        
        # Encode each model's predictions
        enc_hrnet = self._encode_model(
            coords_hrnet, conf_hrnet, extra_feats_hrnet, self.embed_hrnet, model_idx=0
        )  # (B, K, d_model)
        
        enc_openpose = self._encode_model(
            coords_openpose, conf_openpose, extra_feats_openpose, self.embed_openpose, model_idx=1
        )  # (B, K, d_model)
        
        enc_mediapipe = self._encode_model(
            coords_mediapipe, conf_mediapipe, extra_feats_mediapipe, self.embed_mediapipe, model_idx=2
        )  # (B, K, d_model)
        
        # Concatenate encoded features from all models as memory for decoder
        # Shape: (B, 3*K, d_model)
        memory = torch.cat([enc_hrnet, enc_openpose, enc_mediapipe], dim=1)
        
        # Prepare initial decoder queries for each joint
        # Expand queries for batch: (B, K, d_model)
        queries = self.joint_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Add positional encoding to queries
        queries = queries + self.joint_pos_embed.unsqueeze(0)
        
        # Decode: fuse information from all models.
        # We capture cross-attention weights from the *last* decoder layer
        # for interpretability, since earlier layers are more intermediate.
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


class LightweightTransformerFusion(nn.Module):
    """Lightweight version with fewer parameters for faster training.
    
    Args:
        num_joints: Number of body joints
        d_model: Transformer hidden dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_encoder_layers: Number of encoder layers (default: 1)
        num_decoder_layers: Number of decoder layers (default: 1)
        dim_feedforward: FFN hidden dimension (default: 256)
        dropout: Dropout probability (default: 0.1)
    """
    
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
        
        # Use the full transformer but with smaller dimensions
        self.model = TransformerPoseFusion(
            num_joints=num_joints,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(*args, **kwargs)
