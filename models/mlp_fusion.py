"""Multi-model per-joint fusion using multi-head attention.

Implements the MultiModelPoseFusion module described in the prompt.

Key shapes (B=batch, K=num_joints):
- coords_*: (B, K, 2)
- conf_*:   (B, K, 1)
- extra_feats_* (optional): (B, K, F)

The model treats each (model, joint) pair as a token and runs a small
multi-head self-attention over the 3 model-tokens for each joint.

The forward returns:
- coords_fused: (B, K, 2) -- normalized coordinates (no final activation applied)
- conf_fused_logits: (B, K, 1) -- logits; apply sigmoid at inference if needed

"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_joint_and_model_embeddings(num_joints: int, d_model: int, num_models: int = 3):
    """Small helper to build joint and model embedding modules.

    Returns (joint_embed, model_embed) -- nn.Embedding instances.
    """
    joint_embed = nn.Embedding(num_joints, d_model)
    model_embed = nn.Embedding(num_models, d_model)
    return joint_embed, model_embed


class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        z = self.net(x)
        z = self.ln(z)
        return z


class MultiModelPoseFusion(nn.Module):
    """Per-joint fusion of three model predictions using multi-head attention.

    Args:
        num_joints: number of joint types (K)
        d_model: token embedding dimension
        num_heads: attention heads (d_model must be divisible by num_heads)
        hidden_dim: intermediate FFN hidden dim inside token MLPs
    """

    def __init__(
        self,
        num_joints: int,
        d_model: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_joints = num_joints
        self.d_model = d_model
        self.num_models = 3

        # Per-model small MLPs that map [x,y,conf,(extra_feats...)] -> d_model
        # We'll accept variable input dim at forward time; build simple per-model
        # MLPs that expect minimum 3-d inputs. If extra features are used,
        # the forward will create a new linear to project them, or users can
        # modify MLPs directly.
        self.mlp_hrnet = SmallMLP(in_dim=3, hidden_dim=hidden_dim, out_dim=d_model)
        self.mlp_openpose = SmallMLP(in_dim=3, hidden_dim=hidden_dim, out_dim=d_model)
        self.mlp_mediapipe = SmallMLP(in_dim=3, hidden_dim=hidden_dim, out_dim=d_model)

        # Learned embeddings
        self.joint_embed, self.model_embed = build_joint_and_model_embeddings(
            num_joints, d_model, num_models=self.num_models
        )

        # Attention block per-joint (we will run it on flattened B*K batches)
        # Use batch_first=True so input shape is (batch, seq_len, embed)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # Transformer-style FFN after attention
        ffn_hidden = max(d_model * 2, 128)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_hidden), nn.ReLU(), nn.Linear(ffn_hidden, d_model))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Prediction heads from fused embedding -> coords (2) and conf logit (1)
        self.head_coords = nn.Linear(d_model, 2)
        self.head_conf_logits = nn.Linear(d_model, 1)

    def _embed_per_model(self, coords: torch.Tensor, conf: torch.Tensor, extra_feats: Optional[torch.Tensor], mlp: nn.Module) -> torch.Tensor:
        """Build input vector [x,y,conf,(extra...)] and run through mlp.

        coords: (B, K, 2)
        conf:   (B, K, 1)
        extra_feats: optional (B, K, F)

        returns: (B, K, d_model)
        """
        B, K, _ = coords.shape
        base = torch.cat([coords, conf], dim=-1)  # (B, K, 3)
        if extra_feats is not None:
            base = torch.cat([base, extra_feats], dim=-1)  # (B, K, 3+F)

        # Flatten to (B*K, in_dim)
        flat = base.reshape(B * K, -1)

        # If mlp was created for in_dim==3 but we have larger in_dim, create a linear proj
        # on-the-fly to reduce to 3 dims before mlp, or extend mlp externally.
        if flat.shape[-1] != 3 and hasattr(mlp, 'net') and mlp.net[0].in_features == 3:
            # create a small projection (registered to module so it moves with .to(device))
            proj_name = f"_proj_{id(mlp)}_{flat.shape[-1]}"
            if not hasattr(self, proj_name):
                setattr(self, proj_name, nn.Linear(flat.shape[-1], 3))
            proj: nn.Linear = getattr(self, proj_name)
            flat = proj(flat)

        z = mlp(flat)  # (B*K, d_model)
        z = z.view(B, K, -1)
        return z

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

        Returns (coords_fused, conf_fused_logits, [attention_weights])
        
        If return_attention_weights=True, also returns attention_weights:
        - attention_weights: (B, K, 3, 3) - attention matrix averaged over heads
          where attention_weights[b, k, i, j] is how much model i attends to model j
          for joint k in batch b. The importance of model m is the mean of column m.

        coords_fused: (B, K, 2)
        conf_fused_logits: (B, K, 1)
        """
        # Validate shapes
        B, K, _ = coords_hrnet.shape
        assert K == self.num_joints, "coords_hrnet K must equal num_joints"

        # Embed each model per joint -> (B, K, d_model)
        z_hr = self._embed_per_model(coords_hrnet, conf_hrnet, extra_feats_hrnet, self.mlp_hrnet)
        z_op = self._embed_per_model(coords_openpose, conf_openpose, extra_feats_openpose, self.mlp_openpose)
        z_mp = self._embed_per_model(coords_mediapipe, conf_mediapipe, extra_feats_mediapipe, self.mlp_mediapipe)

        # Add joint and model embeddings
        # joint_embed: (K, d_model) -> expand to (B, K, d_model)
        joint_e = self.joint_embed.weight.unsqueeze(0).expand(B, -1, -1)

        model_idxs = torch.tensor([0, 1, 2], device=z_hr.device)
        model_e = self.model_embed(model_idxs)  # (3, d_model)

        # Expand model_e to (B, K, 3, d_model) when stacking
        # First stack per-model embeddings into (B, K, 3, d_model)
        stacked = torch.stack([z_hr, z_op, z_mp], dim=2)  # (B, K, 3, d_model)

        # Add joint and model embeddings
        # joint_e -> (B, K, 1, d_model)
        stacked = stacked + joint_e.unsqueeze(2)
        # model_e -> (1, 1, 3, d_model)
        stacked = stacked + model_e.view(1, 1, 3, -1)

        # Prepare for attention: merge B and K into batch: (B*K, 3, d_model)
        B, K, M, D = stacked.shape
        tokens = stacked.view(B * K, M, D)

        # MultiheadAttention expects (batch, seq_len, embed) with batch_first=True
        if return_attention_weights:
            attn_out, attn_weights = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
            # attn_weights: (B*K, num_heads, 3, 3) - need to average over heads
            # Average over heads: (B*K, 3, 3)
            attn_weights_avg = attn_weights.mean(dim=1)  # (B*K, 3, 3)
            # Reshape back to (B, K, 3, 3)
            attn_weights_reshaped = attn_weights_avg.view(B, K, M, M)
        else:
            attn_out, _ = self.attn(tokens, tokens, tokens)
            attn_weights_reshaped = None

        # Residual + Norm
        tokens = self.norm1(tokens + attn_out)

        # FFN
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        # Aggregate models dimension (mean)
        fused = tokens.mean(dim=1)  # (B*K, d_model)
        fused = fused.view(B, K, D)  # (B, K, d_model)

        coords_fused = self.head_coords(fused)  # (B, K, 2)
        conf_fused_logits = self.head_conf_logits(fused)  # (B, K, 1)

        if return_attention_weights:
            return coords_fused, conf_fused_logits, attn_weights_reshaped
        else:
            return coords_fused, conf_fused_logits, None
