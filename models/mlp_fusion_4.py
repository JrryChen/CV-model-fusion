"""Four-model per-joint fusion (HRNet/DEKR, OpenPose, MediaPipe, ViTPose).

Same architecture as ``MultiModelPoseFusion`` but with four model tokens per joint.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .mlp_fusion import SmallMLP, build_joint_and_model_embeddings


class MultiModelPoseFusion4(nn.Module):
    """Per-joint fusion of four model predictions using multi-head self-attention."""

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
        self.num_models = 4

        self.mlp_hrnet = SmallMLP(in_dim=3, hidden_dim=hidden_dim, out_dim=d_model)
        self.mlp_openpose = SmallMLP(in_dim=3, hidden_dim=hidden_dim, out_dim=d_model)
        self.mlp_mediapipe = SmallMLP(in_dim=3, hidden_dim=hidden_dim, out_dim=d_model)
        self.mlp_vitpose = SmallMLP(in_dim=3, hidden_dim=hidden_dim, out_dim=d_model)

        self.joint_embed, self.model_embed = build_joint_and_model_embeddings(
            num_joints, d_model, num_models=self.num_models
        )

        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        ffn_hidden = max(d_model * 2, 128)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_hidden, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.head_coords = nn.Linear(d_model, 2)
        self.head_conf_logits = nn.Linear(d_model, 1)

    def _embed_per_model(
        self,
        coords: torch.Tensor,
        conf: torch.Tensor,
        extra_feats: Optional[torch.Tensor],
        mlp: nn.Module,
    ) -> torch.Tensor:
        B, K, _ = coords.shape
        base = torch.cat([coords, conf], dim=-1)
        if extra_feats is not None:
            base = torch.cat([base, extra_feats], dim=-1)
        flat = base.reshape(B * K, -1)
        if flat.shape[-1] != 3 and hasattr(mlp, 'net') and mlp.net[0].in_features == 3:
            proj_name = f"_proj_{id(mlp)}_{flat.shape[-1]}"
            if not hasattr(self, proj_name):
                setattr(self, proj_name, nn.Linear(flat.shape[-1], 3))
            proj: nn.Linear = getattr(self, proj_name)
            flat = proj(flat)
        z = mlp(flat)
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

        z_hr = self._embed_per_model(coords_hrnet, conf_hrnet, extra_feats_hrnet, self.mlp_hrnet)
        z_op = self._embed_per_model(coords_openpose, conf_openpose, extra_feats_openpose, self.mlp_openpose)
        z_mp = self._embed_per_model(coords_mediapipe, conf_mediapipe, extra_feats_mediapipe, self.mlp_mediapipe)
        z_pb = self._embed_per_model(coords_vitpose, conf_vitpose, extra_feats_vitpose, self.mlp_vitpose)

        joint_e = self.joint_embed.weight.unsqueeze(0).expand(B, -1, -1)
        model_idxs = torch.tensor([0, 1, 2, 3], device=z_hr.device)
        model_e = self.model_embed(model_idxs)

        stacked = torch.stack([z_hr, z_op, z_mp, z_pb], dim=2)
        stacked = stacked + joint_e.unsqueeze(2)
        stacked = stacked + model_e.view(1, 1, self.num_models, -1)

        B_, K_, M, D = stacked.shape
        tokens = stacked.view(B_ * K_, M, D)

        if return_attention_weights:
            attn_out, attn_weights = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
            attn_weights_avg = attn_weights.mean(dim=1)
            attn_weights_reshaped = attn_weights_avg.view(B_, K_, M, M)
        else:
            attn_out, _ = self.attn(tokens, tokens, tokens)
            attn_weights_reshaped = None

        tokens = self.norm1(tokens + attn_out)
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        fused = tokens.mean(dim=1)
        fused = fused.view(B_, K_, D)

        coords_fused = self.head_coords(fused)
        conf_fused_logits = self.head_conf_logits(fused)

        if return_attention_weights:
            return coords_fused, conf_fused_logits, attn_weights_reshaped
        return coords_fused, conf_fused_logits, None
