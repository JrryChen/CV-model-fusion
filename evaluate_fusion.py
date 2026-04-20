"""Evaluate fusion model on test/val set with detailed metrics.

Computes accuracy, L2 distance, and other diagnostic metrics
to understand model performance and identify overfitting.
"""
from pathlib import Path
import argparse
import json
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import math

import os
import sys

from models.mlp_fusion import MultiModelPoseFusion
from models.mlp_fusion_4 import MultiModelPoseFusion4
from models.transformer_fusion import TransformerPoseFusion, LightweightTransformerFusion
from models.transformer_fusion_4 import TransformerPoseFusion4, LightweightTransformerFusion4
from models.transformer_internal_fusion import (
    TransformerInternalPoseFusion,
    LightweightTransformerInternalFusion,
)
from models.transformer_internal_fusion_4 import (
    TransformerInternalPoseFusion4,
    LightweightTransformerInternalFusion4,
)
from training.train_mlp import build_dataloader_from_list, _read_p2d_cache, _parse_rel_path
from data.pytorch_dataset import VideoFrameKeypointDataset

# Import DEKR modules if available
DEKR_AVAILABLE = False
DEKR_ROOT = None
try:
    DEKR_ROOT = Path(os.environ.get('DEKR_ROOT', ''))  # set DEKR_ROOT env var if needed
    if DEKR_ROOT.exists():
        DEKR_TOOLS = DEKR_ROOT / 'tools'
        if str(DEKR_TOOLS) not in sys.path:
            sys.path.insert(0, str(DEKR_TOOLS))
        import _init_paths  # noqa: F401
        from config import cfg, update_config
        import models
        DEKR_AVAILABLE = True
        print("DEKR modules available for trained model evaluation")
except Exception as e:
    print(f"DEKR modules not available: {e}. Trained DEKR evaluation will be skipped.")


"""
Trained OpenPose model support
"""
OPENPOSE_AVAILABLE = False
OPENPOSE_ROOT = None
try:
    OPENPOSE_ROOT = Path(os.environ.get('OPENPOSE_ROOT', ''))  # set OPENPOSE_ROOT env var if needed
    OPENPOSE_SRC = OPENPOSE_ROOT / 'src'
    if OPENPOSE_SRC.exists():
        if str(OPENPOSE_SRC) not in sys.path:
            sys.path.insert(0, str(OPENPOSE_SRC))
        from model import bodypose_model  # type: ignore
        OPENPOSE_AVAILABLE = True
        print("OpenPose modules available for trained OpenPose evaluation")
except Exception as e:
    print(f"OpenPose modules not available: {e}. Trained OpenPose evaluation will be skipped.")


def load_trained_dekr_model(checkpoint_path: str, device: torch.device, num_joints: int = 3):
    """Load trained DEKR model from checkpoint."""
    if not DEKR_AVAILABLE:
        return None, None
    
    try:
        # Define DEKRWrapper class locally to avoid importing train_dekr_improved.py
        # This prevents dependency issues (e.g., json_tricks not in fusionenv)
        class DEKRWrapper(nn.Module):
            """Wrapper around DEKR model for coordinate regression training."""
            
            def __init__(self, dekr_model, cfg, num_joints=3):
                super().__init__()
                self.dekr_model = dekr_model
                self.cfg = cfg
                self.num_joints = num_joints
                # COCO keypoint indices for left side: left_shoulder=5, left_elbow=7, left_wrist=9
                self.joint_indices = [5, 7, 9]
                
            def forward(self, image):
                """Forward pass that returns keypoint coordinates.
                
                Args:
                    image: (B, 3, H, W) tensor, normalized ImageNet stats
                    
                Returns:
                    coords: (B, K, 2) normalized coordinates [0, 1]
                    conf: (B, K) confidence scores
                """
                B = image.shape[0]
                device = image.device
                
                with torch.set_grad_enabled(self.training):
                    outputs = self.dekr_model(image)
                    if isinstance(outputs, tuple):
                        heatmap, offset = outputs
                    else:
                        heatmap = outputs
                        offset = None
                
                B, K_full, H, W = heatmap.shape
                joint_indices = self.joint_indices[:self.num_joints]
                if max(joint_indices) >= K_full:
                    raise ValueError(f"Model outputs {K_full} joints, but need indices up to {max(joint_indices)}")
                
                heatmap_subset = heatmap[:, joint_indices, :, :]  # (B, num_joints, H, W)
                heatmap_flat = heatmap_subset.view(B, self.num_joints, -1)  # (B, num_joints, H*W)
                max_vals, max_indices = torch.max(heatmap_flat, dim=2)  # (B, num_joints)
                
                y_coords = max_indices // W
                x_coords = max_indices % W
                coords = torch.stack([x_coords.float() / W, y_coords.float() / H], dim=-1)  # (B, num_joints, 2)
                conf = torch.sigmoid(max_vals)  # (B, num_joints)
                
                return coords, conf
        
        # Build DEKR model from pretrained checkpoint
        cfg_file = DEKR_ROOT / 'experiments' / 'coco' / 'w48' / 'w48_4x_reg03_bs5_640_adam_lr1e-3_coco_x140.yaml'
        if not cfg_file.exists():
            cfg_file = DEKR_ROOT / 'experiments' / 'coco' / 'inference_demo_coco.yaml'
            print(f"Warning: W48 config not found, using {cfg_file}")
        
        class Args:
            cfg = str(cfg_file)
            opts = []
            modelDir = ''
            logDir = ''
            dataDir = ''
            prevModelDir = ''
        
        args = Args()
        update_config(cfg, args)
        
        # Build model - use is_train=True to match training, then switch to eval mode
        # This ensures BN layers and other train-time components match the checkpoint
        pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=True)
        
        # Load trained checkpoint
        trained_ckpt = Path(checkpoint_path)
        if trained_ckpt.exists():
            print(f"Loading trained DEKR checkpoint from {trained_ckpt}")
            d = torch.load(str(trained_ckpt), map_location=device)
            if 'model_state' in d:
                # Create wrapper - the trained checkpoint should contain all weights
                wrapper = DEKRWrapper(pose_model, cfg, num_joints=num_joints)
                try:
                    # Try loading directly first (trained checkpoint should be self-contained)
                    try:
                        wrapper.load_state_dict(d['model_state'], strict=True)
                        print("✓ Loaded trained weights with strict=True (perfect match)")
                    except RuntimeError as e:
                        # If strict=True fails, the checkpoint might need base model initialization
                        error_msg = str(e)
                        print(f"⚠️  strict=True failed: {error_msg[:300]}")
                        
                        # Initialize base model with pretrained weights, then load trained weights
                        pretrained_path = DEKR_ROOT / 'model' / 'pose_coco' / 'pose_dekr_hrnetw48_coco.pth'
                        if pretrained_path.exists():
                            print(f"Initializing base model with pretrained weights, then loading trained weights")
                            state_dict = torch.load(str(pretrained_path), map_location='cpu')
                            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                                state_dict = state_dict['state_dict']
                            pose_model.load_state_dict(state_dict, strict=False)
                            # Recreate wrapper with initialized base model
                            wrapper = DEKRWrapper(pose_model, cfg, num_joints=num_joints)
                        
                        # Now try loading trained weights again
                        missing_keys, unexpected_keys = wrapper.load_state_dict(d['model_state'], strict=False)
                        if missing_keys:
                            print(f"⚠️  {len(missing_keys)} keys still missing after pretrained init")
                            if len(missing_keys) <= 20:
                                print(f"  Missing: {list(missing_keys)}")
                        if unexpected_keys:
                            print(f"⚠️  {len(unexpected_keys)} unexpected keys")
                            if len(unexpected_keys) <= 20:
                                print(f"  Unexpected: {list(unexpected_keys)}")
                    
                    wrapper.eval()
                    wrapper = wrapper.to(device)
                    print("Loaded trained DEKR model successfully")
                    return wrapper, cfg
                except Exception as e:
                    print(f"Warning: Could not load trained DEKR checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, None
            else:
                print("Warning: Checkpoint does not contain 'model_state' key")
                return None, None
        else:
            print(f"Warning: Trained DEKR checkpoint not found at {trained_ckpt}")
            return None, None
    except Exception as e:
        print(f"Warning: Could not load trained DEKR model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_transformer_fusion_model(
    checkpoint_path: str,
    device: torch.device,
    num_joints: int = 3,
    model_type: str = 'auto',
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
):
    """Load transformer fusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        num_joints: Number of joints
        model_type: 'full', 'lightweight', or 'auto' (try to detect from checkpoint)
        d_model, nhead, etc.: Model hyperparameters (used if model_type != 'auto')
    
    Returns:
        model: Loaded transformer fusion model, or None if loading failed
    """
    try:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"Warning: Transformer fusion checkpoint not found at {ckpt_path}")
            return None
        
        print(f"Loading transformer fusion checkpoint from {ckpt_path}")
        d = torch.load(str(ckpt_path), map_location=device)
        
        if 'model_state' not in d:
            print("Warning: Transformer fusion checkpoint does not contain 'model_state' key")
            return None
        
        # Try to detect model type from checkpoint keys
        state_dict = d['model_state']
        if state_dict:
            # Check if any key starts with 'model.' (lightweight has nested model)
            first_key = list(state_dict.keys())[0]
            is_lightweight = first_key.startswith('model.')
        else:
            is_lightweight = False
        
        if model_type == 'auto':
            # Try lightweight first (has 'model.' prefix)
            if is_lightweight:
                model_type = 'lightweight'
                print(f"Auto-detected lightweight transformer fusion model from checkpoint keys")
            else:
                model_type = 'full'
                print(f"Auto-detected full transformer fusion model from checkpoint keys")
        
        # Create model
        if model_type == 'lightweight':
            model = LightweightTransformerFusion(
                num_joints=num_joints,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        else:
            model = TransformerPoseFusion(
                num_joints=num_joints,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ Loaded transformer fusion model ({model_type}) with strict=True")
        except RuntimeError as e:
            print(f"⚠️  strict=True failed: {str(e)[:300]}")
            # Try strict=False
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️  {len(missing_keys)} keys missing after strict=False")
                if len(missing_keys) <= 20:
                    print(f"  Missing: {list(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️  {len(unexpected_keys)} unexpected keys")
                if len(unexpected_keys) <= 20:
                    print(f"  Unexpected: {list(unexpected_keys)}")
        
        model.eval()
        model = model.to(device)
        print(f"Loaded transformer fusion model successfully ({model_type})")
        return model
        
    except Exception as e:
        print(f"Warning: Could not load transformer fusion model: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_transformer_internal_fusion_model(
    checkpoint_path: str,
    device: torch.device,
    num_joints: int = 3,
    model_type: str = 'auto',
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
) -> Optional[nn.Module]:
    """Load transformer *internal* fusion model from checkpoint."""
    try:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"Warning: Transformer INTERNAL fusion checkpoint not found at {ckpt_path}")
            return None

        print(f"Loading transformer INTERNAL fusion checkpoint from {ckpt_path}")
        d = torch.load(str(ckpt_path), map_location=device)

        if 'model_state' not in d:
            print("Warning: Transformer INTERNAL fusion checkpoint does not contain 'model_state' key")
            return None

        state_dict = d['model_state']
        model_cfg = d.get('model_config', {})

        # Feature dims used during training (default to 3 = [x,y,conf] if not present)
        dekr_feat_dim = int(model_cfg.get('dekr_feat_dim', 3))
        openpose_feat_dim = int(model_cfg.get('openpose_feat_dim', 3))

        # Try to detect lightweight vs full from state_dict keys
        if state_dict:
            first_key = list(state_dict.keys())[0]
            is_lightweight = first_key.startswith('model.')
        else:
            is_lightweight = False

        if model_type == 'auto':
            if is_lightweight:
                model_type = 'lightweight'
                print("Auto-detected LIGHTWEIGHT internal transformer fusion model from checkpoint keys")
            else:
                model_type = 'full'
                print("Auto-detected FULL internal transformer fusion model from checkpoint keys")

        # Create model
        if model_type == 'lightweight':
            model = LightweightTransformerInternalFusion(
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
        else:
            model = TransformerInternalPoseFusion(
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

        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ Loaded INTERNAL transformer fusion model ({model_type}) with strict=True")
        except RuntimeError as e:
            print(f"⚠️  strict=True failed for INTERNAL transformer fusion: {str(e)[:300]}")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️  {len(missing_keys)} keys missing after strict=False")
                if len(missing_keys) <= 20:
                    print(f"  Missing: {list(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️  {len(unexpected_keys)} unexpected keys")
                if len(unexpected_keys) <= 20:
                    print(f"  Unexpected: {list(unexpected_keys)}")

        model.eval()
        model = model.to(device)
        print(f"Loaded INTERNAL transformer fusion model successfully ({model_type})")
        return model

    except Exception as e:
        print(f"Warning: Could not load INTERNAL transformer fusion model: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_transformer_fusion_4_model(
    checkpoint_path: str,
    device: torch.device,
    num_joints: int = 3,
    model_type: str = 'auto',
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
):
    """Load 4-input transformer fusion (TransformerPoseFusion4) from checkpoint."""
    try:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"Warning: Transformer fusion-4 checkpoint not found at {ckpt_path}")
            return None
        print(f"Loading transformer fusion-4 checkpoint from {ckpt_path}")
        d = torch.load(str(ckpt_path), map_location=device)
        if 'model_state' not in d:
            print("Warning: checkpoint does not contain 'model_state'")
            return None
        state_dict = d['model_state']
        cfg = d.get('model_config', {})
        if cfg:
            d_model = int(cfg.get('d_model', d_model))
            nhead = int(cfg.get('nhead', nhead))
            num_encoder_layers = int(cfg.get('num_encoder_layers', num_encoder_layers))
            num_decoder_layers = int(cfg.get('num_decoder_layers', num_decoder_layers))
            dim_feedforward = int(cfg.get('dim_feedforward', dim_feedforward))
            dropout = float(cfg.get('dropout', dropout))

        if state_dict:
            first_key = list(state_dict.keys())[0]
            is_lightweight = first_key.startswith('model.')
        else:
            is_lightweight = False

        if model_type == 'auto':
            model_type = 'lightweight' if is_lightweight else 'full'

        if model_type == 'lightweight':
            model = LightweightTransformerFusion4(
                num_joints=num_joints,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        else:
            model = TransformerPoseFusion4(
                num_joints=num_joints,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ Loaded transformer fusion-4 ({model_type}) strict=True")
        except RuntimeError as e:
            print(f"⚠️  strict=True failed fusion-4: {str(e)[:200]}")
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model.to(device)
    except Exception as e:
        print(f"Warning: Could not load transformer fusion-4: {e}")
        return None


def load_transformer_internal_fusion_4_model(
    checkpoint_path: str,
    device: torch.device,
    num_joints: int = 3,
    model_type: str = 'auto',
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
) -> Optional[nn.Module]:
    """Load 4-model internal transformer fusion from checkpoint."""
    try:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            return None
        print(f"Loading INTERNAL transformer fusion-4 from {ckpt_path}")
        d = torch.load(str(ckpt_path), map_location=device)
        if 'model_state' not in d:
            return None
        state_dict = d['model_state']
        cfg = d.get('model_config', {})
        dekr_feat_dim = int(cfg.get('dekr_feat_dim', 3))
        openpose_feat_dim = int(cfg.get('openpose_feat_dim', 3))
        vitpose_feat_dim = int(cfg.get('vitpose_feat_dim', cfg.get('posebert_feat_dim', 3)))
        if cfg:
            d_model = int(cfg.get('d_model', d_model))
            nhead = int(cfg.get('nhead', nhead))
            num_encoder_layers = int(cfg.get('num_encoder_layers', num_encoder_layers))
            num_decoder_layers = int(cfg.get('num_decoder_layers', num_decoder_layers))
            dim_feedforward = int(cfg.get('dim_feedforward', dim_feedforward))
            dropout = float(cfg.get('dropout', dropout))

        if state_dict:
            first_key = list(state_dict.keys())[0]
            is_lightweight = first_key.startswith('model.')
        else:
            is_lightweight = False
        if model_type == 'auto':
            model_type = 'lightweight' if is_lightweight else 'full'

        if model_type == 'lightweight':
            model = LightweightTransformerInternalFusion4(
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
        else:
            model = TransformerInternalPoseFusion4(
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
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model.to(device)
    except Exception as e:
        print(f"Warning: Could not load internal fusion-4: {e}")
        return None


def load_trained_openpose_model(checkpoint_path: str, device: torch.device, num_joints: int = 3):
    """Load trained OpenPose model from checkpoint_best.pt."""
    if not OPENPOSE_AVAILABLE:
        return None
    
    try:
        class OpenPoseWrapper(nn.Module):
            """Wrapper around OpenPose model for coordinate evaluation."""
            def __init__(self, openpose_model, num_joints=3):
                super().__init__()
                self.openpose_model = openpose_model
                self.num_joints = num_joints
                # OpenPose BODY_25 indices for left arm: shoulder=5, elbow=6, wrist=7
                self.joint_indices = [5, 6, 7]

            def forward(self, image):
                # image: (B, 3, H, W)
                B = image.shape[0]
                device_local = image.device
                with torch.set_grad_enabled(self.training):
                    outputs = self.openpose_model(image)
                    if isinstance(outputs, tuple):
                        paf, heatmap = outputs
                    else:
                        heatmap = outputs
                        paf = None  # noqa: F841

                B, num_heatmaps, H, W = heatmap.shape
                coords_list = []
                conf_list = []
                for joint_idx in self.joint_indices:
                    if joint_idx < num_heatmaps:
                        joint_heatmap = heatmap[:, joint_idx, :, :]
                        joint_flat = joint_heatmap.view(B, -1)
                        max_vals, max_indices = torch.max(joint_flat, dim=1)
                        y_coords = max_indices // W
                        x_coords = max_indices % W
                        x_norm = x_coords.float() / W
                        y_norm = y_coords.float() / H
                        coords_list.append(torch.stack([x_norm, y_norm], dim=-1))
                        conf_list.append(torch.sigmoid(max_vals))
                    else:
                        coords_list.append(torch.zeros(B, 2, device=device_local))
                        conf_list.append(torch.zeros(B, device=device_local))

                coords = torch.stack(coords_list, dim=1)  # (B, K, 2)
                conf = torch.stack(conf_list, dim=1)      # (B, K)
                return coords, conf

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"Warning: Trained OpenPose checkpoint not found at {ckpt_path}")
            return None

        print(f"Loading trained OpenPose checkpoint from {ckpt_path}")
        d = torch.load(str(ckpt_path), map_location='cpu')
        if 'model_state' not in d:
            print("Warning: OpenPose checkpoint does not contain 'model_state' key")
            return None

        base_model = bodypose_model()
        wrapper = OpenPoseWrapper(base_model, num_joints=num_joints)
        try:
            wrapper.load_state_dict(d['model_state'], strict=False)
        except Exception as e:
            print(f"Warning: Could not load OpenPose state dict: {e}")
            import traceback
            traceback.print_exc()
            return None

        wrapper.to(device)
        wrapper.eval()
        print("Loaded trained OpenPose model successfully")
        return wrapper
    except Exception as e:
        print(f"Warning: Could not load trained OpenPose model: {e}")
        import traceback
        traceback.print_exc()
        return None


def _softmax2(a: float, b: float) -> Tuple[float, float]:
    """Stable softmax for 2 scalars."""
    m = a if a >= b else b
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    z = ea + eb
    return ea / z, eb / z


def _dataset_index_for_importance_row(
    row_idx: int,
    frame_info: Optional[List[Dict[str, Any]]],
    dataset_len: Optional[int] = None,
) -> int:
    """Map an importance-weights row to ``VideoFrameKeypointDataset`` index.

    ``extract_importance_weights`` stores ``global_frame_idx`` in ``frame_info``.
    ``eval_all_models`` often omits ``frame_info``; then row order matches the
    fusion dataloader with ``shuffle=False``, so the row index is the dataset index.

    If ``global_frame_idx`` is present but outside ``[0, dataset_len)`` (stale JSON,
    different ``batch_size`` used when the file was written, or ``--list``/``--root``
    mismatch vs eval), we fall back to the sequential row index so thumbnails still
    align with eval order when possible.
    """
    row_idx = int(row_idx)
    fallback = row_idx
    if frame_info and 0 <= row_idx < len(frame_info):
        g = frame_info[row_idx].get('global_frame_idx', None)
        if g is not None:
            gi = int(g)
            if dataset_len is None:
                return gi
            if 0 <= gi < dataset_len:
                return gi
            # global index does not match this dataset; row order still matches eval
    return fallback


def _get_video_boundaries(dataset) -> List[int]:
    """Return global indices where a new video starts in the dataset.

    Each boundary is the first frame index of a new video (excluding index 0).
    Works by scanning ``dataset.index`` for changes in ``video_rel_path``.
    """
    if dataset is None:
        return []
    index = getattr(dataset, 'index', None)
    if not index:
        return []
    boundaries: List[int] = []
    prev_video = index[0][0]
    for i, entry in enumerate(index[1:], start=1):
        if entry[0] != prev_video:
            boundaries.append(i)
            prev_video = entry[0]
    return boundaries


def _draw_video_boundaries(ax, boundaries: List[int], xmin=None, xmax=None,
                           color: str = 'red', linestyle: str = '-',
                           linewidth: float = 1.0, alpha: float = 0.55,
                           label_first: bool = True):
    """Draw vertical lines at video boundary positions on *ax*.

    Only boundaries falling within [xmin, xmax] (if given) are drawn.
    The first boundary drawn gets the label ``'New video'`` so it appears once
    in the legend.
    """
    labeled = False
    for b in boundaries:
        if xmin is not None and b < xmin:
            continue
        if xmax is not None and b > xmax:
            continue
        lbl = 'New video' if label_first and not labeled else None
        ax.axvline(b, color=color, linestyle=linestyle, linewidth=linewidth,
                   alpha=alpha, label=lbl)
        labeled = True


def _to_uint8_rgb(img_t: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor to HWC uint8 RGB for plotting."""
    if isinstance(img_t, torch.Tensor):
        img = img_t.detach().cpu()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img = img.numpy()
    else:
        img = img_t
    img = np.asarray(img)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi <= 1.01 and lo >= -0.01 and hi > lo:
        img = np.clip(img, 0.0, 1.0)
    elif hi > 1.0 or lo < 0.0:
        # Typical ImageNet / [-1,1] normalization: stretch to [0, 1] for display
        img = (img - lo) / (hi - lo + 1e-8)
    else:
        img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _overlay_keypoints(ax, coords_px: np.ndarray, color: str, label: str):
    """coords_px: (K,2) in pixel space of the shown image."""
    if coords_px is None:
        return
    pts = np.asarray(coords_px, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c=color, label=label, alpha=0.9, edgecolors='k', linewidths=0.4)


def _save_frame_comparison(
    sample: Dict[str, Any],
    preds: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
):
    """Save a single annotated frame with GT + multiple prediction sets."""
    img_t = sample['image']  # (C,H,W)
    kps = sample['keypoints']  # (K,3) in resized pixel space
    img = _to_uint8_rgb(img_t)
    gt_px = kps[:, :2].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

    _overlay_keypoints(ax, gt_px, color='#2ca02c', label='GT')
    # Consistent colors
    color_map = {
        'dekr_untrained': '#1f77b4',
        'dekr_trained': '#ff7f0e',
        'openpose_untrained': '#9467bd',
        'openpose_trained': '#d62728',
        'fusion': '#1f77b4',
        'transformer_fusion': '#ff7f0e',
    }
    for name, coords_px in preds.items():
        _overlay_keypoints(ax, coords_px, color=color_map.get(name, '#7f7f7f'), label=name)

    ax.legend(fontsize=8, loc='lower right', framealpha=0.85)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_fusion_transformer_comparison_artifacts(
    compare: Dict[str, Any],
    image_dataset: VideoFrameKeypointDataset,
    output_dir: str,
    max_frames_to_visualize: int = 12,
):
    """Save time-series + derived 'weights' and a few annotated frames for fusion vs transformer fusion."""
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    with open(outp / 'fusion_transformer_comparison.json', 'w') as f:
        json.dump(compare, f, indent=2)

    N = int(compare.get('num_frames', 0))
    frame_idx = np.arange(N)
    vid_bounds = _get_video_boundaries(image_dataset)

    def _nan(arr):
        a = np.asarray(arr, dtype=np.float32)
        return a

    fusion_err = _nan(compare['per_frame_error_norm'].get('fusion', []))
    transformer_err = _nan(compare['per_frame_error_norm'].get('transformer_fusion', []))

    # Plot errors over time
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame_idx, fusion_err, label='Fusion (MultiModelPoseFusion)', alpha=0.85, linewidth=2)
    ax.plot(frame_idx, transformer_err, label='Transformer Fusion', alpha=0.85, linewidth=2)
    _draw_video_boundaries(ax, vid_bounds, xmin=0, xmax=N - 1)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Per-frame mean error (normalized by couch length)')
    ax.set_title('Fusion vs Transformer Fusion — Per-Frame Mean Error')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outp / 'fusion_transformer_error_over_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # "Weights" per frame: softmax over negative error (lower error => higher weight)
    transformer_w = []
    for i in range(N):
        f_err, t_err = float(fusion_err[i]), float(transformer_err[i])
        if np.isfinite(f_err) and np.isfinite(t_err):
            w_f, w_t = _softmax2(-f_err, -t_err)
            transformer_w.append(float(w_t))
        else:
            transformer_w.append(float('nan'))
    compare.setdefault('derived', {})
    compare['derived']['transformer_fusion_weight'] = transformer_w
    with open(outp / 'fusion_transformer_comparison.json', 'w') as f:
        json.dump(compare, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame_idx, transformer_w, label='Weight on Transformer Fusion vs Fusion', alpha=0.9, linewidth=2)
    _draw_video_boundaries(ax, vid_bounds, xmin=0, xmax=N - 1)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Transformer Fusion weight (softmax over -error)')
    ax.set_ylim([0, 1])
    ax.set_title('Derived Per-Frame "Weight" on Transformer Fusion (lower error ⇒ higher weight)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outp / 'fusion_transformer_weights_over_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Annotated frame comparisons: pick frames with largest improvement (fusion - transformer)
    def _topk_improve(fusion_err, transformer_err, k):
        imp = (fusion_err - transformer_err)
        valid = np.isfinite(imp)
        idxs = np.where(valid)[0]
        if idxs.size == 0:
            return []
        order = np.argsort(imp[idxs])[::-1]  # biggest improvement first
        return idxs[order[:k]].tolist()

    top_frames = _topk_improve(fusion_err, transformer_err, max_frames_to_visualize)

    # Use stored coords (normalized) to overlay on resized images
    W = int(compare.get('img_wh', [256, 256])[0])
    H = int(compare.get('img_wh', [256, 256])[1])
    for rank, gi in enumerate(top_frames):
        sample = image_dataset[int(gi)]
        preds = {}
        cf = compare['coords_norm'].get('fusion', {}).get(str(gi), None)
        ct = compare['coords_norm'].get('transformer_fusion', {}).get(str(gi), None)
        preds['fusion'] = (np.asarray(cf) * np.asarray([W, H])).tolist() if cf is not None else None
        preds['transformer_fusion'] = (np.asarray(ct) * np.asarray([W, H])).tolist() if ct is not None else None

        preds_px = {k: (np.asarray(v, dtype=np.float32) if v is not None else None) for k, v in preds.items()}
        title = f"Frame {gi} (rank {rank+1})"
        _save_frame_comparison(
            sample=sample,
            preds=preds_px,
            out_path=outp / 'frame_comparisons' / f'frame_{gi:06d}.png',
            title=title,
        )


def _save_trained_untrained_comparison_artifacts(
    compare: Dict[str, Any],
    image_dataset: VideoFrameKeypointDataset,
    output_dir: str,
    max_frames_to_visualize: int = 12,
):
    """Save time-series + derived 'weights' and a few annotated frames for trained vs untrained."""
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    with open(outp / 'trained_untrained_comparison.json', 'w') as f:
        json.dump(compare, f, indent=2)

    N = int(compare.get('num_frames', 0))
    frame_idx = np.arange(N)
    vid_bounds = _get_video_boundaries(image_dataset)

    def _nan(arr):
        a = np.asarray(arr, dtype=np.float32)
        return a

    dekr_u = _nan(compare['per_frame_error_norm'].get('dekr_untrained', []))
    dekr_t = _nan(compare['per_frame_error_norm'].get('dekr_trained', []))
    dekr_frz = _nan(compare['per_frame_error_norm'].get('dekr_trained_freeze', []))
    op_u = _nan(compare['per_frame_error_norm'].get('openpose_untrained', []))
    op_t = _nan(compare['per_frame_error_norm'].get('openpose_trained', []))

    # Plot errors over time (downsample for speed if needed)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(frame_idx, dekr_u, label='DEKR untrained (cache)', alpha=0.75)
    ax.plot(frame_idx, dekr_t, label='DEKR trained', alpha=0.85)
    if dekr_frz.shape[0] == N:
        ax.plot(frame_idx, dekr_frz, label='DEKR frozen-backbone trained', alpha=0.85)
    ax.plot(frame_idx, op_u, label='OpenPose untrained (cache)', alpha=0.75)
    ax.plot(frame_idx, op_t, label='OpenPose trained', alpha=0.85)
    _draw_video_boundaries(ax, vid_bounds, xmin=0, xmax=N - 1)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Per-frame mean error (normalized by couch length)')
    ax.set_title('Trained vs Untrained Baselines — Per-Frame Mean Error')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outp / 'trained_untrained_error_over_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # "Weights" per frame: softmax over negative error (lower error => higher weight)
    dekr_w_tr = []
    dekr_w_frz = []
    op_w_tr = []
    for i in range(N):
        du, dt = float(dekr_u[i]), float(dekr_t[i])
        df = float(dekr_frz[i]) if dekr_frz.shape[0] == N else float('nan')
        ou, ot = float(op_u[i]), float(op_t[i])
        if np.isfinite(du) and np.isfinite(dt):
            w_u, w_t = _softmax2(-du, -dt)
            dekr_w_tr.append(float(w_t))
        else:
            dekr_w_tr.append(float('nan'))
        if np.isfinite(du) and np.isfinite(df):
            w_u, w_f = _softmax2(-du, -df)
            dekr_w_frz.append(float(w_f))
        else:
            dekr_w_frz.append(float('nan'))
        if np.isfinite(ou) and np.isfinite(ot):
            w_u, w_t = _softmax2(-ou, -ot)
            op_w_tr.append(float(w_t))
        else:
            op_w_tr.append(float('nan'))
    compare.setdefault('derived', {})
    compare['derived']['dekr_trained_weight'] = dekr_w_tr
    compare['derived']['dekr_trained_freeze_weight'] = dekr_w_frz
    compare['derived']['openpose_trained_weight'] = op_w_tr
    with open(outp / 'trained_untrained_comparison.json', 'w') as f:
        json.dump(compare, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame_idx, dekr_w_tr, label='Weight on DEKR trained vs untrained', alpha=0.9)
    if dekr_frz.shape[0] == N:
        ax.plot(frame_idx, dekr_w_frz, label='Weight on DEKR frozen-backbone vs untrained', alpha=0.9)
    ax.plot(frame_idx, op_w_tr, label='Weight on OpenPose trained vs untrained', alpha=0.9)
    _draw_video_boundaries(ax, vid_bounds, xmin=0, xmax=N - 1)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Trained weight (softmax over -error)')
    ax.set_ylim([0, 1])
    ax.set_title('Derived Per-Frame “Weight” on Trained Model (lower error ⇒ higher weight)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outp / 'trained_untrained_weights_over_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Annotated frame comparisons: pick frames with largest improvement (untrained - trained)
    def _topk_improve(untrained, trained, k):
        imp = (untrained - trained)
        valid = np.isfinite(imp)
        idxs = np.where(valid)[0]
        if idxs.size == 0:
            return []
        order = np.argsort(imp[idxs])[::-1]  # biggest improvement first
        return idxs[order[:k]].tolist()

    top_dekr = _topk_improve(dekr_u, dekr_t, max_frames_to_visualize)
    top_op = _topk_improve(op_u, op_t, max_frames_to_visualize)

    # Use stored coords (normalized) to overlay on resized images
    W = int(compare.get('img_wh', [256, 256])[0])
    H = int(compare.get('img_wh', [256, 256])[1])
    for group_name, frame_list in [('dekr', top_dekr), ('openpose', top_op)]:
        for rank, gi in enumerate(frame_list):
            sample = image_dataset[int(gi)]
            preds = {}
            if group_name == 'dekr':
                cu = compare['coords_norm'].get('dekr_untrained', {}).get(str(gi), None)
                ct = compare['coords_norm'].get('dekr_trained', {}).get(str(gi), None)
                cf = compare['coords_norm'].get('dekr_trained_freeze', {}).get(str(gi), None)
                preds['dekr_untrained'] = (np.asarray(cu) * np.asarray([W, H])).tolist() if cu is not None else None
                preds['dekr_trained'] = (np.asarray(ct) * np.asarray([W, H])).tolist() if ct is not None else None
                if cf is not None:
                    preds['dekr_trained_freeze'] = (np.asarray(cf) * np.asarray([W, H])).tolist()
            else:
                cu = compare['coords_norm'].get('openpose_untrained', {}).get(str(gi), None)
                ct = compare['coords_norm'].get('openpose_trained', {}).get(str(gi), None)
                preds['openpose_untrained'] = (np.asarray(cu) * np.asarray([W, H])).tolist() if cu is not None else None
                preds['openpose_trained'] = (np.asarray(ct) * np.asarray([W, H])).tolist() if ct is not None else None

            preds_px = {k: (np.asarray(v, dtype=np.float32) if v is not None else None) for k, v in preds.items()}
            title = f"{group_name.upper()} frame {gi} (rank {rank+1})"
            _save_frame_comparison(
                sample=sample,
                preds=preds_px,
                out_path=outp / 'frame_comparisons' / group_name / f'frame_{gi:06d}.png',
                title=title,
            )


def evaluate_model(
    checkpoint_path: str,
    list_file: str,
    dataset_root: str,
    batch_size: int = 8,
    device: str = 'cpu',
    num_workers: int = 0,
    pin_memory: bool = False,
    tau: float = 0.05,
    output_json: Optional[str] = None,
    compare_baselines: bool = True,
    trained_dekr_checkpoint: Optional[str] = None,
    trained_dekr_freeze_checkpoint: Optional[str] = None,
    trained_openpose_checkpoint: Optional[str] = None,
    trained_openpose_freeze_checkpoint: Optional[str] = None,
    compare_trained_untrained: bool = False,
    comparison_output_dir: Optional[str] = None,
    max_frames_to_visualize: int = 12,
    use_transformer_fusion: bool = False,
    use_transformer_internal_fusion: bool = False,
    compare_fusion_transformer: bool = False,
    transformer_fusion_checkpoint: Optional[str] = None,
    transformer_model_type: str = 'auto',
    transformer_d_model: int = 256,
    transformer_nhead: int = 8,
    transformer_num_encoder_layers: int = 2,
    transformer_num_decoder_layers: int = 2,
    transformer_dim_feedforward: int = 512,
    transformer_dropout: float = 0.1,
) -> Dict[str, Any]:
    """Evaluate model and compute accuracy metrics.
    
    Returns dict with:
    - accuracy: % of predictions within tau threshold
    - mae: mean absolute error in normalized coords
    - l2_dist_norm: mean L2 distance in normalized coords
    - l2_dist_px: mean L2 distance in pixel coords
    - per_joint_accuracy: dict of accuracy per joint
    - per_joint_l2_dist_px: dict of L2 distance per joint
    - baseline_comparison: accuracy for HRNet, OpenPose, MediaPipe (if compare_baselines=True)
    """
    device_t = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    
    # Load checkpoint
    ck_path = Path(checkpoint_path)
    if not ck_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
    
    d = torch.load(str(ck_path), map_location=device_t)
    
    # Probe dataset for num_joints
    ds_probe = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
    if len(ds_probe) == 0:
        raise RuntimeError("Dataset appears empty")
    K = ds_probe[0]['keypoints'].shape[0]
    print(f"Detected {K} joints from dataset")
    
    # Auto-detect model type from checkpoint path if not explicitly set
    if not use_transformer_fusion and not use_transformer_internal_fusion:
        lower_path = str(ck_path).lower()
        if 'internal' in lower_path:
            use_transformer_internal_fusion = True
            print("Auto-detected INTERNAL transformer fusion model from checkpoint path")
        elif 'transformer' in lower_path:
            use_transformer_fusion = True
            print("Auto-detected transformer fusion model from checkpoint path")
    
    # Create model and load state
    if use_transformer_internal_fusion:
        model = load_transformer_internal_fusion_model(
            checkpoint_path=checkpoint_path,
            device=device_t,
            num_joints=K,
            model_type=transformer_model_type,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if model is None:
            raise RuntimeError(f"Failed to load INTERNAL transformer fusion model from {ck_path}")
    elif use_transformer_fusion:
        model = load_transformer_fusion_model(
            checkpoint_path=checkpoint_path,
            device=device_t,
            num_joints=K,
            model_type=transformer_model_type,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if model is None:
            raise RuntimeError(f"Failed to load transformer fusion model from {ck_path}")
    else:
        model = MultiModelPoseFusion(num_joints=K).to(device_t)
        model.load_state_dict(d['model_state'])
        model.eval()
        print(f"Loaded MultiModelPoseFusion checkpoint from {ck_path}")
    
    # Create dataloader
    loader = build_dataloader_from_list(
        list_file,
        dataset_root,
        batch_size=batch_size,
        output_size=(256, 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        require_caches=False
    )
    
    # Accumulate metrics
    all_correct = []
    all_l2_dist_norm = []
    all_l2_dist_px = []
    all_mae = []
    per_joint_correct = {j: [] for j in range(K)}
    per_joint_l2_dist_px = {j: [] for j in range(K)}
    per_joint_mae = {j: [] for j in range(K)}
    all_confidences = []
    correct_confidences = []
    incorrect_confidences = []
    
    # Load trained DEKR model if checkpoint provided
    trained_dekr_model = None
    if trained_dekr_checkpoint:
        trained_dekr_model, _ = load_trained_dekr_model(trained_dekr_checkpoint, device_t, num_joints=K)
        if trained_dekr_model is None:
            print("Warning: Could not load trained DEKR model, skipping trained DEKR evaluation")

    # Load DEKR checkpoint trained with frozen backbone (optional additional baseline)
    trained_dekr_freeze_model = None
    if trained_dekr_freeze_checkpoint:
        trained_dekr_freeze_model, _ = load_trained_dekr_model(
            trained_dekr_freeze_checkpoint, device_t, num_joints=K
        )
        if trained_dekr_freeze_model is None:
            print("Warning: Could not load frozen-backbone DEKR model, skipping frozen-backbone DEKR evaluation")

    # Load trained OpenPose model if checkpoint provided
    trained_openpose_model = None
    if trained_openpose_checkpoint:
        trained_openpose_model = load_trained_openpose_model(trained_openpose_checkpoint, device_t, num_joints=K)
        if trained_openpose_model is None:
            print("Warning: Could not load trained OpenPose model, skipping trained OpenPose evaluation")

    # Load OpenPose model trained with frozen backbone if checkpoint provided
    trained_openpose_freeze_model = None
    if trained_openpose_freeze_checkpoint:
        trained_openpose_freeze_model = load_trained_openpose_model(trained_openpose_freeze_checkpoint, device_t, num_joints=K)
        if trained_openpose_freeze_model is None:
            print("Warning: Could not load frozen-backbone OpenPose model, skipping frozen-backbone OpenPose evaluation")

    # Create image dataset for trained model inference if needed
    image_dataset = None
    if (trained_dekr_model is not None) or (trained_dekr_freeze_model is not None) or (trained_openpose_model is not None) or (trained_openpose_freeze_model is not None):
        image_dataset = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
        print(f"Created image dataset for trained model inference: {len(image_dataset)} samples")

    # Load transformer fusion model for comparison if requested
    transformer_fusion_model = None
    if compare_fusion_transformer:
        if transformer_fusion_checkpoint is None:
            raise ValueError("--transformer-fusion-checkpoint required when --compare-fusion-transformer is set")
        transformer_fusion_model = load_transformer_fusion_model(
            checkpoint_path=transformer_fusion_checkpoint,
            device=device_t,
            num_joints=K,
            model_type=transformer_model_type,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if transformer_fusion_model is None:
            raise RuntimeError(f"Failed to load transformer fusion model from {transformer_fusion_checkpoint}")
        print(f"Loaded transformer fusion model for comparison from {transformer_fusion_checkpoint}")
        if image_dataset is None:
            image_dataset = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
    
    # Optional: collect per-frame comparison artifacts for trained vs untrained
    compare_payload: Optional[Dict[str, Any]] = None
    if compare_trained_untrained:
        if image_dataset is None:
            image_dataset = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
        N = len(image_dataset)
        compare_payload = {
            'num_frames': N,
            'img_wh': [256, 256],
            'tau': float(tau),
            'per_frame_error_norm': {
                'dekr_untrained': [float('nan')] * N,   # cache "dekr" used as HRNet/DEKR baseline in fusion
                'openpose_untrained': [float('nan')] * N,
                'dekr_trained': [float('nan')] * N,
                'dekr_trained_freeze': [float('nan')] * N,
                'openpose_trained': [float('nan')] * N,
            },
            # store a subset of coords for overlay (normalized coords, key=str(frame_idx))
            'coords_norm': {
                'dekr_untrained': {},
                'openpose_untrained': {},
                'dekr_trained': {},
                'dekr_trained_freeze': {},
                'openpose_trained': {},
            },
        }
    
    # Optional: collect per-frame comparison artifacts for fusion vs transformer fusion
    fusion_transformer_payload: Optional[Dict[str, Any]] = None
    if compare_fusion_transformer and transformer_fusion_model is not None:
        if image_dataset is None:
            image_dataset = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
        N = len(image_dataset)
        fusion_transformer_payload = {
            'num_frames': N,
            'img_wh': [256, 256],
            'tau': float(tau),
            'per_frame_error_norm': {
                'fusion': [float('nan')] * N,
                'transformer_fusion': [float('nan')] * N,
            },
            'coords_norm': {
                'fusion': {},
                'transformer_fusion': {},
            },
        }
    
    # Baseline model metrics (if comparing)
    baseline_metrics = {
        'hrnet': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': []},
        'openpose': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': []},
        'mediapipe': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': []},
    }
    
    # Trained model metrics
    if trained_dekr_model is not None:
        baseline_metrics['dekr_trained'] = {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': []}
    if trained_dekr_freeze_model is not None:
        baseline_metrics['dekr_trained_freeze'] = {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': []}
    if trained_openpose_model is not None:
        baseline_metrics['openpose_trained'] = {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': []}
    if trained_openpose_freeze_model is not None:
        baseline_metrics['openpose_trained_freeze'] = {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': []}
    
    # Per-frame mean error (average across all K joints per frame)
    # These track frames where all K joints were detected
    fusion_frame_errors = []
    baseline_frame_errors = {
        'hrnet': [],
        'openpose': [],
        'mediapipe': [],
    }
    if trained_dekr_model is not None:
        baseline_frame_errors['dekr_trained'] = []
    if trained_dekr_freeze_model is not None:
        baseline_frame_errors['dekr_trained_freeze'] = []
    if trained_openpose_model is not None:
        baseline_frame_errors['openpose_trained'] = []
    if trained_openpose_freeze_model is not None:
        baseline_frame_errors['openpose_trained_freeze'] = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device_t)
            
            # Forward pass (using same normalization as training)
            if use_transformer_internal_fusion:
                # Build internal features from cached coords+conf
                feats_dekr = torch.cat([batch['coords_hrnet'], batch['conf_hrnet']], dim=-1)
                feats_openpose = torch.cat([batch['coords_openpose'], batch['conf_openpose']], dim=-1)
                coords_fused, conf_fused_logits, _ = model(
                    feats_dekr,
                    feats_openpose,
                    batch['coords_mediapipe'],
                    batch['conf_mediapipe'],
                    return_attention_weights=False,
                )
            else:
                coords_fused, conf_fused_logits, _ = model(
                    batch['coords_hrnet'],
                    batch['conf_hrnet'],
                    batch['coords_openpose'],
                    batch['conf_openpose'],
                    batch['coords_mediapipe'],
                    batch['conf_mediapipe'],
                    return_attention_weights=False,
                )
            
            # Get ground truth
            coords_gt = batch['coords_gt']  # (B, K, 2) normalized
            coords_gt_px = batch.get('coords_gt_px')  # (B, K, 2) pixels
            img_wh = batch.get('img_wh')  # (B, 2)
            couch_len = batch.get('couch_len')  # (B,)
            mask_hr = batch.get('mask_hrnet')  # (B, K, 1)
            mask_op = batch.get('mask_openpose')
            mask_mp = batch.get('mask_mediapipe')
            
            # Compute per-joint accuracy
            mask_any = ((mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1)) > 0.0)  # (B, K)
            
            # APPROACH 2: Everything is now in 256x256 space
            # coords_fused is normalized [0,1] based on 256x256
            # coords_gt_px is in 256x256 pixel space
            # Convert fusion predictions to pixel space using img_wh from batch (consistent with training)
            # CRITICAL: Force img_wh to [256, 256] to match training coordinate space
            # (Some batches might have img_wh set to original resolution, causing coordinate space mismatch)
            img_wh_tensor = torch.tensor([256.0, 256.0], device=coords_fused.device).view(1, 1, 2)  # (1, 1, 2)
            if img_wh is not None:
                # Verify img_wh is [256, 256] - if not, warn and use [256, 256] instead
                img_wh_check = img_wh[0].cpu().numpy()
                if not (abs(img_wh_check[0] - 256.0) < 1.0 and abs(img_wh_check[1] - 256.0) < 1.0):
                    print(f"WARNING: img_wh in batch is {img_wh_check}, expected [256, 256]. Using [256, 256] instead.")
            else:
                    img_wh_tensor = img_wh.view(-1, 1, 2)  # (B, 1, 2)
            coords_fused_px = coords_fused * img_wh_tensor  # (B, K, 2) in pixel space
            
            # L2 distance in pixels (both in 256x256 space)
            l2_dist_px = torch.norm(coords_fused_px - coords_gt_px, dim=-1)  # (B, K)
            
            # Normalize by couch length (256x256 diagonal)
            dists_norm = l2_dist_px / couch_len.view(-1, 1)  # (B, K)
            
            # Accuracy (within tau)
            correct = (dists_norm < tau).float()  # (B, K)
            
            # Confidence scores
            conf_fused = torch.sigmoid(conf_fused_logits.squeeze(-1))  # (B, K)
            
            # Baseline model distances (if comparing)
            # NOTE: All models and ground truth are now in 256x256 space
            if compare_baselines:
                coords_hr = batch['coords_hrnet']  # Normalized (0-1) in 256x256 space
                coords_op = batch['coords_openpose']  # Normalized (0-1) in 256x256 space
                coords_mp = batch['coords_mediapipe']  # Normalized (0-1) in 256x256 space
                
                # Convert to pixel space using img_wh (consistent with training)
                coords_hr_px = coords_hr * img_wh_tensor  # (B, K, 2) in pixel space
                coords_op_px = coords_op * img_wh_tensor
                coords_mp_px = coords_mp * img_wh_tensor
                
                # Compare directly in 256x256 space (coords_gt_px is also in 256x256)
                l2_dist_hr = torch.norm(coords_hr_px - coords_gt_px, dim=-1)
                l2_dist_op = torch.norm(coords_op_px - coords_gt_px, dim=-1)
                l2_dist_mp = torch.norm(coords_mp_px - coords_gt_px, dim=-1)
                
                dists_norm_hr = l2_dist_hr / couch_len.view(-1, 1)
                dists_norm_op = l2_dist_op / couch_len.view(-1, 1)
                dists_norm_mp = l2_dist_mp / couch_len.view(-1, 1)
                
                correct_hr = (dists_norm_hr < tau).float()
                correct_op = (dists_norm_op < tau).float()
                correct_mp = (dists_norm_mp < tau).float()

            # Store per-frame errors for fusion vs transformer fusion comparison
            if fusion_transformer_payload is not None:
                B = coords_gt.shape[0]
                for b in range(B):
                    gi = batch_idx * batch_size + b
                    if gi >= fusion_transformer_payload['num_frames']:
                        continue
                    # Store fusion model errors and coords (only when all joints present)
                    if mask_any[b].sum().item() == K:
                        err = float(dists_norm[b].mean().item())
                        fusion_transformer_payload['per_frame_error_norm']['fusion'][gi] = err
                        fusion_transformer_payload['coords_norm']['fusion'][str(gi)] = coords_fused[b].detach().cpu().numpy().tolist()
            
            # Store per-frame baseline errors for trained-vs-untrained comparison (only when all joints present)
            if compare_payload is not None:
                B = coords_gt.shape[0]
                for b in range(B):
                    gi = batch_idx * batch_size + b
                    if gi >= compare_payload['num_frames']:
                        continue
                    # DEKR "untrained" is the cache baseline stored in coords_hrnet
                    if mask_hr[b].squeeze(-1).sum().item() == K:
                        err = float((dists_norm_hr[b].mean().item()) if compare_baselines else float('nan'))
                        compare_payload['per_frame_error_norm']['dekr_untrained'][gi] = err
                        compare_payload['coords_norm']['dekr_untrained'][str(gi)] = batch['coords_hrnet'][b].detach().cpu().numpy().tolist()
                    # OpenPose untrained
                    if compare_baselines and mask_op[b].squeeze(-1).sum().item() == K:
                        err = float(dists_norm_op[b].mean().item())
                        compare_payload['per_frame_error_norm']['openpose_untrained'][gi] = err
                        compare_payload['coords_norm']['openpose_untrained'][str(gi)] = batch['coords_openpose'][b].detach().cpu().numpy().tolist()
            
            # Accumulate metrics only where at least one model has the joint
            for b in range(coords_fused.shape[0]):
                # Per-frame mean error computation (for frames where all K joints detected)
                fusion_frame_dists = []
                hr_frame_dists = []
                op_frame_dists = []
                mp_frame_dists = []
                
                for j in range(K):
                    if mask_any[b, j] > 0:
                        is_correct = bool(correct[b, j].item())
                        all_correct.append(is_correct)
                        all_l2_dist_px.append(float(l2_dist_px[b, j].item()))
                        all_l2_dist_norm.append(float(dists_norm[b, j].item()))
                        
                        mae = float(torch.abs(coords_fused[b, j] - coords_gt[b, j]).mean().item())
                        all_mae.append(mae)
                        
                        per_joint_correct[j].append(is_correct)
                        per_joint_l2_dist_px[j].append(float(l2_dist_px[b, j].item()))
                        per_joint_mae[j].append(mae)
                        
                        conf_score = float(conf_fused[b, j].item())
                        all_confidences.append(conf_score)
                        if is_correct:
                            correct_confidences.append(conf_score)
                        else:
                            incorrect_confidences.append(conf_score)
                        
                        # For per-frame mean, collect normalized distances for this joint
                        fusion_frame_dists.append(float(dists_norm[b, j].item()))
                        
                        # Baseline comparisons
                        if compare_baselines:
                            if mask_hr[b, j] > 0:
                                baseline_metrics['hrnet']['correct'].append(bool(correct_hr[b, j].item()))
                                baseline_metrics['hrnet']['l2_dist_px'].append(float(l2_dist_hr[b, j].item()))
                                baseline_metrics['hrnet']['l2_dist_norm'].append(float(dists_norm_hr[b, j].item()))
                                hr_frame_dists.append(float(dists_norm_hr[b, j].item()))
                            if mask_op[b, j] > 0:
                                baseline_metrics['openpose']['correct'].append(bool(correct_op[b, j].item()))
                                baseline_metrics['openpose']['l2_dist_px'].append(float(l2_dist_op[b, j].item()))
                                baseline_metrics['openpose']['l2_dist_norm'].append(float(dists_norm_op[b, j].item()))
                                op_frame_dists.append(float(dists_norm_op[b, j].item()))
                            if mask_mp[b, j] > 0:
                                baseline_metrics['mediapipe']['correct'].append(bool(correct_mp[b, j].item()))
                                baseline_metrics['mediapipe']['l2_dist_px'].append(float(l2_dist_mp[b, j].item()))
                                baseline_metrics['mediapipe']['l2_dist_norm'].append(float(dists_norm_mp[b, j].item()))
                                mp_frame_dists.append(float(dists_norm_mp[b, j].item()))
                
                # Compute per-frame mean error (only if all K joints were detected)
                if len(fusion_frame_dists) == K:
                    fusion_frame_errors.append(float(np.mean(fusion_frame_dists)))
                
                if compare_baselines:
                    if len(hr_frame_dists) == K:
                        baseline_frame_errors['hrnet'].append(float(np.mean(hr_frame_dists)))
                    if len(op_frame_dists) == K:
                        baseline_frame_errors['openpose'].append(float(np.mean(op_frame_dists)))
                    if len(mp_frame_dists) == K:
                        baseline_frame_errors['mediapipe'].append(float(np.mean(mp_frame_dists)))
    
    # Evaluate trained DEKR model if provided
    if trained_dekr_model is not None and image_dataset is not None:
        print("\nEvaluating trained DEKR model...")
        from torchvision import transforms as T
        
        # ImageNet normalization for DEKR
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        dekr_trained_frame_dists = []
        sample_idx = 0
        
        # Create a dataloader for images
        image_loader = DataLoader(
            image_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        for img_batch_idx, sample_batch in enumerate(image_loader):
            # Get images and ground truth
            images = sample_batch['image']  # (B, 3, H, W) - tensor in [0, 1] range
            keypoints_gt = sample_batch['keypoints']  # (B, K, 3) - (x, y, conf) in pixel space (resized image space)
            
            B = images.shape[0]
            H, W = images.shape[2], images.shape[3]  # Should be (256, 256) based on output_size
            
            # Get metadata from batch
            meta_list = sample_batch.get('meta', [])
            if not isinstance(meta_list, list):
                meta_list = [meta_list] if meta_list else [{}] * B
            
            img_wh_list = []
            couch_len_list = []
            
            for b in range(B):
                meta = meta_list[b] if b < len(meta_list) else {}
                # Image size is the resized size (256x256)
                img_wh_list.append([W, H])
                
                # Use 256x256 diagonal as couch_len (since everything is in 256x256 space)
                couch_len_256 = np.sqrt(float(W)**2 + float(H)**2)  # sqrt(256^2 + 256^2)
                couch_len_list.append(couch_len_256)
            
            img_wh = torch.tensor(img_wh_list, dtype=torch.float32)  # (B, 2)
            couch_len = torch.tensor(couch_len_list, dtype=torch.float32)  # (B,)
            
            # Normalize images for DEKR (ImageNet stats)
            # Images are in [0, 1], convert to ImageNet normalized
            images_norm = normalize(images)  # (B, 3, H, W)
            images_norm = images_norm.to(device_t)
            img_wh = img_wh.to(device_t)
            couch_len = couch_len.to(device_t)
            
            # Run trained DEKR model
            with torch.no_grad():
                coords_dekr_trained, conf_dekr_trained = trained_dekr_model(images_norm)
                # coords_dekr_trained: (B, K, 2) normalized [0, 1]
                # conf_dekr_trained: (B, K) confidence scores
            
            # Convert to pixel space
            img_wh_tensor = img_wh.view(-1, 1, 2)  # (B, 1, 2)
            coords_dekr_trained_px = coords_dekr_trained * img_wh_tensor  # (B, K, 2)
            
            # Get ground truth in pixel space
            coords_gt_px = keypoints_gt[:, :, :2].to(device_t)  # (B, K, 2)
            
            # Compute distances
            l2_dist_px = torch.norm(coords_dekr_trained_px - coords_gt_px, dim=-1)  # (B, K)
            dists_norm = l2_dist_px / couch_len.view(-1, 1)  # (B, K)
            correct = (dists_norm < tau).float()  # (B, K)
            
            # Accumulate metrics
            for b in range(B):
                if compare_payload is not None:
                    gi = img_batch_idx * batch_size + b
                    if gi < compare_payload['num_frames'] and (keypoints_gt[b, :, 2] > 0.5).sum().item() == K:
                        compare_payload['per_frame_error_norm']['dekr_trained'][gi] = float(dists_norm[b].mean().item())
                        compare_payload['coords_norm']['dekr_trained'][str(gi)] = coords_dekr_trained[b].detach().cpu().numpy().tolist()
                dekr_trained_frame_dists_batch = []
                for j in range(K):
                    # Only count if confidence > 0 (joint is present in GT)
                    if keypoints_gt[b, j, 2] > 0.5:  # GT confidence threshold
                        baseline_metrics['dekr_trained']['correct'].append(bool(correct[b, j].item()))
                        baseline_metrics['dekr_trained']['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                        baseline_metrics['dekr_trained']['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                        dekr_trained_frame_dists_batch.append(float(dists_norm[b, j].item()))
                
                # Per-frame mean error (if all K joints detected)
                if len(dekr_trained_frame_dists_batch) == K:
                    baseline_frame_errors['dekr_trained'].append(float(np.mean(dekr_trained_frame_dists_batch)))
        
        print(f"Trained DEKR evaluation complete: {len(baseline_metrics['dekr_trained']['correct'])} samples")

    # Evaluate DEKR model trained with frozen backbone (if provided)
    if trained_dekr_freeze_model is not None and image_dataset is not None:
        print("\nEvaluating DEKR (Frozen Backbone) model...")
        from torchvision import transforms as T

        # ImageNet normalization for DEKR
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        image_loader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        for img_batch_idx, sample_batch in enumerate(image_loader):
            images = sample_batch['image']  # (B, 3, H, W) in [0, 1]
            keypoints_gt = sample_batch['keypoints']  # (B, K, 3) in pixel space of resized image

            B = images.shape[0]
            H, W = images.shape[2], images.shape[3]

            img_wh_list = [[W, H] for _ in range(B)]
            couch_len_256 = np.sqrt(float(W) ** 2 + float(H) ** 2)
            couch_len_list = [couch_len_256 for _ in range(B)]

            img_wh = torch.tensor(img_wh_list, dtype=torch.float32).to(device_t)
            couch_len = torch.tensor(couch_len_list, dtype=torch.float32).to(device_t)

            images_norm = normalize(images).to(device_t)

            with torch.no_grad():
                coords_dekr_frz, _conf_dekr_frz = trained_dekr_freeze_model(images_norm)

            coords_dekr_frz_px = coords_dekr_frz * img_wh.view(-1, 1, 2)
            coords_gt_px = keypoints_gt[:, :, :2].to(device_t)

            l2_dist_px = torch.norm(coords_dekr_frz_px - coords_gt_px, dim=-1)
            dists_norm = l2_dist_px / couch_len.view(-1, 1)
            correct = (dists_norm < tau).float()

            for b in range(B):
                if compare_payload is not None:
                    gi = img_batch_idx * batch_size + b
                    if gi < compare_payload['num_frames'] and (keypoints_gt[b, :, 2] > 0.5).sum().item() == K:
                        compare_payload['per_frame_error_norm']['dekr_trained_freeze'][gi] = float(
                            dists_norm[b].mean().item()
                        )
                        compare_payload['coords_norm']['dekr_trained_freeze'][str(gi)] = coords_dekr_frz[
                            b
                        ].detach().cpu().numpy().tolist()

                frame_dists = []
                for j in range(K):
                    if keypoints_gt[b, j, 2] > 0.5:
                        baseline_metrics['dekr_trained_freeze']['correct'].append(bool(correct[b, j].item()))
                        baseline_metrics['dekr_trained_freeze']['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                        baseline_metrics['dekr_trained_freeze']['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                        frame_dists.append(float(dists_norm[b, j].item()))
                if len(frame_dists) == K:
                    baseline_frame_errors['dekr_trained_freeze'].append(float(np.mean(frame_dists)))

        print(
            f"DEKR (Frozen Backbone) evaluation complete: {len(baseline_metrics['dekr_trained_freeze']['correct'])} samples"
        )

    # Evaluate trained OpenPose model if provided
    if trained_openpose_model is not None and image_dataset is not None:
        print("\nEvaluating trained OpenPose model...")
        image_loader = DataLoader(
            image_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        for img_batch_idx, sample_batch in enumerate(image_loader):
            images = sample_batch['image']          # (B, 3, H, W)
            keypoints_gt = sample_batch['keypoints']  # (B, K, 3) in pixel space of resized image

            B = images.shape[0]
            H, W = images.shape[2], images.shape[3]

            meta_list = sample_batch.get('meta', [])
            if not isinstance(meta_list, list):
                meta_list = [meta_list] if meta_list else [{}] * B

            img_wh_list = []
            couch_len_list = []
            for b in range(B):
                meta = meta_list[b] if b < len(meta_list) else {}
                img_wh_list.append([W, H])
                # Use 256x256 diagonal as couch_len (since everything is in 256x256 space)
                couch_len_256 = np.sqrt(float(W)**2 + float(H)**2)  # sqrt(256^2 + 256^2)
                couch_len_list.append(couch_len_256)

            img_wh = torch.tensor(img_wh_list, dtype=torch.float32).to(device_t)
            couch_len = torch.tensor(couch_len_list, dtype=torch.float32).to(device_t)
            images = images.to(device_t)

            with torch.no_grad():
                coords_op_trained, conf_op_trained = trained_openpose_model(images)

            img_wh_tensor = img_wh.view(-1, 1, 2)
            coords_op_trained_px = coords_op_trained * img_wh_tensor
            coords_gt_px = keypoints_gt[:, :, :2].to(device_t)

            l2_dist_px = torch.norm(coords_op_trained_px - coords_gt_px, dim=-1)
            dists_norm = l2_dist_px / couch_len.view(-1, 1)
            correct = (dists_norm < tau).float()

            for b in range(B):
                if compare_payload is not None:
                    gi = img_batch_idx * batch_size + b
                    if gi < compare_payload['num_frames'] and (keypoints_gt[b, :, 2] > 0.5).sum().item() == K:
                        compare_payload['per_frame_error_norm']['openpose_trained'][gi] = float(dists_norm[b].mean().item())
                        compare_payload['coords_norm']['openpose_trained'][str(gi)] = coords_op_trained[b].detach().cpu().numpy().tolist()
                frame_dists = []
                for j in range(K):
                    if keypoints_gt[b, j, 2] > 0.5:
                        baseline_metrics['openpose_trained']['correct'].append(bool(correct[b, j].item()))
                        baseline_metrics['openpose_trained']['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                        baseline_metrics['openpose_trained']['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                        frame_dists.append(float(dists_norm[b, j].item()))
                if len(frame_dists) == K:
                    baseline_frame_errors['openpose_trained'].append(float(np.mean(frame_dists)))

        print(f"Trained OpenPose evaluation complete: {len(baseline_metrics['openpose_trained']['correct'])} samples")

    # Evaluate OpenPose model trained with frozen backbone (if provided)
    if trained_openpose_freeze_model is not None and image_dataset is not None:
        print("\nEvaluating OpenPose (Frozen Backbone) model...")
        image_loader = DataLoader(
            image_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        for img_batch_idx, sample_batch in enumerate(image_loader):
            images = sample_batch['image']          # (B, 3, H, W)
            keypoints_gt = sample_batch['keypoints']  # (B, K, 3) in pixel space of resized image

            B = images.shape[0]
            H, W = images.shape[2], images.shape[3]

            img_wh_list = [[W, H] for _ in range(B)]
            couch_len_256 = np.sqrt(float(W)**2 + float(H)**2)
            couch_len_list = [couch_len_256 for _ in range(B)]

            img_wh = torch.tensor(img_wh_list, dtype=torch.float32).to(device_t)
            couch_len = torch.tensor(couch_len_list, dtype=torch.float32).to(device_t)
            images = images.to(device_t)

            with torch.no_grad():
                coords_op_frz, _conf_op_frz = trained_openpose_freeze_model(images)

            img_wh_tensor = img_wh.view(-1, 1, 2)
            coords_op_frz_px = coords_op_frz * img_wh_tensor
            coords_gt_px = keypoints_gt[:, :, :2].to(device_t)

            l2_dist_px = torch.norm(coords_op_frz_px - coords_gt_px, dim=-1)
            dists_norm = l2_dist_px / couch_len.view(-1, 1)
            correct = (dists_norm < tau).float()

            for b in range(B):
                frame_dists = []
                for j in range(K):
                    if keypoints_gt[b, j, 2] > 0.5:
                        baseline_metrics['openpose_trained_freeze']['correct'].append(bool(correct[b, j].item()))
                        baseline_metrics['openpose_trained_freeze']['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                        baseline_metrics['openpose_trained_freeze']['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                        frame_dists.append(float(dists_norm[b, j].item()))
                if len(frame_dists) == K:
                    baseline_frame_errors['openpose_trained_freeze'].append(float(np.mean(frame_dists)))

        print(f"OpenPose (Frozen Backbone) evaluation complete: {len(baseline_metrics['openpose_trained_freeze']['correct'])} samples")
    
    # Evaluate transformer fusion model for comparison if provided
    if transformer_fusion_model is not None and fusion_transformer_payload is not None:
        print("\nEvaluating transformer fusion model for comparison...")
        for batch_idx, batch in enumerate(loader):
            # Move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device_t)
            
            # Forward pass with transformer fusion model
            coords_transformer, conf_transformer_logits, _ = transformer_fusion_model(
                batch['coords_hrnet'],
                batch['conf_hrnet'],
                batch['coords_openpose'],
                batch['conf_openpose'],
                batch['coords_mediapipe'],
                batch['conf_mediapipe'],
                return_attention_weights=False,
            )
            
            # Get ground truth
            coords_gt = batch['coords_gt']
            coords_gt_px = batch.get('coords_gt_px')
            img_wh = batch.get('img_wh')
            couch_len = batch.get('couch_len')
            mask_hr = batch.get('mask_hrnet')
            mask_op = batch.get('mask_openpose')
            mask_mp = batch.get('mask_mediapipe')
            mask_any = ((mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1)) > 0.0)
            
            # Convert to pixel space
            img_wh_tensor = torch.tensor([256.0, 256.0], device=coords_transformer.device).view(1, 1, 2)
            coords_transformer_px = coords_transformer * img_wh_tensor
            
            # Compute distances
            l2_dist_px = torch.norm(coords_transformer_px - coords_gt_px, dim=-1)
            dists_norm = l2_dist_px / couch_len.view(-1, 1)
            
            # Store per-frame errors and coords
            B = coords_gt.shape[0]
            for b in range(B):
                gi = batch_idx * batch_size + b
                if gi >= fusion_transformer_payload['num_frames']:
                    continue
                # Only store when all joints present
                if mask_any[b].sum().item() == K:
                    err = float(dists_norm[b].mean().item())
                    fusion_transformer_payload['per_frame_error_norm']['transformer_fusion'][gi] = err
                    fusion_transformer_payload['coords_norm']['transformer_fusion'][str(gi)] = coords_transformer[b].detach().cpu().numpy().tolist()
        
        print(f"Transformer fusion evaluation complete for comparison")
    
    # Compute summary stats
    if len(all_correct) == 0:
        print("WARNING: No valid predictions (mask_any was all zeros)")
        return {}
    
    accuracy = float(np.mean(all_correct))
    mae = float(np.mean(all_mae))
    l2_dist_norm = float(np.mean(all_l2_dist_norm))
    l2_dist_px = float(np.mean(all_l2_dist_px))
    
    # Confidence stats
    mean_conf_all = float(np.mean(all_confidences))
    mean_conf_correct = float(np.mean(correct_confidences)) if correct_confidences else 0.0
    mean_conf_incorrect = float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0
    
    results = {
        'accuracy': accuracy,
        'mae': mae,
        'l2_dist_norm': l2_dist_norm,
        'l2_dist_px': l2_dist_px,
        'mean_confidence': mean_conf_all,
        'mean_confidence_correct': mean_conf_correct,
        'mean_confidence_incorrect': mean_conf_incorrect,
        'num_samples': len(all_correct),
        'per_joint': {},
        'per_frame_mean_error': {},
    }
    
    # Per-frame mean error statistics (similar to OpenPose analysis)
    if fusion_frame_errors:
        results['per_frame_mean_error']['fusion'] = {
            'mean': float(np.mean(fusion_frame_errors)),
            'std': float(np.std(fusion_frame_errors)),
            'median': float(np.median(fusion_frame_errors)),
            'max': float(np.max(fusion_frame_errors)),
            'min': float(np.min(fusion_frame_errors)),
            'count': len(fusion_frame_errors),
        }
    
    for j in range(K):
        if len(per_joint_correct[j]) > 0:
            results['per_joint'][f'joint_{j}'] = {
                'accuracy': float(np.mean(per_joint_correct[j])),
                'l2_dist_px': float(np.mean(per_joint_l2_dist_px[j])),
                'mae': float(np.mean(per_joint_mae[j])),
                'count': len(per_joint_correct[j])
            }
    
    # Compute baseline comparisons
    if compare_baselines:
        results['baselines'] = {}
        for model_name, metrics in baseline_metrics.items():
            if len(metrics['correct']) > 0:
                results['baselines'][model_name] = {
                    'accuracy': float(np.mean(metrics['correct'])),
                    'l2_dist_px': float(np.mean(metrics['l2_dist_px'])),
                    'l2_dist_norm': float(np.mean(metrics['l2_dist_norm'])),
                    'count': len(metrics['correct'])
                }
        
        # Per-frame mean error for baselines
        for model_name, frame_errs in baseline_frame_errors.items():
            if frame_errs:
                results['per_frame_mean_error'][model_name] = {
                    'mean': float(np.mean(frame_errs)),
                    'std': float(np.std(frame_errs)),
                    'median': float(np.median(frame_errs)),
                    'max': float(np.max(frame_errs)),
                    'min': float(np.min(frame_errs)),
                    'count': len(frame_errs),
                }
    
    # Print results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS (threshold tau={tau})")
    print("="*60)
    print(f"FUSION MODEL:")
    print(f"  Accuracy:                 {accuracy*100:.2f}%")
    print(f"  Mean L2 distance (px):    {l2_dist_px:.3f}")
    print(f"  Mean L2 distance (norm):  {l2_dist_norm:.4f}")
    print(f"  Mean Absolute Error:      {mae:.4f}")
    print(f"  Total samples:            {len(all_correct)}")
    
    # Per-frame mean error (OpenPose-style analysis)
    if 'per_frame_mean_error' in results and 'fusion' in results['per_frame_mean_error']:
        fm = results['per_frame_mean_error']['fusion']
        print(f"\n  Per-Frame Mean Error (avg of {K} joints per frame):")
        print(f"    Mean:                   {fm['mean']:.4f}")
        print(f"    Std Dev:                {fm['std']:.4f}")
        print(f"    Median:                 {fm['median']:.4f}")
        print(f"    Min/Max:                {fm['min']:.4f} / {fm['max']:.4f}")
        print(f"    Frames (all joints):    {fm['count']}")
    
    if compare_baselines and 'baselines' in results:
        print(f"\nBASELINE MODELS:")
        for model_name, metrics in results['baselines'].items():
            if model_name == 'dekr_trained':
                model_display = 'DEKR (Trained)'
            elif model_name == 'dekr_trained_freeze':
                model_display = 'DEKR (Frozen Backbone)'
            elif model_name == 'openpose_trained':
                model_display = 'OpenPose (Trained)'
            elif model_name == 'openpose_trained_freeze':
                model_display = 'OpenPose (Frozen Backbone)'
            else:
                model_display = model_name.upper().replace('HRNET', 'HRNet/DEKR')
            print(f"  {model_display}:")
            print(f"    Accuracy:               {metrics['accuracy']*100:.2f}%")
            print(f"    Mean L2 distance (px):  {metrics['l2_dist_px']:.3f}")
            print(f"    Mean L2 distance (norm):{metrics['l2_dist_norm']:.4f}")
            print(f"    Samples:                {metrics['count']}")
            
            # Per-frame mean error for this baseline
            if model_name in results['per_frame_mean_error']:
                fm = results['per_frame_mean_error'][model_name]
                print(f"    Per-Frame Mean Error:")
                print(f"      Mean:                 {fm['mean']:.4f}")
                print(f"      Std Dev:              {fm['std']:.4f}")
                print(f"      Frames (all joints):  {fm['count']}")
        
        # Compute improvements
        print(f"\nIMPROVEMENT vs BASELINES (Joint-Level Accuracy):")
        for model_name, metrics in results['baselines'].items():
            improvement = (accuracy - metrics['accuracy']) * 100
            if model_name == 'dekr_trained':
                model_display = 'DEKR (Trained)'
            elif model_name == 'dekr_trained_freeze':
                model_display = 'DEKR (Frozen Backbone)'
            elif model_name == 'openpose_trained':
                model_display = 'OpenPose (Trained)'
            elif model_name == 'openpose_trained_freeze':
                model_display = 'OpenPose (Frozen Backbone)'
            else:
                model_display = model_name.upper().replace('HRNET', 'HRNet/DEKR')
            sign = "+" if improvement >= 0 else ""
            print(f"  vs {model_display}: {sign}{improvement:.2f} percentage points")
        
        # Per-frame mean error improvements
        if 'fusion' in results['per_frame_mean_error']:
            fusion_frame_mean = results['per_frame_mean_error']['fusion']['mean']
            print(f"\nIMPROVEMENT vs BASELINES (Per-Frame Mean Error):")
            for model_name in baseline_frame_errors.keys():
                if model_name in results['per_frame_mean_error']:
                    baseline_frame_mean = results['per_frame_mean_error'][model_name]['mean']
                    improvement = baseline_frame_mean - fusion_frame_mean
                    pct_improvement = (improvement / baseline_frame_mean * 100) if baseline_frame_mean > 0 else 0
                    if model_name == 'dekr_trained':
                        model_display = 'DEKR (Trained)'
                    elif model_name == 'dekr_trained_freeze':
                        model_display = 'DEKR (Frozen Backbone)'
                    elif model_name == 'openpose_trained':
                        model_display = 'OpenPose (Trained)'
                    elif model_name == 'openpose_trained_freeze':
                        model_display = 'OpenPose (Frozen Backbone)'
                    else:
                        model_display = model_name.upper().replace('HRNET', 'HRNet/DEKR')
                    sign = "+" if improvement >= 0 else ""
                    print(f"  vs {model_display}: {sign}{improvement:.4f} ({sign}{pct_improvement:.1f}%)")
    
    print(f"\nCONFIDENCE SCORE STATS:")
    print(f"  Mean (all):             {mean_conf_all:.4f}")
    print(f"  Mean (correct):         {mean_conf_correct:.4f}")
    print(f"  Mean (incorrect):       {mean_conf_incorrect:.4f}")
    print(f"  Calibration gap:        {mean_conf_correct - accuracy:.4f}")
    print("="*60)
    
    # Save to JSON if requested
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    # Save trained-vs-untrained comparison artifacts if requested
    if compare_payload is not None:
        out_dir = comparison_output_dir or (Path(output_json).parent / 'trained_untrained_comparison' if output_json else Path('trained_untrained_comparison'))
        _save_trained_untrained_comparison_artifacts(
            compare=compare_payload,
            image_dataset=image_dataset,
            output_dir=str(out_dir),
            max_frames_to_visualize=int(max_frames_to_visualize),
        )
        print(f"Trained-vs-untrained comparison artifacts saved to {out_dir}")
    
    # Save fusion-vs-transformer-fusion comparison artifacts if requested
    if fusion_transformer_payload is not None:
        out_dir = comparison_output_dir or (Path(output_json).parent / 'fusion_transformer_comparison' if output_json else Path('fusion_transformer_comparison'))
        _save_fusion_transformer_comparison_artifacts(
            compare=fusion_transformer_payload,
            image_dataset=image_dataset,
            output_dir=str(out_dir),
            max_frames_to_visualize=int(max_frames_to_visualize),
        )
        print(f"Fusion-vs-transformer-fusion comparison artifacts saved to {out_dir}")
    
    return results


def extract_importance_weights(
    checkpoint_path: str,
    list_file: str,
    dataset_root: str,
    batch_size: int = 8,
    device: str = 'cpu',
    num_workers: int = 0,
    pin_memory: bool = False,
    output_file: Optional[str] = None,
    use_transformer_fusion: bool = False,
    use_transformer_internal_fusion: bool = False,
    transformer_model_type: str = 'auto',
    transformer_d_model: int = 256,
    transformer_nhead: int = 8,
    transformer_num_encoder_layers: int = 2,
    transformer_num_decoder_layers: int = 2,
    transformer_dim_feedforward: int = 512,
    transformer_dropout: float = 0.1,
) -> Dict[str, Any]:
    """Extract importance weights from attention mechanism for test set.
    
    Returns dict with:
    - importance_weights: List of (B, K, 3) arrays - importance for each model per joint
    - frame_info: List of frame metadata (batch_idx, frame_idx, etc.)
    - model_names: ['hrnet', 'openpose', 'mediapipe']
    - model_type: 'fusion' or 'transformer_fusion'
    """
    device_t = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    
    # Load checkpoint
    ck_path = Path(checkpoint_path)
    if not ck_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
    
    d = torch.load(str(ck_path), map_location=device_t)
    
    # Probe dataset for num_joints
    ds_probe = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
    if len(ds_probe) == 0:
        raise RuntimeError("Dataset appears empty")
    K = ds_probe[0]['keypoints'].shape[0]
    print(f"Detected {K} joints from dataset")
    
    # Auto-detect model type from checkpoint path if not explicitly set
    if not use_transformer_fusion and not use_transformer_internal_fusion:
        lower_path = str(ck_path).lower()
        if 'internal' in lower_path:
            use_transformer_internal_fusion = True
            print("Auto-detected INTERNAL transformer fusion model from checkpoint path")
        elif 'transformer' in lower_path:
            use_transformer_fusion = True
            print("Auto-detected transformer fusion model from checkpoint path")
    
    # Create model and load state
    if use_transformer_internal_fusion:
        model = load_transformer_internal_fusion_model(
            checkpoint_path=checkpoint_path,
            device=device_t,
            num_joints=K,
            model_type=transformer_model_type,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if model is None:
            raise RuntimeError(f"Failed to load INTERNAL transformer fusion model from {ck_path}")
        model_type = 'transformer_internal_fusion'
    elif use_transformer_fusion:
        model = load_transformer_fusion_model(
            checkpoint_path=checkpoint_path,
            device=device_t,
            num_joints=K,
            model_type=transformer_model_type,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if model is None:
            raise RuntimeError(f"Failed to load transformer fusion model from {ck_path}")
        model_type = 'transformer_fusion'
    else:
        model = MultiModelPoseFusion(num_joints=K).to(device_t)
        model.load_state_dict(d['model_state'])
        model.eval()
        model_type = 'fusion'
    print(f"Loaded {model_type} checkpoint from {ck_path}")
    
    # Create dataloader
    loader = build_dataloader_from_list(
        list_file,
        dataset_root,
        batch_size=batch_size,
        output_size=(256, 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        require_caches=False
    )
    
    # Store importance weights and frame info
    all_importance_weights = []  # List of (B, K, 3) arrays
    frame_info = []  # List of dicts with batch_idx, frame_idx, etc.
    global_sample_offset = 0  # robust linear index (not batch_idx * batch_size)

    model_names = ['hrnet', 'openpose', 'mediapipe']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device_t)
            
            # Forward pass with attention weights (using same normalization as training)
            if use_transformer_internal_fusion:
                feats_dekr = torch.cat([batch['coords_hrnet'], batch['conf_hrnet']], dim=-1)
                feats_openpose = torch.cat([batch['coords_openpose'], batch['conf_openpose']], dim=-1)
                coords_fused, conf_fused_logits, attn_weights = model(
                    feats_dekr,
                    feats_openpose,
                    batch['coords_mediapipe'],
                    batch['conf_mediapipe'],
                    return_attention_weights=True,
                )
            else:
                coords_fused, conf_fused_logits, attn_weights = model(
                    batch['coords_hrnet'],
                    batch['conf_hrnet'],
                    batch['coords_openpose'],
                    batch['conf_openpose'],
                    batch['coords_mediapipe'],
                    batch['conf_mediapipe'],
                    return_attention_weights=True,
                )
            
            if model_type in ('transformer_fusion', 'transformer_internal_fusion'):
                # Transformer fusion returns (B, K, 3) directly - per-model attention weights
                # Already normalized to sum to 1 across models
                importance = attn_weights  # (B, K, 3)
            else:
                # Regular fusion: attn_weights: (B, K, 3, 3) - attention matrix from multi-head attention
                # The attention weights come from the MultiModelPoseFusion model's learned attention mechanism.
                # During training, the model learns to attend to different baseline models (HRNet, OpenPose, MediaPipe)
                # based on their predictions. The attention matrix shows how much each model attends to each other model.
                # 
                # Compute importance as mean attention received by each model (mean of columns)
                # attn_weights[b, k, :, m] is how much each model attends to model m for joint k in batch b
                # Importance of model m = mean(attn_weights[b, k, :, m])
                # This represents the learned importance/contribution of each baseline model to the fusion
                importance = attn_weights.mean(dim=2)  # (B, K, 3) - mean over "from" dimension
            
            # Store importance weights
            all_importance_weights.append(importance.cpu().numpy())
            
            # Store frame info (linear index = order in concatenated importance rows)
            B = coords_fused.shape[0]
            for b in range(B):
                frame_info.append({
                    'batch_idx': batch_idx,
                    'frame_idx_in_batch': b,
                    'global_frame_idx': global_sample_offset + b,
                })
            global_sample_offset += B
    
    # Concatenate all batches
    importance_weights_array = np.concatenate(all_importance_weights, axis=0)  # (N, K, 3)
    
    results = {
        'importance_weights': importance_weights_array.tolist(),  # Convert to list for JSON
        'frame_info': frame_info,
        'model_names': model_names,
        'num_frames': len(frame_info),
        'num_joints': K,
        'model_type': model_type,
        'list_file': list_file,
        'dataset_root': dataset_root,
    }
    
    # Save to file if requested
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Importance weights saved to {out_path}")
    
    return results


def visualize_importance_weights(
    importance_data: Dict[str, Any],
    output_dir: str,
    num_examples_per_model: int = 5,
    tau: float = 0.05,
    list_file: Optional[str] = None,
    dataset_root: Optional[str] = None,
):
    """Visualize cases where each model has the highest importance.
    
    Args:
        importance_data: Output from extract_importance_weights
        output_dir: Directory to save visualizations
        num_examples_per_model: Number of examples to show for each model
        tau: Distance threshold for accuracy (needed if loading from evaluation)
        list_file: Optional path to list file for loading actual frames
        dataset_root: Optional dataset root for loading actual frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    importance_weights = np.array(importance_data['importance_weights'])  # (N, K, M)
    model_names = importance_data.get('model_names', ['hrnet', 'openpose', 'mediapipe'])
    num_frames, num_joints = importance_weights.shape[:2]
    num_models = importance_weights.shape[2]

    # Load dataset once for frame visualization and video boundaries
    dataset = None
    if list_file and dataset_root:
        try:
            dataset = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
            print(f"Loaded dataset for importance visualization: {len(dataset)} entries")
            if num_frames > len(dataset):
                print(
                    f"Warning: importance rows ({num_frames}) > dataset length ({len(dataset)}); "
                    f"check --list / --root match the eval run."
                )
        except Exception as e:
            print(f"Warning: Could not load dataset for frame visualization: {e}")
            dataset = None
    vid_bounds = _get_video_boundaries(dataset)
    ds_len = len(dataset) if dataset is not None else None
    
    # For each model, find frames/joints where it has highest importance
    
    # Create summary statistics plot: actual importance weights sampled every 1000 frames
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Average importance across all joints per frame: (N, M)
    avg_importance = importance_weights.mean(axis=1)  # (N, M)
    
    # Sample every 1000 frames
    sample_interval = 1000
    sample_indices = np.arange(0, num_frames, sample_interval)
    if sample_indices[-1] != num_frames - 1:
        # Include the last frame
        sample_indices = np.append(sample_indices, num_frames - 1)
    
    sampled_frames = sample_indices
    sampled_importance = avg_importance[sample_indices, :]  # (num_samples, M)
    
    _palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors = [_palette[i % len(_palette)] for i in range(num_models)]
    
    for model_idx, model_name in enumerate(model_names):
        ax.plot(sampled_frames, sampled_importance[:, model_idx], 
               marker='o', label=model_name.upper(), 
               color=colors[model_idx], 
               linewidth=2, markersize=8, alpha=0.8)
        
        # Annotate each point with its value
        for i, (frame_idx, weight_val) in enumerate(zip(sampled_frames, sampled_importance[:, model_idx])):
            # Alternate annotation positions to reduce overlap
            if model_idx == 0:
                offset_x, offset_y = 5, 15
            elif model_idx == 1:
                offset_x, offset_y = 5, -25
            elif model_idx == 2:
                offset_x = 15 if i % 2 == 0 else -15
                offset_y = 0
            else:
                offset_x = 8
                offset_y = 12 + model_idx * 5
            
            ax.annotate(f'{weight_val:.3f}', 
                       xy=(frame_idx, weight_val),
                       xytext=(offset_x, offset_y),
                       textcoords='offset points',
                       fontsize=8,
                       color=colors[model_idx],
                       alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                               edgecolor=colors[model_idx], linewidth=1))
    
    _draw_video_boundaries(ax, vid_bounds, xmin=0, xmax=num_frames - 1)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Average Importance Weight (across all joints)', fontsize=12)
    caption = importance_data.get(
        'importance_caption',
        'Weights represent learned attention from fusion model\'s multi-head attention mechanism',
    )
    ax.set_title(
        f'Importance Weights Over Time (Sampled Every {sample_interval} Frames)\n{caption}',
        fontsize=14,
        fontweight='bold',
    )
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path / 'importance_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary statistics to {output_path / 'importance_summary.png'}")

    # ------------------------------------------------------------
    # Additional visualization: highlight situations where weights change a lot
    # ------------------------------------------------------------
    # Use the same avg_importance over frames: (N, 3)
    # Compute frame-to-frame change magnitude for each model and overall
    if num_frames > 1:
        deltas = np.abs(np.diff(avg_importance, axis=0))  # (N-1, 3)
        total_change = deltas.sum(axis=1)  # (N-1,)

        # Pick top change events
        num_events = min(8, len(total_change))
        if num_events > 0:
            event_indices = np.argsort(total_change)[-num_events:]  # indices in [0, N-2]
            event_indices = np.sort(event_indices)

            window = 50  # frames before/after the change point to show
            
            # Create figure with subplots: one row per event, two columns (plot + frame)
            fig = plt.figure(figsize=(18, 3 * num_events))
            gs = GridSpec(num_events, 2, figure=fig, width_ratios=[2, 1], hspace=0.3, wspace=0.2)
            
            first_ax = None  # Store reference to first axis for legend
            for row, idx in enumerate(event_indices):
                # Left subplot: weight change plot
                ax_evt = fig.add_subplot(gs[row, 0])
                if first_ax is None:
                    first_ax = ax_evt
                start = max(0, idx - window)
                end = min(num_frames - 1, idx + window)
                t = np.arange(start, end + 1)

                for m_idx, m_name in enumerate(model_names):
                    ax_evt.plot(
                        t,
                        avg_importance[start:end + 1, m_idx],
                        label=m_name.upper() if row == 0 else None,
                        color=colors[m_idx],
                        linewidth=1.5,
                        alpha=0.9,
                    )

                # Mark the primary change frame
                change_frame = idx + 1  # because diff is between idx and idx+1
                ax_evt.axvline(change_frame, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
                _draw_video_boundaries(ax_evt, vid_bounds, xmin=start, xmax=end,
                                       label_first=(row == 0))

                # Try to include global frame index from frame_info if available
                global_idx = None
                frame_info = importance_data.get('frame_info', None)
                if frame_info is not None and 0 <= change_frame < len(frame_info):
                    global_idx = frame_info[change_frame].get('global_frame_idx', None)

                if global_idx is not None:
                    title_suffix = f"(local frame {change_frame}, global frame {global_idx})"
                else:
                    title_suffix = f"(frame {change_frame})"

                ax_evt.set_title(
                    f'Large weight change event {row + 1}/{num_events} {title_suffix}\n'
                    f'Δ(total importance) = {total_change[idx]:.4f}',
                    fontsize=11,
                )
                ax_evt.grid(True, alpha=0.3)
                ax_evt.set_ylim([0, 1])
                ax_evt.set_xlabel('Frame Index')
                ax_evt.set_ylabel('Avg Importance')
                
                # Right subplot: actual frame image
                ax_frame = fig.add_subplot(gs[row, 1])
                ax_frame.axis('off')
                
                # Try to load and display the actual frame (fallback: local row index == dataset idx)
                fi = frame_info if frame_info else None
                ds_idx = _dataset_index_for_importance_row(change_frame, fi, ds_len)
                if dataset is not None and 0 <= ds_idx < len(dataset):
                    try:
                        sample = dataset[ds_idx]
                        img_np = _to_uint8_rgb(sample['image'])
                        ax_frame.imshow(img_np)
                        ax_frame.set_title(
                            f'Frame {ds_idx}' + (f' (global {global_idx})' if global_idx is not None else ''),
                            fontsize=10,
                        )
                        
                        # Get video info from metadata if available
                        meta = sample.get('meta', {})
                        video_path = meta.get('video_rel_path', '')
                        if video_path:
                            ax_frame.text(0.5, -0.1, video_path, transform=ax_frame.transAxes,
                                        fontsize=8, ha='center', wrap=True)
                    except Exception as e:
                        ax_frame.text(0.5, 0.5, f'Could not load frame {ds_idx}\n{str(e)}',
                                    transform=ax_frame.transAxes, ha='center', va='center',
                                    fontsize=9, color='red')
                else:
                    ax_frame.text(0.5, 0.5, 'Frame not available',
                                transform=ax_frame.transAxes, ha='center', va='center',
                                fontsize=10, color='gray')

            # Only add legend once (top subplot)
            if first_ax is not None:
                first_ax.legend(fontsize=9, loc='best')
            try:
                plt.tight_layout()
            except Exception:
                # Some axes may not be compatible with tight_layout, but bbox_inches='tight' handles it
                pass
            plt.savefig(output_path / 'importance_change_events.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved weight change event visualization to {output_path / 'importance_change_events.png'}")


def plot_importance_over_time(
    importance_data: Dict[str, Any],
    output_dir: str,
    average_across_joints: bool = True,
    joint_specific: Optional[List[int]] = None,
    smoothing_window: int = 1,
    list_file: Optional[str] = None,
    dataset_root: Optional[str] = None,
):
    """Create line graphs showing how importance weights change over frames.
    
    Args:
        importance_data: Output from extract_importance_weights
        output_dir: Directory to save visualizations
        average_across_joints: If True, average importance across all joints per frame
        joint_specific: Optional list of joint indices to plot separately
        smoothing_window: Moving average window size (1 = no smoothing)
        list_file: Optional path to list file for computing video boundaries
        dataset_root: Optional dataset root for computing video boundaries
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    importance_weights = np.array(importance_data['importance_weights'])  # (N, K, 3)
    model_names = importance_data['model_names']
    num_frames, num_joints = importance_weights.shape[:2]

    vid_bounds: List[int] = []
    if list_file and dataset_root:
        try:
            _ds = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
            vid_bounds = _get_video_boundaries(_ds)
        except Exception:
            pass
    
    # Colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # HRNet, OpenPose, MediaPipe
    
    # Apply smoothing if requested
    if smoothing_window > 1:
        try:
            from scipy.ndimage import uniform_filter1d
        except ImportError:
            print("Warning: scipy not available, skipping smoothing")
            smoothing_window = 1
    
    if smoothing_window > 1:
        smoothed_weights = np.zeros_like(importance_weights)
        for model_idx in range(3):
            for joint_idx in range(num_joints):
                smoothed_weights[:, joint_idx, model_idx] = uniform_filter1d(
                    importance_weights[:, joint_idx, model_idx], 
                    size=smoothing_window, 
                    mode='nearest'
                )
        importance_weights = smoothed_weights
    
    # Create frame indices
    frame_indices = np.arange(num_frames)
    
    # Plot 1: Average importance across all joints over time
    if average_across_joints:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Average across joints: (N, 3)
        avg_importance = importance_weights.mean(axis=1)  # (N, 3)
        
        for model_idx, model_name in enumerate(model_names):
            ax.plot(frame_indices, avg_importance[:, model_idx], 
                   label=model_name.upper(), color=colors[model_idx], linewidth=2, alpha=0.8)
        
        _draw_video_boundaries(ax, vid_bounds, xmin=0, xmax=num_frames - 1)
        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Average Importance Weight (across all joints)', fontsize=12)
        ax.set_title('Importance Weights Over Time (Averaged Across All Joints)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_path / 'importance_over_time_average.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved average importance over time to {output_path / 'importance_over_time_average.png'}")
    
    # Plot 2: Per-joint importance over time
    if joint_specific is None:
        joint_specific = list(range(num_joints))
    
    # Create subplots for each joint
    n_joints_to_plot = len(joint_specific)
    if n_joints_to_plot > 0:
        # Arrange in a grid
        n_cols = min(3, n_joints_to_plot)
        n_rows = (n_joints_to_plot + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_joints_to_plot == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for plot_idx, joint_idx in enumerate(joint_specific):
            ax = axes[plot_idx]
            
            # Get importance for this joint: (N, 3)
            joint_importance = importance_weights[:, joint_idx, :]
            
            for model_idx, model_name in enumerate(model_names):
                ax.plot(frame_indices, joint_importance[:, model_idx],
                       label=model_name.upper(), color=colors[model_idx], linewidth=1.5, alpha=0.8)
            
            _draw_video_boundaries(ax, vid_bounds, xmin=0, xmax=num_frames - 1,
                                   label_first=(plot_idx == 0))
            ax.set_xlabel('Frame Index', fontsize=10)
            ax.set_ylabel('Importance Weight', fontsize=10)
            ax.set_title(f'Joint {joint_idx}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Hide unused subplots
        for idx in range(n_joints_to_plot, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Importance Weights Over Time (Per Joint)', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path / 'importance_over_time_per_joint.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved per-joint importance over time to {output_path / 'importance_over_time_per_joint.png'}")
    
    # Plot 3: Stacked area chart showing relative contributions
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if average_across_joints:
        avg_importance = importance_weights.mean(axis=1)  # (N, 3)
    else:
        # Use first joint if not averaging
        avg_importance = importance_weights[:, 0, :]
    
    # Stack the areas
    ax.fill_between(frame_indices, 0, avg_importance[:, 0], 
                    label=model_names[0].upper(), color=colors[0], alpha=0.6)
    ax.fill_between(frame_indices, avg_importance[:, 0], 
                    avg_importance[:, 0] + avg_importance[:, 1],
                    label=model_names[1].upper(), color=colors[1], alpha=0.6)
    ax.fill_between(frame_indices, avg_importance[:, 0] + avg_importance[:, 1],
                    avg_importance[:, 0] + avg_importance[:, 1] + avg_importance[:, 2],
                    label=model_names[2].upper(), color=colors[2], alpha=0.6)
    
    _draw_video_boundaries(ax, vid_bounds, xmin=0, xmax=num_frames - 1)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Importance Weight', fontsize=12)
    ax.set_title('Stacked Importance Weights Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path / 'importance_over_time_stacked.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved stacked importance over time to {output_path / 'importance_over_time_stacked.png'}")


def visualize_frame_context(
    importance_data: Dict[str, Any],
    frame_indices: List[int],
    output_dir: str,
    window: int = 5,
    list_file: Optional[str] = None,
    dataset_root: Optional[str] = None,
):
    """For selected frames, plot importance weights in a local window and show the frame."""
    if not frame_indices:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    importance_weights = np.array(importance_data['importance_weights'])  # (N, K, 3)
    model_names = importance_data['model_names']
    num_frames = importance_weights.shape[0]

    # Average across joints per frame
    avg_importance = importance_weights.mean(axis=1)  # (N, 3)

    # Optional dataset for frames
    dataset = None
    if list_file and dataset_root:
        try:
            dataset = VideoFrameKeypointDataset(
                list_file=list_file,
                dataset_root=dataset_root,
                output_size=(256, 256)
            )
            print(f"[inspect] Loaded dataset for frame context: {len(dataset)} entries")
        except Exception as e:
            print(f"[inspect] Could not load dataset for frame context: {e}")
            dataset = None
    vid_bounds = _get_video_boundaries(dataset)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for f_idx in frame_indices:
        if f_idx < 0 or f_idx >= num_frames:
            print(f"[inspect] Frame {f_idx} out of range (0..{num_frames-1}), skipping")
            continue
        start = max(0, f_idx - window)
        end = min(num_frames - 1, f_idx + window)
        t = np.arange(start, end + 1)

        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0], wspace=0.2)

        # Left: weights over window
        ax0 = fig.add_subplot(gs[0, 0])
        for m_idx, m_name in enumerate(model_names):
            ax0.plot(
                t,
                avg_importance[start:end + 1, m_idx],
                label=m_name.upper(),
                color=colors[m_idx],
                linewidth=2,
                alpha=0.85,
            )
        ax0.axvline(f_idx, color='k', linestyle='--', linewidth=1.2, alpha=0.8)
        _draw_video_boundaries(ax0, vid_bounds, xmin=start, xmax=end)
        ax0.set_ylim([0, 1])
        ax0.set_xlabel('Frame')
        ax0.set_ylabel('Avg Importance (across joints)')
        ax0.set_title(f'Importance window around frame {f_idx} (±{window})')
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=9, loc='best')

        # Right: frame image (if available)
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.axis('off')
        if dataset is not None and f_idx < len(dataset):
            try:
                sample = dataset[f_idx]
                img_np = _to_uint8_rgb(sample['image'])
                ax1.imshow(img_np)
                meta = sample.get('meta', {})
                video_path = meta.get('video_rel_path', '')
                frame_id = meta.get('frame_index', f_idx)
                ax1.set_title(f'Frame {frame_id}\n{video_path}', fontsize=9)
            except Exception as e:
                ax1.text(0.5, 0.5, f'Could not load frame {f_idx}\n{e}',
                         ha='center', va='center', fontsize=9, color='red')
        else:
            ax1.text(0.5, 0.5, 'Frame not available',
                     ha='center', va='center', fontsize=10, color='gray')

        plt.tight_layout()
        out_file = output_path / f'frame_context_{f_idx:06d}.png'
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[inspect] Saved frame context to {out_file}")


def visualize_significant_weight_changes(
    importance_data: Dict[str, Any],
    output_dir: str,
    change_threshold: float = 0.005,
    max_changes: int = 20,
    window: int = 5,
    list_file: Optional[str] = None,
    dataset_root: Optional[str] = None,
):
    """Automatically detect significant weight changes and create 10-frame window visualizations.
    
    Args:
        importance_data: Output from extract_importance_weights
        output_dir: Directory to save visualizations
        change_threshold: Minimum change magnitude to consider significant (default 0.005)
        max_changes: Maximum number of changes to visualize (default 20)
        window: Number of frames before/after to show (default 5, total 11 frames)
        list_file: Optional path to list file for loading actual frames
        dataset_root: Optional dataset root for loading actual frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    importance_weights = np.array(importance_data['importance_weights'])  # (N, K, M)
    model_names = importance_data.get('model_names', ['hrnet', 'openpose', 'mediapipe'])
    frame_info = importance_data.get('frame_info') or []
    num_frames = importance_weights.shape[0]
    num_models_wm = importance_weights.shape[2]
    
    # Average across joints per frame
    avg_importance = importance_weights.mean(axis=1)  # (N, M)
    
    # Compute frame-to-frame change magnitude for each model
    if num_frames < 2:
        print("[weight-changes] Not enough frames to detect changes")
        return
    
    deltas = np.abs(np.diff(avg_importance, axis=0))  # (N-1, 3)
    # Total change magnitude (sum across all models)
    total_change = deltas.sum(axis=1)  # (N-1,)
    
    # Find significant changes (above threshold)
    significant_mask = total_change >= change_threshold
    significant_indices = np.where(significant_mask)[0]  # indices in [0, N-2]
    
    if len(significant_indices) == 0:
        print(f"[weight-changes] No significant weight changes found (threshold={change_threshold})")
        return
    
    # Limit to top changes if too many
    if len(significant_indices) > max_changes:
        # Sort by change magnitude and take top N
        change_magnitudes = total_change[significant_indices]
        top_indices = np.argsort(change_magnitudes)[-max_changes:]
        significant_indices = significant_indices[top_indices]
        significant_indices = np.sort(significant_indices)
    
    print(f"[weight-changes] Found {len(significant_indices)} significant weight changes (threshold={change_threshold})")
    
    # Try to load dataset for frame visualization
    dataset = None
    if list_file and dataset_root:
        try:
            dataset = VideoFrameKeypointDataset(
                list_file=list_file,
                dataset_root=dataset_root,
                output_size=(256, 256)
            )
            print(f"[weight-changes] Loaded dataset: {len(dataset)} entries")
            if num_frames > len(dataset):
                print(
                    f"[weight-changes] Warning: importance rows ({num_frames}) > dataset length ({len(dataset)}); "
                    f"check --list / --root match the eval run."
                )
        except Exception as e:
            print(f"[weight-changes] Warning: Could not load dataset: {e}")
            dataset = None
    
    ds_len = len(dataset) if dataset is not None else None
    vid_bounds = _get_video_boundaries(dataset)
    
    _pal_wm = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors = [_pal_wm[i % len(_pal_wm)] for i in range(num_models_wm)]
    
    # Create visualization for each significant change
    for change_idx, idx in enumerate(significant_indices):
        # The change occurs between frame idx and idx+1
        change_frame = idx + 1  # Local frame index where change occurs
        
        # Get window around change frame
        start_idx = max(0, change_frame - window)
        end_idx = min(num_frames - 1, change_frame + window)
        frame_indices = np.arange(start_idx, end_idx + 1)
        
        # Get global frame index if available
        global_idx = None
        if frame_info and 0 <= change_frame < len(frame_info):
            global_idx = frame_info[change_frame].get('global_frame_idx', None)
        
        # Calculate change details
        prev_weights = avg_importance[idx, :]
        curr_weights = avg_importance[change_frame, :]
        weight_changes = curr_weights - prev_weights
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 2], hspace=0.3)
        
        # Top: Weight plot
        ax_weights = fig.add_subplot(gs[0, 0])
        
        for m_idx, m_name in enumerate(model_names):
            ax_weights.plot(
                frame_indices,
                avg_importance[start_idx:end_idx + 1, m_idx],
                label=m_name.upper(),
                color=colors[m_idx],
                linewidth=2.5,
                marker='o',
                markersize=8,
                alpha=0.85,
            )
        
        # Highlight change frame
        ax_weights.axvline(change_frame, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Change Frame')
        _draw_video_boundaries(ax_weights, vid_bounds, xmin=start_idx, xmax=end_idx)
        
        # Annotate weight values at change frame
        for m_idx, m_name in enumerate(model_names):
            weight_val = avg_importance[change_frame, m_idx]
            change_val = weight_changes[m_idx]
            offset_y = 20 + m_idx * 15
            ax_weights.annotate(
                f'{m_name}: {weight_val:.4f} (Δ{change_val:+.4f})',
                xy=(change_frame, weight_val),
                xytext=(10, offset_y),
                textcoords='offset points',
                fontsize=9,
                color=colors[m_idx],
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                        edgecolor=colors[m_idx], linewidth=1),
            )
        
        # Set labels and title
        change_magnitude = total_change[idx]
        title_text = f'Significant Weight Change #{change_idx + 1}/{len(significant_indices)}'
        if global_idx is not None:
            title_text += f' at Frame {global_idx} (local {change_frame})'
        else:
            title_text += f' at Local Frame {change_frame}'
        title_text += f'\nTotal Change Magnitude: {change_magnitude:.6f}'
        
        ax_weights.set_title(title_text, fontsize=13, fontweight='bold')
        ax_weights.set_xlabel('Local Frame Index', fontsize=12, fontweight='bold')
        ax_weights.set_ylabel('Average Importance Weight', fontsize=12, fontweight='bold')
        ax_weights.legend(fontsize=11, loc='best')
        ax_weights.grid(True, alpha=0.3)
        ax_weights.set_ylim([0, 1])
        
        # Set x-axis labels to show both local and global indices
        x_labels = []
        for f_idx in frame_indices:
            if frame_info and 0 <= f_idx < len(frame_info):
                g_idx = frame_info[f_idx].get('global_frame_idx', 'N/A')
                x_labels.append(f'{f_idx}\n({g_idx})')
            else:
                x_labels.append(str(f_idx))
        ax_weights.set_xticks(frame_indices)
        ax_weights.set_xticklabels(x_labels, fontsize=9)
        
        # Bottom: Frame grid (window*2+1 frames)
        ax_frames = fig.add_subplot(gs[1, 0])
        ax_frames.axis('off')
        
        n_frames_to_show = end_idx - start_idx + 1
        n_cols = min(6, n_frames_to_show)
        n_rows = (n_frames_to_show + n_cols - 1) // n_cols
        
        # Create subplot grid for frames
        frame_gs = GridSpec(n_rows, n_cols, figure=fig, 
                           left=0.05, right=0.95, bottom=0.05, top=0.45, 
                           hspace=0.3, wspace=0.2)
        
        for i, f_idx in enumerate(frame_indices):
            row = i // n_cols
            col = i % n_cols
            ax_frame = fig.add_subplot(frame_gs[row, col])
            ax_frame.axis('off')
            
            # Get frame info (optional); dataset index defaults to row f_idx when frame_info absent
            g_idx = None
            batch_idx = None
            frame_in_batch = None
            if frame_info and 0 <= f_idx < len(frame_info):
                g_idx = frame_info[f_idx].get('global_frame_idx', None)
                batch_idx = frame_info[f_idx].get('batch_idx', None)
                frame_in_batch = frame_info[f_idx].get('frame_idx_in_batch', None)

            ds_idx = _dataset_index_for_importance_row(f_idx, frame_info if frame_info else None, ds_len)
            
            weights_at_frame = avg_importance[f_idx, :]
            hrnet_w = weights_at_frame[0]
            
            # Try to load frame image
            if dataset is not None and 0 <= ds_idx < len(dataset):
                try:
                    sample = dataset[ds_idx]
                    img_np = _to_uint8_rgb(sample['image'])
                    ax_frame.imshow(img_np)
                    
                    # Highlight change frame
                    if f_idx == change_frame:
                        for spine in ax_frame.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(3)
                except Exception as e:
                    ax_frame.text(0.5, 0.5, f'Error loading\nframe {ds_idx}\n{str(e)[:30]}',
                                transform=ax_frame.transAxes, ha='center', va='center',
                                fontsize=8, color='red')
            else:
                ax_frame.text(0.5, 0.5, f'Frame {ds_idx}\nNot available',
                            transform=ax_frame.transAxes, ha='center', va='center',
                            fontsize=10, color='gray')
            
            # Add title with weight info
            title = f'Frame {ds_idx}' + (f' (g={g_idx})' if g_idx is not None else '') + '\n'
            if batch_idx is not None:
                title += f'Batch {batch_idx}'
                if frame_in_batch is not None:
                    title += f', Pos {frame_in_batch}'
                title += '\n'
            title += f'HRnet: {hrnet_w:.4f}'
            if f_idx == change_frame:
                title = f'>>> {title} <<<'
            ax_frame.set_title(title, fontsize=9, fontweight='bold' if f_idx == change_frame else 'normal',
                              color='red' if f_idx == change_frame else 'black')
        
        # Add analysis text
        analysis_text = f"Change Analysis:\n"
        analysis_text += f"• Change occurs between local frames {idx} → {change_frame}\n"
        if global_idx is not None:
            prev_global = frame_info[idx].get('global_frame_idx', 'N/A') if frame_info else 'N/A'
            analysis_text += f"• Global frames: {prev_global} → {global_idx}\n"
        analysis_text += f"• Weight changes: "
        for m_idx, m_name in enumerate(model_names):
            analysis_text += f"{m_name.upper()}={weight_changes[m_idx]:+.4f}  "
        analysis_text += f"\n• Total change magnitude: {change_magnitude:.6f}"
        
        fig.text(0.5, 0.02, analysis_text, ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Significant Weight Change #{change_idx + 1} Visualization', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        if global_idx is not None:
            out_file = output_path / f'weight_change_{change_idx+1:03d}_frame_{global_idx:06d}.png'
        else:
            out_file = output_path / f'weight_change_{change_idx+1:03d}_local_{change_frame:06d}.png'
        
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[weight-changes] Saved visualization to {out_file}")
    
    print(f"[weight-changes] Created {len(significant_indices)} weight change visualizations")


def eval_all_models(
    list_file: str,
    dataset_root: str,
    output_dir: str,
    mlp_fusion_checkpoint: Optional[str] = None,
    mlp_fusion_4_checkpoint: Optional[str] = None,
    transformer_fusion_checkpoint: Optional[str] = None,
    transformer_fusion_4_checkpoint: Optional[str] = None,
    internal_transformer_checkpoint: Optional[str] = None,
    internal_transformer_4_checkpoint: Optional[str] = None,
    trained_dekr_checkpoint: Optional[str] = None,
    trained_dekr_freeze_checkpoint: Optional[str] = None,
    trained_openpose_checkpoint: Optional[str] = None,
    trained_openpose_freeze_checkpoint: Optional[str] = None,
    batch_size: int = 16,
    device: str = 'cuda',
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    tau: float = 0.05,
    extract_importance: bool = True,
    visualize_importance: bool = True,
    max_frames_to_visualize: int = 12,
    transformer_d_model: int = 256,
    transformer_nhead: int = 8,
    transformer_num_encoder_layers: int = 2,
    transformer_num_decoder_layers: int = 2,
    transformer_dim_feedforward: int = 512,
    transformer_dropout: float = 0.1,
    include_rule_based_fusion: bool = True,
    max_batches: Optional[int] = None,
    skip_trained_baselines: bool = False,
) -> Dict[str, Any]:
    """Efficiently evaluate ALL fusion models in a single pass.
    
    This function:
    1. Loads all fusion models once (MLP, Transformer, Internal Transformer)
    2. Evaluates trained baseline models (DEKR, OpenPose) ONCE and shares results
    3. Processes all fusion models in a SINGLE data pass through the test set
    4. Uses GPU optimizations: larger batch size, prefetching, persistent workers
    5. Optionally extracts importance weights and creates visualizations
    6. Optionally evaluates uniform and confidence-weighted fusion (rule-based baselines)
    
    Returns:
        Dict with results for each model type
    """
    import time
    from torchvision import transforms as T
    
    start_time = time.time()
    device_t = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("EFFICIENT MULTI-MODEL EVALUATION")
    print("="*70)
    print(f"Device: {device_t}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Pin memory: {pin_memory}")
    print(f"Prefetch factor: {prefetch_factor}")
    print(f"Persistent workers: {persistent_workers}")
    
    # Probe dataset for num_joints
    ds_probe = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
    if len(ds_probe) == 0:
        raise RuntimeError("Dataset appears empty")
    K = ds_probe[0]['keypoints'].shape[0]
    print(f"Detected {K} joints from dataset, {len(ds_probe)} samples")
    
    # =========================================================================
    # PHASE 1: Load all models
    # =========================================================================
    print("\n" + "-"*50)
    print("PHASE 1: Loading all models...")
    print("-"*50)
    
    models_to_eval = {}
    
    # Load MLP Fusion model
    if mlp_fusion_checkpoint and Path(mlp_fusion_checkpoint).exists():
        print(f"Loading MLP Fusion from {mlp_fusion_checkpoint}")
        d = torch.load(mlp_fusion_checkpoint, map_location=device_t)
        mlp_model = MultiModelPoseFusion(num_joints=K).to(device_t)
        mlp_model.load_state_dict(d['model_state'])
        mlp_model.eval()
        models_to_eval['mlp_fusion'] = {
            'model': mlp_model,
            'type': 'mlp',
            'checkpoint': mlp_fusion_checkpoint,
            'output_json': str(output_path / 'mlp_fusion_results.json'),
        }
        print("  ✓ MLP Fusion loaded")
    
    # Load Transformer Fusion model
    if transformer_fusion_checkpoint and Path(transformer_fusion_checkpoint).exists():
        print(f"Loading Transformer Fusion from {transformer_fusion_checkpoint}")
        transformer_model = load_transformer_fusion_model(
            checkpoint_path=transformer_fusion_checkpoint,
            device=device_t,
            num_joints=K,
            model_type='auto',
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if transformer_model is not None:
            models_to_eval['transformer_fusion'] = {
                'model': transformer_model,
                'type': 'transformer',
                'checkpoint': transformer_fusion_checkpoint,
                'output_json': str(output_path / 'transformer_fusion_results.json'),
            }
            print("  ✓ Transformer Fusion loaded")
    
    # Load Internal Transformer Fusion model
    if internal_transformer_checkpoint and Path(internal_transformer_checkpoint).exists():
        print(f"Loading Internal Transformer Fusion from {internal_transformer_checkpoint}")
        internal_model = load_transformer_internal_fusion_model(
            checkpoint_path=internal_transformer_checkpoint,
            device=device_t,
            num_joints=K,
            model_type='auto',
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if internal_model is not None:
            models_to_eval['internal_transformer_fusion'] = {
                'model': internal_model,
                'type': 'internal_transformer',
                'checkpoint': internal_transformer_checkpoint,
                'output_json': str(output_path / 'internal_transformer_fusion_results.json'),
            }
            print("  ✓ Internal Transformer Fusion loaded")

    # ---- 4-model (ViTPose) checkpoints ----
    if mlp_fusion_4_checkpoint and Path(mlp_fusion_4_checkpoint).exists():
        print(f"Loading MLP Fusion (4-model) from {mlp_fusion_4_checkpoint}")
        d = torch.load(mlp_fusion_4_checkpoint, map_location=device_t)
        mlp4 = MultiModelPoseFusion4(num_joints=K).to(device_t)
        mlp4.load_state_dict(d['model_state'])
        mlp4.eval()
        models_to_eval['mlp_fusion_4'] = {
            'model': mlp4,
            'type': 'mlp4',
            'checkpoint': mlp_fusion_4_checkpoint,
            'output_json': str(output_path / 'mlp_fusion_4_results.json'),
        }
        print("  ✓ MLP Fusion (4-model) loaded")

    if transformer_fusion_4_checkpoint and Path(transformer_fusion_4_checkpoint).exists():
        print(f"Loading Transformer Fusion (4-model) from {transformer_fusion_4_checkpoint}")
        tf4 = load_transformer_fusion_4_model(
            checkpoint_path=transformer_fusion_4_checkpoint,
            device=device_t,
            num_joints=K,
            model_type='auto',
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if tf4 is not None:
            models_to_eval['transformer_fusion_4'] = {
                'model': tf4,
                'type': 'transformer4',
                'checkpoint': transformer_fusion_4_checkpoint,
                'output_json': str(output_path / 'transformer_fusion_4_results.json'),
            }
            print("  ✓ Transformer Fusion (4-model) loaded")

    if internal_transformer_4_checkpoint and Path(internal_transformer_4_checkpoint).exists():
        print(f"Loading Internal Transformer Fusion (4-model) from {internal_transformer_4_checkpoint}")
        it4 = load_transformer_internal_fusion_4_model(
            checkpoint_path=internal_transformer_4_checkpoint,
            device=device_t,
            num_joints=K,
            model_type='auto',
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        if it4 is not None:
            models_to_eval['internal_transformer_fusion_4'] = {
                'model': it4,
                'type': 'internal_transformer4',
                'checkpoint': internal_transformer_4_checkpoint,
                'output_json': str(output_path / 'internal_transformer_fusion_4_results.json'),
            }
            print("  ✓ Internal Transformer Fusion (4-model) loaded")

    if not models_to_eval:
        raise RuntimeError("No fusion model checkpoints found!")
    
    print(f"\nLoaded {len(models_to_eval)} fusion models: {list(models_to_eval.keys())}")
    
    # Load trained baseline models (shared across all evaluations)
    trained_dekr_model = None
    trained_dekr_freeze_model = None
    trained_openpose_model = None
    
    if trained_dekr_checkpoint and Path(trained_dekr_checkpoint).exists():
        print(f"Loading trained DEKR from {trained_dekr_checkpoint}")
        trained_dekr_model, _ = load_trained_dekr_model(trained_dekr_checkpoint, device_t, num_joints=K)
        if trained_dekr_model:
            print("  ✓ Trained DEKR loaded")
    
    if trained_dekr_freeze_checkpoint and Path(trained_dekr_freeze_checkpoint).exists():
        print(f"Loading frozen-backbone DEKR from {trained_dekr_freeze_checkpoint}")
        trained_dekr_freeze_model, _ = load_trained_dekr_model(trained_dekr_freeze_checkpoint, device_t, num_joints=K)
        if trained_dekr_freeze_model:
            print("  ✓ Frozen-backbone DEKR loaded")
    
    if trained_openpose_checkpoint and Path(trained_openpose_checkpoint).exists():
        print(f"Loading trained OpenPose from {trained_openpose_checkpoint}")
        trained_openpose_model = load_trained_openpose_model(trained_openpose_checkpoint, device_t, num_joints=K)
        if trained_openpose_model:
            print("  ✓ Trained OpenPose loaded")
    
    trained_openpose_freeze_model = None
    if trained_openpose_freeze_checkpoint and Path(trained_openpose_freeze_checkpoint).exists():
        print(f"Loading frozen-backbone OpenPose from {trained_openpose_freeze_checkpoint}")
        trained_openpose_freeze_model = load_trained_openpose_model(trained_openpose_freeze_checkpoint, device_t, num_joints=K)
        if trained_openpose_freeze_model:
            print("  ✓ Frozen-backbone OpenPose loaded")
    
    # =========================================================================
    # PHASE 2: Create optimized dataloaders
    # =========================================================================
    print("\n" + "-"*50)
    print("PHASE 2: Creating optimized dataloaders...")
    print("-"*50)
    
    # Fusion dataloader (for cached coordinates)
    _four_model_types = {'mlp4', 'transformer4', 'internal_transformer4'}
    _need_vitpose = any(
        models_to_eval[k]['type'] in _four_model_types for k in models_to_eval
    )
    fusion_loader = build_dataloader_from_list(
        list_file,
        dataset_root,
        batch_size=batch_size,
        output_size=(256, 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        require_caches=False,
        include_vitpose=_need_vitpose,
    )
    print(f"  Fusion loader: {len(fusion_loader)} batches")
    
    # Image dataset for trained model inference
    image_dataset = VideoFrameKeypointDataset(
        list_file=list_file,
        dataset_root=dataset_root,
        output_size=(256, 256)
    )
    image_loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    print(f"  Image loader: {len(image_loader)} batches")
    
    # =========================================================================
    # PHASE 3: Evaluate trained baseline models ONCE (shared across fusion models)
    # =========================================================================
    baseline_results = {
        'dekr_trained': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': [], 'frame_errors': []},
        'dekr_trained_freeze': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': [], 'frame_errors': []},
        'openpose_trained': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': [], 'frame_errors': []},
        'openpose_trained_freeze': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': [], 'frame_errors': []},
    }

    if skip_trained_baselines:
        print("\n" + "-"*50)
        print("PHASE 3: Skipping trained baseline models (--skip-trained-baselines).")
        print("-"*50)
    else:
        print("\n" + "-"*50)
        print("PHASE 3: Evaluating trained baseline models (shared)...")
        print("-"*50)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        with torch.no_grad():
            for img_batch_idx, sample_batch in enumerate(image_loader):
                images = sample_batch['image']  # (B, 3, H, W)
                keypoints_gt = sample_batch['keypoints']  # (B, K, 3)
                B = images.shape[0]
                H, W = images.shape[2], images.shape[3]
                
                img_wh = torch.tensor([[W, H]] * B, dtype=torch.float32).to(device_t)
                couch_len = torch.tensor([np.sqrt(W**2 + H**2)] * B, dtype=torch.float32).to(device_t)
                coords_gt_px = keypoints_gt[:, :, :2].to(device_t)
                
                # Evaluate trained DEKR
                if trained_dekr_model is not None:
                    images_norm = normalize(images).to(device_t)
                    coords_dekr, _ = trained_dekr_model(images_norm)
                    coords_dekr_px = coords_dekr * img_wh.view(-1, 1, 2)
                    l2_dist_px = torch.norm(coords_dekr_px - coords_gt_px, dim=-1)
                    dists_norm = l2_dist_px / couch_len.view(-1, 1)
                    correct = (dists_norm < tau).float()
                    
                    for b in range(B):
                        frame_dists = []
                        for j in range(K):
                            if keypoints_gt[b, j, 2] > 0.5:
                                baseline_results['dekr_trained']['correct'].append(bool(correct[b, j].item()))
                                baseline_results['dekr_trained']['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                                baseline_results['dekr_trained']['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                                frame_dists.append(float(dists_norm[b, j].item()))
                        if len(frame_dists) == K:
                            baseline_results['dekr_trained']['frame_errors'].append(float(np.mean(frame_dists)))
                
                # Evaluate frozen-backbone DEKR
                if trained_dekr_freeze_model is not None:
                    images_norm = normalize(images).to(device_t)
                    coords_dekr_frz, _ = trained_dekr_freeze_model(images_norm)
                    coords_dekr_frz_px = coords_dekr_frz * img_wh.view(-1, 1, 2)
                    l2_dist_px = torch.norm(coords_dekr_frz_px - coords_gt_px, dim=-1)
                    dists_norm = l2_dist_px / couch_len.view(-1, 1)
                    correct = (dists_norm < tau).float()
                    
                    for b in range(B):
                        frame_dists = []
                        for j in range(K):
                            if keypoints_gt[b, j, 2] > 0.5:
                                baseline_results['dekr_trained_freeze']['correct'].append(bool(correct[b, j].item()))
                                baseline_results['dekr_trained_freeze']['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                                baseline_results['dekr_trained_freeze']['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                                frame_dists.append(float(dists_norm[b, j].item()))
                        if len(frame_dists) == K:
                            baseline_results['dekr_trained_freeze']['frame_errors'].append(float(np.mean(frame_dists)))
                
                # Evaluate trained OpenPose
                if trained_openpose_model is not None:
                    images_t = images.to(device_t)
                    coords_op, _ = trained_openpose_model(images_t)
                    coords_op_px = coords_op * img_wh.view(-1, 1, 2)
                    l2_dist_px = torch.norm(coords_op_px - coords_gt_px, dim=-1)
                    dists_norm = l2_dist_px / couch_len.view(-1, 1)
                    correct = (dists_norm < tau).float()
                    
                    for b in range(B):
                        frame_dists = []
                        for j in range(K):
                            if keypoints_gt[b, j, 2] > 0.5:
                                baseline_results['openpose_trained']['correct'].append(bool(correct[b, j].item()))
                                baseline_results['openpose_trained']['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                                baseline_results['openpose_trained']['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                                frame_dists.append(float(dists_norm[b, j].item()))
                        if len(frame_dists) == K:
                            baseline_results['openpose_trained']['frame_errors'].append(float(np.mean(frame_dists)))
                
                # Evaluate frozen-backbone OpenPose
                if trained_openpose_freeze_model is not None:
                    images_t = images.to(device_t)
                    coords_op_frz, _ = trained_openpose_freeze_model(images_t)
                    coords_op_frz_px = coords_op_frz * img_wh.view(-1, 1, 2)
                    l2_dist_px = torch.norm(coords_op_frz_px - coords_gt_px, dim=-1)
                    dists_norm = l2_dist_px / couch_len.view(-1, 1)
                    correct = (dists_norm < tau).float()
                    
                    for b in range(B):
                        frame_dists = []
                        for j in range(K):
                            if keypoints_gt[b, j, 2] > 0.5:
                                baseline_results['openpose_trained_freeze']['correct'].append(bool(correct[b, j].item()))
                                baseline_results['openpose_trained_freeze']['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                                baseline_results['openpose_trained_freeze']['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                                frame_dists.append(float(dists_norm[b, j].item()))
                        if len(frame_dists) == K:
                            baseline_results['openpose_trained_freeze']['frame_errors'].append(float(np.mean(frame_dists)))
                
                if (img_batch_idx + 1) % 50 == 0:
                    print(f"  Processed {img_batch_idx + 1}/{len(image_loader)} batches for trained baselines")
    
    # Print baseline results
    for name, metrics in baseline_results.items():
        if metrics['correct']:
            acc = np.mean(metrics['correct']) * 100
            l2_px = np.mean(metrics['l2_dist_px'])
            print(f"  {name}: Accuracy={acc:.2f}%, L2={l2_px:.3f}px, Samples={len(metrics['correct'])}")
    
    # =========================================================================
    # PHASE 4: Evaluate ALL fusion models in SINGLE pass
    # =========================================================================
    print("\n" + "-"*50)
    print("PHASE 4: Evaluating ALL fusion models in single pass...")
    print("-"*50)
    
    # Initialize metrics storage for each model
    model_metrics = {}
    for model_name in models_to_eval:
        model_metrics[model_name] = {
            'correct': [],
            'l2_dist_norm': [],
            'l2_dist_px': [],
            'mae': [],
            'confidences': [],
            'correct_confidences': [],
            'incorrect_confidences': [],
            'frame_errors': [],
            'per_joint_correct': {j: [] for j in range(K)},
            'per_joint_l2_dist_px': {j: [] for j in range(K)},
            'importance_weights': [] if extract_importance else None,
        }

    if include_rule_based_fusion and extract_importance:
        for rb in ('uniform_fusion', 'confidence_weighted_fusion'):
            model_metrics[rb] = {
                'correct': [],
                'l2_dist_norm': [],
                'l2_dist_px': [],
                'mae': [],
                'confidences': [],
                'correct_confidences': [],
                'incorrect_confidences': [],
                'frame_errors': [],
                'per_joint_correct': {j: [] for j in range(K)},
                'per_joint_l2_dist_px': {j: [] for j in range(K)},
                'importance_weights': [],
            }
        print("  Rule-based fusion (uniform + confidence-weighted) enabled for metrics and importance viz.")
    
    # Baseline metrics (from cached predictions)
    cache_baseline_metrics = {
        'hrnet': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': [], 'frame_errors': []},
        'openpose': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': [], 'frame_errors': []},
        'mediapipe': {'correct': [], 'l2_dist_px': [], 'l2_dist_norm': [], 'frame_errors': []},
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(fusion_loader):
            # Move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device_t)
            
            coords_gt = batch['coords_gt']
            coords_gt_px = batch.get('coords_gt_px')
            img_wh = batch.get('img_wh')
            couch_len = batch.get('couch_len')
            mask_hr = batch.get('mask_hrnet')
            mask_op = batch.get('mask_openpose')
            mask_mp = batch.get('mask_mediapipe')
            mask_pb = batch.get('mask_vitpose')
            if mask_pb is not None:
                mask_any = (
                    (mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1) + mask_pb.squeeze(-1)) > 0.0
                )
            else:
                mask_any = ((mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1)) > 0.0)
            
            img_wh_tensor = torch.tensor([256.0, 256.0], device=device_t).view(1, 1, 2)
            B = coords_gt.shape[0]
            
            # ---- Evaluate each fusion model ----
            for model_name, model_info in models_to_eval.items():
                model = model_info['model']
                model_type = model_info['type']
                metrics = model_metrics[model_name]
                
                # Forward pass based on model type
                if model_type == 'internal_transformer':
                    feats_dekr = torch.cat([batch['coords_hrnet'], batch['conf_hrnet']], dim=-1)
                    feats_openpose = torch.cat([batch['coords_openpose'], batch['conf_openpose']], dim=-1)
                    coords_fused, conf_fused_logits, attn_weights = model(
                        feats_dekr, feats_openpose,
                        batch['coords_mediapipe'], batch['conf_mediapipe'],
                        return_attention_weights=extract_importance,
                    )
                elif model_type == 'internal_transformer4':
                    feats_dekr = torch.cat([batch['coords_hrnet'], batch['conf_hrnet']], dim=-1)
                    feats_openpose = torch.cat([batch['coords_openpose'], batch['conf_openpose']], dim=-1)
                    feats_vitpose = torch.cat([batch['coords_vitpose'], batch['conf_vitpose']], dim=-1)
                    coords_fused, conf_fused_logits, attn_weights = model(
                        feats_dekr, feats_openpose, feats_vitpose,
                        batch['coords_mediapipe'], batch['conf_mediapipe'],
                        return_attention_weights=extract_importance,
                    )
                elif model_type == 'mlp4':
                    coords_fused, conf_fused_logits, attn_weights = model(
                        batch['coords_hrnet'], batch['conf_hrnet'],
                        batch['coords_openpose'], batch['conf_openpose'],
                        batch['coords_mediapipe'], batch['conf_mediapipe'],
                        batch['coords_vitpose'], batch['conf_vitpose'],
                        return_attention_weights=extract_importance,
                    )
                elif model_type == 'transformer4':
                    coords_fused, conf_fused_logits, attn_weights = model(
                        batch['coords_hrnet'], batch['conf_hrnet'],
                        batch['coords_openpose'], batch['conf_openpose'],
                        batch['coords_mediapipe'], batch['conf_mediapipe'],
                        batch['coords_vitpose'], batch['conf_vitpose'],
                        return_attention_weights=extract_importance,
                    )
                else:
                    coords_fused, conf_fused_logits, attn_weights = model(
                        batch['coords_hrnet'], batch['conf_hrnet'],
                        batch['coords_openpose'], batch['conf_openpose'],
                        batch['coords_mediapipe'], batch['conf_mediapipe'],
                        return_attention_weights=extract_importance,
                    )
                
                # Store importance weights if requested
                if extract_importance and attn_weights is not None:
                    if model_type in ('transformer', 'internal_transformer', 'transformer4', 'internal_transformer4'):
                        importance = attn_weights  # (B, K, 3) or (B, K, 4)
                    else:
                        importance = attn_weights.mean(dim=2)  # (B, K, 3) or (B, K, 4)
                    metrics['importance_weights'].append(importance.cpu().numpy())
                
                # Convert to pixel space and compute metrics
                coords_fused_px = coords_fused * img_wh_tensor
                l2_dist_px = torch.norm(coords_fused_px - coords_gt_px, dim=-1)
                dists_norm = l2_dist_px / couch_len.view(-1, 1)
                correct = (dists_norm < tau).float()
                conf_fused = torch.sigmoid(conf_fused_logits.squeeze(-1))
                
                # Accumulate metrics
                for b in range(B):
                    frame_dists = []
                    for j in range(K):
                        if mask_any[b, j] > 0:
                            is_correct = bool(correct[b, j].item())
                            metrics['correct'].append(is_correct)
                            metrics['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                            metrics['l2_dist_norm'].append(float(dists_norm[b, j].item()))
                            
                            mae = float(torch.abs(coords_fused[b, j] - coords_gt[b, j]).mean().item())
                            metrics['mae'].append(mae)
                            
                            metrics['per_joint_correct'][j].append(is_correct)
                            metrics['per_joint_l2_dist_px'][j].append(float(l2_dist_px[b, j].item()))
                            
                            conf_score = float(conf_fused[b, j].item())
                            metrics['confidences'].append(conf_score)
                            if is_correct:
                                metrics['correct_confidences'].append(conf_score)
                            else:
                                metrics['incorrect_confidences'].append(conf_score)
                            
                            frame_dists.append(float(dists_norm[b, j].item()))
                    
                    if len(frame_dists) == K:
                        metrics['frame_errors'].append(float(np.mean(frame_dists)))

            # ---- Rule-based fusion (uniform + confidence-weighted) ----
            if include_rule_based_fusion and extract_importance:
                ch = batch['coords_hrnet']
                co = batch['coords_openpose']
                cm = batch['coords_mediapipe']
                c_h = batch['conf_hrnet']
                c_o = batch['conf_openpose']
                c_m = batch['conf_mediapipe']
                cv_tp = batch.get('coords_vitpose')
                c_v_tp = batch.get('conf_vitpose')

                if cv_tp is not None and c_v_tp is not None:
                    n_m = 4.0
                    coords_uni = (ch + co + cm + cv_tp) / n_m
                    c_mean = (
                        c_h.squeeze(-1)
                        + c_o.squeeze(-1)
                        + c_m.squeeze(-1)
                        + c_v_tp.squeeze(-1)
                    ) / n_m
                    conf_logits_uni = torch.logit(
                        torch.clamp(c_mean.unsqueeze(-1), min=1e-4, max=1.0 - 1e-4)
                    )
                    imp_uni = torch.full(
                        (B, K, 4), 1.0 / n_m, device=device_t, dtype=torch.float32
                    )
                    c_stack = torch.cat([c_h, c_o, c_m, c_v_tp], dim=-1)
                    coords_stack = torch.stack([ch, co, cm, cv_tp], dim=2)
                else:
                    coords_uni = (ch + co + cm) / 3.0
                    c_mean = (c_h.squeeze(-1) + c_o.squeeze(-1) + c_m.squeeze(-1)) / 3.0
                    conf_logits_uni = torch.logit(
                        torch.clamp(c_mean.unsqueeze(-1), min=1e-4, max=1.0 - 1e-4)
                    )
                    imp_uni = torch.full((B, K, 3), 1.0 / 3.0, device=device_t, dtype=torch.float32)
                    c_stack = torch.cat([c_h, c_o, c_m], dim=-1)
                    coords_stack = torch.stack([ch, co, cm], dim=2)

                sum_c = c_stack.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                w = c_stack / sum_c
                coords_cw = (w.unsqueeze(-1) * coords_stack).sum(dim=2)
                conf_fused_cw = (w * c_stack).sum(dim=-1, keepdim=True)
                conf_logits_cw = torch.logit(
                    torch.clamp(conf_fused_cw, min=1e-4, max=1.0 - 1e-4)
                )
                imp_cw = w

                rule_variants = [
                    ('uniform_fusion', coords_uni, conf_logits_uni, imp_uni),
                    ('confidence_weighted_fusion', coords_cw, conf_logits_cw, imp_cw),
                ]
                for rb_key, coords_fused, conf_fused_logits, importance_t in rule_variants:
                    metrics = model_metrics[rb_key]
                    metrics['importance_weights'].append(importance_t.detach().cpu().numpy())

                    coords_fused_px = coords_fused * img_wh_tensor
                    l2_dist_px = torch.norm(coords_fused_px - coords_gt_px, dim=-1)
                    dists_norm = l2_dist_px / couch_len.view(-1, 1)
                    correct = (dists_norm < tau).float()
                    conf_fused = torch.sigmoid(conf_fused_logits.squeeze(-1))

                    for b in range(B):
                        frame_dists = []
                        for j in range(K):
                            if mask_any[b, j] > 0:
                                is_correct = bool(correct[b, j].item())
                                metrics['correct'].append(is_correct)
                                metrics['l2_dist_px'].append(float(l2_dist_px[b, j].item()))
                                metrics['l2_dist_norm'].append(float(dists_norm[b, j].item()))

                                mae = float(torch.abs(coords_fused[b, j] - coords_gt[b, j]).mean().item())
                                metrics['mae'].append(mae)

                                metrics['per_joint_correct'][j].append(is_correct)
                                metrics['per_joint_l2_dist_px'][j].append(float(l2_dist_px[b, j].item()))

                                conf_score = float(conf_fused[b, j].item())
                                metrics['confidences'].append(conf_score)
                                if is_correct:
                                    metrics['correct_confidences'].append(conf_score)
                                else:
                                    metrics['incorrect_confidences'].append(conf_score)

                                frame_dists.append(float(dists_norm[b, j].item()))

                        if len(frame_dists) == K:
                            metrics['frame_errors'].append(float(np.mean(frame_dists)))
            
            # ---- Compute cached baseline metrics (only once, first model iteration) ----
            if batch_idx == 0 or len(cache_baseline_metrics['hrnet']['correct']) < len(ds_probe) * K:
                coords_hr = batch['coords_hrnet']
                coords_op = batch['coords_openpose']
                coords_mp = batch['coords_mediapipe']
                
                coords_hr_px = coords_hr * img_wh_tensor
                coords_op_px = coords_op * img_wh_tensor
                coords_mp_px = coords_mp * img_wh_tensor
                
                l2_dist_hr = torch.norm(coords_hr_px - coords_gt_px, dim=-1)
                l2_dist_op = torch.norm(coords_op_px - coords_gt_px, dim=-1)
                l2_dist_mp = torch.norm(coords_mp_px - coords_gt_px, dim=-1)
                
                dists_norm_hr = l2_dist_hr / couch_len.view(-1, 1)
                dists_norm_op = l2_dist_op / couch_len.view(-1, 1)
                dists_norm_mp = l2_dist_mp / couch_len.view(-1, 1)
                
                correct_hr = (dists_norm_hr < tau).float()
                correct_op = (dists_norm_op < tau).float()
                correct_mp = (dists_norm_mp < tau).float()
                
                for b in range(B):
                    hr_frame_dists, op_frame_dists, mp_frame_dists = [], [], []
                    for j in range(K):
                        if mask_hr[b, j] > 0:
                            cache_baseline_metrics['hrnet']['correct'].append(bool(correct_hr[b, j].item()))
                            cache_baseline_metrics['hrnet']['l2_dist_px'].append(float(l2_dist_hr[b, j].item()))
                            cache_baseline_metrics['hrnet']['l2_dist_norm'].append(float(dists_norm_hr[b, j].item()))
                            hr_frame_dists.append(float(dists_norm_hr[b, j].item()))
                        if mask_op[b, j] > 0:
                            cache_baseline_metrics['openpose']['correct'].append(bool(correct_op[b, j].item()))
                            cache_baseline_metrics['openpose']['l2_dist_px'].append(float(l2_dist_op[b, j].item()))
                            cache_baseline_metrics['openpose']['l2_dist_norm'].append(float(dists_norm_op[b, j].item()))
                            op_frame_dists.append(float(dists_norm_op[b, j].item()))
                        if mask_mp[b, j] > 0:
                            cache_baseline_metrics['mediapipe']['correct'].append(bool(correct_mp[b, j].item()))
                            cache_baseline_metrics['mediapipe']['l2_dist_px'].append(float(l2_dist_mp[b, j].item()))
                            cache_baseline_metrics['mediapipe']['l2_dist_norm'].append(float(dists_norm_mp[b, j].item()))
                            mp_frame_dists.append(float(dists_norm_mp[b, j].item()))
                    
                    if len(hr_frame_dists) == K:
                        cache_baseline_metrics['hrnet']['frame_errors'].append(float(np.mean(hr_frame_dists)))
                    if len(op_frame_dists) == K:
                        cache_baseline_metrics['openpose']['frame_errors'].append(float(np.mean(op_frame_dists)))
                    if len(mp_frame_dists) == K:
                        cache_baseline_metrics['mediapipe']['frame_errors'].append(float(np.mean(mp_frame_dists)))
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(fusion_loader)} batches")

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                print(f"  Stopping early: max_batches={max_batches}")
                break
    
    # =========================================================================
    # PHASE 5: Compile and save results
    # =========================================================================
    print("\n" + "-"*50)
    print("PHASE 5: Compiling results...")
    print("-"*50)
    
    all_results = {}
    
    for model_name, metrics in model_metrics.items():
        if not metrics['correct']:
            continue
        
        accuracy = float(np.mean(metrics['correct']))
        mae = float(np.mean(metrics['mae']))
        l2_dist_norm = float(np.mean(metrics['l2_dist_norm']))
        l2_dist_px = float(np.mean(metrics['l2_dist_px']))
        
        mean_conf_all = float(np.mean(metrics['confidences']))
        mean_conf_correct = float(np.mean(metrics['correct_confidences'])) if metrics['correct_confidences'] else 0.0
        mean_conf_incorrect = float(np.mean(metrics['incorrect_confidences'])) if metrics['incorrect_confidences'] else 0.0
        
        results = {
            'accuracy': accuracy,
            'mae': mae,
            'l2_dist_norm': l2_dist_norm,
            'l2_dist_px': l2_dist_px,
            'mean_confidence': mean_conf_all,
            'mean_confidence_correct': mean_conf_correct,
            'mean_confidence_incorrect': mean_conf_incorrect,
            'calibration_gap': mean_conf_correct - accuracy,
            'num_samples': len(metrics['correct']),
            'per_frame_mean_error': {},
            'baselines': {},
        }
        
        # Per-frame mean error
        if metrics['frame_errors']:
            results['per_frame_mean_error']['fusion'] = {
                'mean': float(np.mean(metrics['frame_errors'])),
                'std': float(np.std(metrics['frame_errors'])),
                'median': float(np.median(metrics['frame_errors'])),
                'min': float(np.min(metrics['frame_errors'])),
                'max': float(np.max(metrics['frame_errors'])),
                'count': len(metrics['frame_errors']),
            }
        
        # Add cached baselines
        for baseline_name, baseline_data in cache_baseline_metrics.items():
            if baseline_data['correct']:
                results['baselines'][baseline_name] = {
                    'accuracy': float(np.mean(baseline_data['correct'])),
                    'l2_dist_px': float(np.mean(baseline_data['l2_dist_px'])),
                    'l2_dist_norm': float(np.mean(baseline_data['l2_dist_norm'])),
                    'count': len(baseline_data['correct']),
                }
                if baseline_data['frame_errors']:
                    results['per_frame_mean_error'][baseline_name] = {
                        'mean': float(np.mean(baseline_data['frame_errors'])),
                        'std': float(np.std(baseline_data['frame_errors'])),
                        'count': len(baseline_data['frame_errors']),
                    }
        
        # Add trained baselines
        for baseline_name, baseline_data in baseline_results.items():
            if baseline_data['correct']:
                results['baselines'][baseline_name] = {
                    'accuracy': float(np.mean(baseline_data['correct'])),
                    'l2_dist_px': float(np.mean(baseline_data['l2_dist_px'])),
                    'l2_dist_norm': float(np.mean(baseline_data['l2_dist_norm'])),
                    'count': len(baseline_data['correct']),
                }
                if baseline_data['frame_errors']:
                    results['per_frame_mean_error'][baseline_name] = {
                        'mean': float(np.mean(baseline_data['frame_errors'])),
                        'std': float(np.std(baseline_data['frame_errors'])),
                        'count': len(baseline_data['frame_errors']),
                    }
        
        all_results[model_name] = results
        
        # Save individual results JSON
        if model_name in models_to_eval:
            output_json = models_to_eval[model_name]['output_json']
        else:
            output_json = str(output_path / f'{model_name}_results.json')
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {model_name} results to {output_json}")
        
        # Save importance weights if extracted
        if extract_importance and metrics['importance_weights']:
            importance_array = np.concatenate(metrics['importance_weights'], axis=0)
            rule_based_meta = {
                'uniform_fusion': (
                    'uniform',
                    'Uniform fusion: equal weights (1/3) per baseline; no learning.',
                ),
                'confidence_weighted_fusion': (
                    'confidence_weighted',
                    'Confidence-weighted fusion: weights proportional to detector confidence per joint.',
                ),
            }
            if model_name in models_to_eval:
                model_type_tag = models_to_eval[model_name]['type']
                importance_caption = None
            elif model_name in rule_based_meta:
                model_type_tag, importance_caption = rule_based_meta[model_name]
            else:
                model_type_tag = 'unknown'
                importance_caption = None

            _mn = ['hrnet', 'openpose', 'mediapipe']
            if importance_array.ndim >= 2 and importance_array.shape[-1] == 4:
                _mn = ['hrnet', 'openpose', 'mediapipe', 'vitpose']
            importance_data = {
                'importance_weights': importance_array.tolist(),
                'model_names': _mn,
                'num_frames': int(importance_array.shape[0]),
                'num_joints': K,
                'model_type': model_type_tag,
                'list_file': list_file,
                'dataset_root': dataset_root,
            }
            if importance_caption is not None:
                importance_data['importance_caption'] = importance_caption
            importance_file = str(output_path / f'{model_name}_importance_weights.json')
            with open(importance_file, 'w') as f:
                json.dump(importance_data, f, indent=2)
            print(f"  Saved {model_name} importance weights to {importance_file}")
            
            # Visualize importance if requested
            if visualize_importance:
                vis_dir = str(output_path / f'{model_name}_importance_visualizations')
                visualize_importance_weights(
                    importance_data=importance_data,
                    output_dir=vis_dir,
                    num_examples_per_model=5,
                    tau=tau,
                    list_file=list_file,
                    dataset_root=dataset_root,
                )
                visualize_significant_weight_changes(
                    importance_data=importance_data,
                    output_dir=vis_dir,
                    change_threshold=0.005,
                    max_changes=20,
                    window=5,
                    list_file=list_file,
                    dataset_root=dataset_root,
                )
                print(f"  Saved {model_name} importance visualizations to {vis_dir}")
    
    # =========================================================================
    # Print combined results table
    # =========================================================================
    elapsed = time.time() - start_time
    
    baseline_display_names = {
        'hrnet': 'HRNet/DEKR',
        'openpose': 'OpenPose',
        'mediapipe': 'MediaPipe',
        'dekr_trained': 'DEKR (Trained)',
        'dekr_trained_freeze': 'DEKR (Frozen Backbone)',
        'openpose_trained': 'OpenPose (Trained)',
        'openpose_trained_freeze': 'OpenPose (Frozen Backbone)',
    }
    
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS (threshold tau={tau})")
    print("="*80)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Test samples: {len(ds_probe)} frames, {K} joints each")
    print()
    
    # Combined table header
    print("FUSION MODELS:")
    print("-"*80)
    print(f"{'Model':<28} {'Accuracy':>10} {'L2 (px)':>10} {'L2 (norm)':>10} {'MAE':>8} {'Calib':>8}")
    print("-"*80)
    for model_name, results in all_results.items():
        display = model_name.replace('_', ' ').title()
        print(f"{display:<28} {results['accuracy']*100:>9.2f}% {results['l2_dist_px']:>10.3f} {results['l2_dist_norm']:>10.4f} {results['mae']:>8.4f} {results['calibration_gap']:>+8.4f}")
    
    print()
    print("BASELINE MODELS (Pre-trained / Cached):")
    print("-"*80)
    print(f"{'Model':<28} {'Accuracy':>10} {'L2 (px)':>10} {'L2 (norm)':>10} {'Samples':>10}")
    print("-"*80)
    for baseline_name in ['hrnet', 'openpose', 'mediapipe']:
        if baseline_name in cache_baseline_metrics and cache_baseline_metrics[baseline_name]['correct']:
            data = cache_baseline_metrics[baseline_name]
            acc = np.mean(data['correct']) * 100
            l2_px = np.mean(data['l2_dist_px'])
            l2_norm = np.mean(data['l2_dist_norm'])
            display = baseline_display_names.get(baseline_name, baseline_name)
            print(f"{display:<28} {acc:>9.2f}% {l2_px:>10.3f} {l2_norm:>10.4f} {len(data['correct']):>10}")
    
    print()
    print("BASELINE MODELS (Fine-tuned on Dataset):")
    print("-"*80)
    print(f"{'Model':<28} {'Accuracy':>10} {'L2 (px)':>10} {'L2 (norm)':>10} {'Samples':>10}")
    print("-"*80)
    for baseline_name in ['dekr_trained', 'dekr_trained_freeze', 'openpose_trained', 'openpose_trained_freeze']:
        if baseline_name in baseline_results and baseline_results[baseline_name]['correct']:
            data = baseline_results[baseline_name]
            acc = np.mean(data['correct']) * 100
            l2_px = np.mean(data['l2_dist_px'])
            l2_norm = np.mean(data['l2_dist_norm'])
            display = baseline_display_names.get(baseline_name, baseline_name)
            print(f"{display:<28} {acc:>9.2f}% {l2_px:>10.3f} {l2_norm:>10.4f} {len(data['correct']):>10}")
    
    # Per-frame error comparison (pick first fusion model for reference)
    first_model = list(all_results.keys())[0] if all_results else None
    if first_model and 'fusion' in all_results[first_model].get('per_frame_mean_error', {}):
        print()
        print("PER-FRAME MEAN ERROR (normalized):")
        print("-"*80)
        print(f"{'Model':<28} {'Mean':>10} {'Std Dev':>10} {'Frames':>10}")
        print("-"*80)
        
        # Fusion models
        for model_name, results in all_results.items():
            if 'fusion' in results.get('per_frame_mean_error', {}):
                fm = results['per_frame_mean_error']['fusion']
                display = model_name.replace('_', ' ').title()
                print(f"{display:<28} {fm['mean']:>10.4f} {fm['std']:>10.4f} {fm['count']:>10}")
        
        # Baselines
        for baseline_name in ['hrnet', 'openpose', 'mediapipe', 'dekr_trained', 'dekr_trained_freeze', 'openpose_trained', 'openpose_trained_freeze']:
            data = None
            if baseline_name in cache_baseline_metrics and cache_baseline_metrics[baseline_name]['frame_errors']:
                data = cache_baseline_metrics[baseline_name]
            elif baseline_name in baseline_results and baseline_results[baseline_name]['frame_errors']:
                data = baseline_results[baseline_name]
            
            if data and data['frame_errors']:
                display = baseline_display_names.get(baseline_name, baseline_name)
                mean_err = np.mean(data['frame_errors'])
                std_err = np.std(data['frame_errors'])
                print(f"{display:<28} {mean_err:>10.4f} {std_err:>10.4f} {len(data['frame_errors']):>10}")
    
    print()
    print("="*80)
    print("Evaluation complete!")
    print("="*80)
    
    # Save combined results
    combined_output = str(output_path / 'all_models_results.json')
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_output}")
    
    return all_results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate fusion model on test/val set')
    
    # === EFFICIENT EVAL-ALL MODE (recommended) ===
    p.add_argument('--eval-all', action='store_true',
                   help='Efficiently evaluate ALL fusion models in a single pass (recommended)')
    p.add_argument('--output-dir', default=None,
                   help='Output directory for all results (used with --eval-all)')
    p.add_argument('--mlp-fusion-checkpoint', default=None,
                   help='Path to MLP fusion checkpoint (used with --eval-all)')
    p.add_argument('--transformer-fusion-checkpoint-all', default=None,
                   help='Path to transformer fusion checkpoint (used with --eval-all)')
    p.add_argument('--internal-transformer-checkpoint', default=None,
                   help='Path to internal transformer fusion checkpoint (used with --eval-all)')
    p.add_argument('--mlp-fusion-4-checkpoint', default=None,
                   help='Path to 4-model MLP fusion checkpoint (ViTPose p2d_cache; used with --eval-all)')
    p.add_argument('--transformer-fusion-4-checkpoint', default=None,
                   help='Path to 4-model transformer fusion checkpoint (used with --eval-all)')
    p.add_argument('--internal-transformer-4-checkpoint', default=None,
                   help='Path to 4-model internal transformer fusion checkpoint (used with --eval-all)')
    p.add_argument('--prefetch-factor', type=int, default=2,
                   help='Prefetch factor for DataLoader (used with --eval-all, default: 2)')
    p.add_argument('--persistent-workers', action='store_true',
                   help='Use persistent workers for DataLoader (improves GPU utilization)')
    
    # === SINGLE MODEL MODE (legacy) ===
    p.add_argument('--checkpoint', default=None, help='Path to checkpoint_best.pt (single model mode)')
    p.add_argument('--list', required=True, help='List file (test.txt or val.txt)')
    p.add_argument('--root', required=True, help='Dataset root directory')
    p.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16 for GPU)')
    p.add_argument('--device', default='cuda', help='Device (default: cuda)')
    p.add_argument('--tau', type=float, default=0.05, help='Distance threshold (normalized)')
    p.add_argument('--output-json', default=None, help='Optional output JSON file for results')
    p.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (default: 4)')
    p.add_argument('--pin-memory', action='store_true', help='Pin memory for DataLoader')
    p.add_argument('--no-baseline-comparison', action='store_true', help='Skip baseline model comparison')
    p.add_argument('--trained-dekr-checkpoint', default=None, 
                   help='Path to trained DEKR checkpoint (e.g., Dataset/DEKR/logs_improved/checkpoint_best.pt)')
    p.add_argument('--trained-dekr-freeze-checkpoint', default=None,
                   help='Path to trained DEKR checkpoint with frozen backbone (e.g., Dataset/DEKR/logs_improved_freeze_backbone/checkpoint_best.pt)')
    p.add_argument('--trained-openpose-checkpoint', default=None,
                   help='Path to trained OpenPose checkpoint (e.g., Dataset/pytorch-openpose/logs_improved/checkpoint_best.pt)')
    p.add_argument('--trained-openpose-freeze-checkpoint', default=None,
                   help='Path to trained OpenPose checkpoint with frozen backbone (e.g., Dataset/pytorch-openpose/logs_improved_freeze_backbone/checkpoint_best.pt)')

    # Transformer fusion model support
    p.add_argument('--use-transformer-fusion', action='store_true',
                   help='Use transformer fusion model instead of MultiModelPoseFusion (auto-detected if checkpoint path contains "transformer")')
    p.add_argument('--use-transformer-internal-fusion', action='store_true',
                   help='Use INTERNAL transformer fusion model (auto-detected if checkpoint path contains "internal")')
    p.add_argument('--transformer-model-type', default='auto', choices=['auto', 'full', 'lightweight'],
                   help='Transformer model type: auto (detect from checkpoint), full (TransformerPoseFusion), or lightweight (LightweightTransformerFusion)')
    p.add_argument('--transformer-d-model', type=int, default=256,
                   help='Transformer hidden dimension (default: 256)')
    p.add_argument('--transformer-nhead', type=int, default=8,
                   help='Number of attention heads (default: 8)')
    p.add_argument('--transformer-num-encoder-layers', type=int, default=2,
                   help='Number of encoder layers (default: 2)')
    p.add_argument('--transformer-num-decoder-layers', type=int, default=2,
                   help='Number of decoder layers (default: 2)')
    p.add_argument('--transformer-dim-feedforward', type=int, default=512,
                   help='FFN hidden dimension (default: 512)')
    p.add_argument('--transformer-dropout', type=float, default=0.1,
                   help='Dropout probability (default: 0.1)')

    # Trained vs untrained comparisons + frame overlays
    p.add_argument('--compare-trained-untrained', action='store_true',
                   help='Save trained-vs-untrained baseline comparisons (per-frame errors, derived weights, frame overlays)')
    p.add_argument('--comparison-output-dir', default=None,
                   help='Output directory for trained-vs-untrained artifacts (defaults near --output-json)')
    p.add_argument('--max-frames-to-visualize', type=int, default=12,
                   help='How many top-improvement frames to save per model (DEKR/OpenPose)')
    
    # Fusion vs Transformer Fusion comparison
    p.add_argument('--compare-fusion-transformer', action='store_true',
                   help='Compare fusion model with transformer fusion model (requires --transformer-fusion-checkpoint)')
    p.add_argument('--transformer-fusion-checkpoint', default=None,
                   help='Path to transformer fusion checkpoint for comparison (used with --compare-fusion-transformer)')
    
    # Importance weight extraction and visualization
    p.add_argument('--extract-importance', action='store_true', 
                   help='Extract importance weights from attention mechanism')
    p.add_argument('--importance-output', default=None,
                   help='Output JSON file for importance weights (used with --extract-importance)')
    p.add_argument('--visualize-importance', action='store_true',
                   help='Visualize importance weights (requires --extract-importance or --importance-input)')
    p.add_argument('--importance-input', default=None,
                   help='Input JSON file with importance weights (used with --visualize-importance)')
    p.add_argument('--visualization-output-dir', default='importance_visualizations',
                   help='Output directory for visualization plots')
    p.add_argument('--num-examples', type=int, default=5,
                   help='Number of examples to show per model in visualization')
    p.add_argument('--plot-time-series', action='store_true',
                   help='Create line graphs showing importance weights over time')
    p.add_argument('--smoothing-window', type=int, default=1,
                   help='Moving average window size for time series plots (1 = no smoothing)')
    p.add_argument('--joints-to-plot', nargs='+', type=int, default=None,
                   help='Specific joint indices to plot (default: all joints)')
    p.add_argument('--inspect-frames', nargs='+', type=int, default=None,
                   help='Frame indices to inspect (importance window + frame image)')
    p.add_argument('--inspect-window', type=int, default=5,
                   help='Window (±) around inspect frames for importance plots')
    p.add_argument('--visualize-weight-changes', action='store_true',
                   help='Automatically visualize significant weight changes with 10-frame windows')
    p.add_argument('--weight-change-threshold', type=float, default=0.005,
                   help='Minimum change magnitude to consider significant (default 0.005)')
    p.add_argument('--max-weight-changes', type=int, default=20,
                   help='Maximum number of weight changes to visualize (default 20)')
    p.add_argument('--weight-change-window', type=int, default=5,
                   help='Number of frames before/after change to show (default 5, total 11 frames)')
    p.add_argument('--no-rule-based-fusion', action='store_true',
                   help='Skip uniform and confidence-weighted fusion baselines (--eval-all only)')
    p.add_argument('--max-batches', type=int, default=None,
                   help='Stop after N fusion-loader batches (--eval-all only; optional smoke test)')
    p.add_argument('--skip-trained-baselines', action='store_true',
                   help='Skip Phase 3 trained DEKR/OpenPose (--eval-all only)')
    
    args = p.parse_args()
    
    # =========================================================================
    # EFFICIENT EVAL-ALL MODE (recommended for evaluating multiple models)
    # =========================================================================
    if args.eval_all:
        if not args.output_dir:
            args.output_dir = str(Path(args.root).parent / 'fusion' / 'eval_results')
        
        eval_all_models(
            list_file=args.list,
            dataset_root=args.root,
            output_dir=args.output_dir,
            mlp_fusion_checkpoint=args.mlp_fusion_checkpoint,
            mlp_fusion_4_checkpoint=args.mlp_fusion_4_checkpoint,
            transformer_fusion_checkpoint=args.transformer_fusion_checkpoint_all,
            transformer_fusion_4_checkpoint=args.transformer_fusion_4_checkpoint,
            internal_transformer_checkpoint=args.internal_transformer_checkpoint,
            internal_transformer_4_checkpoint=args.internal_transformer_4_checkpoint,
            trained_dekr_checkpoint=args.trained_dekr_checkpoint,
            trained_dekr_freeze_checkpoint=args.trained_dekr_freeze_checkpoint,
            trained_openpose_checkpoint=args.trained_openpose_checkpoint,
            trained_openpose_freeze_checkpoint=args.trained_openpose_freeze_checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            tau=args.tau,
            extract_importance=args.extract_importance,
            visualize_importance=args.visualize_importance,
            max_frames_to_visualize=args.max_frames_to_visualize,
            include_rule_based_fusion=not args.no_rule_based_fusion,
            max_batches=args.max_batches,
            skip_trained_baselines=args.skip_trained_baselines,
            transformer_d_model=args.transformer_d_model,
            transformer_nhead=args.transformer_nhead,
            transformer_num_encoder_layers=args.transformer_num_encoder_layers,
            transformer_num_decoder_layers=args.transformer_num_decoder_layers,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            transformer_dropout=args.transformer_dropout,
        )
        sys.exit(0)
    
    # =========================================================================
    # LEGACY SINGLE-MODEL MODE
    # =========================================================================
    if not args.checkpoint:
        p.error("--checkpoint is required when not using --eval-all mode")
    
    # Extract importance weights if requested
    if args.extract_importance:
        print("="*60)
        print("EXTRACTING IMPORTANCE WEIGHTS")
        print("="*60)
        importance_data = extract_importance_weights(
            checkpoint_path=args.checkpoint,
            list_file=args.list,
            dataset_root=args.root,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            output_file=args.importance_output,
            use_transformer_fusion=args.use_transformer_fusion,
            use_transformer_internal_fusion=args.use_transformer_internal_fusion,
            transformer_model_type=args.transformer_model_type,
            transformer_d_model=args.transformer_d_model,
            transformer_nhead=args.transformer_nhead,
            transformer_num_encoder_layers=args.transformer_num_encoder_layers,
            transformer_num_decoder_layers=args.transformer_num_decoder_layers,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            transformer_dropout=args.transformer_dropout,
        )
        print(f"Extracted importance weights for {importance_data['num_frames']} frames")
        
        # Visualize if requested
        if args.visualize_importance:
            print("\n" + "="*60)
            print("VISUALIZING IMPORTANCE WEIGHTS")
            print("="*60)
            visualize_importance_weights(
                importance_data=importance_data,
                output_dir=args.visualization_output_dir,
                num_examples_per_model=args.num_examples,
                tau=args.tau,
                list_file=args.list,
                dataset_root=args.root,
            )
            if args.inspect_frames:
                visualize_frame_context(
                    importance_data=importance_data,
                    frame_indices=args.inspect_frames,
                    output_dir=args.visualization_output_dir,
                    window=args.inspect_window,
                    list_file=args.list,
                    dataset_root=args.root,
                )
            if args.visualize_weight_changes:
                print("\n" + "="*60)
                print("VISUALIZING SIGNIFICANT WEIGHT CHANGES")
                print("="*60)
                visualize_significant_weight_changes(
                    importance_data=importance_data,
                    output_dir=args.visualization_output_dir,
                    change_threshold=args.weight_change_threshold,
                    max_changes=args.max_weight_changes,
                    window=args.weight_change_window,
                    list_file=args.list,
                    dataset_root=args.root,
                )
    elif args.visualize_importance:
        # Load from file if provided
        if args.importance_input:
            print("="*60)
            print("LOADING IMPORTANCE WEIGHTS FROM FILE")
            print("="*60)
            with open(args.importance_input, 'r') as f:
                importance_data = json.load(f)
            print(f"Loaded importance weights for {importance_data['num_frames']} frames")
            
            print("\n" + "="*60)
            print("VISUALIZING IMPORTANCE WEIGHTS")
            print("="*60)
            visualize_importance_weights(
                importance_data=importance_data,
                output_dir=args.visualization_output_dir,
                num_examples_per_model=args.num_examples,
                tau=args.tau,
                list_file=args.list,
                dataset_root=args.root,
            )
            
            # Plot time series if requested
            if args.plot_time_series:
                print("\n" + "="*60)
                print("CREATING TIME SERIES PLOTS")
                print("="*60)
                plot_importance_over_time(
                    importance_data=importance_data,
                    output_dir=args.visualization_output_dir,
                    average_across_joints=True,
                    joint_specific=args.joints_to_plot,
                    smoothing_window=args.smoothing_window,
                    list_file=args.list,
                    dataset_root=args.root,
                )
            if args.inspect_frames:
                visualize_frame_context(
                    importance_data=importance_data,
                    frame_indices=args.inspect_frames,
                    output_dir=args.visualization_output_dir,
                    window=args.inspect_window,
                    list_file=args.list,
                    dataset_root=args.root,
                )
            if args.visualize_weight_changes:
                print("\n" + "="*60)
                print("VISUALIZING SIGNIFICANT WEIGHT CHANGES")
                print("="*60)
                visualize_significant_weight_changes(
                    importance_data=importance_data,
                    output_dir=args.visualization_output_dir,
                    change_threshold=args.weight_change_threshold,
                    max_changes=args.max_weight_changes,
                    window=args.weight_change_window,
                    list_file=args.list,
                    dataset_root=args.root,
                )
        else:
            print("ERROR: --visualize-importance requires either --extract-importance or --importance-input")
    
    # Standalone time series plotting
    if args.plot_time_series and args.importance_input and not args.visualize_importance:
        print("="*60)
        print("LOADING IMPORTANCE WEIGHTS FROM FILE FOR TIME SERIES")
        print("="*60)
        with open(args.importance_input, 'r') as f:
            importance_data = json.load(f)
        print(f"Loaded importance weights for {importance_data['num_frames']} frames")
        
        print("\n" + "="*60)
        print("CREATING TIME SERIES PLOTS")
        print("="*60)
        plot_importance_over_time(
            importance_data=importance_data,
            output_dir=args.visualization_output_dir,
            average_across_joints=True,
            joint_specific=args.joints_to_plot,
            smoothing_window=args.smoothing_window,
            list_file=getattr(args, 'list', None),
            dataset_root=getattr(args, 'root', None),
        )
        if args.inspect_frames:
            visualize_frame_context(
                importance_data=importance_data,
                frame_indices=args.inspect_frames,
                output_dir=args.visualization_output_dir,
                window=args.inspect_window,
                list_file=args.list,
                dataset_root=args.root,
            )
    elif not args.extract_importance and not args.visualize_importance and not args.plot_time_series:
        # Regular evaluation
        evaluate_model(
            checkpoint_path=args.checkpoint,
            list_file=args.list,
            dataset_root=args.root,
            batch_size=args.batch_size,
            device=args.device,
            tau=args.tau,
            output_json=args.output_json,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            compare_baselines=not args.no_baseline_comparison,
            trained_dekr_checkpoint=args.trained_dekr_checkpoint,
            trained_dekr_freeze_checkpoint=args.trained_dekr_freeze_checkpoint,
            trained_openpose_checkpoint=args.trained_openpose_checkpoint,
            trained_openpose_freeze_checkpoint=args.trained_openpose_freeze_checkpoint,
            compare_trained_untrained=args.compare_trained_untrained,
            comparison_output_dir=args.comparison_output_dir,
            max_frames_to_visualize=args.max_frames_to_visualize,
            use_transformer_fusion=args.use_transformer_fusion,
            use_transformer_internal_fusion=args.use_transformer_internal_fusion,
            compare_fusion_transformer=args.compare_fusion_transformer,
            transformer_fusion_checkpoint=args.transformer_fusion_checkpoint,
            transformer_model_type=args.transformer_model_type,
            transformer_d_model=args.transformer_d_model,
            transformer_nhead=args.transformer_nhead,
            transformer_num_encoder_layers=args.transformer_num_encoder_layers,
            transformer_num_decoder_layers=args.transformer_num_decoder_layers,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            transformer_dropout=args.transformer_dropout,
        )
