"""Focused loader + dry-run for fusion using precomputed p2d_cache files.

This module provides a minimal, cache-first dataloader that reads per-video
p2d caches produced by the analysis scripts and yields batches suitable for
`MultiModelPoseFusion` training. It purposely avoids any runtime inference
and synthetic prediction code — missing model predictions are zero-filled
and have mask=0.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.mlp_fusion import MultiModelPoseFusion
from data.pytorch_dataset import VideoFrameKeypointDataset


def _parse_rel_path(rel: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        p = Path(rel)
        parts = p.parts
        if len(parts) >= 4:
            patient = parts[-3]
            exercise = parts[-2]
            cam = Path(parts[-1]).stem
            return patient, exercise, cam
    except Exception:
        pass
    return None, None, None


def _read_p2d_cache(dataset_root: str, rel: str, ann_idx: int, model_name: str):
    patient, exercise, cam = _parse_rel_path(rel)
    if patient is None:
        return None
    cache_path = Path(dataset_root) / 'clips_mp4' / 'p2d_cache' / str(patient) / str(exercise) / f"{cam}_{model_name}_p2d.txt"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'r') as f:
            lines = f.readlines()
        if ann_idx < 0 or ann_idx >= len(lines):
            return None
        parts = lines[ann_idx].strip().split()
        vals = []
        for x in parts:
            try:
                vals.append(float(x))
            except Exception:
                vals.append(0.0)
        if len(vals) < 9:
            vals += [0.0] * (9 - len(vals))
        arr = np.array(vals, dtype=np.float32).reshape(3, 3)
        coords = arr[:, :2]
        conf = arr[:, 2]
        mask = (conf > 0.0).astype(np.float32)
        return coords, conf, mask
    except Exception:
        return None


def _cache_file_path(dataset_root: str, rel: str, model_name: str) -> Path:
    """Return the expected p2d cache file path for a given sample rel path.

    This helper is used for strict existence checks when `--require-caches` is
    enabled.
    """
    patient, exercise, cam = _parse_rel_path(rel)
    if patient is None:
        return Path(dataset_root) / 'clips_mp4' / 'p2d_cache' / 'MISSING_REL'
    return Path(dataset_root) / 'clips_mp4' / 'p2d_cache' / str(patient) / str(exercise) / f"{cam}_{model_name}_p2d.txt"


def build_dataloader_from_list(
    list_file: str,
    dataset_root: str,
    batch_size: int = 4,
    output_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    require_caches: bool = False,
    include_vitpose: bool = False,
) -> DataLoader:
    ds = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=output_size)

    K = None

    def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        nonlocal K
        if len(samples) == 0:
            return {}
        if K is None:
            K = samples[0]['keypoints'].shape[0]

        coords_hr = []
        conf_hr = []
        mask_hr = []
        coords_op = []
        conf_op = []
        mask_op = []
        coords_mp = []
        conf_mp = []
        mask_mp = []
        coords_pb = []
        conf_pb = []
        mask_pb = []
        coords_gt = []
        coords_gt_px = []
        couch_len_list = []

        H, W = output_size

        for s in samples:
            rel = s['meta']['video_rel_path']
            frame_idx = int(s['meta']['frame_index'])
            ann_idx = frame_idx - ds.frame_offset

            # If caller requested strict cache presence, check expected cache files
            if require_caches:
                missing = []
                models = ('dekr', 'openpose', 'mediapipe')
                if include_vitpose:
                    models = models + ('vitpose',)
                for m in models:
                    cp = _cache_file_path(dataset_root, rel, m)
                    if not cp.exists():
                        missing.append(str(cp))
                if missing:
                    raise FileNotFoundError(
                        f"Missing p2d cache files for sample {rel} frame {frame_idx}: " + ", ".join(missing)
                    )

            p2d_dekr = _read_p2d_cache(dataset_root, rel, ann_idx, 'dekr')
            p2d_op = _read_p2d_cache(dataset_root, rel, ann_idx, 'openpose')
            p2d_mp = _read_p2d_cache(dataset_root, rel, ann_idx, 'mediapipe')
            p2d_vp = _read_p2d_cache(dataset_root, rel, ann_idx, 'vitpose') if include_vitpose else None

            def _make_tensors_from_cache(entry, orig_size_hw: Tuple[int, int]):
                if entry is None:
                    coords_px = np.zeros((K, 2), dtype=np.float32)
                    conf = np.zeros((K,), dtype=np.float32)
                    mask = np.zeros((K,), dtype=np.float32)
                else:
                    coords_px, conf, mask = entry
                    if coords_px.shape[0] < K:
                        pad = K - coords_px.shape[0]
                        coords_px = np.vstack([coords_px, np.zeros((pad, 2), dtype=np.float32)])
                        conf = np.concatenate([conf, np.zeros((pad,), dtype=np.float32)])
                        mask = np.concatenate([mask, np.zeros((pad,), dtype=np.float32)])
                # Cache coords are stored in ORIGINAL resolution space (e.g., 1280x720).
                # Scale them down to 256x256 space to match the training/evaluation coordinate space.
                # This is consistent with how ground truth is scaled (see lines 198-207).
                orig_h, orig_w = orig_size_hw
                if orig_w > 0 and orig_h > 0:
                    # Scale from original resolution to 256x256
                    scale_x = float(W) / float(orig_w)  # 256 / orig_w
                    scale_y = float(H) / float(orig_h)  # 256 / orig_h
                    coords_px[:, 0] = coords_px[:, 0] * scale_x  # Scale x coordinates
                    coords_px[:, 1] = coords_px[:, 1] * scale_y  # Scale y coordinates
                # Now coords_px is in 256x256 pixel space, normalize to [0, 1] based on 256x256
                coords_norm = coords_px / np.array([W, H], dtype=np.float32)
                coords_t = torch.from_numpy(coords_norm).float()
                conf_t = torch.from_numpy(conf).unsqueeze(-1).float()
                mask_t = torch.from_numpy(mask).unsqueeze(-1).float()
                return coords_t, conf_t, mask_t, coords_px

            # Original size is available from dataset metadata
            orig_h, orig_w = s['meta'].get('orig_size', (H, W))
            coords_hr_t, conf_hr_t, mask_hr_t, raw_hr_px = _make_tensors_from_cache(p2d_dekr, (int(orig_h), int(orig_w)))
            coords_op_t, conf_op_t, mask_op_t, raw_op_px = _make_tensors_from_cache(p2d_op, (int(orig_h), int(orig_w)))
            coords_mp_t, conf_mp_t, mask_mp_t, raw_mp_px = _make_tensors_from_cache(p2d_mp, (int(orig_h), int(orig_w)))
            if include_vitpose:
                # ViTPose cache is already in 256x256 space, so pass (H, W)
                # as orig_size to make the scale factor 1.0 (no-op).
                coords_pb_t, conf_pb_t, mask_pb_t, _ = _make_tensors_from_cache(
                    p2d_vp, (H, W)
                )
            else:
                coords_pb_t = conf_pb_t = mask_pb_t = None

            coords_hr.append(coords_hr_t)
            conf_hr.append(conf_hr_t)
            mask_hr.append(mask_hr_t)
            coords_op.append(coords_op_t)
            conf_op.append(conf_op_t)
            mask_op.append(mask_op_t)
            coords_mp.append(coords_mp_t)
            conf_mp.append(conf_mp_t)
            mask_mp.append(mask_mp_t)
            if include_vitpose:
                coords_pb.append(coords_pb_t)
                conf_pb.append(conf_pb_t)
                mask_pb.append(mask_pb_t)

            kps = s['keypoints']
            # kps is a torch tensor (K,3) on CPU; convert to numpy for augmentation
            kp_px_np = kps[:, :2].numpy().astype(np.float32)
            # NOTE: augmentation is applied by passing flags into build_dataloader_from_list
            # (handled by wrapper callers). If augment fields are not set, this is a no-op.
            if getattr(build_dataloader_from_list, '_augment_flags', None):
                af = build_dataloader_from_list._augment_flags
                if af.get('augment_keypoints', False):
                    jitter = af.get('jitter_px', 0.0)
                    occp = af.get('occlusion_prob', 0.0)
                    if jitter and jitter > 0.0:
                        kp_px_np = kp_px_np + np.random.normal(0.0, float(jitter), size=kp_px_np.shape).astype(np.float32)
                    if occp and occp > 0.0:
                        occ_mask = np.random.rand(kp_px_np.shape[0]) < float(occp)
                        kp_px_np[occ_mask, :] = 0.0
            gt_px = torch.from_numpy(kp_px_np).float()
            
            # APPROACH 2: Scale ground truth to 256x256 space to match baseline model predictions
            # Ground truth is loaded in original pixel space, scale it down to 256x256
            orig_h, orig_w = s['meta'].get('orig_size', (H, W))
            if orig_w > 0 and orig_h > 0:
                # Scale from original resolution to 256x256
                scale_x = float(W) / float(orig_w)  # 256 / orig_w
                scale_y = float(H) / float(orig_h)  # 256 / orig_h
                gt_px[:, 0] = gt_px[:, 0] * scale_x  # Scale down to 256x256
                gt_px[:, 1] = gt_px[:, 1] * scale_y
            # Now gt_px is in 256x256 pixel space
            
            # Normalize to [0, 1] based on 256x256
            gt = gt_px.clone()
            gt[:, 0] = gt[:, 0] / float(W)  # W = 256
            gt[:, 1] = gt[:, 1] / float(H)  # H = 256
            coords_gt.append(gt)
            coords_gt_px.append(gt_px)  # Already in 256x256 space

            rel_patient, rel_ex, rel_cam = _parse_rel_path(rel)
            cam = rel_cam
            # Use 256x256 diagonal as couch_len (since everything is now in 256x256 space)
            couch_len = np.sqrt(float(H)**2 + float(W)**2)  # sqrt(256^2 + 256^2)
            couch_len_list.append(torch.tensor([couch_len], dtype=torch.float32))

        # Use 256x256 for img_wh to match the 256x256 space we're training in
        # (consistent with how coords_gt_px and couch_len are in 256x256 space)
        img_wh_list = []
        for s in samples:
            img_wh_list.append([float(W), float(H)])  # [256, 256] for all samples
        
        batch = {
            'coords_hrnet': torch.stack(coords_hr, dim=0),
            'conf_hrnet': torch.stack(conf_hr, dim=0),
            'mask_hrnet': torch.stack(mask_hr, dim=0),
            'coords_openpose': torch.stack(coords_op, dim=0),
            'conf_openpose': torch.stack(conf_op, dim=0),
            'mask_openpose': torch.stack(mask_op, dim=0),
            'coords_mediapipe': torch.stack(coords_mp, dim=0),
            'conf_mediapipe': torch.stack(conf_mp, dim=0),
            'mask_mediapipe': torch.stack(mask_mp, dim=0),
            'coords_gt': torch.stack(coords_gt, dim=0),
            'coords_gt_px': torch.stack(coords_gt_px, dim=0),
            'couch_len': torch.cat(couch_len_list, dim=0),
            'img_wh': torch.tensor(img_wh_list, dtype=torch.float32),  # (B, 2) with [256, 256] for all samples (256x256 space)
        }
        if include_vitpose:
            batch['coords_vitpose'] = torch.stack(coords_pb, dim=0)
            batch['conf_vitpose'] = torch.stack(conf_pb, dim=0)
            batch['mask_vitpose'] = torch.stack(mask_pb, dim=0)
        return batch

    # Only enable persistent_workers when using >0 workers
    persistent = persistent_workers if (num_workers and persistent_workers) else False
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, persistent_workers=persistent)
    return loader


def train_step(batch: Dict[str, torch.Tensor], model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, lambda_conf: float = 0.1, tau: float = 0.05) -> float:
    model.train()
    for k in list(batch.keys()):
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    coords_hrnet = batch['coords_hrnet']
    conf_hrnet = batch['conf_hrnet']
    coords_openpose = batch['coords_openpose']
    conf_openpose = batch['conf_openpose']
    coords_mediapipe = batch['coords_mediapipe']
    conf_mediapipe = batch['conf_mediapipe']
    coords_gt = batch['coords_gt']
    coords_gt_px = batch.get('coords_gt_px', None)
    couch_len = batch.get('couch_len', None)
    img_wh = batch.get('img_wh', None)
    mask_hr = batch.get('mask_hrnet', None)
    mask_op = batch.get('mask_openpose', None)
    mask_mp = batch.get('mask_mediapipe', None)

    optimizer.zero_grad()
    coords_fused, conf_fused_logits, _ = model(
        coords_hrnet, conf_hrnet,
        coords_openpose, conf_openpose,
        coords_mediapipe, conf_mediapipe,
        return_attention_weights=False,
    )

    # supervise where at least one model provided the joint
    mask_any = ((mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1)) > 0.0)
    if mask_any.sum() > 0:
        coords_fused_masked = coords_fused[mask_any]
        coords_gt_masked = coords_gt[mask_any]
        loss_coords = torch.nn.functional.smooth_l1_loss(coords_fused_masked, coords_gt_masked)

        # Everything is in 256x256 space now:
        # - coords_fused: normalized [0,1] w.r.t (W,H) = (256,256)
        # - coords_gt_px: pixel coordinates in 256x256 space
        # - img_wh: [256, 256] for all samples (from dataloader)
        # Convert fused coords to 256x256 pixel space using img_wh
        img_wh_tensor = img_wh.view(-1, 1, 2)  # (B, 1, 2)
        coords_fused_px = coords_fused * img_wh_tensor  # (B, K, 2) in 256x256 pixel space
        dists_px = torch.norm(coords_fused_px - coords_gt_px, dim=-1)  # (B, K)
        # couch_len is the 256x256 diagonal for all samples
        dists_norm = dists_px / couch_len.view(-1, 1)
        good = (dists_norm < tau).float()

        conf_logits = conf_fused_logits.squeeze(-1)
        pos_logits = conf_logits[mask_any]
        pos_targets = good[mask_any]
        if pos_logits.numel() > 0:
            bce = torch.nn.BCEWithLogitsLoss()
            loss_conf = bce(pos_logits, pos_targets)
        else:
            loss_conf = torch.tensor(0.0, device=coords_fused.device)

        loss = loss_coords + lambda_conf * loss_conf
    else:
        loss = torch.tensor(0.0, device=coords_fused.device)

    loss.backward()
    optimizer.step()
    return float(loss.item())


def dry_run(list_file: str, dataset_root: str, batch_size: int = 4, output_size: Tuple[int, int] = (256, 256), num_batches: int = 1, require_caches: bool = False, num_workers: int = 0, pin_memory: bool = False, persistent_workers: bool = False, augment_keypoints: bool = False, jitter_px: float = 0.0, occlusion_prob: float = 0.0):
    # attach augment flags for collate (read by collate_fn)
    build_dataloader_from_list._augment_flags = {'augment_keypoints': augment_keypoints, 'jitter_px': float(jitter_px), 'occlusion_prob': float(occlusion_prob)}
    loader = build_dataloader_from_list(list_file, dataset_root, batch_size=batch_size, output_size=output_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, require_caches=require_caches)
    # clear augment flags to avoid leaking to other callers
    build_dataloader_from_list._augment_flags = None
    it = iter(loader)
    for i in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            print('No batches available')
            return
        print(f"Batch {i}:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)}")


def quick_train(list_file: str, dataset_root: str, epochs: int = 1, max_iters: int = 100, batch_size: int = 8, lr: float = 1e-3, device: str = 'cpu', num_joints: Optional[int] = None, require_caches: bool = False, save_dir: Optional[str] = None, resume: Optional[str] = None, val_list: Optional[str] = None, num_workers: int = 0, pin_memory: bool = False, persistent_workers: bool = False, weight_decay: float = 0.0, lr_scheduler_patience: int = 3, lr_scheduler_factor: float = 0.1, early_stop_patience: int = 5, augment_keypoints: bool = False, jitter_px: float = 0.0, occlusion_prob: float = 0.0):
    """Simple training loop using `train_step` and `MultiModelPoseFusion`.

    This is intended as a lightweight runner for experimentation. It performs
    epoch-based loops and stops once `max_iters` iterations have been run.
    """
    device_t = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    # Inspect dataset to determine number of joints (K) if not provided
    ds_probe = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256,256))
    if len(ds_probe) == 0:
        raise RuntimeError('Dataset appears empty; cannot determine number of joints')
    K = ds_probe[0]['keypoints'].shape[0]
    if num_joints is None:
        used_num_joints = K
    else:
        if num_joints != K:
            print(f'Warning: requested num_joints={num_joints} differs from dataset K={K}; using K={K}')
        used_num_joints = K

    # attach augment flags for training loader (collate will read these)
    build_dataloader_from_list._augment_flags = {'augment_keypoints': augment_keypoints, 'jitter_px': float(jitter_px), 'occlusion_prob': float(occlusion_prob)}
    loader = build_dataloader_from_list(list_file, dataset_root, batch_size=batch_size, output_size=(256,256), shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, require_caches=require_caches)
    # clear after creating loader
    build_dataloader_from_list._augment_flags = None

    model = MultiModelPoseFusion(num_joints=used_num_joints).to(device_t)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))

    # scheduler & early stopping (only meaningful if we have a validation set)
    scheduler = None
    if val_list is not None:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=int(lr_scheduler_patience), factor=float(lr_scheduler_factor))

    start_epoch = 0
    it = 0
    best_val = float('inf')
    epochs_since_improve = 0
    # resume from checkpoint if provided
    if resume is not None:
        ck_path = Path(resume)
        if ck_path.exists():
            d = torch.load(str(ck_path), map_location=device_t)
            if 'model_state' in d:
                model.load_state_dict(d['model_state'])
            if 'optimizer' in d:
                try:
                    optimizer.load_state_dict(d['optimizer'])
                except Exception:
                    print('Warning: optimizer state could not be fully restored from checkpoint')
            start_epoch = int(d.get('epoch', 0))
            it = int(d.get('iter', 0))
            print(f'Resuming from checkpoint {ck_path} at epoch {start_epoch} iter {it}')
        else:
            raise FileNotFoundError(f'Requested resume checkpoint not found: {resume}')
    for epoch in range(start_epoch, epochs):
        for batch in loader:
            loss = train_step(batch, model, optimizer, device_t)
            it += 1
            if it % 10 == 0 or it == 1:
                print(f"Epoch {epoch} iter {it} loss {loss:.6f}")
            if it >= max_iters:
                break
        # save checkpoint per epoch if requested
        if save_dir is not None:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            ck = p / f'checkpoint_epoch{epoch+1}.pt'
            torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'iter': it}, str(ck))
            print(f"Saved checkpoint: {ck}")
        # run validation if requested
        if val_list is not None:
            # create validation loader (no augmentation)
            val_loader = build_dataloader_from_list(val_list, dataset_root, batch_size=batch_size, output_size=(256,256), shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, require_caches=require_caches)
            val_loss = validate(model, val_loader, device_t)
            print(f"Validation loss after epoch {epoch}: {val_loss:.6f}")
            # scheduler step
            if scheduler is not None:
                scheduler.step(val_loss)
            # early stopping + save best checkpoint
            if val_loss < best_val:
                best_val = float(val_loss)
                epochs_since_improve = 0
                if save_dir is not None:
                    best_ck = Path(save_dir) / 'checkpoint_best.pt'
                    torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'iter': it, 'val_loss': val_loss}, str(best_ck))
                    print(f"Saved new best checkpoint: {best_ck}")
            else:
                epochs_since_improve += 1
                print(f"No improvement for {epochs_since_improve} epochs (patience={early_stop_patience})")
            if early_stop_patience is not None and epochs_since_improve >= int(early_stop_patience):
                print(f"Early stopping triggered after {epochs_since_improve} epochs without improvement")
                break
        if it >= max_iters:
            break
    return model


def compute_val_loss_batch(batch: Dict[str, torch.Tensor], model: torch.nn.Module, device: torch.device, lambda_conf: float = 0.1, tau: float = 0.05) -> float:
    """Compute loss for a single batch in evaluation mode (no grad)."""
    model.eval()
    for k in list(batch.keys()):
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    coords_hrnet = batch['coords_hrnet']
    conf_hrnet = batch['conf_hrnet']
    coords_openpose = batch['coords_openpose']
    conf_openpose = batch['conf_openpose']
    coords_mediapipe = batch['coords_mediapipe']
    conf_mediapipe = batch['conf_mediapipe']
    coords_gt = batch['coords_gt']
    coords_gt_px = batch.get('coords_gt_px', None)  # 256x256 pixel space
    couch_len = batch.get('couch_len', None)        # 256x256 diagonal per sample
    img_wh = batch.get('img_wh', None)              # [256, 256] for all samples (256x256 space)
    mask_hr = batch.get('mask_hrnet', None)
    mask_op = batch.get('mask_openpose', None)
    mask_mp = batch.get('mask_mediapipe', None)

    with torch.no_grad():
        coords_fused, conf_fused_logits, _ = model(
            coords_hrnet, conf_hrnet,
            coords_openpose, conf_openpose,
            coords_mediapipe, conf_mediapipe,
            return_attention_weights=False,
        )

        mask_any = ((mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1)) > 0.0)
        if mask_any.sum() > 0:
            coords_fused_masked = coords_fused[mask_any]
            coords_gt_masked = coords_gt[mask_any]
            loss_coords = torch.nn.functional.smooth_l1_loss(coords_fused_masked, coords_gt_masked)

            # Everything is in 256x256 space now:
            # - coords_fused: normalized [0,1] w.r.t (W,H) = (256,256)
            # - coords_gt_px: pixel coordinates in 256x256 space
            # - img_wh: [256, 256] for all samples (from dataloader)
            # Convert fused coords to 256x256 pixel space using img_wh
            img_wh_tensor = img_wh.view(-1, 1, 2)  # (B, 1, 2)
            coords_fused_px = coords_fused * img_wh_tensor  # (B, K, 2) in 256x256 pixel space
            dists_px = torch.norm(coords_fused_px - coords_gt_px, dim=-1)
            dists_norm = dists_px / couch_len.view(-1, 1)
            good = (dists_norm < tau).float()

            conf_logits = conf_fused_logits.squeeze(-1)
            pos_logits = conf_logits[mask_any]
            pos_targets = good[mask_any]
            if pos_logits.numel() > 0:
                bce = torch.nn.BCEWithLogitsLoss()
                loss_conf = bce(pos_logits, pos_targets)
            else:
                loss_conf = torch.tensor(0.0, device=coords_fused.device)

            loss = loss_coords + lambda_conf * loss_conf
        else:
            loss = torch.tensor(0.0, device=coords_fused.device)

    return float(loss.item())


def validate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Run validation over `val_loader` and return average loss."""
    total = 0.0
    n = 0
    for batch in val_loader:
        l = compute_val_loss_batch(batch, model, device)
        total += l
        n += 1
    return total / max(1, n)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--list', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-batches', type=int, default=1)
    p.add_argument('--require-caches', action='store_true', help='Fail if any expected p2d cache file is missing (do not zero-fill)')
    p.add_argument('--train', action='store_true', help='Run a short training loop instead of dry-run')
    p.add_argument('--epochs', type=int, default=1, help='Number of epochs to run in train mode')
    p.add_argument('--max-iters', type=int, default=100, help='Maximum training iterations to run')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam')
    p.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu or cuda)')
    p.add_argument('--save-dir', type=str, default=None, help='Directory to save checkpoints')
    p.add_argument('--num-joints', type=int, default=None, help='Number of joints for the fusion model (default: auto-detect from dataset)')
    p.add_argument('--val-list', type=str, default=None, help='Validation list file (for --validate)')
    p.add_argument('--validate', action='store_true', help='Run validation after each epoch using --val-list')
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    p.add_argument('--num-workers', type=int, default=0, help='Number of DataLoader workers')
    p.add_argument('--pin-memory', action='store_true', help='Use pin_memory for DataLoader')
    p.add_argument('--persistent-workers', action='store_true', help='Use persistent_workers for DataLoader (requires num-workers>0)')
    p.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (L2) for Adam optimizer')
    p.add_argument('--lr-scheduler-patience', type=int, default=3, help='Patience (epochs) for ReduceLROnPlateau')
    p.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='LR reduction factor for ReduceLROnPlateau')
    p.add_argument('--early-stop-patience', type=int, default=5, help='Early stopping patience in epochs (no improvement)')
    p.add_argument('--augment-keypoints', action='store_true', help='Apply keypoint augmentation (jitter/occlusion) in collate')
    p.add_argument('--jitter-px', type=float, default=0.0, help='Standard deviation (px) for keypoint jitter')
    p.add_argument('--occlusion-prob', type=float, default=0.0, help='Probability to zero-out a keypoint (per-joint)')
    args = p.parse_args()
    if args.train:
        print('Starting quick training run...')
        quick_train(
            args.list, args.root,
            epochs=args.epochs, max_iters=args.max_iters, batch_size=args.batch_size,
            lr=args.lr, device=args.device, num_joints=args.num_joints, require_caches=args.require_caches,
            save_dir=args.save_dir, resume=args.resume, val_list=args.val_list if args.validate else None,
            num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers,
            weight_decay=args.weight_decay, lr_scheduler_patience=args.lr_scheduler_patience, lr_scheduler_factor=args.lr_scheduler_factor, early_stop_patience=args.early_stop_patience,
            augment_keypoints=args.augment_keypoints, jitter_px=args.jitter_px, occlusion_prob=args.occlusion_prob,
        )
    else:
        dry_run(args.list, args.root, batch_size=args.batch_size, num_batches=args.num_batches, require_caches=args.require_caches, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, augment_keypoints=args.augment_keypoints, jitter_px=args.jitter_px, occlusion_prob=args.occlusion_prob)

