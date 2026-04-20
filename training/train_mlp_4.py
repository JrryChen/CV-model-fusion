"""Improved training for 4-model fusion (DEKR, OpenPose, MediaPipe, ViTPose).

Pass ``--save-dir`` to choose the checkpoint directory.
"""

from typing import Dict, Tuple, Optional
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.mlp_fusion_4 import MultiModelPoseFusion4
from training.train_mlp import build_dataloader_from_list
from data.pytorch_dataset import VideoFrameKeypointDataset
from training.train_mlp_improved import compute_accuracy_and_loss


def train_step_4(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_conf: float = 0.1,
    tau: float = 0.05,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
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
    coords_vitpose = batch['coords_vitpose']
    conf_vitpose = batch['conf_vitpose']
    coords_gt = batch['coords_gt']
    coords_gt_px = batch.get('coords_gt_px')
    couch_len = batch.get('couch_len')
    img_wh = batch.get('img_wh')
    mask_hr = batch.get('mask_hrnet')
    mask_op = batch.get('mask_openpose')
    mask_mp = batch.get('mask_mediapipe')
    mask_pb = batch.get('mask_vitpose')

    optimizer.zero_grad()

    coords_fused, conf_fused_logits, _ = model(
        coords_hrnet, conf_hrnet,
        coords_openpose, conf_openpose,
        coords_mediapipe, conf_mediapipe,
        coords_vitpose, conf_vitpose,
        return_attention_weights=False,
    )

    mask_any = (
        (mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1) + mask_pb.squeeze(-1)) > 0.0
    )

    loss, metrics = compute_accuracy_and_loss(
        coords_fused, conf_fused_logits, coords_gt, coords_gt_px, img_wh, couch_len,
        mask_any, tau=tau, lambda_conf=lambda_conf
    )

    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item()), metrics


def compute_val_metrics_4(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    lambda_conf: float = 0.1,
    tau: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    for k in list(batch.keys()):
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    mask_hr = batch.get('mask_hrnet')
    mask_op = batch.get('mask_openpose')
    mask_mp = batch.get('mask_mediapipe')
    mask_pb = batch.get('mask_vitpose')

    with torch.no_grad():
        coords_fused, conf_fused_logits, _ = model(
            batch['coords_hrnet'], batch['conf_hrnet'],
            batch['coords_openpose'], batch['conf_openpose'],
            batch['coords_mediapipe'], batch['conf_mediapipe'],
            batch['coords_vitpose'], batch['conf_vitpose'],
            return_attention_weights=False,
        )

        mask_any = (
            (mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1) + mask_pb.squeeze(-1)) > 0.0
        )
        loss, metrics = compute_accuracy_and_loss(
            coords_fused, conf_fused_logits,
            batch['coords_gt'], batch.get('coords_gt_px'), batch.get('img_wh'), batch.get('couch_len'),
            mask_any, tau=tau, lambda_conf=lambda_conf
        )

    return float(loss.item()), metrics


def validate_4(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    tau: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    total_loss = 0.0
    total_accuracy = 0.0
    total_dist_px = 0.0
    n = 0

    for batch in val_loader:
        loss, metrics = compute_val_metrics_4(batch, model, device, tau=tau)
        total_loss += loss
        total_accuracy += metrics.get('accuracy', 0.0)
        total_dist_px += metrics.get('mean_dist_px', 0.0)
        n += 1

    if n == 0:
        return float('inf'), {}

    return total_loss / n, {
        'accuracy': total_accuracy / n,
        'mean_dist_px': total_dist_px / n,
    }


def improved_train_4(
    list_file: str,
    dataset_root: str,
    epochs: int = 5,
    max_iters: int = 10000,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = 'cpu',
    num_joints: Optional[int] = None,
    save_dir: Optional[str] = None,
    resume: Optional[str] = None,
    val_list: Optional[str] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    weight_decay: float = 1e-4,
    lr_scheduler_patience: int = 3,
    lr_scheduler_factor: float = 0.5,
    early_stop_patience: int = 5,
    grad_clip: float = 1.0,
    tau: float = 0.05,
    lambda_conf: float = 0.1,
    augment_keypoints: bool = False,
    jitter_px: float = 0.0,
    occlusion_prob: float = 0.0,
):
    device_t = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    ds_probe = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
    if len(ds_probe) == 0:
        raise RuntimeError('Dataset appears empty')
    K = ds_probe[0]['keypoints'].shape[0]
    used_num_joints = num_joints if num_joints is not None else K
    if num_joints is not None and num_joints != K:
        print(f'Warning: requested num_joints={num_joints} differs from dataset K={K}; using K={K}')
        used_num_joints = K

    print(f"[4-model fusion] Detected K={K}; device={device_t}")

    build_dataloader_from_list._augment_flags = {
        'augment_keypoints': augment_keypoints,
        'jitter_px': float(jitter_px),
        'occlusion_prob': float(occlusion_prob),
    }
    loader = build_dataloader_from_list(
        list_file, dataset_root, batch_size=batch_size, output_size=(256, 256),
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, require_caches=False,
        include_vitpose=True,
    )
    build_dataloader_from_list._augment_flags = None

    model = MultiModelPoseFusion4(num_joints=used_num_joints).to(device_t)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))

    scheduler = None
    if val_list is not None:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=int(lr_scheduler_patience),
            factor=float(lr_scheduler_factor)
        )

    start_epoch = 0
    it = 0
    best_val_accuracy = 0.0
    epochs_since_improve = 0

    if resume is not None:
        ck_path = Path(resume)
        if ck_path.exists():
            d = torch.load(str(ck_path), map_location=device_t)
            model.load_state_dict(d['model_state'])
            if 'optimizer' in d:
                try:
                    optimizer.load_state_dict(d['optimizer'])
                except Exception:
                    print('Warning: optimizer state could not be fully restored')
            start_epoch = int(d.get('epoch', 0))
            it = int(d.get('iter', 0))
            best_val_accuracy = float(d.get('best_val_accuracy', 0.0))
            print(f'Resumed from {ck_path} at epoch {start_epoch} iter {it}')

    print("\n" + "="*70)
    print("Starting 4-model fusion training (ViTPose p2d_cache required)")
    print("="*70)

    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        epoch_accuracies = []

        for batch_idx, batch in enumerate(loader):
            loss, metrics = train_step_4(
                batch, model, optimizer, device_t,
                lambda_conf=lambda_conf, tau=tau, grad_clip=grad_clip
            )
            epoch_losses.append(loss)
            epoch_accuracies.append(metrics.get('accuracy', 0.0))
            it += 1

            if it % 10 == 0 or it == 1:
                acc = metrics.get('accuracy', 0.0)
                dist_px = metrics.get('mean_dist_px', 0.0)
                print(f"Epoch {epoch} iter {it:5d} | loss {loss:.4f} | acc {acc*100:5.1f}% | dist {dist_px:6.2f}px")

            if it >= max_iters:
                break

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        print(f"\n>>> Epoch {epoch} summary: loss={avg_loss:.4f}, accuracy={avg_acc*100:.1f}%")

        if save_dir is not None:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            ck = p / f'checkpoint_epoch{epoch+1}.pt'
            torch.save({
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'iter': it,
                'best_val_accuracy': best_val_accuracy,
            }, str(ck))
            print(f"Saved checkpoint: {ck}")

        if val_list is not None:
            val_loader = build_dataloader_from_list(
                val_list, dataset_root, batch_size=batch_size, output_size=(256, 256),
                shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                persistent_workers=persistent_workers, require_caches=False,
                include_vitpose=True,
            )
            val_loss, val_metrics = validate_4(model, val_loader, device_t, tau=tau)
            val_accuracy = val_metrics.get('accuracy', 0.0)
            val_dist = val_metrics.get('mean_dist_px', 0.0)
            print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy*100:.1f}% | Val dist: {val_dist:.2f}px")

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_since_improve = 0
                if save_dir is not None:
                    best_ck = Path(save_dir) / 'checkpoint_best.pt'
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'iter': it,
                        'best_val_accuracy': best_val_accuracy,
                    }, str(best_ck))
                    print(f"✓ New best checkpoint saved (accuracy={best_val_accuracy*100:.1f}%)")
            else:
                epochs_since_improve += 1
                print(f"No improvement ({epochs_since_improve}/{early_stop_patience} epochs)")

            if early_stop_patience is not None and epochs_since_improve >= int(early_stop_patience):
                print("Early stopping triggered")
                break

        if it >= max_iters:
            break

    print("="*70)
    print("Training complete!")
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train 4-model MLP fusion (ViTPose)')
    p.add_argument('--list', required=True, help='Training list file')
    p.add_argument('--root', required=True, help='Dataset root')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max-iters', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--save-dir', default=None)
    p.add_argument('--resume', default=None)
    p.add_argument('--val-list', default=None)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--pin-memory', action='store_true')
    p.add_argument('--persistent-workers', action='store_true')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--tau', type=float, default=0.05)
    p.add_argument('--lambda-conf', type=float, default=0.1)
    p.add_argument('--augment-keypoints', action='store_true')
    p.add_argument('--jitter-px', type=float, default=0.0)
    p.add_argument('--occlusion-prob', type=float, default=0.0)
    args = p.parse_args()

    improved_train_4(
        list_file=args.list,
        dataset_root=args.root,
        epochs=args.epochs,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        resume=args.resume,
        val_list=args.val_list,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        tau=args.tau,
        lambda_conf=args.lambda_conf,
        augment_keypoints=args.augment_keypoints,
        jitter_px=args.jitter_px,
        occlusion_prob=args.occlusion_prob,
    )
