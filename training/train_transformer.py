"""Training script for transformer-based fusion model.

Uses TransformerPoseFusion or LightweightTransformerFusion with
TransformerEncoderLayer and TransformerDecoderLayer.

Key improvements:
1. Full transformer architecture with encoder-decoder attention
2. Better loss computation with accuracy metrics
3. L2 regularization and gradient clipping
4. Learning rate scheduling and early stopping
5. Detailed logging and checkpointing
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.transformer_fusion import TransformerPoseFusion, LightweightTransformerFusion
from training.train_mlp import build_dataloader_from_list
from data.pytorch_dataset import VideoFrameKeypointDataset


def compute_accuracy_and_loss(
    coords_fused: torch.Tensor,
    conf_fused_logits: torch.Tensor,
    coords_gt: torch.Tensor,
    coords_gt_px: torch.Tensor,
    img_wh: torch.Tensor,
    couch_len: torch.Tensor,
    mask_any: torch.Tensor,
    tau: float = 0.05,
    lambda_conf: float = 0.1,
    lambda_l2_coords: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss and accuracy metrics.
    
    Returns:
        loss: scalar tensor
        metrics: dict with 'accuracy', 'loss_coords', 'loss_conf'
    """
    metrics = {}
    
    if mask_any.sum() == 0:
        return torch.tensor(0.0, device=coords_fused.device), metrics
    
    coords_fused_masked = coords_fused[mask_any]
    coords_gt_masked = coords_gt[mask_any]
    
    # Coordinate loss: L2 distance between predictions and ground truth
    loss_coords = torch.nn.functional.mse_loss(coords_fused_masked, coords_gt_masked)
    
    # Compute pixel-space distance for accuracy computation
    img_wh_tensor = img_wh.view(-1, 1, 2)
    coords_fused_px = coords_fused * img_wh_tensor
    dists_px = torch.norm(coords_fused_px - coords_gt_px, dim=-1)
    dists_norm = dists_px / couch_len.view(-1, 1)
    good = (dists_norm < tau).float()
    
    # Confidence loss: binary cross-entropy
    conf_logits = conf_fused_logits.squeeze(-1)
    pos_logits = conf_logits[mask_any]
    pos_targets = good[mask_any]
    
    if pos_logits.numel() > 0:
        bce = torch.nn.BCEWithLogitsLoss()
        loss_conf = bce(pos_logits, pos_targets)
    else:
        loss_conf = torch.tensor(0.0, device=coords_fused.device)
    
    # Compute accuracy (for monitoring)
    accuracy = float(good[mask_any].mean().item())
    
    # Combine losses
    loss = loss_coords + lambda_conf * loss_conf
    
    metrics['loss_coords'] = float(loss_coords.item())
    metrics['loss_conf'] = float(loss_conf.item())
    metrics['accuracy'] = accuracy
    metrics['mean_dist_px'] = float(dists_px[mask_any].mean().item())
    
    return loss, metrics


def train_step(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_conf: float = 0.1,
    tau: float = 0.05,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Single training step.
    
    Returns:
        total_loss: scalar float
        metrics: dict with diagnostic info
    """
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
    coords_gt_px = batch.get('coords_gt_px')
    couch_len = batch.get('couch_len')
    img_wh = batch.get('img_wh')
    mask_hr = batch.get('mask_hrnet')
    mask_op = batch.get('mask_openpose')
    mask_mp = batch.get('mask_mediapipe')
    
    optimizer.zero_grad()
    
    coords_fused, conf_fused_logits, _ = model(
        coords_hrnet, conf_hrnet,
        coords_openpose, conf_openpose,
        coords_mediapipe, conf_mediapipe,
        return_attention_weights=False,
    )
    mask_any = ((mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1)) > 0.0)
    
    loss, metrics = compute_accuracy_and_loss(
        coords_fused, conf_fused_logits, coords_gt, coords_gt_px, img_wh, couch_len,
        mask_any, tau=tau, lambda_conf=lambda_conf
    )
    
    loss.backward()
    
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    
    return float(loss.item()), metrics


def compute_val_metrics(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    lambda_conf: float = 0.1,
    tau: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """Compute validation loss and metrics."""
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
    coords_gt_px = batch.get('coords_gt_px')
    couch_len = batch.get('couch_len')
    img_wh = batch.get('img_wh')
    mask_hr = batch.get('mask_hrnet')
    mask_op = batch.get('mask_openpose')
    mask_mp = batch.get('mask_mediapipe')
    
    with torch.no_grad():
        coords_fused, conf_fused_logits, _ = model(
            coords_hrnet, conf_hrnet,
            coords_openpose, conf_openpose,
            coords_mediapipe, conf_mediapipe,
            return_attention_weights=False,
        )
        mask_any = ((mask_hr.squeeze(-1) + mask_op.squeeze(-1) + mask_mp.squeeze(-1)) > 0.0)
        
        loss, metrics = compute_accuracy_and_loss(
            coords_fused, conf_fused_logits, coords_gt, coords_gt_px, img_wh, couch_len,
            mask_any, tau=tau, lambda_conf=lambda_conf
        )
    
    return float(loss.item()), metrics


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    tau: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """Run validation over entire loader and return average metrics."""
    total_loss = 0.0
    total_accuracy = 0.0
    total_dist_px = 0.0
    n = 0
    
    for batch in val_loader:
        loss, metrics = compute_val_metrics(batch, model, device, tau=tau)
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


def train_transformer_fusion(
    list_file: str,
    dataset_root: str,
    epochs: int = 10,
    max_iters: int = 20000,
    batch_size: int = 8,
    lr: float = 1e-4,
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
    early_stop_patience: int = 7,
    grad_clip: float = 1.0,
    tau: float = 0.05,
    lambda_conf: float = 0.1,
    augment_keypoints: bool = False,
    jitter_px: float = 0.0,
    occlusion_prob: float = 0.0,
    model_type: str = 'full',
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
):
    """Training loop for transformer-based fusion model.
    
    Args:
        model_type: 'full' or 'lightweight'
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
    """
    device_t = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    
    # Probe dataset
    ds_probe = VideoFrameKeypointDataset(list_file=list_file, dataset_root=dataset_root, output_size=(256, 256))
    if len(ds_probe) == 0:
        raise RuntimeError('Dataset appears empty')
    K = ds_probe[0]['keypoints'].shape[0]
    if num_joints is None:
        used_num_joints = K
    else:
        if num_joints != K:
            print(f'Warning: requested num_joints={num_joints} differs from dataset K={K}; using K={K}')
        used_num_joints = K
    
    print(f"\nDetected {K} joints")
    print(f"Model type: {model_type}")
    print(f"Transformer config: d_model={d_model}, nhead={nhead}, enc_layers={num_encoder_layers}, dec_layers={num_decoder_layers}")
    print(f"Learning rate={lr}, weight_decay={weight_decay}")
    
    # Create dataloaders
    build_dataloader_from_list._augment_flags = {
        'augment_keypoints': augment_keypoints,
        'jitter_px': float(jitter_px),
        'occlusion_prob': float(occlusion_prob)
    }
    loader = build_dataloader_from_list(
        list_file, dataset_root, batch_size=batch_size, output_size=(256, 256),
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, require_caches=False,
    )
    build_dataloader_from_list._augment_flags = None
    
    # Create model
    if model_type == 'lightweight':
        model = LightweightTransformerFusion(
            num_joints=used_num_joints,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device_t)
    else:
        model = TransformerPoseFusion(
            num_joints=used_num_joints,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device_t)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer with L2 regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    
    # LR scheduler and early stopping
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
    
    # Resume checkpoint if provided
    if resume is not None:
        ck_path = Path(resume)
        if ck_path.exists():
            d = torch.load(str(ck_path), map_location=device_t)
            model.load_state_dict(d['model_state'])
            if 'optimizer' in d:
                try:
                    optimizer.load_state_dict(d['optimizer'])
                except Exception as e:
                    print(f'Warning: optimizer state could not be fully restored: {e}')
            start_epoch = int(d.get('epoch', 0))
            it = int(d.get('iter', 0))
            best_val_accuracy = float(d.get('best_val_accuracy', 0.0))
            print(f'Resumed from {ck_path} at epoch {start_epoch} iter {it}')
    
    print("\n" + "="*70, flush=True)
    print("Starting transformer fusion training...", flush=True)
    print("="*70, flush=True)
    
    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        epoch_accuracies = []
        epoch_dist_px = []
        
        for batch_idx, batch in enumerate(loader):
            loss, metrics = train_step(
                batch, model, optimizer, device_t,
                lambda_conf=lambda_conf, tau=tau, grad_clip=grad_clip,
            )
            
            epoch_losses.append(loss)
            epoch_accuracies.append(metrics.get('accuracy', 0.0))
            epoch_dist_px.append(metrics.get('mean_dist_px', 0.0))
            it += 1
            
            if it % 10 == 0 or it == 1:
                acc = metrics.get('accuracy', 0.0)
                dist_px = metrics.get('mean_dist_px', 0.0)
                loss_coords = metrics.get('loss_coords', 0.0)
                loss_conf = metrics.get('loss_conf', 0.0)
                print(f"Epoch {epoch} iter {it:5d} | loss {loss:.4f} (coord {loss_coords:.4f} + conf {loss_conf:.4f}) | acc {acc*100:5.1f}% | dist {dist_px:6.2f}px", flush=True)
            
            if it >= max_iters:
                break
        
        # Per-epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        avg_dist = np.mean(epoch_dist_px)
        print(f"\n>>> Epoch {epoch} summary: loss={avg_loss:.4f}, accuracy={avg_acc*100:.1f}%, dist={avg_dist:.2f}px", flush=True)
        
        # Save checkpoint per epoch
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
                'model_config': {
                    'num_joints': used_num_joints,
                    'model_type': model_type,
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_encoder_layers': num_encoder_layers,
                    'num_decoder_layers': num_decoder_layers,
                    'dim_feedforward': dim_feedforward,
                    'dropout': dropout,
                }
            }, str(ck))
            print(f"Saved checkpoint: {ck}", flush=True)
        
        # Validation
        if val_list is not None:
            val_loader = build_dataloader_from_list(
                val_list, dataset_root, batch_size=batch_size, output_size=(256, 256),
                shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                persistent_workers=persistent_workers, require_caches=False,
            )
            val_loss, val_metrics = validate(model, val_loader, device_t, tau=tau)
            val_accuracy = val_metrics.get('accuracy', 0.0)
            val_dist = val_metrics.get('mean_dist_px', 0.0)
            
            print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy*100:.1f}% | Val dist: {val_dist:.2f}px", flush=True)
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}", flush=True)
            
            # Early stopping + best checkpoint
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
                        'model_config': {
                            'num_joints': used_num_joints,
                            'model_type': model_type,
                            'd_model': d_model,
                            'nhead': nhead,
                            'num_encoder_layers': num_encoder_layers,
                            'num_decoder_layers': num_decoder_layers,
                            'dim_feedforward': dim_feedforward,
                            'dropout': dropout,
                        }
                    }, str(best_ck))
                    print(f"✓ New best checkpoint saved (accuracy={best_val_accuracy*100:.1f}%)", flush=True)
            else:
                epochs_since_improve += 1
                print(f"No improvement ({epochs_since_improve}/{early_stop_patience} epochs)", flush=True)
            
            if early_stop_patience is not None and epochs_since_improve >= int(early_stop_patience):
                print(f"Early stopping triggered after {epochs_since_improve} epochs without improvement")
                break
        
        if it >= max_iters:
            break
    
    print("="*70)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_accuracy*100:.1f}%")
    print("="*70)
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train transformer-based pose fusion model')
    
    # Data args
    p.add_argument('--list', required=True, help='Training list file')
    p.add_argument('--root', required=True, help='Dataset root')
    p.add_argument('--val-list', default=None, help='Validation list file')
    
    # Training args
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--max-iters', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4 for transformer)')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    
    # Model args
    p.add_argument('--model-type', default='full', choices=['full', 'lightweight'],
                    help='Model variant: full (d_model=256) or lightweight (d_model=128)')
    p.add_argument('--d-model', type=int, default=256, help='Transformer hidden dimension')
    p.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    p.add_argument('--num-encoder-layers', type=int, default=2)
    p.add_argument('--num-decoder-layers', type=int, default=2)
    p.add_argument('--dim-feedforward', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.1)
    
    # Loss args
    p.add_argument('--tau', type=float, default=0.05, help='Accuracy threshold (normalized coords)')
    p.add_argument('--lambda-conf', type=float, default=0.1, help='Confidence loss weight')
    
    # System args
    p.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--pin-memory', action='store_true', help='Pin memory for DataLoader')
    p.add_argument('--persistent-workers', action='store_true', help='Use persistent workers')
    
    # Checkpoint args
    p.add_argument('--save-dir', default=None, help='Directory to save checkpoints')
    p.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    # Scheduler args
    p.add_argument('--lr-scheduler-patience', type=int, default=3)
    p.add_argument('--lr-scheduler-factor', type=float, default=0.5)
    p.add_argument('--early-stop-patience', type=int, default=7)
    
    # Augmentation args
    p.add_argument('--augment-keypoints', action='store_true')
    p.add_argument('--jitter-px', type=float, default=0.0)
    p.add_argument('--occlusion-prob', type=float, default=0.0)
    
    args = p.parse_args()
    
    # Adjust defaults for lightweight model
    if args.model_type == 'lightweight':
        if args.d_model == 256:  # User didn't override
            args.d_model = 128
        if args.nhead == 8:  # User didn't override
            args.nhead = 4
        if args.dim_feedforward == 512:  # User didn't override
            args.dim_feedforward = 256
    
    train_transformer_fusion(
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
        model_type=args.model_type,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
