"""4-model internal transformer fusion (ViTPose features + three baselines).

Pass ``--save-dir`` to choose the checkpoint directory.

ViTPose uses the same synthetic ``[x,y,conf]`` features as DEKR/OpenPose from p2d_cache.
"""

from typing import Dict, Tuple, Optional
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.transformer_internal_fusion_4 import (
    TransformerInternalPoseFusion4,
    LightweightTransformerInternalFusion4,
)
from training.train_mlp import build_dataloader_from_list
from data.pytorch_dataset import VideoFrameKeypointDataset
from training.train_internal import compute_accuracy_and_loss


def _build_internal_features_4(
    coords_hrnet: torch.Tensor,
    conf_hrnet: torch.Tensor,
    coords_openpose: torch.Tensor,
    conf_openpose: torch.Tensor,
    coords_vitpose: torch.Tensor,
    conf_vitpose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats_dekr = torch.cat([coords_hrnet, conf_hrnet], dim=-1)
    feats_openpose = torch.cat([coords_openpose, conf_openpose], dim=-1)
    feats_vitpose = torch.cat([coords_vitpose, conf_vitpose], dim=-1)
    return feats_dekr, feats_openpose, feats_vitpose


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

    feats_dekr, feats_openpose, feats_vitpose = _build_internal_features_4(
        batch["coords_hrnet"],
        batch["conf_hrnet"],
        batch["coords_openpose"],
        batch["conf_openpose"],
        batch["coords_vitpose"],
        batch["conf_vitpose"],
    )

    optimizer.zero_grad()

    coords_fused, conf_fused_logits, _ = model(
        feats_dekr,
        feats_openpose,
        feats_vitpose,
        batch["coords_mediapipe"],
        batch["conf_mediapipe"],
        return_attention_weights=False,
    )

    mask_any = (
        batch["mask_hrnet"].squeeze(-1)
        + batch["mask_openpose"].squeeze(-1)
        + batch["mask_mediapipe"].squeeze(-1)
        + batch["mask_vitpose"].squeeze(-1)
    ) > 0.0

    loss, metrics = compute_accuracy_and_loss(
        coords_fused,
        conf_fused_logits,
        batch["coords_gt"],
        batch.get("coords_gt_px"),
        batch.get("img_wh"),
        batch.get("couch_len"),
        mask_any,
        tau=tau,
        lambda_conf=lambda_conf,
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

    with torch.no_grad():
        feats_dekr, feats_openpose, feats_vitpose = _build_internal_features_4(
            batch["coords_hrnet"],
            batch["conf_hrnet"],
            batch["coords_openpose"],
            batch["conf_openpose"],
            batch["coords_vitpose"],
            batch["conf_vitpose"],
        )

        coords_fused, conf_fused_logits, _ = model(
            feats_dekr,
            feats_openpose,
            feats_vitpose,
            batch["coords_mediapipe"],
            batch["conf_mediapipe"],
            return_attention_weights=False,
        )

        mask_any = (
            batch["mask_hrnet"].squeeze(-1)
            + batch["mask_openpose"].squeeze(-1)
            + batch["mask_mediapipe"].squeeze(-1)
            + batch["mask_vitpose"].squeeze(-1)
        ) > 0.0

        loss, metrics = compute_accuracy_and_loss(
            coords_fused,
            conf_fused_logits,
            batch["coords_gt"],
            batch.get("coords_gt_px"),
            batch.get("img_wh"),
            batch.get("couch_len"),
            mask_any,
            tau=tau,
            lambda_conf=lambda_conf,
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
        total_accuracy += metrics.get("accuracy", 0.0)
        total_dist_px += metrics.get("mean_dist_px", 0.0)
        n += 1

    if n == 0:
        return float("inf"), {}

    return total_loss / n, {
        "accuracy": total_accuracy / n,
        "mean_dist_px": total_dist_px / n,
    }


def train_transformer_internal_fusion_4(
    list_file: str,
    dataset_root: str,
    epochs: int = 10,
    max_iters: int = 20000,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cpu",
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
    model_type: str = "full",
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    dekr_feat_dim: int = 3,
    openpose_feat_dim: int = 3,
    vitpose_feat_dim: int = 3,
) -> torch.nn.Module:
    device_t = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    ds_probe = VideoFrameKeypointDataset(
        list_file=list_file, dataset_root=dataset_root, output_size=(256, 256)
    )
    if len(ds_probe) == 0:
        raise RuntimeError("Dataset appears empty")
    K = ds_probe[0]["keypoints"].shape[0]
    if num_joints is not None and num_joints != K:
        print(f"Warning: requested num_joints={num_joints} differs from dataset K={K}; using K={K}")
    used_num_joints = K

    print(f"\n[internal-4] K={K} model_type={model_type} vitpose_feat_dim={vitpose_feat_dim}")

    build_dataloader_from_list._augment_flags = {
        "augment_keypoints": augment_keypoints,
        "jitter_px": float(jitter_px),
        "occlusion_prob": float(occlusion_prob),
    }
    loader = build_dataloader_from_list(
        list_file,
        dataset_root,
        batch_size=batch_size,
        output_size=(256, 256),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        require_caches=False,
        include_vitpose=True,
    )
    build_dataloader_from_list._augment_flags = None

    if model_type == "lightweight":
        model = LightweightTransformerInternalFusion4(
            num_joints=used_num_joints,
            dekr_feat_dim=dekr_feat_dim,
            openpose_feat_dim=openpose_feat_dim,
            vitpose_feat_dim=vitpose_feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device_t)
    else:
        model = TransformerInternalPoseFusion4(
            num_joints=used_num_joints,
            dekr_feat_dim=dekr_feat_dim,
            openpose_feat_dim=openpose_feat_dim,
            vitpose_feat_dim=vitpose_feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device_t)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    scheduler = None
    if val_list is not None:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(lr_scheduler_patience),
            factor=float(lr_scheduler_factor),
        )

    start_epoch = 0
    it = 0
    best_val_accuracy = 0.0
    epochs_since_improve = 0

    if resume is not None:
        ck_path = Path(resume)
        if ck_path.exists():
            d = torch.load(str(ck_path), map_location=device_t)
            model.load_state_dict(d["model_state"])
            if "optimizer" in d:
                try:
                    optimizer.load_state_dict(d["optimizer"])
                except Exception as e:
                    print(f"Warning: optimizer not restored: {e}")
            start_epoch = int(d.get("epoch", 0))
            it = int(d.get("iter", 0))
            best_val_accuracy = float(d.get("best_val_accuracy", 0.0))
            print(f"Resumed from {ck_path} epoch={start_epoch} iter={it}")

    cfg = {
        "num_joints": used_num_joints,
        "num_models": 4,
        "model_type": model_type,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "dekr_feat_dim": dekr_feat_dim,
        "openpose_feat_dim": openpose_feat_dim,
        "vitpose_feat_dim": vitpose_feat_dim,
    }

    print("\n" + "=" * 70, flush=True)
    print("Starting 4-model INTERNAL transformer fusion", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        epoch_accuracies = []
        epoch_dist_px = []

        for batch_idx, batch in enumerate(loader):
            loss, metrics = train_step_4(
                batch,
                model,
                optimizer,
                device_t,
                lambda_conf=lambda_conf,
                tau=tau,
                grad_clip=grad_clip,
            )
            epoch_losses.append(loss)
            epoch_accuracies.append(metrics.get("accuracy", 0.0))
            epoch_dist_px.append(metrics.get("mean_dist_px", 0.0))
            it += 1

            if it % 10 == 0 or it == 1:
                acc = metrics.get("accuracy", 0.0)
                dist_px = metrics.get("mean_dist_px", 0.0)
                lc = metrics.get("loss_coords", 0.0)
                lf = metrics.get("loss_conf", 0.0)
                print(
                    f"Epoch {epoch} iter {it:5d} | loss {loss:.4f} "
                    f"(coord {lc:.4f} + conf {lf:.4f}) | acc {acc*100:5.1f}% | dist {dist_px:6.2f}px",
                    flush=True,
                )

            if it >= max_iters:
                break

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        avg_dist = np.mean(epoch_dist_px)
        print(
            f"\n>>> Epoch {epoch} summary: loss={avg_loss:.4f}, acc={avg_acc*100:.1f}%, dist={avg_dist:.2f}px",
            flush=True,
        )

        if save_dir is not None:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            ck = p / f"checkpoint_epoch{epoch+1}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "iter": it,
                    "best_val_accuracy": best_val_accuracy,
                    "model_config": cfg,
                },
                str(ck),
            )
            print(f"Saved checkpoint: {ck}", flush=True)

        if val_list is not None:
            val_loader = build_dataloader_from_list(
                val_list,
                dataset_root,
                batch_size=batch_size,
                output_size=(256, 256),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                require_caches=False,
                include_vitpose=True,
            )
            val_loss, val_metrics = validate_4(model, val_loader, device_t, tau=tau)
            val_accuracy = val_metrics.get("accuracy", 0.0)
            val_dist = val_metrics.get("mean_dist_px", 0.0)
            print(
                f"Val loss: {val_loss:.4f} | Val acc: {val_accuracy*100:.1f}% | dist {val_dist:.2f}px",
                flush=True,
            )

            if scheduler is not None:
                scheduler.step(val_loss)
                print(f"LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_since_improve = 0
                if save_dir is not None:
                    best_ck = Path(save_dir) / "checkpoint_best.pt"
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch + 1,
                            "iter": it,
                            "best_val_accuracy": best_val_accuracy,
                            "model_config": cfg,
                        },
                        str(best_ck),
                    )
                    print(f"✓ Best checkpoint (acc={best_val_accuracy*100:.1f}%)", flush=True)
            else:
                epochs_since_improve += 1
                print(f"No improvement ({epochs_since_improve}/{early_stop_patience})", flush=True)

            if early_stop_patience is not None and epochs_since_improve >= int(early_stop_patience):
                print("Early stopping", flush=True)
                break

        if it >= max_iters:
            break

    print("=" * 70)
    print(f"Done. Best val acc: {best_val_accuracy*100:.1f}%")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train 4-model internal transformer fusion")
    p.add_argument("--list", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--val-list", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--model-type", default="full", choices=["full", "lightweight"])
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--num-decoder-layers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--lambda-conf", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--lr-scheduler-patience", type=int, default=3)
    p.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=7)
    p.add_argument("--augment-keypoints", action="store_true")
    p.add_argument("--jitter-px", type=float, default=0.0)
    p.add_argument("--occlusion-prob", type=float, default=0.0)
    p.add_argument("--dekr-feat-dim", type=int, default=3)
    p.add_argument("--openpose-feat-dim", type=int, default=3)
    p.add_argument("--vitpose-feat-dim", type=int, default=3)
    args = p.parse_args()

    if args.model_type == "lightweight":
        if args.d_model == 256:
            args.d_model = 128
        if args.nhead == 8:
            args.nhead = 4
        if args.dim_feedforward == 512:
            args.dim_feedforward = 256

    train_transformer_internal_fusion_4(
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
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        early_stop_patience=args.early_stop_patience,
        dekr_feat_dim=args.dekr_feat_dim,
        openpose_feat_dim=args.openpose_feat_dim,
        vitpose_feat_dim=args.vitpose_feat_dim,
    )
