"""Run inference using saved `checkpoint_best.pt`(s) on a list (test/val).

Produces one JSONL file per checkpoint (or a single JSONL when ensembling)
with per-sample predictions: normalized coords, optional pixel coords, and
confidence scores. Supports ensembling multiple checkpoints by averaging
logits before sigmoid and averaging coordinates.
"""
from pathlib import Path
import argparse
import json
from typing import List

import torch
from torch.utils.data import DataLoader

import numpy as np

from models.mlp_fusion import MultiModelPoseFusion
from data.pytorch_dataset import VideoFrameKeypointDataset
import cv2


def _parse_rel_path(rel: str):
    p = Path(rel)
    parts = p.parts
    # expect clips_mp4/<patient>/<exercise>/<cam>.mp4
    if len(parts) >= 4:
        patient = parts[-3]
        exercise = parts[-2]
        cam = Path(parts[-1]).stem
        return patient, exercise, cam
    return None, None, None


def _read_full_p2d_file(path: Path):
    if not path.exists():
        return []
    lines = [l.strip() for l in path.open('r') if l.strip()]
    parsed = []
    for line in lines:
        parts = line.split()
        vals = []
        for x in parts:
            try:
                vals.append(float(x))
            except Exception:
                vals.append(0.0)
        parsed.append(vals)
    return parsed


def load_checkpoints(paths: List[Path], device: torch.device):
    dicts = []
    for p in paths:
        d = torch.load(str(p), map_location=device)
        dicts.append(d)
    return dicts


def make_model(num_joints: int, device: torch.device):
    model = MultiModelPoseFusion(num_joints=num_joints)
    model.to(device)
    model.eval()
    return model


def run_inference(checkpoints: List[str], list_file: str, dataset_root: str, out_dir: str, batch_size: int = 8, device: str = 'cpu', apply_sigmoid: bool = True, save_px: bool = True, require_caches: bool = False, num_workers: int = 0, pin_memory: bool = False, norm_ref: str = 'video', frame_offset: int = 5, output_size=(256,256)):
    """Run inference per-video and write per-video p2d fusion files.

    - `norm_ref`: 'video' to normalize by original video width/height, or 'output' to normalize by `output_size` used during training.
    - Output files are written to <dataset_root>/p2d_cache/<patient>/<exercise>/<cam>_fusion_p2d.txt
    """
    device_t = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    ck_paths = [Path(x) for x in checkpoints]
    for p in ck_paths:
        if not p.exists():
            raise FileNotFoundError(f'Checkpoint not found: {p}')

    ck_dicts = load_checkpoints(ck_paths, device_t)

    # We'll create one model instance and repeatedly load state dicts for each checkpoint when computing predictions
    # Probe number of joints from any available p2d reference or default to 3
    # Try to infer K from dataset reference p2d file of first listed video
    lines = [l.strip() for l in Path(list_file).read_text().splitlines() if l.strip()]
    if len(lines) == 0:
        raise RuntimeError('Empty list file')

    # default K
    K = 3
    # attempt to read first video's annotation line to determine K
    first_rel = lines[0].split()[0]
    vpath = Path(dataset_root) / first_rel
    ref_p2d = vpath.with_name(vpath.stem + '_p2d.txt')
    ref_lines = _read_full_p2d_file(ref_p2d)
    if len(ref_lines) > 0 and len(ref_lines[0]) >= 6:
        # each line may contain 6 or 9 values: take length/3
        K = max(1, len(ref_lines[0]) // 3)

    model = make_model(K, device_t)

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # process each video listed
    for li, raw in enumerate(lines):
        parts = raw.split()
        rel = parts[0]
        patient, exercise, cam = _parse_rel_path(rel)
        video_path = Path(dataset_root) / rel
        if not video_path.exists():
            print(f'Skipping missing video: {video_path}')
            continue

        # determine annotation length using reference p2d file
        ref_p2d = video_path.with_name(video_path.stem + '_p2d.txt')
        ref_ann = _read_full_p2d_file(ref_p2d)
        n_ann = len(ref_ann)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        frame_off = max(0, total_frames - n_ann) if n_ann > 0 else frame_offset

        # Read per-model p2d cache files (may be missing)
        cache_root = Path(dataset_root) / 'p2d_cache' / str(patient) / str(exercise)
        model_files = {}
        for m in ('dekr', 'openpose', 'mediapipe'):
            fp = cache_root / f"{cam}_{m}_p2d.txt"
            model_files[m] = _read_full_p2d_file(fp)

        # Prepare output file
        out_cache_dir = Path(dataset_root) / 'p2d_cache' / str(patient) / str(exercise)
        out_cache_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_cache_dir / f"{cam}_fusion_p2d.txt"

        print(f'Processing video [{li+1}/{len(lines)}]: {rel} frames={total_frames} ann_lines={n_ann} -> {out_file}')

        # iterate annotation lines (if no ref annotations, fallback to length of longest model cache)
        if n_ann == 0:
            n_ann = max(len(model_files['dekr']), len(model_files['openpose']), len(model_files['mediapipe']))

        with out_file.open('w') as outf:
            for ann_idx in range(n_ann):
                # read per-model entries for this ann_idx
                def read_entry(lst, idx):
                    if idx < 0 or idx >= len(lst):
                        return [0.0] * (K * 3)
                    vals = lst[idx]
                    if len(vals) < K * 3:
                        vals = vals + [0.0] * (K * 3 - len(vals))
                    return vals[:K * 3]

                dekr_vals = read_entry(model_files['dekr'], ann_idx)
                op_vals = read_entry(model_files['openpose'], ann_idx)
                mp_vals = read_entry(model_files['mediapipe'], ann_idx)

                def to_coords_conf(vals):
                    arr = np.array(vals, dtype=np.float32).reshape(K, 3)
                    coords_px = arr[:, :2]
                    conf = arr[:, 2:3]
                    return coords_px, conf

                coords_dekr_px, conf_dekr = to_coords_conf(dekr_vals)
                coords_op_px, conf_op = to_coords_conf(op_vals)
                coords_mp_px, conf_mp = to_coords_conf(mp_vals)

                # choose normalization reference
                cap = cv2.VideoCapture(str(video_path))
                ret, frame0 = cap.read()
                if not ret:
                    orig_wh = np.array([output_size[0], output_size[1]], dtype=np.float32)
                else:
                    h, w = frame0.shape[:2]
                    orig_wh = np.array([w, h], dtype=np.float32)
                cap.release()

                if norm_ref == 'video':
                    ref_wh = orig_wh
                else:
                    ref_wh = np.array([output_size[1], output_size[0]], dtype=np.float32)

                # normalize for model input (K,2) -> (1,K,2)
                coords_dekr_norm = torch.from_numpy(coords_dekr_px / ref_wh).unsqueeze(0).float().to(device_t)
                coords_op_norm = torch.from_numpy(coords_op_px / ref_wh).unsqueeze(0).float().to(device_t)
                coords_mp_norm = torch.from_numpy(coords_mp_px / ref_wh).unsqueeze(0).float().to(device_t)
                conf_dekr_t = torch.from_numpy(conf_dekr).unsqueeze(0).float().to(device_t)
                conf_op_t = torch.from_numpy(conf_op).unsqueeze(0).float().to(device_t)
                conf_mp_t = torch.from_numpy(conf_mp).unsqueeze(0).float().to(device_t)

                coords_preds = []
                logits_preds = []
                for ck in ck_dicts:
                    if 'model_state' in ck:
                        try:
                            model.load_state_dict(ck['model_state'], strict=True)
                        except Exception:
                            try:
                                model.load_state_dict(ck['model_state'], strict=False)
                            except Exception:
                                pass
                    else:
                        try:
                            model.load_state_dict(ck, strict=False)
                        except Exception:
                            pass

                    coords_fused, conf_logits = model(
                        coords_dekr_norm, conf_dekr_t,
                        coords_op_norm, conf_op_t,
                        coords_mp_norm, conf_mp_t,
                    )
                    coords_preds.append(coords_fused.detach().cpu().numpy()[0])
                    logits_preds.append(conf_logits.detach().cpu().numpy()[0])

                coords_avg = np.mean(np.stack(coords_preds, axis=0), axis=0)  # (K,2)
                logits_avg = np.mean(np.stack(logits_preds, axis=0), axis=0)  # (K,1)
                if apply_sigmoid:
                    conf_probs = 1.0 / (1.0 + np.exp(-logits_avg)).squeeze(-1)
                else:
                    conf_probs = logits_avg.squeeze(-1)

                # Convert normalized coords back to pixel coordinates in the chosen ref space
                coords_px_out = (coords_avg * ref_wh).tolist()

                # Write line as 3*(x y conf)
                out_vals = []
                for j in range(K):
                    x, y = coords_px_out[j]
                    c = float(conf_probs[j])
                    out_vals.extend([float(x), float(y), float(c)])
                # ensure string formatted like other p2d writers
                line = ' '.join(str(v) for v in out_vals)
                outf.write(line + '\n')

        print(f'Wrote fusion p2d for {rel} -> {out_file}')

    print('Inference complete')


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True, help='One or more checkpoint paths (checkpoint_best.pt)')
    p.add_argument('--list', required=True, help='List file of samples (test/val)')
    p.add_argument('--root', required=True, help='Dataset root path')
    p.add_argument('--out-dir', required=True, help='Directory to write predictions')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--no-sigmoid', dest='apply_sigmoid', action='store_false', help='Do not apply sigmoid to confidence logits')
    p.add_argument('--no-px', dest='save_px', action='store_false', help='Do not save pixel coordinates (only normalized)')
    p.add_argument('--require-caches', action='store_true', help='Require p2d cache files to exist')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--pin-memory', action='store_true')
    p.add_argument('--norm-ref', choices=['video', 'output'], default='video', help="Normalization reference for model inputs: 'video' or 'output'")
    p.add_argument('--frame-offset', type=int, default=5, help='Frame offset between video frames and annotation lines')
    p.add_argument('--output-size', nargs=2, type=int, default=[256, 256], help='Output size (H W) used for output normalization when norm-ref=output')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_inference(
        args.checkpoints, args.list, args.root, args.out_dir,
        batch_size=args.batch_size, device=args.device, apply_sigmoid=args.apply_sigmoid, save_px=args.save_px,
        require_caches=args.require_caches, num_workers=args.num_workers, pin_memory=args.pin_memory,
        norm_ref=args.norm_ref, frame_offset=args.frame_offset, output_size=(int(args.output_size[0]), int(args.output_size[1]))
    )
