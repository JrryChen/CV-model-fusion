#!/usr/bin/env python3
"""
Minimal PyTorch Dataset for frame-level keypoint training.

Features:
- Reads list files produced by `split_dataset.py` (lines like `clips_mp4/1/09/cam0.mp4`).
- Reads corresponding `cam#_p2d.txt` which is one line per annotated frame.
- Supports a fixed frame offset between video frames and annotation lines (default 5).
- Optionally resizes frames to a fixed output size and generates keypoint heatmaps
    for heatmap-based models.
- Builds an index of (video_path, frame_idx, annotation_idx) for fast __getitem__.
- Returns image tensor and keypoints tensor shaped (K,3) with (x,y,v). If heatmaps
    are requested, returns 'heatmaps' of shape (K, Hh, Wh).

Limitations / assumptions:
- p2d files are space-separated floats, and the first 6 values correspond to
    left_shoulder(x,y), left_elbow(x,y), left_wrist(x,y). Extra columns are ignored.
- No per-frame frame-index column in p2d files; annotations are sequential and align with
    video frames after accounting for `frame_offset`.
"""
# Note: Removed 'from __future__ import annotations' for Python 3.6 compatibility

import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def _parse_p2d_file(p2d_path: Path) -> List[Tuple[float, float, float, float, float, float]]:
    """Parse a p2d file and return list of tuples for (ls_x, ls_y, le_x, le_y, lw_x, lw_y).

    Returns empty list if file missing or empty.
    """
    if not p2d_path.exists():
        return []
    pts = []
    with p2d_path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                pts.append(None)
                continue
            try:
                vals = [float(x) for x in parts[:6]]
            except ValueError:
                pts.append(None)
                continue
            pts.append(tuple(vals))
    return pts


class VideoFrameKeypointDataset(Dataset):
    def __init__(self, list_file: str, dataset_root: str, frame_offset: int = 5, transform=None,
                 joints=['left_shoulder','left_elbow','left_wrist'], output_size: Tuple[int,int]=None,
                 generate_heatmaps: bool=False, heatmap_size: Tuple[int,int]=(64,64), sigma: float=2.0):
        """Create dataset.

        Args:
            list_file: path to train/val/test .txt produced by split_dataset.py
            dataset_root: path to dataset folder which contains `clips_mp4`
            frame_offset: number of video frames before the first annotation line (default 5)
            transform: torchvision-like transform applied to PIL image tensor; NOTE: keypoints are scaled manually for Resize
            joints: list of joint names (for metadata only)
        """
        self.dataset_root = Path(dataset_root)
        self.lines = [l.strip() for l in Path(list_file).open('r') if l.strip()]
        self.frame_offset = int(frame_offset)
        self.transform = transform
        self.joints = joints
        self.output_size = tuple(output_size) if output_size is not None else None
        self.generate_heatmaps = bool(generate_heatmaps)
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = float(sigma)

        # Build an index of (video_rel_path, frame_idx, ann_idx)
        self.index = []  # list of tuples
        for rel in self.lines:
            # rel looks like 'clips_mp4/1/09/cam0.mp4'
            video_path = self.dataset_root / rel
            if not video_path.exists():
                # skip missing videos but warn
                print(f"Warning: video not found, skipping: {video_path}")
                continue

            # p2d file assumed at same folder named cam#_p2d.txt
            p2d_path = video_path.with_name(video_path.stem + '_p2d.txt')
            ann_lines = _parse_p2d_file(p2d_path)
            n_ann = len(ann_lines)
            if n_ann == 0:
                # no annotations; skip
                continue

            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

            # valid annotated frame indices are [frame_offset, frame_offset + n_ann - 1]
            start = self.frame_offset
            end = self.frame_offset + n_ann - 1
            # clip to available frames
            start = max(0, start)
            end = min(frame_count - 1, end)
            if end < start:
                continue

            for frame_idx in range(start, end + 1):
                ann_idx = frame_idx - self.frame_offset
                if not (0 <= ann_idx < n_ann):
                    continue
                # only keep frames where annotation line is present (non-None)
                if ann_lines[ann_idx] is None:
                    continue
                self.index.append((rel, frame_idx, ann_idx))

        if len(self.index) == 0:
            print("Warning: dataset index is empty; check list_file and annotations.")

        # default image transform if none provided
        if self.transform is None:
            self.to_tensor = T.Compose([T.ToTensor()])
        else:
            self.to_tensor = self.transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rel, frame_idx, ann_idx = self.index[idx]
        video_path = self.dataset_root / rel

        # read frame
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # read annotation
        p2d_path = video_path.with_name(video_path.stem + '_p2d.txt')
        ann_lines = _parse_p2d_file(p2d_path)
        ann = ann_lines[ann_idx]
        # ann is tuple of 6 floats: ls_x, ls_y, le_x, le_y, lw_x, lw_y
        # Build keypoints array shape (K,3) where v=1 (visible)
        kps = np.zeros((len(self.joints), 3), dtype=np.float32)
        if ann is not None:
            xs = [ann[0], ann[2], ann[4]]
            ys = [ann[1], ann[3], ann[5]]
            for i in range(len(self.joints)):
                kps[i, 0] = float(xs[i])
                kps[i, 1] = float(ys[i])
                kps[i, 2] = 1.0

        # If requested, resize image and scale keypoints accordingly
        if self.output_size is not None:
            out_h, out_w = self.output_size
            scale_x = out_w / float(w)
            scale_y = out_h / float(h)
            # scale keypoints
            kps[:, 0] = kps[:, 0] * scale_x
            kps[:, 1] = kps[:, 1] * scale_y
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Convert image to tensor
        img_t = self.to_tensor(img)  # expects HWC->CHW and 0-1 float

        sample = {
            'image': img_t,  # float tensor [C,H,W]
            'keypoints': torch.from_numpy(kps),  # float tensor [K,3]
            'meta': {
                'video_rel_path': rel,
                'frame_index': frame_idx,
                'orig_size': (h, w),
            }
        }

        # If heatmaps are requested, generate them on the heatmap grid
        if self.generate_heatmaps:
            hh, ww = self.heatmap_size
            # scale from image space to heatmap space
            if self.output_size is not None:
                img_h, img_w = self.output_size
            else:
                img_h, img_w = h, w
            scale_x = ww / float(img_w)
            scale_y = hh / float(img_h)
            heatmaps = np.zeros((len(self.joints), hh, ww), dtype=np.float32)
            for i in range(len(self.joints)):
                vx = kps[i, 0] * scale_x
                vy = kps[i, 1] * scale_y
                vvis = kps[i, 2]
                if vvis <= 0:
                    continue
                # gaussian on a grid
                # create meshgrid
                xs = np.arange(0, ww, 1, np.float32)
                ys = np.arange(0, hh, 1, np.float32)
                ys = ys[:, None]
                d2 = (xs - vx) ** 2 + (ys - vy) ** 2
                heat = np.exp(-d2 / (2 * (self.sigma ** 2)))
                # normalize max to 1
                heat = heat / (heat.max() + 1e-8)
                heatmaps[i] = heat
            sample['heatmaps'] = torch.from_numpy(heatmaps)

        return sample


if __name__ == '__main__':
    # Quick smoke test — do not run heavy IO in import-time; only when executed directly
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--list', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--offset', type=int, default=5)
    args = p.parse_args()

    ds = VideoFrameKeypointDataset(args.list, args.root, frame_offset=args.offset)
    print(f"Dataset built with {len(ds)} examples")
    if len(ds) > 0:
        s = ds[0]
        print('sample keys:', list(s.keys()))
