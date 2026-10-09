"""
Supporting module for ENG 2440 Assignment 2 (whole-tumor segmentation, MSD Task01_BrainTumour).

Array convention used everywhere
--------------------------------
nibabel returns MSD images as (X, Y, Z, C) in RAS orientation with 1 mm spacing.
After preprocessing we store channels-first arrays (C, X, Y, Z) and treat them as the
PyTorch [C, D, H, W] layout:  D = X (R-L axis), H = Y (A-P axis), W = Z (S-I axis).
No axes are swapped, so voxel spacing stays (sx, sy, sz) in the same order as the array.

Channel order (from dataset.json): 0 FLAIR, 1 T1w, 2 T1gd (T1ce), 3 T2w.
Label values: 0 background, 1 edema, 2 non-enhancing tumor, 3 enhancing tumor.
Binary whole-tumor target: y = (label > 0).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

CHANNELS = ["FLAIR", "T1", "T1ce", "T2"]
CROP_MARGIN = 4          # voxels of background kept around the brain bounding box


# ----------------------------------------------------------------------------------------
# Loading / preprocessing (Section C)
# ----------------------------------------------------------------------------------------
def load_case(data_root: Path, image_relpath: str, label_relpath: str | None = None):
    """Return image (X,Y,Z,C) float32, label (X,Y,Z) uint8 or None, affine, spacing (3,)."""
    img = nib.load(str(Path(data_root) / image_relpath))
    image = np.asarray(img.dataobj, dtype=np.float32)
    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    label = None
    if label_relpath is not None:
        label = np.asarray(nib.load(str(Path(data_root) / label_relpath)).dataobj).astype(np.uint8)
    return image, label, img.affine, spacing


def brain_mask_from_image(image_xyzc: np.ndarray) -> np.ndarray:
    """Brain = voxels that are non-zero in ANY channel (the MSD background is exactly 0)."""
    return np.any(image_xyzc != 0, axis=-1)


def brain_bbox(mask: np.ndarray, margin: int = CROP_MARGIN):
    """Bounding box (tuple of slices) of the brain mask, expanded by `margin` voxels.
    Computed from the IMAGE only, never the label, so it can be applied to test cases."""
    idx = np.nonzero(mask)
    sl = []
    for ax, n in enumerate(mask.shape):
        lo = max(int(idx[ax].min()) - margin, 0)
        hi = min(int(idx[ax].max()) + 1 + margin, n)
        sl.append(slice(lo, hi))
    return tuple(sl)


def zscore_nonzero(image_xyzc: np.ndarray, mask: np.ndarray):
    """Per-channel z-score using ONLY brain voxels. Background stays exactly 0.
    Returns normalized image and the (mean, std) used for each channel."""
    out = np.zeros_like(image_xyzc, dtype=np.float32)
    stats = []
    for c in range(image_xyzc.shape[-1]):
        ch = image_xyzc[..., c]
        v = ch[mask]
        mu, sd = float(v.mean()), float(v.std()) + 1e-8
        out[..., c][mask] = (v - mu) / sd
        stats.append((mu, sd))
    return out, stats


def pad_to_min(arr: np.ndarray, min_shape, spatial_axes):
    """Zero-pad (at the end) so every spatial axis is >= min_shape. Returns array and pad widths."""
    pads = [(0, 0)] * arr.ndim
    for ax, m in zip(spatial_axes, min_shape):
        if arr.shape[ax] < m:
            pads[ax] = (0, m - arr.shape[ax])
    return np.pad(arr, pads), pads


def preprocess_case(args):
    """Worker for multiprocessing. Loads one case, crops to the brain box, z-scores each
    channel on brain voxels, pads to >= patch size, and caches:
        <cache>/<case_id>_img.npy   float16 (C, X, Y, Z)
        <cache>/<case_id>_lbl.npy   uint8   (X, Y, Z)  original multi-class labels (train/val only)
        <cache>/<case_id>_fg.npy    int16   (N, 3)     tumor voxel coords (subsampled) for tumor-aware sampling
        <cache>/<case_id>_meta.json crop box, original shape, spacing, normalization stats
    Test labels are NOT read here (kept untouched until Section F)."""
    data_root, cache_dir, case_id, image_relpath, label_relpath, split, patch = args
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / f"{case_id}_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    use_label = split != "test"
    image, label, affine, spacing = load_case(data_root, image_relpath, label_relpath if use_label else None)
    mask = brain_mask_from_image(image)
    box = brain_bbox(mask)
    norm, stats = zscore_nonzero(image, mask)
    crop = np.moveaxis(norm[box], -1, 0)                      # (C, X, Y, Z)
    crop, pads = pad_to_min(crop, (patch,) * 3, spatial_axes=(1, 2, 3))
    np.save(cache_dir / f"{case_id}_img.npy", crop.astype(np.float16))

    meta = dict(case_id=case_id, split=split, orig_shape=list(image.shape[:3]),
                box=[[s.start, s.stop] for s in box], pads=[list(p) for p in pads[1:]],
                spacing=spacing.tolist(), affine=np.asarray(affine).tolist(),
                norm_stats=stats, cached_shape=list(crop.shape))
    if use_label:
        lbl = np.pad(label[box], pads[1:])
        np.save(cache_dir / f"{case_id}_lbl.npy", lbl.astype(np.uint8))
        fg = np.argwhere(lbl > 0)
        if len(fg) > 20000:                                    # subsample to keep files small
            fg = fg[np.random.default_rng(0).choice(len(fg), 20000, replace=False)]
        np.save(cache_dir / f"{case_id}_fg.npy", fg.astype(np.int16))
        meta["tumor_voxels"] = int((label > 0).sum())
        meta["tumor_ml"] = float((label > 0).sum() * np.prod(spacing) / 1000.0)
    meta_path.write_text(json.dumps(meta))
    return meta


# ----------------------------------------------------------------------------------------
# Augmentation (training only) (Section C4)
# ----------------------------------------------------------------------------------------
def augment(img: np.ndarray, lbl: np.ndarray, rng: np.random.Generator):
    """Safe, label-preserving augmentation applied to TRAINING patches only.
    1) Left-right flip (axis X / D) with p=0.5: brains are roughly bilaterally symmetric and
       tumors occur in either hemisphere; flipping moves voxels without interpolation, so
       label values stay exactly {0,1}.
    2) Per-channel intensity scale U(0.9,1.1) and shift U(-0.1,0.1) inside the brain only:
       mimics scanner/protocol intensity variation; labels are untouched.
    No rotations/elastic warps: they need interpolation and cost time; skipped on purpose."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
        lbl = lbl[:, ::-1]
    brain = np.any(img != 0, axis=0)
    img = img.copy()
    for c in range(img.shape[0]):
        a, b = rng.uniform(0.9, 1.1), rng.uniform(-0.1, 0.1)
        img[c][brain] = img[c][brain] * a + b
    return np.ascontiguousarray(img), np.ascontiguousarray(lbl)


# ----------------------------------------------------------------------------------------
# Dataset with uniform / tumor-aware patch sampling (Section D)
# ----------------------------------------------------------------------------------------
class PatchDataset(torch.utils.data.Dataset):
    """Each item is ONE random patch. Item `idx` uses its own RNG seeded by (seed, idx), so the
    sequence of (case, patch location, augmentation) is identical regardless of worker count:
    this is what keeps the two sampler runs controlled.

    sampler = "uniform": patch origin uniform over every valid location in the cropped volume.
    sampler = "tumor":   with probability p_fg the patch is centred on a random tumor voxel
                         (jittered, clamped to the volume), otherwise uniform.
    Returns image [C,D,H,W] float32, mask [1,D,H,W] float32 (binary whole tumor), and the
    fraction of tumor voxels in the patch (> 0 means the patch contains tumor)."""

    def __init__(self, case_ids, cache_dir, patch=64, n_items=1000, sampler="uniform",
                 p_fg=0.5, seed=0, augment_on=True):
        self.case_ids = list(case_ids)
        self.cache_dir = Path(cache_dir)
        self.patch = patch
        self.n_items = n_items
        self.sampler = sampler
        self.p_fg = p_fg
        self.seed = seed
        self.augment_on = augment_on
        self._cache = {}

    def __len__(self):
        return self.n_items

    def _arrays(self, cid):
        if cid not in self._cache:   # memory-mapped: only the patch is actually read from disk
            self._cache[cid] = (np.load(self.cache_dir / f"{cid}_img.npy", mmap_mode="r"),
                                np.load(self.cache_dir / f"{cid}_lbl.npy", mmap_mode="r"),
                                np.load(self.cache_dir / f"{cid}_fg.npy"))
        return self._cache[cid]

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, idx])
        cid = self.case_ids[rng.integers(len(self.case_ids))]
        img, lbl, fg = self._arrays(cid)
        shape = np.array(img.shape[1:])
        p = self.patch
        if self.sampler == "tumor" and len(fg) > 0 and rng.random() < self.p_fg:
            centre = fg[rng.integers(len(fg))] + rng.integers(-p // 4, p // 4 + 1, size=3)
            start = np.clip(centre - p // 2, 0, shape - p)
        else:
            start = np.array([rng.integers(0, s - p + 1) for s in shape])
        sl = tuple(slice(int(s), int(s) + p) for s in start)
        x = np.asarray(img[(slice(None),) + sl], dtype=np.float32)
        y = (np.asarray(lbl[sl]) > 0).astype(np.float32)[None]
        if self.augment_on:
            x, y = augment(x, y, rng)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(float(y.mean()))


# ----------------------------------------------------------------------------------------
# Lightweight 3D U-Net (Section D2)
# ----------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """4-level 3D U-Net: encoder widths base*(1,2,4,8), strided-conv downsampling,
    transposed-conv upsampling, skip connections by concatenation, 1x1x1 output head (logits)."""

    def __init__(self, in_ch=4, out_ch=1, base=16):
        super().__init__()
        w = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([ConvBlock(in_ch, w[0]), ConvBlock(w[0], w[1]),
                                  ConvBlock(w[1], w[2]), ConvBlock(w[2], w[3])])
        self.down = nn.ModuleList([nn.Conv3d(w[i], w[i], 2, stride=2) for i in range(3)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(w[i + 1], w[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([ConvBlock(2 * w[i], w[i]) for i in range(3)])
        self.head = nn.Conv3d(w[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i in range(3):
            x = self.enc[i](x)
            skips.append(x)
            x = self.down[i](x)
        x = self.enc[3](x)
        for i in reversed(range(3)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], dim=1))
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------
def dice_bce_loss(logits, target, eps=1e-5):
    """BCE-with-logits + soft Dice (Dice computed over the whole batch so patches with no
    tumor do not produce a 0/0 term). Equal weights."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = (2 * inter + eps) / (p.sum() + target.sum() + eps)
    return bce + (1 - dice)


# ----------------------------------------------------------------------------------------
# Sliding-window inference (Section F1)
# ----------------------------------------------------------------------------------------
def gaussian_weight(patch, sigma_scale=0.125):
    """3D Gaussian importance map: centre voxels of a window (full context) get weight 1,
    border voxels (truncated context) get small weight. sigma = patch * 1/8."""
    c = (patch - 1) / 2
    g = np.exp(-((np.arange(patch) - c) ** 2) / (2 * (patch * sigma_scale) ** 2))
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    w = w / w.max()
    return np.maximum(w, 1e-3).astype(np.float32)   # never exactly 0 -> no division by 0


def window_starts(size, patch, stride):
    """Window start positions along one axis; last window is aligned to the end so every
    voxel is covered."""
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def sliding_window_predict(model, image_cxyz, patch=64, overlap=0.5, device="cpu",
                           batch_size=8, blend="gaussian"):
    """Overlapping sliding-window inference on a (C, X, Y, Z) volume.
    stride = round(patch * (1 - overlap)). Each window's sigmoid PROBABILITIES are multiplied
    by the blending weight and accumulated; the sum is divided by the accumulated weights.
    Thresholding happens only afterwards, on the blended probability map."""
    model.eval()
    C, X, Y, Z = image_cxyz.shape
    stride = max(1, int(round(patch * (1 - overlap))))
    w = gaussian_weight(patch) if blend == "gaussian" else np.ones((patch,) * 3, np.float32)
    w_t = torch.from_numpy(w).to(device)
    prob = torch.zeros((X, Y, Z), device=device)
    norm = torch.zeros((X, Y, Z), device=device)
    vol = torch.from_numpy(np.ascontiguousarray(image_cxyz, dtype=np.float32)).to(device)
    coords = [(a, b, c) for a in window_starts(X, patch, stride)
              for b in window_starts(Y, patch, stride) for c in window_starts(Z, patch, stride)]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = torch.stack([vol[:, a:a + patch, b:b + patch, c:c + patch] for a, b, c in chunk])
        p = torch.sigmoid(model(batch))[:, 0]
        for k, (a, b, c) in enumerate(chunk):
            prob[a:a + patch, b:b + patch, c:c + patch] += p[k] * w_t
            norm[a:a + patch, b:b + patch, c:c + patch] += w_t
    return (prob / norm).cpu().numpy(), dict(stride=stride, n_windows=len(coords))


def uncrop(prob_crop, meta):
    """Paste a prediction computed on the cropped/padded grid back into the ORIGINAL
    240x240x155 grid (outside the brain box = 0), so metrics use the native geometry."""
    full = np.zeros(meta["orig_shape"], dtype=np.float32)
    (x0, x1), (y0, y1), (z0, z1) = meta["box"]
    full[x0:x1, y0:y1, z0:z1] = prob_crop[:x1 - x0, :y1 - y0, :z1 - z0]
    return full


# ----------------------------------------------------------------------------------------
# Metrics (Sections E, F)
# ----------------------------------------------------------------------------------------
def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)), border_value=0)


def hd95(pred, ref, spacing):
    """95th-percentile symmetric Hausdorff distance in mm (medpy convention: 95th percentile of
    the pooled surface-to-surface distances in both directions). Distances use the voxel
    spacing. Undefined (NaN) if either mask is empty. Computed on a bounding box around both
    masks for speed (identical result)."""
    if pred.sum() == 0 or ref.sum() == 0:
        return float("nan")
    union = pred | ref
    idx = np.nonzero(union)
    sl = tuple(slice(max(int(i.min()) - 2, 0), int(i.max()) + 3) for i in idx)
    p, r = pred[sl], ref[sl]
    sp, sr = _surface(p), _surface(r)
    dt_r = ndimage.distance_transform_edt(~sr, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_r[sp], dt_p[sr]])
    return float(np.percentile(d, 95))


def seg_metrics(pred, ref, spacing):
    """Whole-tumor metrics for one case. pred/ref are boolean arrays on the same grid."""
    pred, ref = pred.astype(bool), ref.astype(bool)
    tp = np.count_nonzero(pred & ref)
    fp = np.count_nonzero(pred & ~ref)
    fn = np.count_nonzero(~pred & ref)
    vox_ml = float(np.prod(spacing)) / 1000.0
    ps, rs = tp + fp, tp + fn
    dice = 2 * tp / (ps + rs) if (ps + rs) > 0 else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    sens = tp / rs if rs > 0 else float("nan")
    pred_ml, ref_ml = ps * vox_ml, rs * vox_ml
    return dict(dice=dice, iou=iou, sensitivity=sens, hd95_mm=hd95(pred, ref, spacing),
                pred_ml=pred_ml, ref_ml=ref_ml, fp_ml=fp * vox_ml, fn_ml=fn * vox_ml,
                abs_vol_err_ml=abs(pred_ml - ref_ml),
                pct_vol_err=100.0 * (pred_ml - ref_ml) / ref_ml if ref_ml > 0 else float("nan"),
                empty_pred=ps == 0)


# ----------------------------------------------------------------------------------------
# Connected-component postprocessing (Section F4)
# ----------------------------------------------------------------------------------------
def remove_small_components(mask, spacing, min_ml=0.5):
    """Remove 26-connected components smaller than `min_ml` millilitres. Uses physical volume
    (not voxel count) so the rule means the same thing on any grid."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1)) * np.prod(spacing) / 1000.0
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_ml
    return keep[lab]


# ----------------------------------------------------------------------------------------
# Visualisation helpers (Sections A, G)
# ----------------------------------------------------------------------------------------
def plane_slices(vol, centre):
    """Return (axial, coronal, sagittal) 2D slices through `centre`=(x,y,z), oriented for
    display with origin='lower': rows = second axis, columns = first axis."""
    x, y, z = centre
    return {"Axial (z=%d)" % z: vol[:, :, z].T,
            "Coronal (y=%d)" % y: vol[:, y, :].T,
            "Sagittal (x=%d)" % x: vol[x, :, :].T}


def error_rgb(pred, ref):
    """RGBA overlay: TP green, FP red, FN blue, else transparent."""
    rgba = np.zeros(pred.shape + (4,), np.float32)
    rgba[pred & ref] = (0.1, 0.85, 0.2, 0.75)
    rgba[pred & ~ref] = (1.0, 0.15, 0.1, 0.85)
    rgba[~pred & ref] = (0.15, 0.4, 1.0, 0.85)
    return rgba


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0


# ----------------------------------------------------------------------------------------
# Training loop (Section D3/D4) - identical for both samplers
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_patches(model, loader, device):
    """Mean loss over a FIXED set of validation patches (same patches for every run)."""
    model.eval()
    losses = []
    for x, y, _ in loader:
        losses.append(dice_bce_loss(model(x.to(device)), y.to(device)).item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_volumes(model, case_ids, cache_dir, patch, device, overlap=0.5, thr=0.5):
    """Full-volume validation Dice (sliding window) on the cached cropped grid. The crop box
    contains the whole brain, so Dice here equals Dice on the original grid."""
    dices = []
    for cid in case_ids:
        img = np.load(Path(cache_dir) / f"{cid}_img.npy").astype(np.float32)
        ref = np.load(Path(cache_dir) / f"{cid}_lbl.npy") > 0
        prob, _ = sliding_window_predict(model, img, patch, overlap, device)
        pred = prob >= thr
        s = pred.sum() + ref.sum()
        dices.append(2 * (pred & ref).sum() / s if s > 0 else 1.0)
    return float(np.mean(dices))


def train_run(name, model, train_ds, val_patch_loader, val_case_ids, cache_dir, device,
              iters=2000, batch_size=4, lr=1e-3, eval_every=200, num_workers=4, out_dir="outputs",
              log=print):
    """Train `model` for exactly `iters` optimizer steps on patches from `train_ds`.
    Adam, poly LR decay (1 - t/T)^0.9, grad-norm clip 5. Every `eval_every` steps record
    train loss (mean over the window), validation patch loss and validation full-volume Dice.
    Keeps the checkpoint with the best validation Dice (validation data only)."""
    out_dir = Path(out_dir)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (1 - t / iters) ** 0.9)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, persistent_workers=num_workers > 0,
                                         prefetch_factor=4 if num_workers > 0 else None)
    hist = dict(iter=[], train_loss=[], val_loss=[], val_dice=[], lr=[], time_s=[])
    tumor_frac, window = [], []
    best, t0, it = -1.0, time.perf_counter(), 0
    model.train()
    for x, y, has_t in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = dice_bce_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        it += 1
        window.append(loss.item())
        tumor_frac.extend(has_t.tolist())
        if it % eval_every == 0 or it == iters:
            vl = evaluate_patches(model, val_patch_loader, device)
            vd = evaluate_volumes(model, val_case_ids, cache_dir, train_ds.patch, device)
            model.train()
            for k, v in zip(hist, [it, np.mean(window), vl, vd, sched.get_last_lr()[0],
                                   time.perf_counter() - t0]):
                hist[k].append(float(v))
            window = []
            if vd > best:
                best = vd
                torch.save(model.state_dict(), out_dir / f"{name}_best.pt")
            log(f"[{name}] it {it:5d}/{iters}  train {hist['train_loss'][-1]:.4f}  "
                f"val_loss {vl:.4f}  val_dice {vd:.4f}  ({hist['time_s'][-1] / 60:.1f} min)")
        if it >= iters:
            break
    torch.save(model.state_dict(), out_dir / f"{name}_last.pt")
    tf = np.array(tumor_frac)
    hist["tumor_patch_fraction"] = float(np.mean(tf > 0))       # patches containing any tumor
    hist["tumor_voxel_fraction"] = float(np.mean(tf))           # mean share of tumor voxels
    hist["best_val_dice_subset"] = best
    hist["train_minutes"] = (time.perf_counter() - t0) / 60
    (out_dir / f"{name}_history.json").write_text(json.dumps(hist))
    return hist
