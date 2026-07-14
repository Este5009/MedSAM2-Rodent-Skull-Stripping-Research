#!/usr/bin/env python
"""Run one oracle-box MedSAM2 smoke test on a CAMRI rat MRI volume.

This deliberately remains a single project-owned script: it exercises the official
``SAM2VideoPredictorNPZ`` API without changing any upstream MedSAM2 source file.
SimpleITK returns arrays as (z, y, x); that order is retained throughout inference,
metrics, figures, and NIfTI reconstruction. No display rotation/transposition occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.signal import find_peaks
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_ROOT = REPO_ROOT / "Image_Database" / "CAMRI Rat Brain MRI Data"
DEFAULT_MASK_ROOT = REPO_ROOT / "Mask_Database" / "RodentBrainMask" / "CAMRI Rat"
DEFAULT_CHECKPOINT = REPO_ROOT / "MedSAM2" / "checkpoints" / "MedSAM2_latest.pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "single_subject"
DEFAULT_CONFIG = "configs/sam2.1_hiera_t512.yaml"
MODEL_SIZE = 512


@dataclass(frozen=True)
class Case:
    subject_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class NormalizationResult:
    """Normalized MedSAM2 input and fully auditable estimation diagnostics."""

    image_uint8: np.ndarray
    method_requested: str
    method_applied: str
    diagnostics: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", help="Subject ID such as sub-001; default: first matched case")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Hydra config name in sam2/configs")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--normalization", choices=("percentile", "hybrid_whitestripe"), default="percentile",
        help="The controlled MRI normalization condition.",
    )
    parser.add_argument("--clip-lower-percentile", type=float, default=0.5)
    parser.add_argument("--clip-upper-percentile", type=float, default=99.5)
    parser.add_argument("--max-montage-slices", type=int, default=48)
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true", help="Reduce VRAM at a speed cost")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("medsam2_camri_single")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(output_dir / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def locate_case(image_root: Path, mask_root: Path, requested_subject: str | None) -> Case:
    """Match CAMRI rat RARE images and masks by the BIDS subject/session/acquisition key."""
    images: dict[str, Path] = {}
    pattern = re.compile(r"(sub-\d+_ses-\d+_acq-RARE_T2w)")
    for path in sorted(image_root.rglob("*.nii*")):
        match = pattern.search(path.name)
        if match:
            images[match.group(1)] = path
    pairs: list[Case] = []
    for mask_path in sorted(mask_root.glob("*.nii*")):
        match = pattern.search(mask_path.name)
        if match and match.group(1) in images:
            subject = match.group(1).split("_")[0]
            pairs.append(Case(subject, images[match.group(1)], mask_path))
    if requested_subject:
        pairs = [case for case in pairs if case.subject_id == requested_subject]
    if not pairs:
        raise FileNotFoundError(f"No matching CAMRI rat image/mask pair found (subject={requested_subject!r}).")
    return pairs[0]


def geometry_signature(image: sitk.Image) -> dict[str, object]:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def validate_geometry(image: sitk.Image, mask: sitk.Image) -> None:
    if image.GetSize() != mask.GetSize():
        raise ValueError(f"Image/mask size mismatch: {image.GetSize()} vs {mask.GetSize()}")
    for name, left, right in (
        ("spacing", image.GetSpacing(), mask.GetSpacing()),
        ("origin", image.GetOrigin(), mask.GetOrigin()),
        ("direction", image.GetDirection(), mask.GetDirection()),
    ):
        if not np.allclose(left, right, rtol=0.0, atol=1e-6):
            raise ValueError(f"Image/mask {name} mismatch: {left} vs {right}")


def finite_foreground(volume: np.ndarray) -> np.ndarray:
    """Return the common 3D foreground definition used by both conditions."""
    return np.isfinite(volume) & (volume != 0)


def scale_foreground_to_uint8(
    values_volume: np.ndarray, foreground: np.ndarray, low: float, high: float
) -> np.ndarray:
    """Clip and scale foreground only, leaving all background exactly zero."""
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError(f"Invalid clipping range: low={low}, high={high}")
    result = np.zeros(values_volume.shape, dtype=np.uint8)
    clipped = np.clip(values_volume[foreground], low, high)
    result[foreground] = np.rint((clipped - low) * (255.0 / (high - low))).astype(np.uint8)
    return result


def percentile_normalize_mri(
    volume: np.ndarray, lower_percentile: float, upper_percentile: float
) -> NormalizationResult:
    """Apply the established whole-volume robust percentile baseline."""
    volume = volume.astype(np.float32, copy=False)
    foreground = finite_foreground(volume)
    count = int(foreground.sum())
    if count == 0:
        raise ValueError("MRI has no finite, nonzero foreground voxels.")
    low, high = np.percentile(volume[foreground], [lower_percentile, upper_percentile])
    image_uint8 = scale_foreground_to_uint8(volume, foreground, float(low), float(high))
    diagnostics = {
        "foreground_definition": "finite and image != 0",
        "foreground_voxel_count": count,
        "clip_percentiles": [lower_percentile, upper_percentile],
        "final_clipping_limits": [float(low), float(high)],
        "output_range": [0, 255],
        "fallback_used": False,
        "fallback_reason": None,
    }
    return NormalizationResult(image_uint8, "percentile", "percentile", diagnostics)


def hybrid_whitestripe_normalize_mri(
    volume: np.ndarray, lower_percentile: float, upper_percentile: float
) -> NormalizationResult:
    """Estimate one robust WhiteStripe reference from the full 3D foreground.

    The estimator adapts the earlier rodent-MRI diagnostic's smoothed-histogram
    approach. Unlike that manual ROI tool, this controlled experiment uses the
    complete finite/nonzero 3D foreground and deterministically chooses the most
    prominent mode inside the central 5th--95th percentile intensity range.
    """
    volume = volume.astype(np.float32, copy=False)
    foreground = finite_foreground(volume)
    values = volume[foreground]
    diagnostics: dict[str, object] = {
        "foreground_definition": "finite and image != 0",
        "foreground_voxel_count": int(values.size),
        "histogram_mode_selection_method": (
            "256-bin histogram after 0.5-99.5% raw clipping; Gaussian sigma=2; "
            "highest-prominence peak within raw foreground 5-95% range"
        ),
        "fallback_used": False,
        "fallback_reason": None,
    }

    def fallback(reason: str) -> NormalizationResult:
        baseline = percentile_normalize_mri(volume, lower_percentile, upper_percentile)
        combined = {**diagnostics, **baseline.diagnostics, "fallback_used": True, "fallback_reason": reason}
        return NormalizationResult(baseline.image_uint8, "hybrid_whitestripe", "percentile_fallback", combined)

    if values.size < 1_000:
        return fallback(f"too few foreground voxels ({values.size} < 1000)")
    histogram_low, histogram_high = np.percentile(values, [0.5, 99.5])
    if not np.isfinite([histogram_low, histogram_high]).all() or histogram_high <= histogram_low:
        return fallback("invalid robust intensity range for histogram")
    histogram_values = values[(values >= histogram_low) & (values <= histogram_high)]
    counts, edges = np.histogram(histogram_values, bins=256, range=(histogram_low, histogram_high))
    centers = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=2.0)
    prominence = max(float(smoothed.max()) * 0.03, 1.0)
    peaks, properties = find_peaks(smoothed, prominence=prominence, distance=6)
    central_low, central_high = np.percentile(values, [5.0, 95.0])
    eligible = [i for i, peak in enumerate(peaks) if central_low <= centers[peak] <= central_high]
    diagnostics.update({
        "histogram_robust_limits": [float(histogram_low), float(histogram_high)],
        "histogram_bin_count": 256,
        "candidate_peak_count": int(len(peaks)),
    })
    if not eligible:
        return fallback("no prominent histogram peak in the central 5-95% foreground range")
    chosen_position = max(eligible, key=lambda i: float(properties["prominences"][i]))
    peak_index = int(peaks[chosen_position])
    peak_center = float(centers[peak_index])
    bin_width = float(edges[1] - edges[0])
    half_width = max(2.0 * bin_width, 0.025 * float(histogram_high - histogram_low))
    stripe_low = peak_center - half_width
    stripe_high = peak_center + half_width
    stripe_values = values[(values >= stripe_low) & (values <= stripe_high)]
    stripe_mean = float(np.mean(stripe_values)) if stripe_values.size else float("nan")
    stripe_std = float(np.std(stripe_values)) if stripe_values.size else float("nan")
    minimum_stripe_voxels = max(100, int(math.ceil(values.size * 0.001)))
    minimum_std = max(1e-6, float(histogram_high - histogram_low) * 0.001)
    diagnostics.update({
        "selected_peak_intensity": peak_center,
        "selected_peak_prominence": float(properties["prominences"][chosen_position]),
        "selected_stripe_intensity_bounds": [float(stripe_low), float(stripe_high)],
        "stripe_voxel_count": int(stripe_values.size),
        "minimum_required_stripe_voxels": minimum_stripe_voxels,
        "stripe_mean": stripe_mean,
        "stripe_standard_deviation": stripe_std,
        "minimum_required_stripe_standard_deviation": minimum_std,
    })
    if stripe_values.size < minimum_stripe_voxels:
        return fallback(f"stripe too narrow: {stripe_values.size} voxels < {minimum_stripe_voxels}")
    if not np.isfinite([stripe_mean, stripe_std]).all() or stripe_std <= minimum_std:
        return fallback(f"stripe standard deviation invalid or too small ({stripe_std})")

    normalized = (volume - stripe_mean) / stripe_std
    normalized_values = normalized[foreground]
    clip_low, clip_high = np.percentile(normalized_values, [lower_percentile, upper_percentile])
    if not np.isfinite([clip_low, clip_high]).all() or clip_high <= clip_low:
        return fallback("invalid robust clipping limits after WhiteStripe z-scoring")
    image_uint8 = scale_foreground_to_uint8(normalized, foreground, float(clip_low), float(clip_high))
    diagnostics.update({
        "clip_percentiles": [lower_percentile, upper_percentile],
        "final_clipping_limits": [float(clip_low), float(clip_high)],
        "output_range": [0, 255],
    })
    return NormalizationResult(image_uint8, "hybrid_whitestripe", "hybrid_whitestripe", diagnostics)


def normalize_mri(
    volume: np.ndarray, method: str, lower_percentile: float, upper_percentile: float
) -> NormalizationResult:
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Clipping percentiles must satisfy 0 <= lower < upper <= 100.")
    if method == "percentile":
        return percentile_normalize_mri(volume, lower_percentile, upper_percentile)
    if method == "hybrid_whitestripe":
        return hybrid_whitestripe_normalize_mri(volume, lower_percentile, upper_percentile)
    raise ValueError(f"Unknown normalization method: {method}")


def tight_box(mask_2d: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_2d)
    if xs.size == 0:
        raise ValueError("Cannot create an oracle box from an empty initialization mask.")
    # Inclusive extrema match the official upstream inference scripts.
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def resize_for_model(volume_u8: np.ndarray) -> np.ndarray:
    """Reproduce the official PIL grayscale->RGB->512 resize, returning D,C,H,W."""
    resized = np.empty((volume_u8.shape[0], 3, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
    for z, image in enumerate(volume_u8):
        rgb = Image.fromarray(image, mode="L").convert("RGB")
        rgb = rgb.resize((MODEL_SIZE, MODEL_SIZE), resample=Image.Resampling.BILINEAR)
        resized[z] = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)
    return resized


def normalize_for_model(resized: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(resized).cuda().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda")[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device="cuda")[:, None, None]
    return tensor.sub_(mean).div_(std)


def transformed_box(box: np.ndarray, height: int, width: int) -> np.ndarray:
    transformed = box.copy()
    transformed[[0, 2]] *= MODEL_SIZE / width
    transformed[[1, 3]] *= MODEL_SIZE / height
    if np.any(transformed < 0) or np.any(transformed > MODEL_SIZE):
        raise AssertionError(f"Transformed box outside 512x512 model bounds: {transformed.tolist()}")
    return transformed


def infer_bidirectionally(predictor: object, images: torch.Tensor, original_hw: tuple[int, int], init_z: int,
                          box: np.ndarray, offload_state: bool) -> np.ndarray:
    """Prompt once, then propagate from the initialization slice in both directions."""
    depth, height, width = images.shape[0], *original_hw
    prediction = np.zeros((depth, height, width), dtype=bool)
    state = predictor.init_state(images, height, width, offload_state_to_cpu=offload_state)
    _, _, init_logits = predictor.add_new_points_or_box(state, frame_idx=init_z, obj_id=1, box=box)
    init_mask = (init_logits[0] > 0).cpu().numpy()[0]
    prediction[init_z] = init_mask
    for z, _, logits in predictor.propagate_in_video(state, start_frame_idx=init_z, reverse=False):
        prediction[z] = (logits[0] > 0).cpu().numpy()[0]
    predictor.reset_state(state)

    # A fresh state prevents forward memories from contaminating reverse propagation.
    state = predictor.init_state(images, height, width, offload_state_to_cpu=offload_state)
    predictor.add_new_points_or_box(state, frame_idx=init_z, obj_id=1, box=box)
    for z, _, logits in predictor.propagate_in_video(state, start_frame_idx=init_z, reverse=True):
        prediction[z] = (logits[0] > 0).cpu().numpy()[0]
    predictor.reset_state(state)
    prediction[init_z] = init_mask
    return prediction


def dice_iou(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    intersection = np.count_nonzero(pred & truth)
    total = np.count_nonzero(pred) + np.count_nonzero(truth)
    union = np.count_nonzero(pred | truth)
    dice = 1.0 if total == 0 else 2.0 * intersection / total
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def per_slice_rows(prediction: np.ndarray, truth: np.ndarray, init_z: int) -> list[dict[str, object]]:
    rows = []
    for z in range(truth.shape[0]):
        dice, iou = dice_iou(prediction[z], truth[z])
        rows.append({
            "slice_index": z, "side": "initialization" if z == init_z else ("backward" if z < init_z else "forward"),
            "distance_from_initialization": abs(z - init_z), "dice": dice, "iou": iou,
            "ground_truth_voxels": int(truth[z].sum()), "predicted_voxels": int(prediction[z].sum()),
            "ground_truth_nonempty": bool(truth[z].any()), "prediction_nonempty": bool(prediction[z].any()),
        })
    return rows


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def show_base(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def contour(ax: plt.Axes, mask: np.ndarray, color: str, linewidth: float = 1.2) -> None:
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def overlay(ax: plt.Axes, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.35) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask, :3] = color
    rgba[mask, 3] = alpha
    ax.imshow(rgba, origin="upper")


def save_initialization_qc(raw: np.ndarray, truth: np.ndarray, pred: np.ndarray, z: int, box: np.ndarray,
                           dice: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    titles = ["Original MRI", "Oracle initialization box", "Expert mask", "MedSAM2 prediction", "Errors", "Expert vs prediction contours"]
    for ax, title in zip(axes.flat, titles):
        show_base(ax, raw[z], title)
    x0, y0, x1, y1 = box
    axes.flat[1].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="deepskyblue", linewidth=2))
    overlay(axes.flat[2], truth[z], (0.1, 0.9, 0.2))
    overlay(axes.flat[3], pred[z], (1.0, 0.8, 0.0))
    overlay(axes.flat[4], pred[z] & ~truth[z], (1.0, 0.1, 0.1)); overlay(axes.flat[4], truth[z] & ~pred[z], (0.1, 0.4, 1.0))
    contour(axes.flat[5], truth[z], "lime"); contour(axes.flat[5], pred[z], "gold")
    fig.suptitle(f"Initialization slice z={z} | Dice={dice:.4f}", fontsize=15)
    fig.legend(handles=[Patch(color="red", label="False positive"), Patch(color="royalblue", label="False negative")], loc="lower center", ncol=2)
    fig.savefig(path, dpi=220); plt.close(fig)


def sampled_indices(depth: int, maximum: int) -> np.ndarray:
    return np.arange(depth) if depth <= maximum else np.unique(np.rint(np.linspace(0, depth-1, maximum)).astype(int))


def save_full_montage(raw: np.ndarray, truth: np.ndarray, pred: np.ndarray, rows: list[dict[str, object]], path: Path,
                      maximum: int) -> None:
    indices = sampled_indices(raw.shape[0], maximum)
    cols = 6; nrows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(18, 3*nrows), squeeze=False, constrained_layout=True)
    for ax in axes.flat: ax.set_axis_off()
    for ax, z in zip(axes.flat, indices):
        show_base(ax, raw[z], f"z={z} | Dice={rows[z]['dice']:.3f}")
        contour(ax, truth[z], "lime"); contour(ax, pred[z], "magenta")
    fig.suptitle("Full-volume continuity (native z,y,x display; uniformly sampled when needed)")
    fig.legend(handles=[Line2D([0],[0], color="lime", label="Expert"), Line2D([0],[0], color="magenta", label="MedSAM2")], loc="lower center", ncol=2)
    fig.savefig(path, dpi=180); plt.close(fig)


def save_ranked_montage(raw: np.ndarray, truth: np.ndarray, pred: np.ndarray, rows: list[dict[str, object]], indices: Iterable[int],
                         title: str, path: Path) -> None:
    indices = list(indices); fig, axes = plt.subplots(len(indices), 4, figsize=(14, 3.2*len(indices)), squeeze=False, constrained_layout=True)
    for row_axes, z in zip(axes, indices):
        for ax, label in zip(row_axes, ("MRI", "Expert mask", "Prediction", "FP red / FN blue")): show_base(ax, raw[z], label)
        overlay(row_axes[1], truth[z], (0.1, 0.9, 0.2)); overlay(row_axes[2], pred[z], (1.0, 0.8, 0.0))
        overlay(row_axes[3], pred[z] & ~truth[z], (1.0, 0.1, 0.1)); overlay(row_axes[3], truth[z] & ~pred[z], (0.1, 0.4, 1.0))
        row_axes[0].set_ylabel(f"z={z}\nDice={rows[z]['dice']:.4f}", fontsize=10)
    fig.suptitle(title); fig.savefig(path, dpi=180); plt.close(fig)


def save_profiles(rows: list[dict[str, object]], init_z: int, dice_path: Path, area_path: Path) -> None:
    z = np.array([r["slice_index"] for r in rows]); dice = np.array([r["dice"] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 5)); ax.axvspan(z.min(), init_z, color="royalblue", alpha=.08, label="Backward side")
    ax.axvspan(init_z, z.max(), color="darkorange", alpha=.08, label="Forward side")
    ax.plot(z, dice, color="black", marker=".", linewidth=1); ax.axvline(init_z, color="crimson", linestyle="--", label=f"Initialization z={init_z}")
    ax.set(xlabel="Slice index (native z order)", ylabel="Dice", ylim=(-.02, 1.02), title="MedSAM2 propagation profile"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(dice_path, dpi=220); plt.close(fig)
    gt = np.array([r["ground_truth_voxels"] for r in rows]); pred = np.array([r["predicted_voxels"] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 5)); ax.plot(z, gt, color="limegreen", label="Expert", linewidth=2); ax.plot(z, pred, color="magenta", label="MedSAM2", linewidth=2)
    ax.axvline(init_z, color="crimson", linestyle="--", label=f"Initialization z={init_z}"); ax.set(xlabel="Slice index (native z order)", ylabel="Foreground pixels", title="Foreground-area profile")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(area_path, dpi=220); plt.close(fig)


def save_input_debug(preprocessed: np.ndarray, resized: np.ndarray, z: int, box: np.ndarray, model_box: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    show_base(axes[0], preprocessed[z], f"Preprocessed slice z={z}\nuint8 [0,255]")
    show_base(axes[1], resized[z, 0], "Exact resized channel 0 passed to normalization\n512 x 512")
    show_base(axes[2], resized[z, 0], f"Transformed box\n{np.round(model_box, 2).tolist()}")
    x0,y0,x1,y1=model_box; axes[2].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="deepskyblue",linewidth=2))
    fig.suptitle(f"Input/coordinate inspection | original box={box.tolist()}"); fig.savefig(path, dpi=220); plt.close(fig)


def save_gif(raw: np.ndarray, truth: np.ndarray, pred: np.ndarray, rows: list[dict[str, object]], path: Path, logger: logging.Logger) -> str:
    try:
        import imageio.v2 as imageio
        frames = []
        for z in range(raw.shape[0]):
            fig, ax = plt.subplots(figsize=(5, 5)); show_base(ax, raw[z], f"z={z} | Dice={rows[z]['dice']:.3f}")
            contour(ax, truth[z], "lime", 1.5); contour(ax, pred[z], "magenta", 1.5); fig.tight_layout()
            fig.canvas.draw(); frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); frames.append(frame); plt.close(fig)
        imageio.mimsave(path, frames, duration=0.12, loop=0)
        return "saved"
    except (ImportError, MemoryError, OSError) as exc:
        logger.warning("Skipping optional GIF: %s", exc)
        return f"skipped: {type(exc).__name__}: {exc}"


def enclosed_hole_area(prediction: np.ndarray, truth: np.ndarray) -> int:
    """Count false-negative voxels inside 2D holes enclosed by each predicted mask.

    This is measurement only. The filled masks are never used as predictions or
    fed back into MedSAM2, so the experiment applies no post-processing.
    """
    total = 0
    for pred_slice, truth_slice in zip(prediction, truth):
        holes = binary_fill_holes(pred_slice) & ~pred_slice
        total += int(np.count_nonzero(holes & truth_slice))
    return total


def save_normalization_diagnostic(
    raw: np.ndarray, percentile_result: NormalizationResult,
    whitestripe_result: NormalizationResult, init_z: int, path: Path,
) -> None:
    """Save side-by-side images and histograms for normalization auditability."""
    foreground = finite_foreground(raw)
    raw_values = raw[foreground].astype(np.float32)
    raw_low, raw_high = np.percentile(raw_values, [0.5, 99.5])
    display_raw = scale_foreground_to_uint8(raw.astype(np.float32), foreground, float(raw_low), float(raw_high))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    show_base(axes[0, 0], display_raw[init_z], f"Original MRI z={init_z}\nrobust display only")
    show_base(axes[0, 1], percentile_result.image_uint8[init_z], "Percentile-normalized input")
    show_base(axes[0, 2], whitestripe_result.image_uint8[init_z], "Hybrid WhiteStripe input")
    axes[1, 0].hist(raw_values, bins=256, range=(raw_low, raw_high), color="0.45")
    ws = whitestripe_result.diagnostics
    bounds = ws.get("selected_stripe_intensity_bounds")
    if isinstance(bounds, list) and len(bounds) == 2:
        axes[1, 0].axvspan(bounds[0], bounds[1], color="tab:orange", alpha=.35, label="Detected stripe")
        axes[1, 0].legend()
    axes[1, 0].set(title="3D foreground histogram", xlabel="Original intensity", ylabel="Voxel count")
    axes[1, 1].hist(percentile_result.image_uint8[foreground], bins=256, range=(0, 255), color="steelblue")
    axes[1, 1].set(title="Final percentile intensity histogram", xlabel="uint8 intensity", ylabel="Voxel count")
    axes[1, 2].hist(whitestripe_result.image_uint8[foreground], bins=256, range=(0, 255), color="darkorange")
    applied = whitestripe_result.method_applied
    axes[1, 2].set(title=f"Final WhiteStripe histogram\napplied={applied}", xlabel="uint8 intensity", ylabel="Voxel count")
    fig.suptitle("Controlled normalization diagnostic (same volume, slice, and orientation)", fontsize=15)
    fig.savefig(path, dpi=220); plt.close(fig)


def load_slice_dice(path: Path) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["slice_index"]): float(row["dice"]) for row in csv.DictReader(handle)}


def save_normalization_comparison(
    subject_dir: Path, raw: np.ndarray, truth: np.ndarray, init_z: int
) -> bool:
    """Generate controlled comparison artifacts once both condition outputs exist."""
    percentile_dir = subject_dir / "percentile"
    whitestripe_dir = subject_dir / "hybrid_whitestripe"
    required = [
        percentile_dir / "metrics.json", whitestripe_dir / "metrics.json",
        percentile_dir / "per_slice_metrics.csv", whitestripe_dir / "per_slice_metrics.csv",
        percentile_dir / "prediction.nii.gz", whitestripe_dir / "prediction.nii.gz",
    ]
    if not all(path.is_file() for path in required):
        return False
    p_metrics = json.loads((percentile_dir / "metrics.json").read_text(encoding="utf-8"))
    w_metrics = json.loads((whitestripe_dir / "metrics.json").read_text(encoding="utf-8"))
    if p_metrics["oracle_box_xyxy_original"] != w_metrics["oracle_box_xyxy_original"] or p_metrics["initialization_slice"] != w_metrics["initialization_slice"]:
        raise AssertionError("Normalization comparison is invalid: prompts differ between conditions.")
    p_pred = sitk.GetArrayFromImage(sitk.ReadImage(str(percentile_dir / "prediction.nii.gz"))) > 0
    w_pred = sitk.GetArrayFromImage(sitk.ReadImage(str(whitestripe_dir / "prediction.nii.gz"))) > 0
    p_dice, w_dice = load_slice_dice(required[2]), load_slice_dice(required[3])
    slices = sorted(p_dice)
    if slices != sorted(w_dice):
        raise AssertionError("Per-slice indices differ between normalization conditions.")

    summary_fields = (
        "volumetric_dice", "volumetric_iou", "predicted_foreground_voxels",
        "false_positive_voxels", "false_negative_voxels", "internal_enclosed_hole_area",
        "runtime_seconds", "peak_cuda_memory_mb",
    )
    comparison_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"row_type": "summary", "slice_index": ""}
    for field in summary_fields:
        summary[f"percentile_{field}"] = p_metrics[field]
        summary[f"hybrid_whitestripe_{field}"] = w_metrics[field]
    summary["percentile_mean_nonempty_slice_dice"] = p_metrics["nonempty_slice_dice"]["mean"]
    summary["hybrid_whitestripe_mean_nonempty_slice_dice"] = w_metrics["nonempty_slice_dice"]["mean"]
    summary["percentile_minimum_nonempty_slice_dice"] = p_metrics["nonempty_slice_dice"]["minimum"]
    summary["hybrid_whitestripe_minimum_nonempty_slice_dice"] = w_metrics["nonempty_slice_dice"]["minimum"]
    summary["whitestripe_fallback_used"] = w_metrics["normalization"]["fallback_used"]
    summary["whitestripe_fallback_reason"] = w_metrics["normalization"]["fallback_reason"]
    comparison_rows.append(summary)
    for z in slices:
        comparison_rows.append({
            "row_type": "slice", "slice_index": z, "percentile_slice_dice": p_dice[z],
            "hybrid_whitestripe_slice_dice": w_dice[z], "dice_change_whitestripe_minus_percentile": w_dice[z] - p_dice[z],
        })
    fieldnames = list(dict.fromkeys(key for row in comparison_rows for key in row))
    with (subject_dir / "normalization_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(comparison_rows)

    nonempty = [z for z in slices if truth[z].any()]
    p_worst = min(nonempty, key=lambda z: p_dice[z]); w_worst = min(nonempty, key=lambda z: w_dice[z])
    improvement = max(nonempty, key=lambda z: w_dice[z] - p_dice[z])
    regression = min(nonempty, key=lambda z: w_dice[z] - p_dice[z])
    # Keep all five requested analytical roles even when ties point to the same
    # slice. Repeated rows make the tie explicit instead of silently omitting a
    # requested category from the comparison figure.
    selected = [
        ("Initialization", init_z), ("Baseline worst", p_worst),
        ("WhiteStripe worst", w_worst), ("Largest improvement", improvement),
        ("Largest regression", regression),
    ]
    fig, axes = plt.subplots(len(selected), 5, figsize=(18, 3.5 * len(selected)), squeeze=False, constrained_layout=True)
    for row_axes, (label, z) in zip(axes, selected):
        titles = ("Expert mask", "Percentile prediction", "WhiteStripe prediction", "Percentile errors", "WhiteStripe errors")
        for ax, title in zip(row_axes, titles): show_base(ax, raw[z], title)
        overlay(row_axes[0], truth[z], (0.1, .9, .2)); overlay(row_axes[1], p_pred[z], (1, .8, 0)); overlay(row_axes[2], w_pred[z], (1, .8, 0))
        overlay(row_axes[3], p_pred[z] & ~truth[z], (1, .1, .1)); overlay(row_axes[3], truth[z] & ~p_pred[z], (.1, .4, 1))
        overlay(row_axes[4], w_pred[z] & ~truth[z], (1, .1, .1)); overlay(row_axes[4], truth[z] & ~w_pred[z], (.1, .4, 1))
        row_axes[0].set_title(f"{label} | z={z} | P={p_dice[z]:.3f} W={w_dice[z]:.3f}\nExpert mask", fontsize=9)
    fig.suptitle("Percentile vs hybrid WhiteStripe (red FP, blue FN)")
    fig.savefig(subject_dir / "normalization_comparison.png", dpi=220); plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    case = locate_case(args.image_root, args.mask_root, args.subject)
    subject_dir = args.output_root / case.subject_id
    output_dir = subject_dir / args.normalization
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    logger.info("Selected %s | image=%s | mask=%s", case.subject_id, case.image_path, case.mask_path)

    image_itk, mask_itk = sitk.ReadImage(str(case.image_path)), sitk.ReadImage(str(case.mask_path))
    validate_geometry(image_itk, mask_itk)
    raw = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
    truth = sitk.GetArrayFromImage(mask_itk) > 0
    if raw.shape != truth.shape or not truth.any():
        raise ValueError(f"Invalid arrays: image={raw.shape}, mask={truth.shape}, mask_nonempty={truth.any()}")
    normalization = normalize_mri(raw, args.normalization, args.clip_lower_percentile, args.clip_upper_percentile)
    preprocessed = normalization.image_uint8
    init_z = int(np.argmax(truth.sum(axis=(1, 2))))
    box = tight_box(truth[init_z]); x0,y0,x1,y1 = box.astype(int)
    if not truth[init_z, y0:y1+1, x0:x1+1].any():
        raise AssertionError("Selected oracle box does not contain ground-truth foreground.")
    resized = resize_for_model(preprocessed)
    model_box = transformed_box(box, raw.shape[1], raw.shape[2])
    save_input_debug(preprocessed, resized, init_z, box, model_box, output_dir / "input_debug.png")
    # The diagnostic is computed from both mappings but does not alter the chosen
    # condition's inference input. This makes contrast changes directly auditable.
    percentile_diagnostic = percentile_normalize_mri(raw, args.clip_lower_percentile, args.clip_upper_percentile)
    whitestripe_diagnostic = hybrid_whitestripe_normalize_mri(raw, args.clip_lower_percentile, args.clip_upper_percentile)
    save_normalization_diagnostic(raw, percentile_diagnostic, whitestripe_diagnostic, init_z, subject_dir / "normalization_diagnostic.png")

    from sam2.build_sam import build_sam2_video_predictor_npz
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    try:
        predictor = build_sam2_video_predictor_npz(args.config, str(args.checkpoint), device="cuda")
        model_input = normalize_for_model(resized)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = infer_bidirectionally(predictor, model_input, raw.shape[1:], init_z, box, args.offload_state_to_cpu)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("Insufficient CUDA memory. No CPU fallback was attempted; try --offload-state-to-cpu.") from exc
    runtime = time.perf_counter() - start
    peak_cuda_mb = torch.cuda.max_memory_allocated() / 1024**2
    if prediction.shape != raw.shape:
        raise AssertionError(f"Reconstructed shape {prediction.shape} != original {raw.shape}")

    pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8)); pred_itk.CopyInformation(image_itk)
    prediction_path = output_dir / "prediction.nii.gz"; sitk.WriteImage(pred_itk, str(prediction_path))
    saved = sitk.ReadImage(str(prediction_path)); validate_geometry(image_itk, saved)
    rows = per_slice_rows(prediction, truth, init_z); save_csv(rows, output_dir / "per_slice_metrics.csv")
    volume_dice, volume_iou = dice_iou(prediction, truth)
    false_positive_voxels = int(np.count_nonzero(prediction & ~truth))
    false_negative_voxels = int(np.count_nonzero(truth & ~prediction))
    internal_holes = enclosed_hole_area(prediction, truth)
    nonempty = [r for r in rows if r["ground_truth_nonempty"]]
    nonempty_dice = np.array([r["dice"] for r in nonempty], dtype=float)
    metrics = {
        "subject_id": case.subject_id, "image_path": str(case.image_path), "mask_path": str(case.mask_path),
        "checkpoint": str(args.checkpoint), "config": args.config, "array_shape_zyx": list(raw.shape),
        "image_geometry": geometry_signature(image_itk), "initialization_slice": init_z,
        "oracle_box_xyxy_original": box.tolist(), "oracle_box_xyxy_model_512": model_box.tolist(),
        "normalization_requested": args.normalization,
        "normalization_applied": normalization.method_applied,
        "normalization": normalization.diagnostics,
        "volumetric_dice": volume_dice, "volumetric_iou": volume_iou,
        "nonempty_slice_dice": {"mean": float(nonempty_dice.mean()), "median": float(np.median(nonempty_dice)), "minimum": float(nonempty_dice.min()), "maximum": float(nonempty_dice.max())},
        "empty_predictions_on_nonempty_ground_truth_slices": sum(not r["prediction_nonempty"] for r in nonempty),
        "false_positive_predictions_on_empty_ground_truth_slices": sum(r["prediction_nonempty"] and not r["ground_truth_nonempty"] for r in rows),
        "predicted_foreground_voxels": int(prediction.sum()), "ground_truth_foreground_voxels": int(truth.sum()),
        "false_positive_voxels": false_positive_voxels, "false_negative_voxels": false_negative_voxels,
        "internal_enclosed_hole_area": internal_holes,
        "runtime_seconds": runtime, "peak_cuda_memory_mb": peak_cuda_mb, "device": torch.cuda.get_device_name(0),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_initialization_qc(raw, truth, prediction, init_z, box, float(rows[init_z]["dice"]), output_dir / "initialization_qc.png")
    save_full_montage(raw, truth, prediction, rows, output_dir / "full_volume_montage.png", args.max_montage_slices)
    ranked = sorted((int(r["slice_index"]) for r in nonempty), key=lambda z: float(rows[z]["dice"]))
    save_ranked_montage(raw, truth, prediction, rows, ranked[:min(10,len(ranked))], "Lowest-Dice non-empty slices (worst to better)", output_dir / "worst_slices.png")
    # Uniformly sample six from the upper Dice quartile to avoid six nearly identical adjacent slices.
    high = ranked[max(0, math.floor(.75*len(ranked))):]
    best = [high[i] for i in np.unique(np.rint(np.linspace(0, len(high)-1, min(6,len(high)))).astype(int))]
    best.sort(key=lambda z: float(rows[z]["dice"]), reverse=True)
    save_ranked_montage(raw, truth, prediction, rows, best, "Representative high-Dice non-empty slices", output_dir / "best_slices.png")
    save_profiles(rows, init_z, output_dir / "dice_by_slice.png", output_dir / "foreground_area_by_slice.png")
    gif_status = "skipped by --skip-gif" if args.skip_gif else save_gif(raw, truth, prediction, rows, output_dir / "propagation.gif", logger)
    metrics["gif"] = gif_status; (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparison_saved = save_normalization_comparison(subject_dir, raw, truth, init_z)
    logger.info("shape_zyx=%s spacing_xyz=%s init_z=%d box_xyxy=%s", raw.shape, image_itk.GetSpacing(), init_z, box.tolist())
    logger.info("normalization=%s applied=%s fallback=%s", args.normalization, normalization.method_applied, normalization.diagnostics["fallback_used"])
    logger.info("Dice=%.6f IoU=%.6f runtime=%.2fs peak_cuda=%.1f MiB GIF=%s comparison=%s", volume_dice, volume_iou, runtime, peak_cuda_mb, gif_status, comparison_saved)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Smoke test failed: %s", exc)
        raise
