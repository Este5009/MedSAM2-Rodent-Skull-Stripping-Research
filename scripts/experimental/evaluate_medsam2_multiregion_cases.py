#!/usr/bin/env python
"""Evaluate MedSAM2 on pre-identified difficult CAMRI multi-region slices.

The experiment keeps the validated single-volume MedSAM2 pipeline unchanged:
whole-volume percentile normalization, one tight oracle box on the largest-area
expert slice, and bidirectional propagation. Only the evaluated subjects/slices
and topology-focused reporting are new.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.ndimage import binary_fill_holes, label
import SimpleITK as sitk
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL_DIR))

# Reuse the already validated project-owned loader, normalization, predictor,
# metrics, coordinate, and full-volume visualization functions.
from run_medsam2_camri_single import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MASK_ROOT,
    dice_iou,
    infer_bidirectionally,
    normalize_for_model,
    per_slice_rows,
    percentile_normalize_mri,
    resize_for_model,
    save_full_montage,
    tight_box,
    transformed_box,
    validate_geometry,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "multiregion_cases"
DEFAULT_PREVIOUS_ROOT = Path(r"C:\Users\esteb\Projects\Imaging\Rodent-Skull-Stripping-Research")
CASE_TARGETS: dict[str, list[int]] = {
    "sub-086": [6, 53, 55],
    "sub-112": [4, 54, 56],
    "sub-066": [6, 57, 59],
    "sub-050": [6, 53, 54],
    "sub-109": [6, 54, 56],
}
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class SubjectPaths:
    subject_id: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--previous-project-root", type=Path, default=DEFAULT_PREVIOUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="*", choices=sorted(CASE_TARGETS), default=sorted(CASE_TARGETS))
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    return parser.parse_args()


def locate_subject_paths(subject_id: str, image_root: Path, mask_root: Path) -> SubjectPaths:
    image_candidates = sorted(image_root.glob(f"{subject_id}/ses-1/anat/*RARE_T2w.nii*"))
    mask_candidates = sorted(mask_root.glob(f"CAMRI_Rat-{subject_id}_*RARE_T2w*.nii*"))
    if len(image_candidates) != 1 or len(mask_candidates) != 1:
        raise FileNotFoundError(
            f"Expected one image and mask for {subject_id}; found "
            f"images={len(image_candidates)}, masks={len(mask_candidates)}"
        )
    return SubjectPaths(subject_id, image_candidates[0], mask_candidates[0])


def read_csv_index(path: Path, key_fields: tuple[str, str]) -> dict[tuple[str, int], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    subject_field, slice_field = key_fields
    return {(row[subject_field], int(row[slice_field])): row for row in rows}


def load_previous_results(previous_root: Path) -> dict[str, dict[tuple[str, int], dict[str, str]]]:
    failure = read_csv_index(
        previous_root / "outputs/failure_analysis/latest/tables/per_slice_metrics.csv",
        ("subject", "slice_index"),
    )
    multibox = read_csv_index(
        previous_root / "outputs/multibox_oracle/per_slice_comparison.csv",
        ("subject_id", "slice_index"),
    )
    combined = read_csv_index(
        previous_root / "outputs/mask_prompt_oracle/tables/per_slice_comparison.csv",
        ("subject_id", "slice_index"),
    )
    return {"failure": failure, "multibox": multibox, "combined": combined}


def optional_float(row: dict[str, str] | None, field: str) -> float | None:
    if row is None or field not in row or row[field] in ("", "nan", "NaN"):
        return None
    return float(row[field])


def component_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """Measure topology without altering the raw prediction."""
    gt_labels, gt_count = label(truth, structure=CONNECTIVITY_8)
    pred_labels, pred_count = label(pred, structure=CONNECTIVITY_8)
    missed = sum(not np.any(pred[gt_labels == component]) for component in range(1, gt_count + 1))
    spurious = sum(not np.any(truth[pred_labels == component]) for component in range(1, pred_count + 1))
    holes = binary_fill_holes(pred) & ~pred
    return {
        "expert_component_count": int(gt_count),
        "predicted_component_count": int(pred_count),
        "expert_components_missed_entirely": int(missed),
        "spurious_predicted_components": int(spurious),
        "enclosed_hole_area": int(holes.sum()),
        "enclosed_false_negative_area": int(np.count_nonzero(holes & truth)),
    }


def target_metrics(
    subject_id: str, z: int, init_z: int, pred: np.ndarray, truth: np.ndarray,
    previous: dict[str, dict[tuple[str, int], dict[str, str]]],
) -> dict[str, object]:
    intersection = int(np.count_nonzero(pred & truth))
    pred_area, gt_area = int(pred.sum()), int(truth.sum())
    dice, iou = dice_iou(pred, truth)
    precision = intersection / pred_area if pred_area else (1.0 if gt_area == 0 else 0.0)
    recall = intersection / gt_area if gt_area else (1.0 if pred_area == 0 else 0.0)
    topology = component_metrics(pred, truth)
    key = (subject_id, z)
    failure_row = previous["failure"].get(key)
    multibox_row = previous["multibox"].get(key)
    combined_row = previous["combined"].get(key)
    return {
        "subject_id": subject_id,
        "slice_index": z,
        "propagation_side": "initialization" if z == init_z else ("backward" if z < init_z else "forward"),
        "distance_from_initialization": abs(z - init_z),
        "medsam2_dice": dice,
        "medsam2_iou": iou,
        "medsam2_precision": precision,
        "medsam2_recall": recall,
        "false_positive_pixels": int(np.count_nonzero(pred & ~truth)),
        "false_negative_pixels": int(np.count_nonzero(truth & ~pred)),
        "expert_foreground_area": gt_area,
        "predicted_foreground_area": pred_area,
        **topology,
        "original_medsam_single_box_dice": optional_float(failure_row, "dice"),
        "original_medsam_raw_multibox_dice": optional_float(multibox_row, "multibox_dice"),
        "original_medsam_combined_prompt_dice": optional_float(combined_row, "box_mask_dice"),
        "single_box_comparison_available": failure_row is not None,
        "raw_multibox_comparison_available": multibox_row is not None,
        "combined_prompt_comparison_available": combined_row is not None,
    }


def show_base(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def overlay(ax: plt.Axes, mask: np.ndarray, color: tuple[float, float, float], alpha: float = .35) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask, :3] = color
    rgba[mask, 3] = alpha
    ax.imshow(rgba, origin="upper")


def contour(ax: plt.Axes, mask: np.ndarray, color: str) -> None:
    if mask.any():
        ax.contour(mask.astype(float), levels=[.5], colors=[color], linewidths=1.2)


def save_target_figure(
    path: Path, image: np.ndarray, truth: np.ndarray, pred: np.ndarray, row: dict[str, object]
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for ax, title in zip(axes.flat, (
        "MRI", "Expert mask", "Raw MedSAM2 prediction", "Expert vs prediction contours",
        "Errors (red FP, blue FN)", "Connected-component labels",
    )):
        show_base(ax, image, title)
    overlay(axes[0, 1], truth, (.1, .9, .2)); overlay(axes[0, 2], pred, (1, .8, 0))
    contour(axes[1, 0], truth, "lime"); contour(axes[1, 0], pred, "magenta")
    overlay(axes[1, 1], pred & ~truth, (1, .1, .1)); overlay(axes[1, 1], truth & ~pred, (.1, .4, 1))
    gt_labels, _ = label(truth, structure=CONNECTIVITY_8); pred_labels, _ = label(pred, structure=CONNECTIVITY_8)
    combined = np.zeros_like(gt_labels, dtype=np.int32); combined[gt_labels > 0] = gt_labels[gt_labels > 0]
    offset = int(gt_labels.max()) + 1; combined[pred_labels > 0] = pred_labels[pred_labels > 0] + offset
    axes[1, 2].imshow(np.ma.masked_where(combined == 0, combined), cmap=ListedColormap(plt.cm.tab20.colors), alpha=.65, origin="upper")
    previous_text = (
        f"original single={row['original_medsam_single_box_dice'] if row['original_medsam_single_box_dice'] is not None else 'unavailable'} | "
        f"raw multi={row['original_medsam_raw_multibox_dice'] if row['original_medsam_raw_multibox_dice'] is not None else 'unavailable'} | "
        f"combined={row['original_medsam_combined_prompt_dice'] if row['original_medsam_combined_prompt_dice'] is not None else 'unavailable'}"
    )
    fig.suptitle(
        f"{row['subject_id']} z={row['slice_index']} | MedSAM2 Dice={row['medsam2_dice']:.4f} | "
        f"GT/pred components={row['expert_component_count']}/{row['predicted_component_count']} | "
        f"missed={row['expert_components_missed_entirely']} | distance={row['distance_from_initialization']}\n{previous_text}",
        fontsize=12,
    )
    fig.savefig(path, dpi=220); plt.close(fig)


def save_summary_montage(
    output_path: Path, cases: list[dict[str, object]], images: dict[str, np.ndarray],
    truths: dict[str, np.ndarray], predictions: dict[str, np.ndarray],
) -> None:
    ordered = sorted(cases, key=lambda row: float(row["medsam2_dice"]))
    cols = 4; nrows = math.ceil(len(ordered) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(16, 4 * nrows), squeeze=False, constrained_layout=True)
    for ax in axes.flat: ax.set_axis_off()
    for ax, row in zip(axes.flat, ordered):
        subject, z = str(row["subject_id"]), int(row["slice_index"])
        show_base(ax, images[subject][z], f"{subject} z={z} | Dice={row['medsam2_dice']:.3f}\nGT/pred CC={row['expert_component_count']}/{row['predicted_component_count']}")
        contour(ax, truths[subject][z], "lime"); contour(ax, predictions[subject][z], "magenta")
    fig.suptitle("Difficult target slices: MedSAM2 Dice worst to best")
    fig.savefig(output_path, dpi=200); plt.close(fig)


def save_method_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [f"{r['subject_id']}:{r['slice_index']}" for r in rows]
    x = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(15, 6))
    series = (
        ("Original single-box", "original_medsam_single_box_dice", "o"),
        ("Original raw multi-box", "original_medsam_raw_multibox_dice", "s"),
        ("Original combined prompt", "original_medsam_combined_prompt_dice", "^"),
        ("MedSAM2", "medsam2_dice", "D"),
    )
    for name, field, marker in series:
        values = np.array([np.nan if r[field] is None else float(r[field]) for r in rows])
        ax.plot(x, values, marker=marker, linewidth=1, label=name)
    ax.set_xticks(x, labels, rotation=60, ha="right"); ax.set(ylabel="Dice", ylim=(-.02, 1.02), title="Target-slice method comparison")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def save_component_recovery(path: Path, rows: list[dict[str, object]]) -> None:
    gt = np.array([r["expert_component_count"] for r in rows]); pred = np.array([r["predicted_component_count"] for r in rows])
    recovered = np.array([r["expert_components_missed_entirely"] == 0 for r in rows])
    fig, ax = plt.subplots(figsize=(7, 6)); limit = max(int(gt.max()), int(pred.max())) + 1
    ax.plot([0, limit], [0, limit], "--", color="0.5", label="Equal count")
    for state, color, name in ((True, "green", "All expert regions overlapped"), (False, "red", "One or more expert regions missed")):
        ax.scatter(gt[recovered == state], pred[recovered == state], color=color, s=70, label=name)
    ax.set(xlabel="Expert connected components", ylabel="Predicted connected components", xlim=(0, limit), ylim=(0, limit), title="Component recovery")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def save_distance_plot(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for side, color, marker in (("backward", "royalblue", "o"), ("forward", "darkorange", "s"), ("initialization", "crimson", "D")):
        selected = [r for r in rows if r["propagation_side"] == side]
        if selected:
            ax.scatter([r["distance_from_initialization"] for r in selected], [r["medsam2_dice"] for r in selected], color=color, marker=marker, s=70, label=side)
    ax.set(xlabel="Absolute distance from initialization slice", ylabel="MedSAM2 Dice", ylim=(-.02, 1.02), title="Difficult-slice Dice versus propagation distance")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = args.output_dir / "target_figures"; target_dir.mkdir(exist_ok=True)
    montage_dir = args.output_dir / "subject_montages"; montage_dir.mkdir(exist_ok=True)
    prediction_dir = args.output_dir / "predictions"; prediction_dir.mkdir(exist_ok=True)
    previous = load_previous_results(args.previous_project_root)

    from sam2.build_sam import build_sam2_video_predictor_npz
    predictor = build_sam2_video_predictor_npz(args.config, str(args.checkpoint), device="cuda")
    all_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    images: dict[str, np.ndarray] = {}; truths: dict[str, np.ndarray] = {}; predictions: dict[str, np.ndarray] = {}

    for subject_id in args.subjects:
        paths = locate_subject_paths(subject_id, args.image_root, args.mask_root)
        image_itk, mask_itk = sitk.ReadImage(str(paths.image_path)), sitk.ReadImage(str(paths.mask_path))
        validate_geometry(image_itk, mask_itk)
        raw = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
        truth = sitk.GetArrayFromImage(mask_itk) > 0
        if raw.shape != truth.shape or not truth.any():
            raise ValueError(f"Invalid image/mask arrays for {subject_id}: {raw.shape}/{truth.shape}")
        normalization = percentile_normalize_mri(raw, .5, 99.5)
        init_z = int(np.argmax(truth.sum(axis=(1, 2))))
        box = tight_box(truth[init_z]); model_box = transformed_box(box, raw.shape[1], raw.shape[2])
        resized = resize_for_model(normalization.image_uint8)
        model_input = normalize_for_model(resized)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = infer_bidirectionally(predictor, model_input, raw.shape[1:], init_z, box, args.offload_state_to_cpu)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(f"Insufficient CUDA memory while processing {subject_id}; no CPU fallback attempted.") from exc
        runtime = time.perf_counter() - start; peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        if prediction.shape != raw.shape:
            raise AssertionError(f"Prediction shape mismatch for {subject_id}: {prediction.shape} vs {raw.shape}")
        pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8)); pred_itk.CopyInformation(image_itk)
        sitk.WriteImage(pred_itk, str(prediction_dir / f"{subject_id}_prediction.nii.gz"))
        slice_rows = per_slice_rows(prediction, truth, init_z)
        volume_dice, volume_iou = dice_iou(prediction, truth)
        nonempty_dice = np.array([r["dice"] for r in slice_rows if r["ground_truth_nonempty"]], dtype=float)
        subject_rows.append({
            "subject_id": subject_id, "image_path": str(paths.image_path), "mask_path": str(paths.mask_path),
            "initialization_slice": init_z, "oracle_box_xyxy": json.dumps(box.tolist()), "model_box_xyxy_512": json.dumps(model_box.tolist()),
            "volumetric_dice": volume_dice, "volumetric_iou": volume_iou,
            "mean_nonempty_slice_dice": float(nonempty_dice.mean()), "minimum_nonempty_slice_dice": float(nonempty_dice.min()),
            "empty_predictions_on_nonempty_slices": sum(r["ground_truth_nonempty"] and not r["prediction_nonempty"] for r in slice_rows),
            "false_positive_predictions_on_empty_slices": sum(not r["ground_truth_nonempty"] and r["prediction_nonempty"] for r in slice_rows),
            "runtime_seconds": runtime, "peak_cuda_memory_mb": peak_mb,
        })
        for z in CASE_TARGETS[subject_id]:
            if not 0 <= z < raw.shape[0]:
                raise IndexError(f"Target slice {subject_id}:{z} outside volume depth {raw.shape[0]}")
            row = target_metrics(subject_id, z, init_z, prediction[z], truth[z], previous)
            all_rows.append(row)
            save_target_figure(target_dir / f"{subject_id}_slice-{z:03d}.png", raw[z], truth[z], prediction[z], row)
        save_full_montage(raw, truth, prediction, slice_rows, montage_dir / f"{subject_id}_full_volume.png", maximum=64)
        images[subject_id], truths[subject_id], predictions[subject_id] = raw, truth, prediction
        del model_input, resized; torch.cuda.empty_cache()
        print(f"{subject_id}: init={init_z}, box={box.tolist()}, volume Dice={volume_dice:.4f}, runtime={runtime:.2f}s")

    write_csv(args.output_dir / "comparison.csv", all_rows)
    write_csv(args.output_dir / "per_subject_metrics.csv", subject_rows)
    save_summary_montage(args.output_dir / "summary_montage.png", all_rows, images, truths, predictions)
    save_method_comparison(args.output_dir / "method_comparison.png", all_rows)
    save_component_recovery(args.output_dir / "component_recovery.png", all_rows)
    save_distance_plot(args.output_dir / "dice_vs_distance.png", all_rows)
    manifest = {
        "checkpoint": str(args.checkpoint), "config": args.config, "normalization": "percentile_0.5_99.5",
        "initialization_strategy": "largest-area non-empty expert slice; one tight oracle box per volume",
        "subjects_and_target_slices": {subject: CASE_TARGETS[subject] for subject in args.subjects},
        "previous_project_root": str(args.previous_project_root), "output_dir": str(args.output_dir),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved focused comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
