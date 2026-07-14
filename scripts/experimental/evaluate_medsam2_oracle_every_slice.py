#!/usr/bin/env python
"""Compare independent and shared-state MedSAM2 oracle boxes on every slice.

The shared-state condition follows the official video-predictor interaction:
all boxes are added before tracking, making them conditioning frames, and the
volume is then propagated without resetting the state.  Consequently, prompted
slice outputs are intentionally reported separately from unprompted empty-slice
outputs: the official predictor reuses conditioning-frame outputs during
propagation, so memory cannot refine those prompted masks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from scipy.stats import wilcoxon
import SimpleITK as sitk
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL_DIR))

from run_medsam2_camri_single import (  # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_IMAGE_ROOT, DEFAULT_MASK_ROOT,
    dice_iou, normalize_for_model, percentile_normalize_mri, resize_for_model,
    tight_box, validate_geometry,
)

SUBJECTS = ("sub-050", "sub-066", "sub-086", "sub-109", "sub-112")
OUTPUT_DIR = REPO_ROOT / "outputs" / "oracle_every_slice"
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


@dataclass
class Subject:
    subject_id: str
    image_path: Path
    mask_path: Path
    image_itk: sitk.Image
    raw: np.ndarray
    image_u8: np.ndarray
    truth: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def locate(subject_id: str, image_root: Path, mask_root: Path) -> tuple[Path, Path]:
    images = sorted(image_root.glob(f"{subject_id}/ses-1/anat/*RARE_T2w.nii*"))
    masks = sorted(mask_root.glob(f"CAMRI_Rat-{subject_id}_*RARE_T2w*.nii*"))
    if len(images) != 1 or len(masks) != 1:
        raise FileNotFoundError(f"{subject_id}: expected one image/mask, found {len(images)}/{len(masks)}")
    return images[0], masks[0]


def load_subject(subject_id: str, image_root: Path, mask_root: Path) -> Subject:
    image_path, mask_path = locate(subject_id, image_root, mask_root)
    image_itk, mask_itk = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(mask_path))
    validate_geometry(image_itk, mask_itk)
    raw = sitk.GetArrayFromImage(image_itk).astype(np.float32, copy=False)
    truth = sitk.GetArrayFromImage(mask_itk) > 0
    if raw.shape != truth.shape or not truth.any():
        raise ValueError(f"{subject_id}: invalid aligned arrays {raw.shape}/{truth.shape}")
    normalized = percentile_normalize_mri(raw, .5, 99.5)
    return Subject(subject_id, image_path, mask_path, image_itk, raw, normalized.image_uint8, truth)


def independent_inference(predictor: object, subject: Subject) -> tuple[np.ndarray, float, float]:
    """Run every non-empty slice as a separate image; set_image resets features."""
    prediction = np.zeros_like(subject.truth)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for z in np.flatnonzero(subject.truth.any(axis=(1, 2))):
            rgb = np.repeat(subject.image_u8[z, :, :, None], 3, axis=2)
            predictor.set_image(rgb)
            masks, _, _ = predictor.predict(box=tight_box(subject.truth[z]), multimask_output=False)
            prediction[z] = masks[0]
    torch.cuda.synchronize()
    return prediction, time.perf_counter() - start, torch.cuda.max_memory_allocated() / 1024**2


def shared_state_inference(predictor: object, subject: Subject, offload: bool) -> tuple[np.ndarray, float, float]:
    """Add all oracle boxes before tracking, then propagate in one official state."""
    prediction = np.zeros_like(subject.truth)
    images = normalize_for_model(resize_for_model(subject.image_u8))
    nonempty = np.flatnonzero(subject.truth.any(axis=(1, 2)))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    state = predictor.init_state(images, *subject.truth.shape[1:], offload_state_to_cpu=offload)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for z in nonempty:
            _, _, logits = predictor.add_new_points_or_box(
                state, frame_idx=int(z), obj_id=1, box=tight_box(subject.truth[z]))
            prediction[z] = (logits[0] > 0).cpu().numpy()[0]
        start_z = int(nonempty[0])
        for z, _, logits in predictor.propagate_in_video(state, start_frame_idx=start_z, reverse=False):
            prediction[z] = (logits[0] > 0).cpu().numpy()[0]
        for z, _, logits in predictor.propagate_in_video(state, start_frame_idx=start_z, reverse=True):
            prediction[z] = (logits[0] > 0).cpu().numpy()[0]
    torch.cuda.synchronize(); runtime = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**2
    predictor.reset_state(state)
    del images, state
    return prediction, runtime, peak


def components(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    gt_labels, gt_n = label(truth, structure=CONNECTIVITY)
    pred_labels, pred_n = label(pred, structure=CONNECTIVITY)
    recovered = sum(np.any(pred[gt_labels == i]) for i in range(1, gt_n + 1))
    spurious = sum(not np.any(truth[pred_labels == i]) for i in range(1, pred_n + 1))
    return int(gt_n), int(recovered), int(pred_n), int(spurious)


def slice_row(subject: Subject, z: int, a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    truth = subject.truth[z]; ad, ai = dice_iou(a, truth); bd, bi = dice_iou(b, truth)
    ag, ar, ap, asp = components(a, truth); _, br, bp, bsp = components(b, truth)
    return {
        "subject_id": subject.subject_id, "slice_index": z, "ground_truth_nonempty": bool(truth.any()),
        "oracle_box_xyxy": json.dumps(tight_box(truth).tolist()) if truth.any() else "",
        "independent_dice": ad, "shared_state_dice": bd, "dice_delta_shared_minus_independent": bd-ad,
        "independent_iou": ai, "shared_state_iou": bi,
        "independent_false_positive_pixels": int(np.count_nonzero(a & ~truth)),
        "shared_state_false_positive_pixels": int(np.count_nonzero(b & ~truth)),
        "independent_false_negative_pixels": int(np.count_nonzero(truth & ~a)),
        "shared_state_false_negative_pixels": int(np.count_nonzero(truth & ~b)),
        "expert_components": ag, "independent_components_recovered": ar,
        "shared_state_components_recovered": br, "independent_predicted_components": ap,
        "shared_state_predicted_components": bp, "independent_spurious_components": asp,
        "shared_state_spurious_components": bsp,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_prediction(path: Path, prediction: np.ndarray, reference: sitk.Image) -> None:
    image = sitk.GetImageFromArray(prediction.astype(np.uint8)); image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def save_profile(path: Path, rows: list[dict[str, object]], subject_id: str) -> None:
    selected = [r for r in rows if r["subject_id"] == subject_id and r["ground_truth_nonempty"]]
    fig, ax = plt.subplots(figsize=(10, 4)); z = [r["slice_index"] for r in selected]
    ax.plot(z, [r["independent_dice"] for r in selected], label="Independent", marker=".")
    ax.plot(z, [r["shared_state_dice"] for r in selected], label="Shared state", marker=".")
    ax.set(xlabel="Slice", ylabel="Dice", ylim=(-.02, 1.02), title=f"{subject_id}: oracle box on every non-empty slice")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def save_summary(path: Path, subject_rows: list[dict[str, object]]) -> None:
    ids = [r["subject_id"] for r in subject_rows]; x = np.arange(len(ids)); width = .38
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, akey, bkey, title in (
        (axes[0], "independent_volume_dice", "shared_state_volume_dice", "Volume Dice"),
        (axes[1], "independent_mean_nonempty_slice_dice", "shared_state_mean_nonempty_slice_dice", "Mean non-empty-slice Dice")):
        ax.bar(x-width/2, [r[akey] for r in subject_rows], width, label="Independent")
        ax.bar(x+width/2, [r[bkey] for r in subject_rows], width, label="Shared state")
        ax.set_xticks(x, ids, rotation=30); ax.set_ylim(0, 1); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    axes[0].legend(); fig.savefig(path, dpi=220); plt.close(fig)


def statistics(rows: list[dict[str, object]], samples: int) -> dict[str, object]:
    prompted = [r for r in rows if r["ground_truth_nonempty"]]
    delta = np.asarray([r["dice_delta_shared_minus_independent"] for r in prompted], dtype=float)
    nonzero = delta[~np.isclose(delta, 0, atol=1e-12)]
    test = wilcoxon(nonzero) if nonzero.size else None
    rng = np.random.default_rng(2024)
    means = np.asarray([rng.choice(delta, delta.size, replace=True).mean() for _ in range(samples)])
    ordered = sorted(prompted, key=lambda r: r["dice_delta_shared_minus_independent"])
    return {
        "analysis_unit": "paired non-empty slices", "paired_slice_count": int(delta.size),
        "mean_dice_delta": float(delta.mean()), "median_dice_delta": float(np.median(delta)),
        "bootstrap_95_percent_ci_mean_delta": np.quantile(means, [.025, .975]).tolist(),
        "wins_ties_losses": {"wins": int((delta > 1e-12).sum()), "ties": int(np.isclose(delta, 0, atol=1e-12).sum()), "losses": int((delta < -1e-12).sum())},
        "wilcoxon_statistic": None if test is None else float(test.statistic),
        "wilcoxon_p_value": 1.0 if test is None else float(test.pvalue),
        "best_improved_slices": ordered[-5:][::-1], "worst_changed_slices": ordered[:5],
        "interpretation_caveat": "Prompted frames are conditioning frames reused during propagation; memory can only alter unprompted slices in this official interaction pattern.",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is disabled.")
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "predictions"; fig_dir = args.output_dir / "figures"
    pred_dir.mkdir(exist_ok=True); fig_dir.mkdir(exist_ok=True)
    subjects = [load_subject(s, args.image_root, args.mask_root) for s in args.subjects]

    from sam2.build_sam import build_sam2, build_sam2_video_predictor_npz
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    image_predictor = SAM2ImagePredictor(build_sam2(args.config, str(args.checkpoint), device="cuda"))
    results: dict[str, dict[str, object]] = {}
    for subject in subjects:
        pred, runtime, peak = independent_inference(image_predictor, subject)
        results[subject.subject_id] = {"a": pred, "a_runtime": runtime, "a_peak": peak}
        print(f"{subject.subject_id} independent: {runtime:.2f}s, {peak:.1f} MiB")
    del image_predictor; gc.collect(); torch.cuda.empty_cache()

    video_predictor = build_sam2_video_predictor_npz(args.config, str(args.checkpoint), device="cuda")
    for subject in subjects:
        pred, runtime, peak = shared_state_inference(video_predictor, subject, args.offload_state_to_cpu)
        results[subject.subject_id].update({"b": pred, "b_runtime": runtime, "b_peak": peak})
        print(f"{subject.subject_id} shared: {runtime:.2f}s, {peak:.1f} MiB")

    slice_rows: list[dict[str, object]] = []; subject_rows: list[dict[str, object]] = []
    for subject in subjects:
        result = results[subject.subject_id]; a = result["a"]; b = result["b"]
        rows = [slice_row(subject, z, a[z], b[z]) for z in range(subject.truth.shape[0])]
        slice_rows.extend(rows); nonempty = [r for r in rows if r["ground_truth_nonempty"]]
        ad, ai = dice_iou(a, subject.truth); bd, bi = dice_iou(b, subject.truth)
        subject_rows.append({
            "subject_id": subject.subject_id, "image_path": str(subject.image_path), "mask_path": str(subject.mask_path),
            "independent_volume_dice": ad, "shared_state_volume_dice": bd, "volume_dice_delta": bd-ad,
            "independent_volume_iou": ai, "shared_state_volume_iou": bi,
            "independent_mean_nonempty_slice_dice": float(np.mean([r["independent_dice"] for r in nonempty])),
            "shared_state_mean_nonempty_slice_dice": float(np.mean([r["shared_state_dice"] for r in nonempty])),
            "independent_runtime_seconds": result["a_runtime"], "shared_state_runtime_seconds": result["b_runtime"],
            "independent_peak_cuda_memory_mb": result["a_peak"], "shared_state_peak_cuda_memory_mb": result["b_peak"],
            "independent_false_positive_pixels": int(np.count_nonzero(a & ~subject.truth)),
            "shared_state_false_positive_pixels": int(np.count_nonzero(b & ~subject.truth)),
            "independent_false_negative_pixels": int(np.count_nonzero(subject.truth & ~a)),
            "shared_state_false_negative_pixels": int(np.count_nonzero(subject.truth & ~b)),
            "independent_empty_slice_false_positives": sum(not r["ground_truth_nonempty"] and r["independent_false_positive_pixels"] > 0 for r in rows),
            "shared_state_empty_slice_false_positives": sum(not r["ground_truth_nonempty"] and r["shared_state_false_positive_pixels"] > 0 for r in rows),
            "expert_components": sum(r["expert_components"] for r in nonempty),
            "independent_components_recovered": sum(r["independent_components_recovered"] for r in nonempty),
            "shared_state_components_recovered": sum(r["shared_state_components_recovered"] for r in nonempty),
        })
        save_prediction(pred_dir / f"{subject.subject_id}_independent.nii.gz", a, subject.image_itk)
        save_prediction(pred_dir / f"{subject.subject_id}_shared_state.nii.gz", b, subject.image_itk)
        save_profile(fig_dir / f"{subject.subject_id}_slice_profile.png", rows, subject.subject_id)

    write_csv(args.output_dir / "per_slice_comparison.csv", slice_rows)
    write_csv(args.output_dir / "per_subject_comparison.csv", subject_rows)
    stats = statistics(slice_rows, args.bootstrap_samples)
    (args.output_dir / "statistical_comparison.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    save_summary(fig_dir / "summary_comparison.png", subject_rows)
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = "unavailable"
    manifest = {
        "experiment": "oracle_every_slice", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "subjects": args.subjects, "checkpoint": str(args.checkpoint.resolve()),
        "config": args.config, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "preprocessing": "whole-volume finite/nonzero 0.5-99.5 percentile clipping to uint8",
        "independent_method": "SAM2ImagePredictor.set_image + predict per non-empty slice",
        "shared_method": "one NPZ video inference state; add all non-empty boxes in z order before bidirectional propagation",
        "official_api_semantics": stats["interpretation_caveat"], "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "statistics": stats}, indent=2))


if __name__ == "__main__":
    main()
