#!/usr/bin/env python
"""Oracle multi-anchor ceiling experiment for difficult CAMRI volumes."""

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
import numpy as np
from scipy.ndimage import label
import SimpleITK as sitk
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL_DIR))

from evaluate_medsam2_multiregion_cases import (  # noqa: E402
    CASE_TARGETS, CONNECTIVITY_8, component_metrics, locate_subject_paths,
)
from run_medsam2_camri_single import (  # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_IMAGE_ROOT, DEFAULT_MASK_ROOT,
    dice_iou, infer_bidirectionally, normalize_for_model, percentile_normalize_mri,
    resize_for_model, tight_box, validate_geometry,
)

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "multi_anchor"
CONDITIONS = ("single_anchor", "three_fixed_anchors", "adaptive_oracle", "three_anchor_multibox_oracle")
ADAPTIVE_TARGET_DICE = 0.90
ADAPTIVE_MAX_ANCHORS = 7


@dataclass
class RunResult:
    prediction: np.ndarray
    anchors: list[int]
    boxes_by_anchor: dict[int, list[np.ndarray]]
    runtime_seconds: float
    peak_cuda_memory_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", nargs="*", choices=sorted(CASE_TARGETS), default=sorted(CASE_TARGETS))
    parser.add_argument("--adaptive-target-dice", type=float, default=ADAPTIVE_TARGET_DICE)
    parser.add_argument("--adaptive-max-anchors", type=int, default=ADAPTIVE_MAX_ANCHORS)
    return parser.parse_args()


def fixed_anchor_indices(truth: np.ndarray) -> tuple[int, int, int]:
    nonempty = np.flatnonzero(truth.sum(axis=(1, 2)) > 0)
    if nonempty.size < 3:
        raise ValueError("At least three non-empty slices are required for fixed anchors.")
    central = int(np.argmax(truth.sum(axis=(1, 2))))
    inferior = int(np.quantile(nonempty, .20, method="nearest"))
    superior = int(np.quantile(nonempty, .80, method="nearest"))
    anchors = sorted({inferior, central, superior})
    if len(anchors) != 3:
        raise ValueError(f"Fixed anchor quantiles were not unique: {anchors}")
    return anchors[0], central, anchors[-1]


def component_boxes(mask: np.ndarray) -> list[np.ndarray]:
    labels, count = label(mask, structure=CONNECTIVITY_8)
    boxes = [tight_box(labels == component) for component in range(1, count + 1)]
    return sorted(boxes, key=lambda b: (float(b[0]), float(b[1])))


def single_boxes(truth: np.ndarray, anchors: list[int]) -> dict[int, list[np.ndarray]]:
    return {z: [tight_box(truth[z])] for z in anchors}


def logits_union(logits: torch.Tensor) -> np.ndarray:
    """Union official per-object logits without connected-component postprocessing."""
    return np.any((logits > 0).cpu().numpy()[:, 0], axis=0)


def infer_multi_anchor(
    predictor: object, images: torch.Tensor, original_hw: tuple[int, int],
    anchors: list[int], boxes_by_anchor: dict[int, list[np.ndarray]], multi_object: bool,
) -> np.ndarray:
    """Use one conditioning state and a non-overlapping directional coverage rule."""
    depth, height, width = images.shape[0], *original_hw
    prediction = np.zeros((depth, height, width), dtype=bool)
    state = predictor.init_state(images, height, width)
    if multi_object:
        max_objects = max(len(boxes_by_anchor[z]) for z in anchors)
        for z in anchors:
            for object_index, box in enumerate(boxes_by_anchor[z], start=1):
                predictor.add_new_points_or_box(state, frame_idx=z, obj_id=object_index, box=box)
        # Objects without a component on a given anchor intentionally receive no
        # prompt there; their predictions are still consolidated by official API.
        assert max_objects == len(state["obj_ids"])
    else:
        for z in anchors:
            predictor.add_new_points_or_box(state, frame_idx=z, obj_id=1, box=boxes_by_anchor[z][0])

    lowest = min(anchors)
    for z, _, logits in predictor.propagate_in_video(state, start_frame_idx=lowest, reverse=False):
        prediction[z] = logits_union(logits)
    # Only fill the uncovered prefix. This avoids an arbitrary overlap merge.
    for z, _, logits in predictor.propagate_in_video(state, start_frame_idx=lowest, reverse=True):
        if z < lowest:
            prediction[z] = logits_union(logits)
    predictor.reset_state(state)
    return prediction


def timed_inference(
    predictor: object, images: torch.Tensor, truth: np.ndarray, anchors: list[int],
    boxes: dict[int, list[np.ndarray]], multi_object: bool = False,
) -> RunResult:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if len(anchors) == 1 and not multi_object:
                prediction = infer_bidirectionally(predictor, images, truth.shape[1:], anchors[0], boxes[anchors[0]][0], False)
            else:
                prediction = infer_multi_anchor(predictor, images, truth.shape[1:], anchors, boxes, multi_object)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("Insufficient CUDA memory; CPU fallback is disabled.") from exc
    return RunResult(prediction, list(anchors), boxes, time.perf_counter() - start, torch.cuda.max_memory_allocated() / 1024**2)


def slice_dice(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.array([dice_iou(prediction[z], truth[z])[0] for z in range(truth.shape[0])])


def adaptive_inference(
    predictor: object, images: torch.Tensor, truth: np.ndarray, central: int,
    baseline: RunResult, target_dice: float, max_anchors: int,
) -> tuple[RunResult, list[dict[str, object]]]:
    anchors = [central]; result = baseline; history: list[dict[str, object]] = []
    cumulative_runtime = baseline.runtime_seconds
    peak_memory = baseline.peak_cuda_memory_mb
    nonempty = np.flatnonzero(truth.sum(axis=(1, 2)) > 0)
    while True:
        dices = slice_dice(result.prediction, truth); active = dices[nonempty]
        volume_dice, _ = dice_iou(result.prediction, truth)
        history.append({"anchor_count": len(anchors), "anchors": json.dumps(sorted(anchors)), "volumetric_dice": volume_dice,
                        "mean_nonempty_slice_dice": float(active.mean()), "minimum_nonempty_slice_dice": float(active.min())})
        if float(active.min()) >= target_dice or len(anchors) >= max_anchors:
            return RunResult(result.prediction, result.anchors, result.boxes_by_anchor, cumulative_runtime, peak_memory), history
        candidates = [int(z) for z in nonempty if int(z) not in anchors]
        worst = min(candidates, key=lambda z: (dices[z], -abs(z - central)))
        anchors.append(worst); anchors.sort()
        result = timed_inference(predictor, images, truth, anchors, single_boxes(truth, anchors))
        cumulative_runtime += result.runtime_seconds
        peak_memory = max(peak_memory, result.peak_cuda_memory_mb)


def nearest_anchor_distance(depth: int, anchors: list[int]) -> np.ndarray:
    return np.array([min(abs(z - anchor) for anchor in anchors) for z in range(depth)])


def aggregate_subject(
    subject: str, condition: str, result: RunResult, truth: np.ndarray, targets: list[int]
) -> dict[str, object]:
    dices = slice_dice(result.prediction, truth); nonempty = truth.sum(axis=(1, 2)) > 0
    vd, vi = dice_iou(result.prediction, truth)
    recovered = missed = pred_components = 0
    for z in targets:
        topology = component_metrics(result.prediction[z], truth[z])
        recovered += topology["expert_component_count"] - topology["expert_components_missed_entirely"]
        missed += topology["expert_components_missed_entirely"]
        pred_components += topology["predicted_component_count"]
    return {
        "subject_id": subject, "condition": condition, "volumetric_dice": vd, "volumetric_iou": vi,
        "mean_nonempty_slice_dice": float(dices[nonempty].mean()), "minimum_nonempty_slice_dice": float(dices[nonempty].min()),
        "mean_target_slice_dice": float(dices[targets].mean()), "minimum_target_slice_dice": float(dices[targets].min()),
        "target_expert_components_recovered": recovered, "target_expert_components_missed": missed,
        "target_predicted_component_count": pred_components,
        "false_positive_voxels": int(np.count_nonzero(result.prediction & ~truth)),
        "false_negative_voxels": int(np.count_nonzero(truth & ~result.prediction)),
        "empty_predictions_on_nonempty_slices": int(sum(nonempty[z] and not result.prediction[z].any() for z in range(len(nonempty)))),
        "runtime_seconds": result.runtime_seconds, "peak_cuda_memory_mb": result.peak_cuda_memory_mb,
        "anchor_count": len(result.anchors), "box_count": sum(len(v) for v in result.boxes_by_anchor.values()),
        "anchor_indices": json.dumps(result.anchors),
    }


def per_slice_rows(subject: str, condition: str, result: RunResult, truth: np.ndarray, targets: list[int]) -> list[dict[str, object]]:
    distances = nearest_anchor_distance(truth.shape[0], result.anchors); rows = []
    for z in range(truth.shape[0]):
        topology = component_metrics(result.prediction[z], truth[z]); dice, iou = dice_iou(result.prediction[z], truth[z])
        rows.append({"subject_id": subject, "condition": condition, "slice_index": z, "is_difficult_target": z in targets,
                     "distance_to_nearest_anchor": int(distances[z]), "dice": dice, "iou": iou,
                     "expert_area": int(truth[z].sum()), "predicted_area": int(result.prediction[z].sum()), **topology})
    return rows


def show(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", origin="upper"); ax.set_title(title, fontsize=8); ax.set_axis_off()


def contours(ax: plt.Axes, truth: np.ndarray, pred: np.ndarray) -> None:
    if truth.any(): ax.contour(truth, levels=[.5], colors=["lime"], linewidths=1)
    if pred.any(): ax.contour(pred, levels=[.5], colors=["magenta"], linewidths=1)


def save_difficult_montage(path: Path, raw: np.ndarray, truth: np.ndarray, results: dict[str, RunResult], targets: list[int]) -> None:
    fig, axes = plt.subplots(len(targets), 6, figsize=(18, 4 * len(targets)), squeeze=False, constrained_layout=True)
    for row_axes, z in zip(axes, targets):
        show(row_axes[0], raw[z], f"z={z} expert"); contours(row_axes[0], truth[z], np.zeros_like(truth[z]))
        for ax, condition, title in zip(row_axes[1:5], CONDITIONS, ("single", "three", "adaptive", "three multi-box")):
            pred = results[condition].prediction[z]; d = dice_iou(pred, truth[z])[0]; cc = component_metrics(pred, truth[z])
            show(ax, raw[z], f"{title}\nDice={d:.3f} CC={cc['predicted_component_count']}"); contours(ax, truth[z], pred)
        pred = results["adaptive_oracle"].prediction[z]; show(row_axes[5], raw[z], "adaptive errors\nred FP / blue FN")
        for mask, color in ((pred & ~truth[z], (1,0,0,.35)), (truth[z] & ~pred, (0,.3,1,.35))):
            rgba=np.zeros((*mask.shape,4)); rgba[mask]=color; row_axes[5].imshow(rgba, origin="upper")
    fig.suptitle("Difficult slices: oracle anchor conditions"); fig.savefig(path, dpi=200); plt.close(fig)


def save_profiles(path: Path, truth: np.ndarray, results: dict[str, RunResult]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6)); z=np.arange(truth.shape[0])
    for condition in CONDITIONS:
        result=results[condition]; ax.plot(z, slice_dice(result.prediction, truth), label=condition)
        for anchor in result.anchors: ax.axvline(anchor, alpha=.12)
    ax.set(xlabel="Slice index", ylabel="Dice", ylim=(-.02,1.02), title="Dice-by-slice and anchor locations"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)


def save_anchor_figure(path: Path, raw: np.ndarray, truth: np.ndarray, result: RunResult, reasons: dict[int,str]) -> None:
    cols=3; rows=math.ceil(len(result.anchors)/cols); fig,axes=plt.subplots(rows,cols,figsize=(12,4*rows),squeeze=False,constrained_layout=True)
    for ax in axes.flat: ax.set_axis_off()
    for ax,z in zip(axes.flat,result.anchors):
        show(ax,raw[z],f"z={z}: {reasons.get(z,'adaptive worst-Dice anchor')}")
        labels,count=label(truth[z],structure=CONNECTIVITY_8)
        for component in range(1,count+1): ax.contour(labels==component,levels=[.5],linewidths=1)
        for box in result.boxes_by_anchor[z]:
            x0,y0,x1,y1=box; ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="cyan",linewidth=1.5))
    fig.suptitle("Oracle anchors, expert components, and boxes"); fig.savefig(path,dpi=200); plt.close(fig)


def save_full_comparison(path: Path, raw: np.ndarray, truth: np.ndarray, results: dict[str, RunResult]) -> None:
    indices=np.unique(np.rint(np.linspace(0,truth.shape[0]-1,16)).astype(int)); fig,axes=plt.subplots(len(indices),5,figsize=(15,3*len(indices)),constrained_layout=True)
    for row_axes,z in zip(axes,indices):
        show(row_axes[0],raw[z],f"z={z} expert"); contours(row_axes[0],truth[z],np.zeros_like(truth[z]))
        for ax,condition in zip(row_axes[1:],CONDITIONS): show(ax,raw[z],condition); contours(ax,truth[z],results[condition].prediction[z])
    fig.suptitle("Full-volume sampled comparison (expert green, prediction magenta)"); fig.savefig(path,dpi=170); plt.close(fig)


def write_csv(path: Path, rows: list[dict[str,object]]) -> None:
    with path.open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def global_plots(output: Path, slice_rows: list[dict[str,object]], subject_rows: list[dict[str,object]]) -> None:
    fig,ax=plt.subplots(figsize=(9,6))
    for condition in CONDITIONS:
        rows=[r for r in slice_rows if r["condition"]==condition and int(r["expert_area"])>0]
        ax.scatter([r["distance_to_nearest_anchor"] for r in rows],[r["dice"] for r in rows],s=12,alpha=.45,label=condition)
    ax.set(xlabel="Distance to nearest anchor",ylabel="Dice",ylim=(-.02,1.02),title="Dice versus nearest-anchor distance");ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/"dice_vs_nearest_anchor.png",dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6)); x=np.arange(len(subject_rows)); recovered=[r["target_expert_components_recovered"] for r in subject_rows]
    colors=[CONDITIONS.index(str(r["condition"])) for r in subject_rows]; ax.scatter(x,recovered,c=colors,cmap="tab10",s=70)
    ax.set_xticks(x,[f"{r['subject_id']}\n{r['condition']}" for r in subject_rows],rotation=70,ha="right");ax.set_ylabel("Target expert components recovered");ax.set_title("Component recovery by condition");fig.tight_layout();fig.savefig(output/"component_recovery.png",dpi=220);plt.close(fig)


def main() -> None:
    args=parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU fallback disabled.")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in ("difficult_montages","profiles","full_volume_montages","anchors","predictions"): (args.output_dir/name).mkdir(exist_ok=True)
    from sam2.build_sam import build_sam2_video_predictor_npz
    predictor=build_sam2_video_predictor_npz(args.config,str(args.checkpoint),device="cuda")
    subject_rows=[]; all_slice_rows=[]; adaptive_history=[]
    for subject in args.subjects:
        paths=locate_subject_paths(subject,args.image_root,args.mask_root); image_itk=sitk.ReadImage(str(paths.image_path)); mask_itk=sitk.ReadImage(str(paths.mask_path));validate_geometry(image_itk,mask_itk)
        raw=sitk.GetArrayFromImage(image_itk).astype(np.float32);truth=sitk.GetArrayFromImage(mask_itk)>0
        normalized=percentile_normalize_mri(raw,.5,99.5).image_uint8; resized=resize_for_model(normalized); images=normalize_for_model(resized)
        inferior,central,superior=fixed_anchor_indices(truth); fixed=[inferior,central,superior]
        baseline=timed_inference(predictor,images,truth,[central],single_boxes(truth,[central]))
        three=timed_inference(predictor,images,truth,fixed,single_boxes(truth,fixed))
        adaptive,history=adaptive_inference(predictor,images,truth,central,baseline,args.adaptive_target_dice,args.adaptive_max_anchors)
        multi_boxes={z:component_boxes(truth[z]) for z in fixed}; multibox=timed_inference(predictor,images,truth,fixed,multi_boxes,True)
        results=dict(zip(CONDITIONS,(baseline,three,adaptive,multibox)))
        reasons={inferior:"20th percentile non-empty",central:"maximum expert area",superior:"80th percentile non-empty"}
        for condition,result in results.items():
            subject_rows.append(aggregate_subject(subject,condition,result,truth,CASE_TARGETS[subject]));all_slice_rows.extend(per_slice_rows(subject,condition,result,truth,CASE_TARGETS[subject]))
            out=sitk.GetImageFromArray(result.prediction.astype(np.uint8));out.CopyInformation(image_itk);sitk.WriteImage(out,str(args.output_dir/"predictions"/f"{subject}_{condition}.nii.gz"))
        for row in history: adaptive_history.append({"subject_id":subject,**row})
        save_difficult_montage(args.output_dir/"difficult_montages"/f"{subject}.png",raw,truth,results,CASE_TARGETS[subject])
        save_profiles(args.output_dir/"profiles"/f"{subject}.png",truth,results)
        save_full_comparison(args.output_dir/"full_volume_montages"/f"{subject}.png",raw,truth,results)
        save_anchor_figure(args.output_dir/"anchors"/f"{subject}_three_fixed.png",raw,truth,three,reasons)
        save_anchor_figure(args.output_dir/"anchors"/f"{subject}_adaptive.png",raw,truth,adaptive,reasons)
        print(subject,"fixed",fixed,"adaptive",adaptive.anchors)
        del images,resized;torch.cuda.empty_cache()
    write_csv(args.output_dir/"comparison_by_subject.csv",subject_rows);write_csv(args.output_dir/"comparison_by_slice.csv",all_slice_rows);write_csv(args.output_dir/"adaptive_history.csv",adaptive_history)
    global_plots(args.output_dir,all_slice_rows,subject_rows)
    manifest={"checkpoint":str(args.checkpoint),"config":args.config,"normalization":"percentile 0.5-99.5","conditions":CONDITIONS,"fixed_anchor_logic":"20th percentile non-empty, maximum-area, 80th percentile non-empty","adaptive_target_dice":args.adaptive_target_dice,"adaptive_max_anchors":args.adaptive_max_anchors,"subjects":args.subjects}
    (args.output_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Saved",args.output_dir)

if __name__=="__main__": main()
