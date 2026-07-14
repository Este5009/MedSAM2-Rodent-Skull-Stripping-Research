# AGENTS.md

## Read First

Before making changes, read:

```text
AGENTS.md
MEMORY.md
```

Use `AGENTS.md` for project goals, research priorities, and repository rules.
Use `MEMORY.md` for durable user preferences.

Do not store temporary experiment results or one-time tasks in `MEMORY.md`.

---

## Project Title

```text
MedSAM2 Rodent Skull-Stripping Research
```

Repository goal:

```text
Evaluate and extend MedSAM2 for accurate, autonomous volumetric rodent brain MRI segmentation.
```

The CAMRI rat dataset is the initial benchmark. The broader research objective is to understand whether a promptable 3D medical foundation model can approach specialist-model accuracy while reducing or eliminating manual prompting.

---

## Research Motivation

Previous work with the original 2D MedSAM established:

```text
MedSAM with oracle slice-wise boxes ≈ 0.91 Dice
RS2-Net on CAMRI Rat can approach ≈ 0.99 Dice
```

This indicates that prompt engineering alone cannot overcome the original MedSAM checkpoint's performance ceiling for this task.

MedSAM2 is now being investigated because it:

- is designed for 3D medical images and videos;
- initializes segmentation on one representative slice;
- propagates predictions bidirectionally through the volume;
- uses a memory-attention module to exploit continuity between adjacent slices;
- may reduce manual prompting from one box per slice to one initialization prompt per volume.

Do not assume MedSAM2 will outperform specialist models. Establish its actual performance experimentally.

---

## Main Research Questions

### Phase 1 — Reproduction and baseline

1. Can the official MedSAM2 inference pipeline be reproduced without modification?
2. Can MedSAM2 process one CAMRI rat MRI volume correctly?
3. What volumetric Dice, slice-wise Dice, NSD, and failure rate does it achieve with an oracle initialization box?
4. Does it outperform the original MedSAM baseline?

### Phase 2 — Propagation analysis

1. How does segmentation quality change as distance from the initialization slice increases?
2. Does forward propagation behave differently from backward propagation?
3. On which slices does the target disappear, fragment, or drift?
4. How sensitive is propagation to the initialization slice?
5. How sensitive is propagation to box margin and perturbation?

### Phase 3 — Autonomous initialization

1. Can the best initialization slice be selected automatically?
2. Can an automatic coarse localizer generate the initialization box?
3. Can the complete volume be segmented without human interaction?
4. How much accuracy is lost relative to oracle initialization?

### Phase 4 — Specialist comparison

Compare MedSAM2 against:

- original MedSAM;
- RS2-Net;
- an appropriate U-Net/nnU-Net baseline if implemented;
- any additional model only when it answers a clear research question.

---

## Scientific Priorities

Follow this order:

1. Reproduce the official implementation.
2. Establish a trustworthy oracle-prompt baseline.
3. Validate geometry, orientation, preprocessing, and metrics.
4. Analyze propagation behavior and failure cases.
5. Test automatic initialization.
6. Consider fine-tuning only after inference limitations are understood.

Do not begin by training a new model.

Do not add complexity merely because a method is newer or more advanced.

Every experiment must answer one explicit research question.

---

## Repository Organization

Preferred structure:

```text
configs/
docs/
external/
outputs/
    figures/
    logs/
    metrics/
    predictions/
scripts/
    core/
    experimental/
src/
tests/
```

Rules:

- `external/MedSAM2/` contains the upstream MedSAM2 repository.
- Treat upstream code as read-only by default.
- Do not commit large checkpoints, datasets, generated predictions, or temporary caches.
- `scripts/core/` contains validated, reproducible workflows.
- `scripts/experimental/` contains new or unstable experiments.
- `src/` contains reusable project-specific modules.
- `tests/` contains focused tests for loading, preprocessing, prompting, propagation, and metrics.
- `outputs/` contains generated results and should be organized by experiment.
- Prefer configuration files over hard-coded paths and parameters.

If upstream changes are unavoidable:

1. explain why a wrapper or adapter is insufficient;
2. make the smallest possible change;
3. document the exact upstream file and line-level behavior;
4. preserve a clean record of the original implementation.

---

## Data and Split Rules

- Split data by subject, never by slice.
- Never use expert masks to create prompts for an automatic method.
- Oracle masks may be used only for explicitly labeled oracle baselines and evaluation.
- Keep preprocessing identical across compared methods unless preprocessing itself is the controlled experimental variable.
- Preserve original volume orientation, spacing, and affine metadata whenever possible.
- Record all resampling, resizing, normalization, and axis-order operations.
- Use nearest-neighbor interpolation for masks.

---

## MedSAM2-Specific Rules

MedSAM2 treats a 3D volume as an ordered sequence of 2D slices.

The expected baseline workflow is:

```text
volume
→ select initialization slice
→ create one 2D box prompt
→ segment initialization slice
→ propagate toward both ends
→ reconstruct 3D prediction
→ evaluate in 3D
```

Verify all of the following before trusting results:

- slice axis and ordering;
- orientation consistency between image and mask;
- initialization slice selection;
- bounding-box coordinate convention;
- image resizing and coordinate transforms;
- propagation start and end indices;
- forward and backward prediction merging;
- empty-mask handling;
- restoration to original volume shape;
- 3D metric computation.

Do not describe MedSAM2 as a true 3D encoder. It encodes slices in 2D and uses memory attention across the slice sequence. The paper itself notes that it does not explicitly model full 3D spatial continuity.

---

## Evaluation Requirements

At minimum report:

- volumetric Dice per subject;
- mean, median, standard deviation, minimum, and maximum Dice;
- slice-wise Dice distribution;
- normalized surface distance when spacing metadata is available;
- empty-slice false positives;
- non-empty-slice false negatives;
- propagation failure rate;
- inference time;
- initialization strategy and prompt source.

Whenever possible also report:

- Dice versus distance from initialization slice;
- forward versus backward propagation performance;
- worst-subject and worst-slice cases;
- sensitivity to initialization slice;
- sensitivity to prompt perturbation.

Always save per-subject and per-slice CSV files, not only aggregate metrics.

---

## Experiment Reproducibility

Every experiment should record:

- experiment name;
- date;
- git commit;
- checkpoint name or hash;
- dataset split;
- subject list;
- configuration;
- random seed;
- device;
- software versions;
- output paths;
- summary metrics.

Prefer commands such as:

```bash
python scripts/core/evaluate_camri_medsam2.py --config configs/camri_oracle.yaml
```

Avoid manual notebook-only workflows for final results.

Notebooks may be used for exploration, but validated results must be reproducible from scripts.

---

## Coding Standards

- Use Python with clear type hints where practical.
- Prefer `pathlib`, `argparse`, dataclasses, and small testable functions.
- Add descriptive function docstrings.
- Explain both what the code does and why.
- Use assertions and explicit validation for tensor shapes, coordinate ranges, and image-mask alignment.
- Fail loudly on invalid data rather than silently correcting uncertain cases.
- Avoid broad rewrites of working code.
- Reuse existing validated utilities.
- Keep research logic separate from plotting and file I/O.

---

## Testing Expectations

Before a full benchmark, test:

1. one official MedSAM2 example;
2. one CAMRI subject;
3. one slice with a known non-empty mask;
4. one empty boundary slice;
5. forward and backward propagation;
6. reconstructed volume shape;
7. metric correctness on synthetic masks.

A full run should not begin until the single-subject output has been visually inspected.

---

## Visualization Requirements

Quality-control figures should show:

- representative MRI slices;
- expert mask;
- initialization box and slice;
- MedSAM2 prediction;
- false positives and false negatives;
- slices near propagation failure;
- 3D or montage-level continuity when useful.

Figures must clearly distinguish:

- oracle prompts;
- automatic prompts;
- expert annotations;
- model predictions.

Do not use ambiguous titles such as “best result” without specifying the method and metric.

---

## Research Integrity

- Do not overstate generalization from the CAMRI dataset.
- Do not claim clinical readiness.
- Distinguish architectural capability from checkpoint behavior.
- Distinguish oracle prompting from autonomous inference.
- Report negative results and failed propagation cases.
- Do not tune parameters on the test set.
- Avoid comparing metrics produced with incompatible preprocessing or evaluation definitions.

This repository is a research prototype, not clinical software.