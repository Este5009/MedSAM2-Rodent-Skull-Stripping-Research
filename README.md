# MedSAM2 Rodent Skull-Stripping Research

Research repository for evaluating and extending MedSAM2 toward accurate, autonomous volumetric segmentation of rodent brain MRI. The CAMRI rat dataset will serve as the initial benchmark.

## Research motivation

Promptable medical foundation models may reduce the manual effort required for volumetric segmentation, but their accuracy, propagation behavior, and failure modes must be established before autonomous use. This project will investigate whether MedSAM2 can use information across an ordered MRI slice sequence while reducing or ultimately eliminating manual initialization. It will distinguish oracle prompting, semi-automatic methods, and fully automatic inference throughout the evaluation.

## High-level objectives

- Reproduce the official MedSAM2 inference workflow without modifying upstream code.
- Establish a trustworthy oracle-initialization baseline on CAMRI rat MRI volumes.
- Validate image geometry, preprocessing, propagation, reconstruction, and volumetric metrics.
- Characterize forward and backward propagation behavior and failure cases.
- Develop and evaluate automatic initialization and prompt generation.
- Compare MedSAM2 with appropriate specialist segmentation baselines.

## Repository structure

```text
docs/                  Research and methodology documentation
external/              Read-only upstream repositories, including MedSAM2
datasets/              Local datasets (not tracked by Git)
checkpoints/            Local model weights (not tracked by Git)
outputs/
    figures/            Selected quality-control and report figures
    logs/               Runtime logs (not tracked by Git)
    metrics/            Evaluation outputs
    predictions/        Generated model predictions (not tracked by Git)
scripts/
    core/               Validated, reproducible workflows
    experimental/       New or unstable research experiments
src/                    Reusable project-specific modules
tests/                  Focused validation and regression tests
```

## Current roadmap

The project is currently in the repository-setup phase. The next stages are to reproduce official MedSAM2 inference, establish a CAMRI rat benchmark, and analyze volumetric propagation. See [ROADMAP.md](ROADMAP.md) for the complete phased plan.

## Expected future work

Future work is expected to include reproducible MedSAM2 inference, per-subject and per-slice evaluation, propagation sensitivity studies, automatic initialization, automatic prompt generation, specialist-model comparisons, and preparation of research findings for publication. No experimental results are reported yet.

## Research status

This repository currently provides project organization and planning only. It does not yet include MedSAM2 source code, checkpoints, dependencies, or segmentation implementations. Local datasets may exist in ignored directories, but no dataset files are tracked by Git.
