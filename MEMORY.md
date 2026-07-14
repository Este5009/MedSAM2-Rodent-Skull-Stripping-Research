# MEMORY.md

## Communication

- Prefer clear, direct explanations.
- Prefer practical steps, commands, and decision points.
- Explain the strategic reason behind each research step.
- Be proactive: identify when experimental evidence suggests a pivot, a stronger baseline, or a more suitable model.
- Do not remain narrowly focused on the current implementation when the model itself appears to impose a performance ceiling.
- Avoid unnecessary verbosity, but do not omit important caveats.

## Coding

- Use beginner-friendly but professional Python.
- Prioritize readability over clever abstractions.
- Prefer `pathlib`, `argparse`, descriptive names, type hints, and small testable functions.
- Add robust shape, range, path, and alignment checks.
- Use configuration files for experiment parameters when practical.

## Comments

- Use heavy but natural technical commenting.
- Explain what the code does and why the step matters scientifically.
- Avoid generic or repetitive AI-generated comments.
- Include function headers or docstrings describing purpose, inputs, outputs, assumptions, and failure conditions.

## Documentation

- Use a professional research tone.
- Document the purpose, hypothesis, method, assumptions, inputs, outputs, and evaluation for each experiment.
- Keep commands and expected outputs explicit.
- Preserve enough detail for another researcher to reproduce the experiment.

## Repository

- Keep upstream MedSAM2 code under `external/MedSAM2/` and treat it as read-only by default.
- Put validated workflows in `scripts/core/`.
- Put new or unstable work in `scripts/experimental/`.
- Put reusable project code in `src/`.
- Preserve working code and prefer incremental changes over rewrites.
- Do not commit datasets, checkpoints, large predictions, environments, or caches.

## Research

- The immediate project is MedSAM2-based rodent MRI skull stripping.
- The broader goal is highly accurate, autonomous, and generalizable biomedical image segmentation.
- Establish the model's oracle-prompt ceiling before investing in automatic prompt generation.
- Understand results, failure modes, data geometry, and evaluation before training new architectures.
- Compare against strong specialist models and be honest when a foundation model does not add accuracy.
- Favor experiments that isolate one variable and answer one clear scientific question.
- Keep oracle, semi-automatic, and fully automatic methods explicitly separated.
- Never split data by slice; split by subject.
- Do not tune on the test set.

## Updating Memory

- Store only durable preferences and long-term project principles.
- Do not store temporary tasks, individual experiment results, current bugs, or one-time instructions.