# Project Criteria: CXR-LLaVA Improvement

## Scope and project boundary

This file applies only to `/storage/student4/hanhpm/CXR_LLaVA_Improvement`.

The separate `E:/Downloads/USTH_ICTLab/patchcore-inspection-experiment` project is out of scope. Do not route CXR-LLaVA tasks through its instructions, experiment plans, roadmaps, runbooks, datasets, or commands.

At the start of work in this repository, read:

1. `CXR_LLaVA_Improvement/AGENTS.md`
2. `CXR_LLaVA_Improvement/CRITERIA.md`
3. `CXR_LLaVA_Improvement/docs/MODEL_CARD.md` when model behavior or limitations are relevant

The repository-local `AGENTS.md` and this file are authoritative for CXR-LLaVA work. Do not infer requirements from the PatchCore project.

## Project goal

Use CXR-LLaVA as a research-only multimodal model for chest X-ray interpretation and improve the reproducibility and usability of its inference workflow. Prioritize:

- Running the existing model on a GPU notebook, especially Google Colab with a T4.
- Reusing `main.py`, `CXR_LLAVA_HF/`, `requirements.txt`, sample images, and the test CSV where applicable.
- Supporting a user-owned fork through an explicit `git clone` or repository-path configuration.
- Making small, verifiable improvements to loading, inference, prompts, and evaluation.
- Separating smoke tests from metric or research claims.

## Model and data assumptions

- Model family: `ECOFRI/CXR-LLAVA-v2` unless another checkpoint is explicitly selected.
- Input: chest X-ray images, primarily 512x512 grayscale as described by the model card.
- Sample assets are under `CXR_LLaVA_Improvement/IMG/`.
- Test metadata is `CXR_LLaVA_Improvement/MIMIC_CXR_DATASET_TESTSET.csv`.
- Dataset paths must be configurable; never assume a Windows, WSL, Kaggle, or Colab path exists elsewhere.
- Do not expose or commit private patient data, credentials, access tokens, or personal Git URLs.
- A generated report is experimental model output, not a clinical diagnosis.

## Google Colab and T4 requirements

Colab notebooks should:

- Detect and report the active device, using CUDA when a T4 is available and CPU fallback only for smoke testing.
- Install only dependencies needed by this repository.
- Keep the personal fork URL in one clearly marked configuration cell; never hard-code a private token.
- Clone the fork into a predictable workspace path and use that path as the active project directory.
- Reuse repository code rather than duplicating model implementation in notebook cells.
- Use flushed progress for long-running cells.
- Run a small sample-image smoke test before optional batch evaluation.
- Avoid downloading or processing the full dataset by default; batch evaluation must be opt-in.
- Free model and tensor memory when switching experiments on a 16 GB T4.

## Safety and medical-use boundary

CXR-LLaVA is for research and educational use only. Notebook outputs must be labeled as model-generated text and must not be presented as verified clinical findings, treatment advice, or patient-specific medical decisions. Do not claim clinical accuracy from a smoke test or a small local sample.

When handling real medical images or reports, use only authorized data, keep identifying information out of logs and committed files, and do not upload private data to external services without explicit authorization.

## Review and validation pipeline

For a code or notebook change:

1. Inspect the existing implementation and documented usage.
2. Define expected behavior and the smallest relevant check.
3. Validate imports and syntax without launching an expensive model run.
4. Run a T4 smoke test only when requested or when it is the direct deliverable.
5. Report whether the result is a smoke test, metric-complete evaluation, or benchmark claim.

When debugging, review dataset handling, model loading, image preprocessing, prompt construction, device placement, output decoding, and failure messages in that order.

## Coding rules

1. Keep changes minimal and directly related to CXR-LLaVA.
2. Reuse existing functions and modules before adding duplicate logic.
3. Keep imports at the top of Python files unless a narrow technical reason requires otherwise.
4. Avoid broad `try/except` blocks; catch only expected I/O or model-loading errors.
5. Do not add unused imports, variables, functions, or dependencies.
6. Do not delete existing comments or unrelated code without an explicit request.
7. Keep command-line arguments flexible but simple.
8. Use clear output labels and avoid unnecessary decorative prints.
9. Do not add PatchCore-specific code, paths, metrics, plans, or runbooks to this repository.
10. Do not add credentials, private repository tokens, or local absolute paths to committed files.

## Command and artifact rules

- Commands for this repository must use `CXR_LLaVA_Improvement` as the working directory or clearly change into it first.
- Colab cells must use `/content/` paths or variables defined in the notebook.
- Do not run training, full-dataset inference, or other long jobs without an explicit scope, dataset path, and output location.
- Do not create experiment manifests, YAML files, benchmark claims, or evaluation artifacts unless requested.
- Generated notebooks belong inside this repository and should have descriptive names.
- When a change is complete, provide the exact command or notebook cell sequence needed to reproduce it.

## Expected implementation style

- Prefer explicit, readable code over abstractions for one-off notebook workflows.
- Preserve the current repository structure and public interfaces unless required.
- Keep notebook cells short enough to rerun independently where practical.
- Make assumptions visible in markdown cells and configuration variables.
