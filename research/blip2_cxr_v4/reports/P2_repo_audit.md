# P2.6 source repository audit

Status PARTIAL for runnable checkpoint readiness; source checkouts COMPLETE. Source-only shallow clones; no LFS pull, checkpoint, install, or script execution. Exact commands: `timeout 20 env GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none https://github.com/synlp/R2-LLM research/blip2_cxr_v4/third_party/R2-LLM` and the same command for `https://github.com/DasithEdirisinghe/META-CXR`. The initial R2 checkout timed out on upstream tracked `__pycache__` files; bounded `timeout 20 git restore --source=HEAD --worktree :/` passes completed it. R2 now has 147 tracked files and a clean worktree at the pinned SHA; META checkout also completed. CPU/read-only audit; no dataset cohort, metrics, or CI.

| Item | R2-LLM | META-CXR |
|---|---|---|
| Official source | [synlp/R2-LLM](https://github.com/synlp/R2-LLM) | [DasithEdirisinghe/META-CXR](https://github.com/DasithEdirisinghe/META-CXR) |
| Cloned SHA | `2167adf5bf3a204ad85e7065aad74fa1ebb9640e` | `e97d70900dd92c2349b98e60decb5c27e6b1e121` |
| License | Root `LICENSE`: MIT | Root license NOT_FOUND; vendored `model/lavis/LICENSE.txt` is not a whole-repository license; reuse terms UNKNOWN |
| Train | `train.py`, stage1/stage2 YAMLs, DeepSpeed option | `pretraining/train.py`; dataset and model package |
| Infer | `generate_reports.py` | `inference.py`, `inference.sh`, Gradio Docker route |
| Dataset/preprocess | `minigpt4/datasets/datasets/iuxray_dataset.py`, `mimic_dataset.py`; JSON `train` records with `image_path`, `id`, `report`; RGB image open | `model/lavis/data/ReportDataset.py`; MIMIC official split/metadata CSV, cleaned report CSV, CheXpert CSV; AP/PA selection and image resize/crop; IU adapter present |
| Eval | `eval_configs/eval.yaml`, inference generation; standalone scoring entrypoint UNKNOWN | Classification/evaluation functions in pretraining; standalone report-eval entrypoint UNKNOWN |
| Required checkpoints | Vicuna 7B/13B per MiniGPT-4 configs, projection, [R2 pre-trained/fine-tuned weights](https://huggingface.co/a-b-c-d-e-g/R2-LLM); exact files/access/license UNKNOWN | [Google Drive folder](https://drive.google.com/drive/folders/1zUT1ogIdmEjOXtBe1Vzw44uFV_ZvlcMF?usp=sharing), expected `pretrainings/output/<subdir>/<file>.pth`; exact name/license UNKNOWN |
| Dependency pins | `environment.yaml`: Python 3.9, torch 2.0.0, transformers 4.30.0, tokenizers 0.13.2, timm 0.6.13, peft 0.2.0, bitsandbytes 0.37.0 | `requirements.txt`: torch 1.13.0, torchvision 0.14.0, timm 0.4.12, peft git SHA, nltk 3.8.1, pycocoevalcap 1.2; Docker GPU route |
| K80/cu102 vs current env | Current cxr-llava-k80 has torch 1.12.1+cu102, transformers 4.36.2; pinned versions differ. Runtime K80 support UNVERIFIED; model size/weights also unresolved | Current torch differs; GPU Docker stack and required weights unverified on K80. Runtime support UNKNOWN |

The presence of repository code does not establish a runnable benchmark. Neither repository's checkpoint was downloaded or executed. Next decision: resolve licensing and exact checkpoint access, then a separately approved compatibility test; no GPU work under this batch.
