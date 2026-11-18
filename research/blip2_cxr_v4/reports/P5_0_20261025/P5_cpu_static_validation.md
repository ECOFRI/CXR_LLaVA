# P5.0 CPU static validation — 2026-10-25

Classification: **static/readiness checks only**. No GPU allocation, complete-model load, training, inference, downloads or P4 artifact edits.

## Checks performed

| Check | Result | Evidence / limit |
|---|---|---|
| P4 config/index/three manifest SHA-256 | PASS | Exact hashes in `P5_matched_protocol.md`; recomputed from files. |
| P4 train/validation/Indiana manifest rows | PASS | 2,000/500/500 unique study and image IDs, splits train/validate/external_x00. Selected MIMIC train/validation subject overlap 0. |
| P4 checkpoint integrity/metadata | PASS | `P4.2_last.pt` SHA-256 `7f833eea69fb6d2743b19bbc808be949bb1a5e014bf43a81f08090f9a5f0d8af`; CPU metadata has only two projection adapter tensors, optimizer, next position, order hash and RNG states. P4 terminal status separately records reload PASS. |
| P4 full metrics identity | PASS | `metrics.json` SHA-256 `ed6be843294d14deafa1fd9c9366bf302a53c08f12f6804c530e680393fb9be5`; status PASS, 1,000 paired studies, 500 per cohort. No P4 recalculation. |
| Pinned BLIP-2 metadata | PASS | Local `config.json`: 32 queries, ViT 224/14/1408, Q-Former 768×12 layers/12 heads, cross-attention every 2 layers, encoder width 1408, OPT width 2560. Safetensors **slice metadata only**, no full weight tensors read: 105,137,664 Q-Former + 24,576 query + 1,968,640 projection. |
| Q-Former/projection wrapper path | PASS for small CPU surrogate | Local Transformers 4.36.2 Q-Former instantiated at 2 layers/24 hidden, six 32-wide visual tokens; existing `FP32QFormer` + `FP32Projection` returned `[1,4,24] → [1,4,40]`; Q-Former, query and projection gradients finite/nonzero. This does **not** prove full BLIP-2 training compatibility or K80 fit. |
| Existing relevant CPU tests | PASS | `test_p3_preflight.py`: 3/3; `test_p4_projection_gate.py`: 3/3. These protect old code paths and P4 target masking; they do not validate a future P5 trainer. |
| Legacy Q-Former training protocol | FAIL readiness | `X06_qformer.yaml` has 100-step cap and heuristic 11.5 GiB preflight. `train_blip2_bridges.py` does not match P4 target mask/cohort/resume/frozen eval mode. Must not be used for P5.1 as-is. |

CPU software context used only for inspection: `/home/student4/anaconda3/envs/cxr-llava-k80/bin/python`, PyTorch 1.12.1+cu102, Transformers 4.36.2. No environment changes. The small surrogate checks local wrapper dtype and gradient wiring; it does not test original checkpoint integration, real image preprocessing, optimizer updates, or generation. P5.1 remains HOLD pending implementation, tests, scientific approval and resource authorization.

Reproduce the metadata checks using the same environment's Python with `json`, `hashlib`, `torch.load(..., map_location='cpu')` for the 23 MiB P4 adapter checkpoint, and `safetensors.safe_open(...).get_slice(key).get_shape()` for pinned model shape metadata. Run from the repository root; do not load the full model or touch CUDA.
