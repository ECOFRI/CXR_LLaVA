# P5.0 architecture specification — 2026-10-25

Status: **CPU DESIGN ONLY; P5 GPU HOLD.** Hypothesis: trainable BLIP-2 Q-Former may add value over the completed P4 projection adapter. This is a local BLIP-2 ablation, not CLEAR grounding or a reproduction of another paper.

## Exact proposed path

`RGB 224×224 image → frozen BLIP-2 ViT → trainable pretrained Q-Former with trainable pretrained query_tokens → trainable pretrained language_projection → frozen OPT-2.7B → report`.

Pinned base: `Salesforce/blip2-opt-2.7b`, revision `59a1ef6c1e5117b3f65523d1c6066825bcf315e3`, local checkpoint from P4. Initialize **all three trainable components from that original pretrained checkpoint**, not P4 M1 or P4.1. The comparison therefore tests architecture and optimization together; the projection initialization is matched, but the resulting trained projection weights need not match P4. Load the complete model only in a separately approved GPU gate.

| Boundary | Shape, batch 1 | Precision / state |
|---|---|---|
| Processor/ViT | `[1,3,224,224] → [1,257,1408]` (CLS + 16×16 patches) | ViT frozen FP16, eval |
| Query tokens | `[1,32,768]` | trainable FP32; expand batch view |
| Q-Former | cross-attends 32 queries to 257 visual tokens; `[1,32,768]` | pretrained 12 layers, 12 heads, cross-attention every 2 layers, encoder width 1408; trainable FP32 |
| Language projection | `[1,32,768] → [1,32,2560]` | pretrained trainable FP32; output cast to OPT FP16 |
| OPT prefix + text | 32 visual tokens + ≤256 text tokens, at most 288 sequence positions | OPT frozen FP16, eval; retain autograd through OPT into prefix |

The local `config.json` supplies 224/14/1408 vision, 32 queries, Q-Former 768/12 layers/12 heads/cross-attention frequency 2/encoder width 1408, and OPT width 2560. The `257` count follows 224/14=16 patches per side plus CLS; it must be checked against an actual P5.1 forward. Decoder causal attention and the same P4 target-only/EOS labels apply; no text-token loss on prompt or visual prefix.

## Parameter allowlist and initialization

Freeze **every** parameter except exactly: all `qformer.inner.*` weights after wrapping, `query_tokens`, `language_projection.layer.weight`, and `language_projection.layer.bias`. In the pinned checkpoint these correspond to `qformer.*`, `query_tokens`, and `language_projection.*`. Reject missing/unexpected keys before wrapping; compare initial tensors against the pinned source after the native FP16 load and FP32 conversion as P4 did. Preserve frozen ViT/OPT `.eval()` while Q-Former and projection are `.train()`; inspect gradients and frozen parameter versions after backward. No OPT LoRA, vision adaptation, extra query tokens, or changed attention layout.

Checkpoint metadata inspection, without loading tensor payloads, confirms **105,137,664 Q-Former + 24,576 query-token + 1,968,640 projection = 107,130,880 trainable parameters**. P4 had 1,968,640; P5 has 54.42× as many. Trainable FP32 weights alone are 408.67 MiB; gradient and Adam moments bring the nominal trainable-state footprint to 1.596 GiB (16 bytes/parameter), excluding activations and allocator overhead.

## Existing code disposition

- Reusable: pinned checkpoint/processor; `FP32QFormer` and `FP32Projection` wrappers; P4 manifest, target tokenizer, runtime lease/budget and P4 checkpoint mechanics as implementation references; P4 paired inference/evaluation logic as comparison reference.
- **Not P5-ready:** `scripts/train/train_blip2_bridges.py` uses `X06_qformer.yaml` with `max_steps=100`, generic eligible-record selection, unmasked `labels=input_ids`, 128-token combined text truncation, `model.train()` on frozen modules, and resume that restores step/optimizer but not data cursor or RNG. Its synthetic `QueryBridge` dry-run is not the pretrained BLIP-2 Q-Former. Its `required_free_gib=11.5` exceeds the K80 capacity reported by the P3 audit. The old inference entrypoint targets official MIMIC test and must not select P5 validation cases as-is.
- Before P5.1 approval: implement a new P5 contract and trainer/inference entrypoints by adapting P4 behavior, with exact allowlist, original initialization, frozen eval modes, same target masking, checkpoint cursor/order/RNG, distinct P5 run IDs and files, atomic status, shared GPU lease, cumulative time ledger and no P4 writes. Add CPU tests for the wrapper, token mask, checkpoint identity and inference record join. Version and freeze source/config hashes. This P5.0 task does not change executable training code.

Likely gate failures: Q-Former gradient missing through the frozen OPT, FP32/FP16 mismatch at cross-attention/projection, Adam state or extra activation OOM, 288-position attention cost, legacy PyTorch 1.12.1/cu102/Transformers 4.36.2 operations on K80, and generation EOS reduction (P4 uses a tested single-sequence EOS2 workaround). Any workaround that changes scientific behavior needs a new approval.
