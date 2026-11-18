# P5.0 matched protocol — 2026-10-25

Status: design frozen against P4 inputs; **no P5 scientific approval or GPU authority**. P4 sources remain immutable.

## Cohorts and immutable provenance

Reuse `runs/P4_projection_pilot_20261023/manifest_index.approved.json` (SHA-256 `74965a79b31394a65af10dcb7987db0881e5083a8bb0da888bccbe120bd03175`) and its manifests unchanged:

| Cohort | Count | SHA-256 | Use |
|---|---:|---|---|
| MIMIC official train | 2,000 unique studies/images | `ec8bc5fd8eb23c98884671cf139e579bad06cd1b6ea66e316b6320d65f39d4e2` | one epoch, one seeded shuffle |
| MIMIC official validate | 500 unique studies/images | `b0d8f209ae6699bcde215d02652a2da29be59eb7f4936efa34d34ef81b6c21a3` | evaluation only |
| Indiana X00 | 500 unique studies/images | `5878e73e6c20091d1a26bfb474894c30e11894e86f73217342c8ed6bbaf59944` | external evaluation only |

CPU hash/count checks pass, and selected MIMIC train/validation patient overlap is zero. The source-order convenience selection is not random. Keep first available AP/PA image in official MIMIC split order; keep Indiana's saved frontal image and matching X00 reference. Do not use locked MIMIC/other test data for model selection. Recheck source file/image/reference identity before future execution; P5.0 did not rehash all image contents.

## Parity matrix

| Component | P4 M1 | Proposed P5 Q-Former | Rule |
|---|---|---|---|
| Pretrained checkpoint | pinned BLIP-2 revision above | same original checkpoint | identical; no P4 trained adapter initialization |
| ViT / OPT | frozen FP16 eval | frozen FP16 eval | identical |
| Q-Former/query tokens | frozen FP16 | trainable FP32 | intentional difference |
| Language projection | trainable FP32 | trainable FP32 | same module/init source, separately trained |
| Image preprocessing/view/reference | P4 approved contract/manifests | same | identical |
| Training prompt/target | QA prompt; FINDINGS + IMPRESSION | same | identical |
| Labels/context | target/EOS only; target ≤128, text ≤256, 32 visual tokens | same | identical |
| Optimizer | AdamW LR 1e-4, β=(.9,.999), ε=1e-8, weight decay .01, no scheduler/clipping; batch 1/accumulation 1, seed 42 | **candidate same settings**, subject to explicit P5 approval | no silent tuning for model size |
| Training schedule | 2,000 updates/one epoch | proposed same | must finish exact manifest, no `max_steps=100` |
| Generation | QA prompt, greedy, EOS2, ≤128 new tokens, P4 single-sequence workaround | same | identical decode and output extraction |
| Evaluation | fixed 500+500, P4 text tokenizer/labeler/ontology and bootstrap | same paired protocol | compare only shared IDs |

Primary comparator M1 is the completed P4 projection checkpoint (SHA-256 `7f833eea69fb6d2743b19bbc808be949bb1a5e014bf43a81f08090f9a5f0d8af`); reuse its saved predictions. Secondary M0 uses saved P4 pretrained outputs. Indiana M2/X00 is descriptive because its model and protocol differ. P4 approved contract SHA-256 is `d010d2cf32cb6cc9b96b834e5eb0a5df1d8bd13985ec31cf5faa447260471eb9`. P5 must get its own config/approval hash; the P4 authorization field does **not** authorize P5.

Resume must restore exact manifest permutation, next record, Python/NumPy/PyTorch/CUDA RNG, optimizer, trainable tensors, elapsed budget and source hashes. Every generation must contain model/run/study/image IDs and report text; reject duplicate or unpaired records. Store new artifacts under a separate P5 run directory without overwriting P4.
