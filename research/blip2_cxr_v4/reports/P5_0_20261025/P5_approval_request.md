# P5.0 decision bundle for a future P5.1 approval — 2026-10-25

**Current decision: GO P5.0 CPU design completed; HOLD P5.1/P5.2/P5.3. This file grants no GPU authorization.**

## Proposed future request

Approve **P5.1 only** after a new P5-specific executable contract and CPU tests are reviewed. Proposed scope: one Tesla K80; original pinned BLIP-2 OPT-2.7B; pretrained Q-Former, query tokens and language projection trainable FP32; vision/OPT frozen FP16 eval; exact P4 2,000-study train manifest and target/prompt/tokenizer/processor rules; 20 actual updates; ≤15 cumulative minutes; peak reserved ≤10.5 GiB; optimizer/update/frozen/reload checks. No P4 artifact writes, no weight download or environment change. P5.1 adapter is a feasibility artifact, **not** the P5.2 initialization.

## Approval prerequisites, not silently assumed

1. Implement and review a P5-specific trainer/gate: P4 target/EOS masking and 256 text/128 target caps; exact allowlist and frozen eval modes; pretrained initialization; deterministic order and full RNG/data-cursor resume; checkpoint/manifest/config/source hashes; atomic status, GPU lease and cumulative watchdog. Replace the legacy 100-step bridge path. Verify CPU tests.
2. Freeze and explicitly approve P5 optimizer settings. P4 AdamW LR 1e-4, β=(.9,.999), ε=1e-8, weight decay .01, constant scheduler, batch/accumulation 1, seed 42 are the **matched candidate**, but their suitability for 107M trainable parameters is unmeasured. Record any scientific change before execution.
3. Specify P5.1 output directory and new run ID; require a fresh initialization check and no reuse of the P4.2 adapter. Verify model/processor revision, checkpoint shard identity, config and manifest hashes.
4. Agree to hard-stop on peak reserved >10.5 GiB, nonfinite/zero gradients, changed frozen parameters, failed checkpoint reload, >900 cumulative seconds, or incompatible CUDA operations. No automatic architecture/resource workaround.

If P5.1 passes, review its measured memory and throughput against `P5_resource_budget.md` before seeking **separate conditional approvals** for P5.2 and P5.3. Their proposed ceilings are 2,000 updates/4 cumulative hours and 1,000 new generations/3 cumulative hours, respectively; neither is approved by this request. P4's technical PASS and BLEU gain motivate the test but do not predict clinical benefit. The hypothesis is Q-Former adaptation, not CLEAR concept grounding.
