# P5.0 resource budget and gates — 2026-10-25

All P5 GPU limits below are **proposals only; HOLD**. Target: one Tesla K80, existing FP16 frozen ViT/OPT and FP32 trainable Q-Former/query/projection. No automatic batch, precision, optimizer, query-count or GPU changes.

| Component | Basis | Planning quantity |
|---|---|---:|
| P4 measured peak reserved | P4.1/2 status | 7.953 GiB |
| Newly trainable Q-Former + queries | 105,162,240 parameters × (16-byte FP32 weight+gradient+two Adam moments minus 2-byte frozen FP16 baseline) | **+1.371 GiB** approximate incremental persistent state |
| P5 P4-based state estimate | 7.953 + 1.371 | **9.324 GiB**, not a measured peak or strict lower bound |
| Extra activation/allocator/Adam transient | no P5 measurement | unknown; rough planning allowance +0.3–2.0 GiB, giving ~9.6–11.4 GiB; can exceed 10.5 GiB gate or physical K80 memory |
| P5 adapter weights | 107,130,880 × 4 bytes | 408.67 MiB |
| P5 optimizer+grad+weights | 107,130,880 × 16 bytes | 1.596 GiB nominal GPU state; checkpoint model+two Adam moments roughly 1.2 GiB before metadata/serialization |

The 7.953 GiB P4 peak already includes full frozen model plus projection training; adding only a state delta avoids double-counting model weights. Peak timing and allocator behavior can differ, and P5 backprop through Q-Former may retain many more activations, so 9.324 GiB must not be used as a PASS prediction. The 10.5 GiB proposed gate leaves only ~1.176 GiB above this comparison estimate. Actual peak reserved, not allocated, controls the gate. Reserve at least ~2.5 GiB disk for rolling/final checkpoints, plus output/log headroom; retaining all 40 periodic 50-step snapshots would cost roughly 48 GiB and is not proposed.

| Phase | Proposed hard gate | Decision |
|---|---|---|
| P5.1 | 20 **actual** updates, ≤900 s cumulative including load/retries, peak reserved ≤10.5 GiB; finite loss and nonzero finite Q-Former/query/projection gradients; unchanged ViT/OPT; all trainables update; checkpoint save/reload | HOLD; separate GO required |
| P5.2 | original checkpoint, exact 2,000-study one-epoch manifest, ≤14,400 s cumulative, one K80; cursor/RNG/order restore and final reload | HOLD; conditional on P5.1 PASS and forecast |
| P5.3 | at most 1,000 **new** Q-Former generations on 500+500, ≤10,800 s cumulative; 10-case validity gate, reuse P4 M0/M1/M2 | HOLD; conditional on P5.2 PASS |

P4.2 measured 4,944.690 s/2,000 = 2.472 s per update including its wall-time overhead. P5 has **no measured update time**. The absolute P5.2 limit is 7.2 s/update only if load/checkpoints take zero time. With 1,800 s held for overhead, a planning target is ≤6.3 s/update; use the measured P5.1 step distribution and load/checkpoint timings to forecast `load + 2,000 × upper-step estimate + checkpoint/verify ≤14,400 s`. If that fails, HOLD P5.2. P4.3 charged 15,243.812 s/2,000 generations = 7.622 s per generation across its mixed arms and retries; 1,000 P5 generations would be ~7,622 s only if P5 inference had similar throughput, leaving ~3,178 s of the proposed 3-hour budget. This is an analogy, not a measurement.

Legacy cu102/K80 FP16 backward, wrapper dtype conversion, OPT prefix autograd, long text, and optimizer allocation may fail. A scientific change such as fewer queries, new optimizer, altered precision or smaller batch requires a separately specified configuration and **new** feasibility gate, never a silent retry. Stop on OOM, budget failure or loss/gradient invalidity; retain atomic status and cumulative time ledger.
