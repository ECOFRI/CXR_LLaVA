# E4-B1 research decision

**E4_B1_PROMPT_ROUTE_REJECTED**. `HOLD_MORE_EVIDENCE`; `HOLD_COMPLEX_FUSION`.

This is a format-selection development gate, not clinical validation. The oracle uses reference-report text and is not image truth. Exact M/S text changes are descriptive only; the frozen sensitivity gate uses extracted finding states.

## Precommitted gate counts

| Format | O collapse | M collapse | M mean mentions | Oracle O/S same cells | M/S changed/eligible | Pass |
|---|---:|---:|---:|---:|---:|---|
| F1 | 10/24 | 17/24 | 0.58/6 | 15/69 vs 8/69 (10.1 pp) | 3/19 | False |
| F2 | 7/24 | 0/24 | 2.42/6 | 10/69 vs 31/69 (-30.4 pp) | 11/19 | False |
| F3 | 0/24 | 0/24 | 2.67/6 | 33/69 vs 36/69 (-4.3 pp) | 13/19 | False |

**VERIFIED:** cohort disjointness, frozen template/hash checks, deterministic smoke status and observed prompt/output/compute artifacts. **AUTOMATED PROXY:** six-finding labeler states and gate rates. **UNVERIFIED:** image-grounded correctness, clinical value and CLEAR benefit. F1 oracle uses categorical table cells because numerical cosine values cannot be fabricated from labels; it is not an exact semantic match to numeric F1-M/S.

The prespecified oracle gate was not met **in this setup**. F2 and F3 nevertheless change extracted finding states between matched and shuffled evidence on 11/19 and 13/19 eligible images, respectively; therefore this decision does not mean the model is completely insensitive to prompt evidence. It means no tested format met the frozen oracle-following threshold. Do not generalize to every prompt or decoding method without a new protocol.

Descriptive exact M/S output-text changes among the same 19 direction-different images: F1 4/19, F2 17/19, F3 15/19. These counts do not enter the pass decision.

E4-B2 recommendation only: to reach 15 reference-positive images per target, this pilot cohort lacks {'Atelectasis': 11, 'Cardiomegaly': 10, 'Consolidation': 11, 'Edema': 11, 'Pleural Effusion': 9, 'Pneumothorax': 11} additional positives (before accounting for overlap and split separation). Keep C0, N1 and a separately selected M/S format after a new gate; oracle is optional and development-only. At 21.08 s/report on four K80 devices, a four-arm cohort of N images needs roughly 84.3N seconds of model-call time plus setup and evaluation. A new cohort/split and GPU budget require separate authorization. No E4-B2 run started.
