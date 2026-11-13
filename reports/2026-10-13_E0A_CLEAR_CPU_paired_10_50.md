# Experiment Analysis Log

## 1. Experiment Information

Run `20261013T203617_E0A_CLEAR_CPU_10_50` COMPLETE; nested gates 10 then 50. Approved CPU continuation, existing env reused; no install/training/GPU job. Commit `f6535c6d9d7eb23c2e9a2ac24f48771a1d276efc` plus runner/config/input/source hashes in manifest.

## 2. Objective

Validate paired frozen representation extraction at 10/50 and obtain measured post-warm-up CPU resources, minimal reusable features and diverse technical evidence.

## 3. Survey / Technical Motivation

Released CLEAR official loader/encode_image and native DINOv2 forward_features; original source unmodified. Reuse prior CXR E0A artifacts. This is a representation harness, not an evaluation of pathology recognition.

## 4. Environment

Existing `/storage/student4/hanhpm/CXR_LLaVA_Improvement/research/envs/cxr-clear-cpu`; torch 2.3.1+cpu; FP32 CPU, 4 intra-op / 1 inter-op threads. Prior environment manifest linked and hashed; no repeated comprehensive inventory.

## 5. Dependency Setup

No dependencies installed/reinstalled. GPU hidden; CUDA build absent. Local checkpoint/architecture cache used without downloads.

## 6. Code / Notebook Changes

Added isolated paired CPU gate runner and report/evidence generator. Original CLEAR/CXR/DINO implementations, original caches and successful runs preserved.

## 7. Experiment Configuration

Exactly the 50 IDs/studies/hashes used by CXR E0A, in its original order. First 10 form gate 10; remaining 40 continue only after checkpoint, shape, finite/repeat/API checks and RAM gate pass. Batch=1; seed42; 3 untimed warm-ups on first paired image. Virtual address space capped at 8 GiB plus per-phase/sample RSS checks, stop on any mismatch. Only three features cached; no register/mean/attention artifacts.

## 8. Code Executed

From project root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 research/envs/cxr-clear-cpu/bin/python -m research.clear_cpu_gates
```

Creates a new run; do not duplicate completed work unless needed for reproducibility.

## 9. Output

Run `/storage/student4/hanhpm/CXR_LLaVA_Improvement/research/runs/20261013T203617_E0A_CLEAR_CPU_10_50`; cached features 150.950 MiB, 50 files. Source/mapping/cache hashes, sample timings, staged metrics, shape checks, logs and registry saved. Earlier one-image and CXR caches untouched.

## 10. Output Analysis

Gate10 mean 1.009615s; gate50 mean 1.013899s. All 50 native/official outputs finite and shape-consistent; max repeat=0.0, max official API difference=0.0. Checkpoint/source/env unchanged. Registers [1,4,768] verified on every image, not cached.

## 11. Quantitative Results

| Thành phần | CXR-LLaVA | CLEAR |
|---|---|---|
| Native CLS feature | 1024-D | 768-D |
| Native patch features | 1024 x 1024 (32x32 spatial grid) | 1024 x 768 (32x32 spatial grid) |
| Register tokens | 0 dedicated registers; 1 CLS | 4 (verified; not cached) |
| Projected global feature | 128-D tower global head | 768-D CLEAR visual projection |
| Preprocessing | 512x512 grayscale; aspect resize + center crop | 448x448 RGB; exact square resize |
| Precision / device | FP32 / Tesla K80 GPU0 | FP32 / CPU, 4 intra-op threads |
| Mean forward latency | 0.532728 s (historical gate; no explicit warm-up) | 1.013899 s (3 untimed warm-up forwards) |
| Peak memory | 1.339895 GiB GPU allocated; 1.394531 GiB reserved | 1.395298 GiB process peak RSS |
| Wall time | 66.019 s historical harness | 181.347 s current harness; 3 forwards/image for verification |
| Determinism | PASS at gate endpoints 1/10/50; max difference=0 | PASS on all 50 images; max difference=0.0 |
| Official output consistency | Native patches equal captured final normalized sequence | encode_image agreement on all 50; max difference=0.0 |
| Cached fields in this CLEAR run | Existing CXR cache preserved | CLS + projected global + native patches only |

CLEAR median 1.011075s, p95 1.032024s; warm-up 3.060s; model loading 3.301s; RAM budget8GiB. Latency excludes preprocessing, validation/repeat/API checks and disk IO. Wall includes these checks but excludes setup/import overhead before harness starts. [CPU resource figure](/storage/student4/hanhpm/CXR_LLaVA_Improvement/research/figures/2026-10-13/20261013T203617_E0A_CLEAR_CPU_10_50/cpu_latency_ram.png).

## 12. Qualitative Results

Five [paired contact-sheet pages](/storage/student4/hanhpm/CXR_LLaVA_Improvement/research/figures/2026-10-13/20261013T203617_E0A_CLEAR_CPU_10_50) cover all 50 images. Each image is linked to both feature caches and native preprocessing metadata. `12` diverse review candidates selected from low/mid/high preview contrast x official GT device flag; original dimensions/aspect ratio included. [Review metadata](/storage/student4/hanhpm/CXR_LLaVA_Improvement/research/runs/20261013T203617_E0A_CLEAR_CPU_10_50/technical_review.csv). Contrast/geometry are technical proxies, not validated quality scores. Device GT does not prove vector recognition; AP/PA posture unverified because source metadata lacks it. Burned-in view/position annotations remain for manual review, not inferred labels. All are frontal source images; no clinical adjudication or lesion heatmap.

## 13. Comparison with Baseline

CPU CLEAR timings and GPU CXR timings use different hardware, output heads and warm-up protocols. Do not rank encoder efficiency from this table. A fair efficiency study requires identical hardware/precision/threads/batch/warm-up/timing scope. Global head projections also have different training objectives; dimensions do not establish clinical superiority.

## 14. Problems Encountered

No numerical/shape/checkpoint/resource errors. xFormers unavailable warnings use standard CPU SDPA; no code patch. AP/PA metadata and pixel-level clinical quality adjudication are missing; do not fabricate posture strata. No re-selection of cohort for examples.

## 15. Conclusion

Both paired encoder harnesses now pass technical 50-image gates. GO for technical acceptance and independent data-protocol preparation. H1 encoder superiority and H2 concept utility remain inconclusive. No accuracy claim from vectors or equipment-heavy examples.

## 16. Next Experiment

Review paired metadata/caches and frozen dataset split risks before E1/E2. Any efficiency comparison must use matched hardware/protocol. No full-cohort or training run started.

## 17. Reproducibility Checklist

50/50 identity/study/hash match; cached tensors reloaded and verified; original env/source fingerprints verified; staged pass artifacts, warm-up timing policy, RSS ceiling, checkpoint hashes, image evidence, command and logs retained.

## 18. Short Lab Update

CLEAR CPU passed 10 then 50 on exact CXR E0A IDs. Post-warm-up mean 1.013899s, peak RSS 1.395GiB. Minimal cache 150.9MiB. Paired technical table/evidence ready; no clinical or cross-hardware efficiency conclusions.


### Position annotation review within the unchanged cohort

Two source images were inspected at original resolution for literal burned-in text: `e981c159cf28f6da`: **AP PORT SUPINE**; `8da2100d6ad45c72`: **UPRIGHT; AP**. These provide technical examples of differing position annotations in the same 50 IDs, supplementing the 12 contrast/device-GT review candidates. This transcribes source text; it is not independent clinical/DICOM validation of acquisition posture. Evidence pointers/hashes are in `manual_position_annotation_review.csv`; no new images or inference were added.
