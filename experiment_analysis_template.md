# Experiment Analysis Log

## 1. Experiment Information

- **Experiment ID:** 
- **Date:** 
- **Researcher:** 
- **Project:** CXR_LLaVA_Improvement
- **Repository:** https://github.com/hanhpm/CXR_LLaVA_Improvement
- **Branch:** 
- **Commit:** 
- **Notebook / Script:** 
- **Status:** Planned / Running / Completed / Failed

---

## 2. Objective

Describe the main objective of this experiment.

Example:

> Evaluate whether CXR-LLaVA can be loaded and executed successfully in Google Colab before applying further model improvements.

---

## 3. Survey / Technical Motivation

### References reviewed

- Paper / documentation:
- Repository:
- Model card:
- Related implementation:

### Key idea

Summarize the technique or implementation idea that motivates this experiment.

### Hypothesis

State what you expect to observe.

Example:

> The model should run successfully with a compatible Transformers version and a small compatibility patch for the tokenizer loader.

---

## 4. Environment

| Component | Version / Configuration |
|---|---|
| Platform | Google Colab |
| OS | |
| GPU | |
| Python | |
| CUDA | |
| PyTorch | |
| Transformers | |
| Tokenizers | |
| Hugging Face Hub | |
| Accelerate | |
| BitsAndBytes | |
| Protobuf | |
| Model | ECOFRI/CXR-LLAVA-v2 |

### Environment verification

```text
Paste the environment-check output here.
```

---

## 5. Dependency Setup

### Installation command

```bash
# Paste the exact pip/uv/conda commands used for this experiment.
```

### Dependency issues

Record dependency conflicts or warnings here.

Example:

```text
diffusers requires a newer huggingface-hub version.
gradio requires a newer huggingface-hub version.
```

### Decision

Explain whether the conflict affects the current experiment.

Example:

> The conflicts are unrelated to CXR-LLaVA inference because neither diffusers nor gradio is used in this notebook.

---

## 6. Code / Notebook Changes

### Files changed

- 
- 
- 

### Main change

Describe exactly what was changed.

Example:

```python
kwargs.pop("add_special_tokens", None)
```

Reason:

> CXR-LLaVA passes `add_special_tokens=False` to `LlamaTokenizer.from_pretrained()`, which conflicts with newer Transformers tokenizer validation.

### Why this change is necessary

Explain whether this is:

- compatibility fix
- bug fix
- model modification
- optimization
- experimental method

---

## 7. Experiment Configuration

| Parameter | Value |
|---|---|
| Input data | |
| Number of samples | |
| Image size | |
| Prompt | |
| Precision | FP16 / BF16 / FP32 / INT8 / INT4 |
| Quantization | None / NF4 / GPTQ / AWQ |
| Device map | |
| Max new tokens | |
| Batch size | |
| Seed | |

Add additional parameters if required.

---

## 8. Code Executed

Paste only the important experiment code, not the entire notebook.

```python
# Main experiment code
```

---

## 9. Output

### Raw output

```text
Paste the important notebook output here.
```

### Errors / warnings

```text
Paste relevant errors or warnings here.
```

Avoid pasting unrelated Colab logs.

---

## 10. Output Analysis

### What worked

- 
- 
- 

### What failed

- 
- 
- 

### Root cause

Describe the technical root cause, not only the error message.

Example:

> The model weights downloaded successfully, but model initialization failed because the custom CXR-LLaVA tokenizer loader passed `add_special_tokens` as a constructor argument. Current Transformers versions reject this because `add_special_tokens` is already a tokenizer method.

### Fix applied

- 
- 

### Result after fix

- 
- 

---

## 11. Quantitative Results

| Metric | Baseline | Current Experiment | Difference |
|---|---:|---:|---:|
| GPU memory | | | |
| Model load time | | | |
| Inference latency | | | |
| Throughput | | | |
| BLEU | | | |
| ROUGE-L | | | |
| Other metric | | | |

Use only metrics relevant to the experiment.

---

## 12. Qualitative Results

### Input

```text
Describe the test input or image.
```

### Model output

```text
Paste the generated report / answer.
```

### Observation

- Correct findings:
- Missing findings:
- Incorrect findings:
- Hallucinated findings:
- Language / formatting issues:

---

## 13. Comparison with Baseline

Describe what changed relative to the previous experiment.

| Aspect | Previous | Current |
|---|---|---|
| Model | | |
| Environment | | |
| Method | | |
| Memory | | |
| Latency | | |
| Output quality | | |

### Interpretation

Explain whether the change represents an improvement, regression, or inconclusive result.

---

## 14. Problems Encountered

### Problem 1

**Error**

```text
Paste error.
```

**Cause**

-

**Solution**

-

**Status:** Fixed / Unresolved / Workaround

### Problem 2

**Error**

```text
Paste error.
```

**Cause**

-

**Solution**

-

**Status:** Fixed / Unresolved / Workaround

---

## 15. Conclusion

Summarize the experiment in 3-5 sentences.

Recommended structure:

1. What was tested.
2. Whether the experiment succeeded.
3. Main quantitative or qualitative result.
4. Important limitation.
5. Whether the method should be continued.

---

## 16. Next Experiment

- [ ] 
- [ ] 
- [ ] 

Example:

- [ ] Establish a clean FP16 baseline.
- [ ] Record peak GPU memory and inference latency.
- [ ] Evaluate the baseline on a fixed CXR validation subset.
- [ ] Compare FP16 against 4-bit NF4 quantization.
- [ ] Measure report-quality degradation after quantization.

---

## 17. Reproducibility Checklist

- [ ] Repository URL recorded
- [ ] Branch recorded
- [ ] Git commit recorded
- [ ] Notebook/script recorded
- [ ] Environment versions recorded
- [ ] Dataset/input version recorded
- [ ] Random seed recorded where applicable
- [ ] Exact configuration recorded
- [ ] Raw output saved
- [ ] Metrics saved
- [ ] Errors and fixes documented
- [ ] Final conclusion recorded

---

## 18. Short Lab Update

Use this section when reporting to the professor.

> **Objective:**  
> 
> **Survey:**  
> 
> **Implementation:**  
> 
> **Experiment:**  
> 
> **Result:**  
> 
> **Issue:**  
> 
> **Next step:**  

This section should normally fit within one short paragraph or one presentation slide.
