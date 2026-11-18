"""Pinned recovery loader and report-helper prompt audit (no inference in audit)."""
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

MODEL_REVISION = "b2224786bb90d54b1e1291171866706cfbb44e2b"


def report_prompt_audit(model, output):
    """Capture the actual helper chat without invoking image processing/generation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    captured = {}

    def capture(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        captured.update({k: v for k, v in bound.arguments.items() if k != "image"})
        return "PROMPT_AUDIT_ONLY"

    original = model.generate_cxr_repsonse
    with patch.object(model, "generate_cxr_repsonse", new=capture):
        model.write_radiologic_report(None)
    rendered = model.apply_chat_template(captured["chat"])
    if not isinstance(rendered, str) or "<image>" not in rendered:
        raise ValueError("Report helper did not render an image prompt")
    (output / "report_prompt.txt").write_text(rendered, encoding="utf-8")
    captured.update(
        rendered_prompt=rendered,
        template=model.tokenizer.chat_template,
        prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        model_revision=MODEL_REVISION,
        generation_config=model.generation_config.to_dict(),
        audit_kind="runtime_helper_capture_without_generation",
    )
    for name in ("write_radiologic_report", "apply_chat_template", "generate_cxr_repsonse"):
        source = inspect.getsource(getattr(model, name))
        (output / f"{name}_source.py").write_text(source, encoding="utf-8")
        captured[name + "_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    if hasattr(model, "tokenizer_image_token"):
        captured["input_token_ids"] = model.tokenizer_image_token(rendered, model.tokenizer)
    (output / "report_prompt_audit.json").write_text(
        json.dumps(captured, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return captured


def load_pinned_model(model_id):
    import torch
    import transformers
    from importlib.metadata import version
    expected = "4.46.3"
    installed = version("transformers")
    if transformers.__version__ != expected or installed != expected:
        raise RuntimeError(
            f"Expected transformers=={expected}; loaded={transformers.__version__}, "
            f"installed={installed}. Run updated Cell 6, restart session, "
            "then rerun Cells 1-5 and 7 (skip Cell 6 after restart)."
        )
    from transformers import AutoModel, BitsAndBytesConfig, GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required; no CPU model fallback")
    tokenizer_loader = transformers.LlamaTokenizer.from_pretrained
    generation_loader = GenerationConfig.from_pretrained

    def tokenizer_compat(*args, **kwargs):
        kwargs.pop("add_special_tokens", None)
        kwargs["revision"] = MODEL_REVISION
        return tokenizer_loader(*args, **kwargs)

    def pinned_generation(*args, **kwargs):
        kwargs["revision"] = MODEL_REVISION
        return generation_loader(*args, **kwargs)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["vision_tower", "mm_projector", "lm_head"],
    )
    with patch.object(transformers.LlamaTokenizer, "from_pretrained", new=tokenizer_compat), \
         patch.object(GenerationConfig, "from_pretrained", new=pinned_generation):
        model = AutoModel.from_pretrained(
            model_id, revision=MODEL_REVISION, code_revision=MODEL_REVISION,
            trust_remote_code=True, quantization_config=quantization,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0},
        )
    model.eval()
    return model
