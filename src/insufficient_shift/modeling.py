from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device


def load_model(model_name_or_path: str, *, dtype: str = "auto") -> LoadedModel:
    """Load a Hugging Face decoder-only LM with hidden-state output enabled."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    requested_dtype: str | torch.dtype
    if dtype == "auto":
        requested_dtype = "auto"
    else:
        requested_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=requested_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def find_final_norm(model: Any) -> Any | None:
    """Resolve the common final-normalization module used by Llama/Qwen-like LMs."""
    candidates = (
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("model", "decoder", "final_layer_norm"),
    )
    for path in candidates:
        node = model
        try:
            for name in path:
                node = getattr(node, name)
            return node
        except AttributeError:
            continue
    return None


def render_prompt(
    tokenizer: Any,
    system_prompt: str,
    question: str,
    context: str,
    *,
    sufficient_token: str = " A",
    insufficient_token: str = " B",
) -> str:
    sufficient_display = sufficient_token.strip()
    insufficient_display = insufficient_token.strip()
    user = (
        f"Question:\n{question}\n\nContext:\n{context}\n\n"
        f"{sufficient_display} = Sufficient\n"
        f"{insufficient_display} = Insufficient\n\n"
        f"Answer with exactly {sufficient_display} or {insufficient_display}.\nLabel:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_prompt}\n\nUser: {user}\n\nAssistant:"
