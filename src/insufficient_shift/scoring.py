from __future__ import annotations

from collections.abc import Sequence

import torch

from .modeling import LoadedModel, find_final_norm
from .schema import LayerScore


def _single_token_id(tokenizer, token_text: str, *, role: str) -> int:
    if not token_text or not token_text.strip():
        raise ValueError(f"{role} token must not be empty or whitespace-only")
    ids = tokenizer.encode(token_text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"{role} token {token_text!r} maps to {len(ids)} tokens ({ids}), not one. "
            "Choose another value with the corresponding CLI option."
        )
    return ids[0]


@torch.inference_mode()
def score_labels_by_layer(
    loaded: LoadedModel,
    prompt: str,
    *,
    sufficient_token: str = " A",
    insufficient_token: str = " B",
    last_k: int | None = None,
) -> list[LayerScore]:
    """Read a constrained A/B decision from each layer through a raw logit lens.

    Both labels must be exactly one tokenizer token. A context therefore needs one
    forward pass, and every layer is compared at the same final prompt position.
    The reported probability is conditional on choosing one of the two label
    tokens; the logit margin remains the primary, unsaturated score.
    """
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokenized to an empty sequence")

    sufficient_id = _single_token_id(tokenizer, sufficient_token, role="Sufficient")
    insufficient_id = _single_token_id(tokenizer, insufficient_token, role="Insufficient")
    if sufficient_id == insufficient_id:
        raise ValueError("Sufficient and insufficient labels map to the same token ID")

    input_ids = torch.tensor([prompt_ids], device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states: Sequence[torch.Tensor] = outputs.hidden_states
    final_norm = find_final_norm(model)
    lm_head = model.get_output_embeddings()

    # hidden_states[0] is the embedding output; indices 1..N are block outputs.
    n_layers = len(hidden_states) - 1
    first_layer = max(0, n_layers - last_k) if last_k else 0
    results: list[LayerScore] = []
    for zero_idx in range(first_layer, n_layers):
        layer_idx = zero_idx + 1
        projected = hidden_states[layer_idx][:, -1, :]
        # The last returned state is already final-normalized in HF Llama/Qwen.
        if final_norm is not None and layer_idx < n_layers:
            projected = final_norm(projected)
        logits = lm_head(projected)[0].float()
        label_logits = torch.stack((logits[sufficient_id], logits[insufficient_id]))
        p_insufficient = torch.softmax(label_logits, dim=0)[1]
        margin = label_logits[1] - label_logits[0]
        results.append(
            LayerScore(
                layer=layer_idx,
                sufficient_logit=label_logits[0].item(),
                insufficient_logit=label_logits[1].item(),
                insufficient_logit_margin=margin.item(),
                p_insufficient_conditional=p_insufficient.item(),
            )
        )
    return results
