"""
sampling.py
-----------
Decoding strategies for next-word selection during generation.

All strategies take logits (1-D tensor of size vocab_size, or 2-D batch)
and return a sampled token index. They are pure functions and can be
swapped at generation time without retraining.

Strategies:
  - proportional   : sample from softmax(logits) directly.
  - temperature    : softmax(logits / T) - higher T = more uniform.
  - top_k          : zero out everything except top K, then sample.
  - nucleus (top_p): sample from the smallest set whose cumulative
                     probability exceeds p.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_proportional(logits: torch.Tensor) -> torch.Tensor:
    """
    Sample from the direct softmax distribution.
    logits: (V,) or (B, V)
    Returns: (1,) or (B,) Long tensor.
    """
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Scale logits by 1/T before softmax. T < 1 sharpens, T > 1 flattens.
    """
    if temperature <= 0:
        # Greedy (argmax) - undefined for T<=0, treat as greedy
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_k(logits: torch.Tensor, k: int = 40,
                  temperature: float = 1.0) -> torch.Tensor:
    """
    Keep only the top K logits, mask the rest to -inf, sample from softmax.
    """
    if k <= 0:
        return sample_temperature(logits, temperature)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    k = min(k, logits.size(-1))
    top_vals, _ = torch.topk(logits, k, dim=-1)
    cutoff = top_vals[..., -1].unsqueeze(-1)
    masked = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    masked = masked / temperature if temperature > 0 else masked
    probs = F.softmax(masked, dim=-1)
    out = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return out.squeeze(0) if squeeze else out


def sample_nucleus(logits: torch.Tensor, p: float = 0.9,
                    temperature: float = 1.0) -> torch.Tensor:
    """
    Top-p (nucleus) sampling. Keeps the smallest set of tokens whose
    cumulative probability exceeds p.
    """
    if p <= 0 or p >= 1:
        return sample_temperature(logits, temperature)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    scaled = logits / temperature if temperature > 0 else logits
    probs = F.softmax(scaled, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Mask: keep tokens until cumulative > p (include the boundary token)
    mask = cumulative > p
    # Shift right so the first token over the threshold is still kept
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    # Sample from the sorted distribution, then map back
    sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
    sampled = sorted_idx.gather(-1, sampled_pos).squeeze(-1)
    return sampled.squeeze(0) if squeeze else sampled


def sample(strategy: str, logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Dispatch by strategy name. Supported:
      - "proportional"
      - "temperature" (kwarg: temperature)
      - "top_k"       (kwargs: k, optional temperature)
      - "nucleus"     (kwargs: p, optional temperature)
    """
    if strategy == "proportional":
        return sample_proportional(logits)
    if strategy == "temperature":
        return sample_temperature(logits, kwargs.get("temperature", 1.0))
    if strategy == "top_k":
        return sample_top_k(logits, kwargs.get("k", 40),
                              kwargs.get("temperature", 1.0))
    if strategy == "nucleus":
        return sample_nucleus(logits, kwargs.get("p", 0.9),
                                kwargs.get("temperature", 1.0))
    raise ValueError(f"Unknown sampling strategy: {strategy}")
