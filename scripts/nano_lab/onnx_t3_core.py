"""Shared Nano GPT2 export wrapper without speech-package imports."""
from __future__ import annotations
from typing import Any
import torch


class NanoT3KV(torch.nn.Module):
    """Single graph for both prefill and one-token KV-cache decode.

    ``inputs_embeds`` is already assembled by the caller.  A zero-length
    ``past_*`` tensor selects prefill; a non-empty cache selects autoregressive
    decode.  This avoids storing two copies of the 110M Nano T3 weights in a
    long-lived ONNX Runtime process.
    """

    def __init__(self, t3: Any):
        super().__init__()
        self.t3 = t3

    def forward(self, inputs_embeds: torch.Tensor, cache_position: torch.Tensor, *past: torch.Tensor) -> tuple[torch.Tensor, ...]:
        positions = cache_position.to(dtype=torch.long)
        hidden = inputs_embeds + self.t3.tfmr.wpe(positions).unsqueeze(0).to(inputs_embeds.device)
        hidden = self.t3.tfmr.drop(hidden)
        present: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, block in enumerate(self.t3.tfmr.h):
            key_past = past[layer_index * 2]
            value_past = past[layer_index * 2 + 1]
            residual = hidden
            hidden_norm = block.ln_1(hidden)
            query, key, value = block.attn.c_attn(hidden_norm).split(block.attn.split_size, dim=2)
            shape = (*query.shape[:-1], -1, block.attn.head_dim)
            query = query.view(shape).transpose(1, 2)
            key = key.view(shape).transpose(1, 2)
            value = value.view(shape).transpose(1, 2)
            key = torch.cat((key_past, key), dim=2)
            value = torch.cat((value_past, value), dim=2)
            scale = float(block.attn.head_dim) ** -0.5 if block.attn.scale_attn_weights else 1.0
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
            # Causal positions are dynamic.  For one-token decode this mask is
            # all true; for prefill it is the standard lower triangle.
            past_len = key_past.shape[2]
            query_len = hidden.shape[1]
            key_positions = torch.arange(past_len + query_len, device=hidden.device)
            query_positions = torch.arange(past_len, past_len + query_len, device=hidden.device)
            causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e30)
            weights = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
            attn = torch.matmul(weights, value)
            attn = attn.transpose(1, 2).contiguous().reshape(attn.shape[0], attn.shape[2], -1)
            attn = block.attn.c_proj(attn)
            hidden = residual + attn
            residual = hidden
            hidden = block.ln_2(hidden)
            hidden = block.mlp(hidden)
            hidden = residual + hidden
            present.append((key, value))
        hidden = self.t3.tfmr.ln_f(hidden)
        logits = self.t3.speech_head(hidden[:, -1, :])
        return (logits, *[value for pair in present for value in pair])

