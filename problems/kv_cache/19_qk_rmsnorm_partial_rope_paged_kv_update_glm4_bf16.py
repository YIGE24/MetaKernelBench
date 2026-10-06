"""GLM-4.5 MoE Q/K RMSNorm, partial NeoX RoPE, and paged KV-cache update."""

# Source: vLLM
# Reference: vllm-project/vllm@b9b6306ebe0f5fa5046553667192e56bb3934af0
# Workload: tests/compile/passes/test_rocm_aiter_qk_norm_rope_kvcache_fusion.py
# License: Apache-2.0

import torch

NUM_TOKENS = 2048
MAX_PAD_TOKENS = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROTARY_DIM = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 256
MAX_POSITION_EMBEDDINGS = 4096
ROPE_BASE = 10000.0
EPS = 1e-5

@torch.no_grad()
def reference(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
    k = k.view(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(num_tokens, NUM_KV_HEADS, HEAD_DIM)

    q = _rms_norm_per_head(q, q_norm_weight, eps)
    k = _rms_norm_per_head(k, k_norm_weight, eps)

    cos_sin = cos_sin_cache.index_select(0, positions.flatten())
    cos, sin = cos_sin.chunk(2, dim=-1)
    q = _apply_partial_neox_rope(q, cos, sin)
    k = _apply_partial_neox_rope(k, cos, sin)

    valid = slot_mapping >= 0
    valid_slots = slot_mapping[valid]
    block_indices = torch.div(valid_slots, BLOCK_SIZE, rounding_mode="floor")
    block_offsets = torch.remainder(valid_slots, BLOCK_SIZE)
    kv_cache[0, block_indices, block_offsets] = k[valid]
    kv_cache[1, block_indices, block_offsets] = v[valid]
    return q, kv_cache

def make_inputs() -> list:
    device = torch.device("cuda")
    qkv_width = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    qkv = torch.randn((NUM_TOKENS, qkv_width), device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, MAX_POSITION_EMBEDDINGS, (NUM_TOKENS,), device=device, dtype=torch.int64)
    q_norm_weight = torch.randn((HEAD_DIM,), device=device, dtype=torch.bfloat16)
    k_norm_weight = torch.randn((HEAD_DIM,), device=device, dtype=torch.bfloat16)
    cos_sin_cache = _make_cos_sin_cache(device)
    slot_mapping = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=device)[:NUM_TOKENS]
    num_pads = int(torch.randint(1, MAX_PAD_TOKENS + 1, (1,), device=device).item())
    slot_mapping[NUM_TOKENS - num_pads:] = -1
    kv_cache = torch.randn((2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
    return [qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, slot_mapping, kv_cache, EPS]

def _rms_norm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    wide = x.to(torch.promote_types(x.dtype, torch.float32))
    variance = wide.pow(2).mean(dim=-1, keepdim=True)
    normalized = wide * torch.rsqrt(variance + eps)
    return (normalized.to(weight.dtype) * weight).to(x.dtype)

def _apply_partial_neox_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_rot = x[..., :ROTARY_DIM]
    x_pass = x[..., ROTARY_DIM:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return torch.cat((rotated, x_pass), dim=-1)

def _make_cos_sin_cache(device: torch.device) -> torch.Tensor:
    rotary_indices = torch.arange(0, ROTARY_DIM, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (ROPE_BASE ** (rotary_indices / ROTARY_DIM))
    positions = torch.arange(MAX_POSITION_EMBEDDINGS, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(torch.bfloat16)
