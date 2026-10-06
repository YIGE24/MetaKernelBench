"""MiniMax-M3 main/index QK normalization, RoPE, and dual-cache insert."""

# Source: vLLM
# Reference: vllm-project/vllm@b9b6306ebe0f5fa5046553667192e56bb3934af0
# Workload: tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py
# License: Apache-2.0

import torch

NUM_TOKENS = 513
NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
NUM_INDEX_Q_HEADS = 4
HEAD_DIM = 128
ROTARY_DIM = 64
BLOCK_SIZE = 64
NUM_BLOCKS = 10
MAX_POSITION = 4096
ROPE_BASE = 5_000_000.0
EPS = 1e-6

@torch.no_grad()
def reference(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    index_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sizes = [NUM_Q_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM,
             NUM_KV_HEADS * HEAD_DIM, NUM_INDEX_Q_HEADS * HEAD_DIM, HEAD_DIM]
    q, k, v, iq, ik = qkv.split(sizes, dim=-1)
    q = _norm_rope(q.view(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM), q_norm_weight, positions, cos_sin_cache)
    k = _norm_rope(k.view(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM), k_norm_weight, positions, cos_sin_cache)
    iq = _norm_rope(iq.view(NUM_TOKENS, NUM_INDEX_Q_HEADS, HEAD_DIM), index_q_norm_weight, positions, cos_sin_cache)
    ik = _norm_rope(ik.view(NUM_TOKENS, 1, HEAD_DIM), index_k_norm_weight, positions, cos_sin_cache).squeeze(1)
    v = v.view(NUM_TOKENS, NUM_KV_HEADS, HEAD_DIM)

    blocks = torch.div(slot_mapping, BLOCK_SIZE, rounding_mode="floor")
    offsets = torch.remainder(slot_mapping, BLOCK_SIZE)
    kv_cache[blocks, :, offsets, :HEAD_DIM] = k
    kv_cache[blocks, :, offsets, HEAD_DIM:] = v

    index_blocks = torch.div(index_slot_mapping, BLOCK_SIZE, rounding_mode="floor")
    index_offsets = torch.remainder(index_slot_mapping, BLOCK_SIZE)
    index_cache[index_blocks, index_offsets] = ik
    return q, iq, kv_cache, index_cache

def make_inputs() -> list:
    device = torch.device("cuda")
    width = (NUM_Q_HEADS + 2 * NUM_KV_HEADS + NUM_INDEX_Q_HEADS + 1) * HEAD_DIM
    qkv = torch.randn(NUM_TOKENS, width, device=device, dtype=torch.bfloat16)
    weights = [torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(4)]
    positions = torch.randint(0, MAX_POSITION, (NUM_TOKENS,), device=device, dtype=torch.int64)
    cos_sin_cache = _make_rope_cache(device)
    slot_mapping = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=device)[:NUM_TOKENS]
    index_slot_mapping = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=device)[:NUM_TOKENS]
    kv_cache = torch.randn(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM, device=device, dtype=torch.bfloat16)
    return [qkv, *weights, positions, cos_sin_cache, slot_mapping, index_slot_mapping, kv_cache, index_cache]

def _norm_rope(x: torch.Tensor, weight: torch.Tensor, positions: torch.Tensor,
               cos_sin_cache: torch.Tensor) -> torch.Tensor:
    wide_dtype = torch.promote_types(x.dtype, torch.float32)
    wide = x.to(wide_dtype)
    wide = wide * torch.rsqrt(wide.pow(2).mean(dim=-1, keepdim=True) + EPS)
    normed = (wide * (1.0 + weight.to(wide_dtype))).to(x.dtype)
    cs = cos_sin_cache.index_select(0, positions).to(wide_dtype)
    cos, sin = cs.chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x1 = normed[..., :ROTARY_DIM // 2].to(wide_dtype)
    x2 = normed[..., ROTARY_DIM // 2:ROTARY_DIM].to(wide_dtype)
    rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)
    return torch.cat((rot, normed[..., ROTARY_DIM:]), dim=-1)

def _make_rope_cache(device: torch.device) -> torch.Tensor:
    dims = torch.arange(0, ROTARY_DIM, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (ROPE_BASE ** (dims / ROTARY_DIM))
    pos = torch.arange(MAX_POSITION, device=device, dtype=torch.float32)
    freq = torch.outer(pos, inv_freq)
    return torch.cat((freq.cos(), freq.sin()), dim=-1).to(torch.bfloat16)
