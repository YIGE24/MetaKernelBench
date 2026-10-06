"""MLA projection, partial RoPE, and compressed paged-cache update."""

# Source: vLLM
# Reference: vllm-project/vllm@b9b6306ebe0f5fa5046553667192e56bb3934af0
# Workload: tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py
# License: Apache-2.0

import torch
import torch.nn.functional as F

NUM_TOKENS = 2048
MAX_PAD_TOKENS = 128
NUM_HEADS = 16
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_DIM = 128
ROPE_DIM = 64
QK_HEAD_DIM = QK_NOPE_DIM + ROPE_DIM
BLOCK_SIZE = 16
NUM_BLOCKS = 256
MAX_POSITION = 4096
ROPE_BASE = 10000.0

@torch.no_grad()
def reference(
    qkv_lora: torch.Tensor,
    q_b_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_c, kv_and_rope = qkv_lora.split([Q_LORA_RANK, KV_LORA_RANK + ROPE_DIM], -1)
    kv_c, k_pe = kv_and_rope.split([KV_LORA_RANK, ROPE_DIM], -1)

    q = F.linear(q_c, q_b_weight)
    q = q.view(NUM_TOKENS, NUM_HEADS, QK_HEAD_DIM)
    q_nope, q_pe = q.split([QK_NOPE_DIM, ROPE_DIM], dim=-1)

    cs = cos_sin_cache.index_select(0, positions)
    cos, sin = cs.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_pe = _rope(q_pe, cos, sin)
    k_pe = _rope(k_pe.unsqueeze(1), cos, sin).squeeze(1)
    q_out = torch.cat((q_nope, q_pe), dim=-1)

    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    blocks = torch.div(slots, BLOCK_SIZE, rounding_mode="floor")
    offsets = torch.remainder(slots, BLOCK_SIZE)
    kv_cache[blocks, offsets, 0, :KV_LORA_RANK] = kv_c[valid]
    kv_cache[blocks, offsets, 0, KV_LORA_RANK:] = k_pe[valid]
    return q_out, kv_cache

def make_inputs() -> list:
    device = torch.device("cuda")
    qkv_lora = torch.randn(NUM_TOKENS, Q_LORA_RANK + KV_LORA_RANK + ROPE_DIM, device=device, dtype=torch.bfloat16)
    q_b_weight = torch.randn(NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK, device=device, dtype=torch.bfloat16) * 0.02
    positions = torch.randint(0, MAX_POSITION, (NUM_TOKENS,), device=device, dtype=torch.int64)
    cos_sin_cache = _make_rope_cache(device)
    slot_mapping = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=device)[:NUM_TOKENS]
    num_pads = int(torch.randint(1, MAX_PAD_TOKENS + 1, (1,), device=device).item())
    slot_mapping[NUM_TOKENS - num_pads:] = -1
    kv_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, 1, KV_LORA_RANK + ROPE_DIM, device=device, dtype=torch.bfloat16)
    return [qkv_lora, q_b_weight, positions, cos_sin_cache, slot_mapping, kv_cache]

def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

def _make_rope_cache(device: torch.device) -> torch.Tensor:
    dims = torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (ROPE_BASE ** (dims / ROPE_DIM))
    pos = torch.arange(MAX_POSITION, device=device, dtype=torch.float32)
    freq = torch.outer(pos, inv_freq)
    return torch.cat((freq.cos(), freq.sin()), dim=-1).to(torch.bfloat16)
