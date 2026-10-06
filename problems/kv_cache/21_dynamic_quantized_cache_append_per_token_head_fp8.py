"""Per-token-head FP8 dynamic quantization and paged KV-cache append."""

# Source: vLLM
# Reference: vllm-project/vllm@b9b6306ebe0f5fa5046553667192e56bb3934af0
# Workload: tests/quantization/test_per_token_kv_cache.py; vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
# License: Apache-2.0

import torch

NUM_TOKENS = 2048
MAX_PAD_TOKENS = 128
NUM_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 256
FP8_MAX = 448.0
FP8_MIN = -448.0

@torch.no_grad()
def reference(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qk, sk = _quantize(key)
    qv, sv = _quantize(value)
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    blocks = torch.div(slots, BLOCK_SIZE, rounding_mode="floor")
    offsets = torch.remainder(slots, BLOCK_SIZE)
    key_cache[blocks, offsets] = qk[valid].view(torch.uint8)
    value_cache[blocks, offsets] = qv[valid].view(torch.uint8)
    k_scale_cache[blocks, offsets] = sk[valid]
    v_scale_cache[blocks, offsets] = sv[valid]
    return key_cache, value_cache, k_scale_cache, v_scale_cache

def make_inputs() -> list:
    device = torch.device("cuda")
    key = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slot_mapping = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=device)[:NUM_TOKENS]
    num_pads = int(torch.randint(1, MAX_PAD_TOKENS + 1, (1,), device=device).item())
    slot_mapping[NUM_TOKENS - num_pads:] = -1
    key_cache = torch.randint(0, 256, (NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM), device=device, dtype=torch.uint8)
    value_cache = torch.randint_like(key_cache, 0, 256)
    k_scale_cache = torch.rand(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, device=device, dtype=torch.float32) + 0.5
    v_scale_cache = torch.rand_like(k_scale_cache) + 0.5
    return [key, value, slot_mapping, key_cache, value_cache, k_scale_cache, v_scale_cache]

def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    wide = x.to(torch.promote_types(x.dtype, torch.float32))
    absmax = wide.abs().amax(dim=-1)
    scale = (absmax / FP8_MAX).clamp(min=1e-6)
    q = (wide * (1.0 / scale.unsqueeze(-1))).clamp(FP8_MIN, FP8_MAX)
    q = q.to(torch.float32).to(torch.float8_e4m3fn)
    return q, scale
