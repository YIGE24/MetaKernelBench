"""Paged MLA-cache gather with FP8 dequantization and sequence compaction."""

# Source: vLLM
# Reference: vllm-project/vllm@b9b6306ebe0f5fa5046553667192e56bb3934af0
# Workload: tests/kernels/attention/test_cache.py::test_gather_and_maybe_dequant_cache_mla;
#   csrc/libtorch_stable/cache_kernels.cu::gather_and_maybe_dequant_cache
# License: Apache-2.0

import torch

KV_LORA_RANK = 512
ROPE_DIM = 64
ENTRY_SIZE = KV_LORA_RANK + ROPE_DIM
BLOCK_SIZE = 16
NUM_BLOCKS = 1024
NUM_REQUESTS = 8
TOTAL_TOKENS = 4096
SCALE = 0.1

@torch.no_grad()
def reference(src_cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor, scale: float) -> torch.Tensor:
    outputs = []
    fp8_cache = src_cache.view(torch.float8_e4m3fn)
    for request in range(seq_lens.numel()):
        length = int(seq_lens[request].item())
        positions = torch.arange(length, device=src_cache.device)
        logical_blocks = torch.div(positions, BLOCK_SIZE, rounding_mode="floor")
        offsets = torch.remainder(positions, BLOCK_SIZE)
        physical_blocks = block_table[request, logical_blocks].long()
        gathered = fp8_cache[physical_blocks, offsets].float() * scale
        outputs.append(gathered)
    return torch.cat(outputs, dim=0)

def make_inputs() -> list:
    device = torch.device("cuda")
    source = torch.randn(NUM_BLOCKS, BLOCK_SIZE, ENTRY_SIZE, device=device, dtype=torch.float16)
    src_cache = (source / SCALE).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    block_table = torch.stack([torch.randperm(NUM_BLOCKS, device=device) for _ in range(NUM_REQUESTS)]).to(torch.int32)
    cuts = torch.randperm(TOTAL_TOKENS - 1, device=device)[:NUM_REQUESTS - 1].sort().values + 1
    bounds = torch.cat([cuts.new_zeros(1), cuts, cuts.new_full((1,), TOTAL_TOKENS)])
    seq_lens = bounds.diff().to(torch.int32)
    return [src_cache, block_table, seq_lens, SCALE]
