"""Context-parallel paged gather and mixed-layout FP8/BF16 upconversion."""

# Source: vLLM
# Reference: vllm-project/vllm@b9b6306ebe0f5fa5046553667192e56bb3934af0
# Workload: tests/kernels/test_cp_gather_fp8.py::test_cp_gather_and_upconvert_fp8_kv_cache;
#   csrc/libtorch_stable/cache_kernels.cu::cp_gather_and_upconvert_fp8_kv_cache
# License: Apache-2.0

import math

import torch

NOPE_DIM = 512
ROPE_DIM = 64
GROUP_SIZE = 128
NUM_GROUPS = 4
ENTRY_BYTES = 656
BLOCK_SIZE = 64
NUM_BLOCKS = 16
NUM_REQUESTS = 3
TOTAL_TOKENS = 641

@torch.no_grad()
def reference(cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    outputs = []
    for request in range(seq_lens.numel()):
        length = int(seq_lens[request].item())
        positions = torch.arange(length, device=cache.device)
        logical_blocks = torch.div(positions, BLOCK_SIZE, rounding_mode="floor")
        offsets = torch.remainder(positions, BLOCK_SIZE)
        physical_blocks = block_table[request, logical_blocks].long()
        entries = cache[physical_blocks, offsets]

        fp8_values = entries[:, :NOPE_DIM].contiguous().view(torch.float8_e4m3fn)
        scales = entries[:, NOPE_DIM:NOPE_DIM + 16].contiguous().view(torch.float32)
        nope_parts = []
        for group in range(NUM_GROUPS):
            start = group * GROUP_SIZE
            part = fp8_values[:, start:start + GROUP_SIZE].float()
            nope_parts.append((part * scales[:, group:group + 1]).to(torch.bfloat16))
        nope = torch.cat(nope_parts, dim=-1)
        rope = entries[:, NOPE_DIM + 16:].contiguous().view(torch.bfloat16)
        outputs.append(torch.cat((nope, rope), dim=-1))
    return torch.cat(outputs, dim=0)

def make_inputs() -> list:
    device = torch.device("cuda")
    cuts = torch.randperm(TOTAL_TOKENS - 1, device=device)[:NUM_REQUESTS - 1].sort().values + 1
    bounds = torch.cat([cuts.new_zeros(1), cuts, cuts.new_full((1,), TOTAL_TOKENS)])
    lengths = bounds.diff().tolist()
    blocks_per_request = [math.ceil(length / BLOCK_SIZE) for length in lengths]
    max_blocks = max(blocks_per_request)
    block_table = torch.zeros(NUM_REQUESTS, max_blocks, device=device, dtype=torch.int32)
    permutation = torch.randperm(NUM_BLOCKS, device=device).to(torch.int32)
    cursor = 0
    for request, count in enumerate(blocks_per_request):
        block_table[request, :count] = permutation[cursor:cursor + count]
        cursor += count

    cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, ENTRY_BYTES, device=device, dtype=torch.uint8)
    flat = cache.view(-1, ENTRY_BYTES)
    fp8 = torch.randn(flat.shape[0], NOPE_DIM, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.rand(flat.shape[0], NUM_GROUPS, device=device, dtype=torch.float32) * 2.0 + 0.1
    rope = torch.randn(flat.shape[0], ROPE_DIM, device=device, dtype=torch.bfloat16)
    flat[:, :NOPE_DIM] = fp8.view(torch.uint8)
    flat[:, NOPE_DIM:NOPE_DIM + 16] = scales.view(torch.uint8)
    flat[:, NOPE_DIM + 16:] = rope.view(torch.uint8)
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    return [cache, block_table, seq_lens]
