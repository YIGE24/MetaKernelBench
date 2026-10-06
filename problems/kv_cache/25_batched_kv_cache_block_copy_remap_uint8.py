"""Batched K/V cache-page copy under a shared source-to-destination remap."""

# Source: vLLM
# Reference: vllm-project/vllm@b9b6306ebe0f5fa5046553667192e56bb3934af0
# Workload: tests/kernels/attention/test_cache.py::test_swap_blocks; csrc/libtorch_stable/cache_kernels.cu::swap_blocks
# License: Apache-2.0

import torch

NUM_BLOCKS = 1024
NUM_MAPPINGS = 256
BLOCK_SIZE = 32
NUM_HEADS = 8
HEAD_DIM = 256

@torch.no_grad()
def reference(
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    block_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_blocks = block_mapping[:, 0].long()
    dst_blocks = block_mapping[:, 1].long()
    dst_k[dst_blocks] = src_k[src_blocks]
    dst_v[dst_blocks] = src_v[src_blocks]
    return dst_k, dst_v

def make_inputs() -> list:
    device = torch.device("cuda")
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
    src_k = torch.randint(0, 256, shape, device=device, dtype=torch.uint8)
    src_v = torch.randint(0, 256, shape, device=device, dtype=torch.uint8)
    dst_k = torch.randint(0, 256, shape, device=device, dtype=torch.uint8)
    dst_v = torch.randint(0, 256, shape, device=device, dtype=torch.uint8)

    permutation = torch.randperm(NUM_BLOCKS, device=device)
    source_blocks = permutation[:NUM_MAPPINGS]
    destination_blocks = permutation[NUM_MAPPINGS:2 * NUM_MAPPINGS]
    block_mapping = torch.stack((source_blocks, destination_blocks), dim=1)
    return [src_k, src_v, dst_k, dst_v, block_mapping]
