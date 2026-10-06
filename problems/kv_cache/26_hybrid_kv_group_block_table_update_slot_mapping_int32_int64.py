"""Hybrid KV-group staged table update followed by token slot translation."""

# Source: vLLM
# Reference: vllm-project/vllm@b9b6306ebe0f5fa5046553667192e56bb3934af0
# Workload: tests/v1/worker/test_gpu_block_table.py::test_block_tables_apply_staged_writes_fuses_kv_groups;
#   vllm/v1/worker/gpu/block_table.py::BlockTables
# License: Apache-2.0

import torch

BLOCK_SIZES = (16, 32, 8)
KERNEL_BLOCK_SIZES = (16, 16, 8)
MAX_NUM_REQS = 64
MAX_NUM_BATCHED_TOKENS = 4096
MAX_BLOCKS = (512, 1024, 512)
NUM_STAGED_WRITES = 2 * MAX_NUM_REQS
STAGE_CAPACITY = 4
MAX_POSITION = NUM_STAGED_WRITES // MAX_NUM_REQS * STAGE_CAPACITY * min(BLOCK_SIZES)
MAX_REQUEST_TOKENS = MAX_NUM_BATCHED_TOKENS // MAX_NUM_REQS
PAD_SLOT_ID = -1

@torch.no_grad()
def reference(
    block_table_0: torch.Tensor,
    block_table_1: torch.Tensor,
    block_table_2: torch.Tensor,
    num_blocks: torch.Tensor,
    write_requests: torch.Tensor,
    overwrite: torch.Tensor,
    staged_ids: torch.Tensor,
    staged_counts: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tables = [block_table_0, block_table_1, block_table_2]
    counts = num_blocks
    for group, table in enumerate(tables):
        _apply_group_writes(table, counts, group, write_requests, overwrite, staged_ids, staged_counts)

    slot_mappings = torch.full(
        (len(tables), MAX_NUM_BATCHED_TOKENS), PAD_SLOT_ID, device=positions.device, dtype=torch.int64
    )
    for batch_idx in range(idx_mapping.numel()):
        request = int(idx_mapping[batch_idx].item())
        start = int(query_start_loc[batch_idx].item())
        end = int(query_start_loc[batch_idx + 1].item())
        request_positions = positions[start:end].long()
        for group, table in enumerate(tables):
            block_size = KERNEL_BLOCK_SIZES[group]
            logical_blocks = torch.div(request_positions, block_size, rounding_mode="floor")
            offsets = torch.remainder(request_positions, block_size)
            physical_blocks = table[request, logical_blocks].long()
            slot_mappings[group, start:end] = physical_blocks * block_size + offsets

    return tables[0], tables[1], tables[2], counts, slot_mappings

def make_inputs() -> list:
    device = torch.device("cuda")
    tables = [torch.randint(0, 2048, (MAX_NUM_REQS, width), device=device, dtype=torch.int32) for width in MAX_BLOCKS]
    num_blocks = torch.zeros((3, MAX_NUM_REQS), device=device, dtype=torch.int32)

    write_requests = torch.cat(
        [torch.randperm(MAX_NUM_REQS, device=device), torch.randperm(MAX_NUM_REQS, device=device)]
    ).to(torch.int32)
    overwrite = torch.randint(0, 2, (NUM_STAGED_WRITES,), device=device, dtype=torch.bool)
    staged_counts = torch.randint(0, STAGE_CAPACITY + 1, (3, NUM_STAGED_WRITES), device=device, dtype=torch.int32)
    staged_ids = torch.full((3, NUM_STAGED_WRITES, STAGE_CAPACITY), -1, device=device, dtype=torch.int32)
    candidates = torch.randint(0, 2048, staged_ids.shape, device=device, dtype=torch.int32)
    filled = torch.arange(STAGE_CAPACITY, device=device)[None, None, :] < staged_counts[:, :, None]
    staged_ids[filled] = candidates[filled]

    idx_mapping = torch.randperm(MAX_NUM_REQS, device=device).to(torch.int32)
    lengths = torch.randint(0, MAX_REQUEST_TOKENS + 1, (MAX_NUM_REQS,), device=device)
    boundaries = lengths.cumsum(0).clamp_(max=MAX_NUM_BATCHED_TOKENS)
    query_start_loc = torch.cat([boundaries.new_zeros(1), boundaries]).to(torch.int32)
    positions = torch.randint(0, MAX_POSITION, (MAX_NUM_BATCHED_TOKENS,), device=device, dtype=torch.int64)
    return [
        tables[0],
        tables[1],
        tables[2],
        num_blocks,
        write_requests,
        overwrite,
        staged_ids,
        staged_counts,
        idx_mapping,
        query_start_loc,
        positions,
    ]

def _apply_group_writes(
    table: torch.Tensor,
    counts: torch.Tensor,
    group: int,
    write_requests: torch.Tensor,
    overwrite: torch.Tensor,
    staged_ids: torch.Tensor,
    staged_counts: torch.Tensor,
) -> None:
    expansion = BLOCK_SIZES[group] // KERNEL_BLOCK_SIZES[group]
    for write in range(write_requests.numel()):
        request = int(write_requests[write].item())
        count = int(staged_counts[group, write].item())
        start = 0 if bool(overwrite[write].item()) else int(counts[group, request].item())
        raw_ids = staged_ids[group, write, :count].long()
        if expansion > 1:
            offsets = torch.arange(expansion, device=table.device)
            block_ids = (raw_ids[:, None] * expansion + offsets[None, :]).reshape(-1)
        else:
            block_ids = raw_ids
        table[request, start:start + block_ids.numel()] = block_ids.to(table.dtype)
        counts[group, request] = start + block_ids.numel()
