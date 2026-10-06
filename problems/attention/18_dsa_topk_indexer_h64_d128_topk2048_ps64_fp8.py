"""Native sparse attention Top-K index selection with FP8 query and KV-cache data from DeepSeek-V3.2."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json
# Workload: flashinfer-trace/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl
#   record a52c09bc-2ee5-4366-be02-457932a80631 (seq_lens and block-table width exact; page assignment
#   re-drawn per draw over the captured pool; index cache re-synthesized per draw with real FP8 payloads
#   and per-token scales; Top-K returned in ascending token order)
# License: Apache-2.0

import torch

NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 64
NUM_PAGES = 11923
MAX_NUM_PAGES = 43
SEQ_LENS = (63, 9, 2693, 212, 11, 25, 6, 50, 77, 22, 25, 10, 52, 76, 11, 30, 1822, 25, 1034, 1811, 7, 13, 23, 1896,
            13, 285, 18, 166, 16, 1421, 12)
DECISION_MARGIN = 1e-3
REDRAW_LIMIT = 16

@torch.no_grad()
def reference(q_index_fp8: torch.Tensor, k_index_cache_fp8: torch.Tensor, weights: torch.Tensor,
              seq_lens: torch.Tensor, block_table: torch.Tensor) -> tuple[torch.Tensor]:
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32)
    K_all = _dequant_fp8_kv_cache(k_index_cache_fp8)

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        K_paged = K_all[page_indices]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]

        q_b = q[b]
        scores = q_b @ K.T
        scores_relu = torch.relu(scores)
        weighted_scores = scores_relu * weights[b][:, None]
        final_scores = weighted_scores.sum(dim=0)

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)
        topk_idx = topk_idx.sort().values

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)

def make_inputs() -> list:
    device = torch.device("cuda")
    batch_size = len(SEQ_LENS)
    seq_lens = torch.tensor(SEQ_LENS, device=device, dtype=torch.int32)
    block_table = _draw_block_table(device)
    for _ in range(REDRAW_LIMIT):
        q_index_fp8 = torch.randn(batch_size, NUM_INDEX_HEADS, INDEX_HEAD_DIM, device=device,
                                  dtype=torch.float32).to(torch.float8_e4m3fn)
        k_index_cache_fp8 = _packed_fp8_cache(NUM_PAGES, PAGE_SIZE, INDEX_HEAD_DIM, device)
        weights = torch.randn([batch_size, NUM_INDEX_HEADS], device=device, dtype=torch.float32)
        if not _unstable_selection(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
            return [q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table]
    raise ValueError(f"selection margins under {DECISION_MARGIN} persist after {REDRAW_LIMIT} redraws")

def _draw_block_table(device: torch.device) -> torch.Tensor:
    perm = torch.randperm(NUM_PAGES - 1, device=device) + 1
    block_table = torch.zeros((len(SEQ_LENS), MAX_NUM_PAGES), device=device, dtype=torch.int32)
    cursor = 0
    for b, seq_len in enumerate(SEQ_LENS):
        num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_table[b, :num_pages_for_seq] = perm[cursor:cursor + num_pages_for_seq].to(torch.int32)
        cursor += num_pages_for_seq
    return block_table

def _unstable_selection(q_index_fp8: torch.Tensor, k_index_cache_fp8: torch.Tensor, weights: torch.Tensor,
                        seq_lens: torch.Tensor, block_table: torch.Tensor) -> bool:
    q = q_index_fp8.to(torch.float32)
    K_all = _dequant_fp8_kv_cache(k_index_cache_fp8)
    for b in range(seq_lens.numel()):
        seq_len = int(seq_lens[b].item())
        if seq_len <= 2048:
            continue
        num_pages_for_seq = (seq_len + 63) // 64
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
        K = K_all[page_indices].reshape(-1, 128)[:seq_len]
        final_scores = (torch.relu(q[b] @ K.T) * weights[b][:, None]).sum(dim=0)
        boundary = torch.topk(final_scores, 2049).values
        if boundary[2047] - boundary[2048] < DECISION_MARGIN:
            return True
    return False

def _dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)

    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale

def _packed_fp8_cache(num_pages: int, page_size: int, head_dim: int, device: torch.device) -> torch.Tensor:
    values = torch.randn(num_pages, page_size, head_dim, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.rand(num_pages, page_size, device=device, dtype=torch.float32) * 0.5 + 0.5
    packed = torch.empty(num_pages, page_size * (head_dim + 4), dtype=torch.uint8, device=device)
    packed[:, :page_size * head_dim] = values.reshape(num_pages, -1).view(torch.uint8)
    packed[:, page_size * head_dim:] = scales.view(torch.uint8)
    return packed.view(torch.int8).reshape(num_pages, page_size, 1, head_dim + 4)
