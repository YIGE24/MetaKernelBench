"""Grouped-query attention decode with a paged KV cache from Llama 3.1 8B (page size 64)."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gqa_paged/gqa_paged_decode_h32_kv8_d128_ps64.json
# Workload: flashinfer-trace/workloads/gqa_paged/gqa_paged_decode_h32_kv8_d128_ps64.jsonl
#   record 8c88a86f-1843-42eb-9251-8a0ef4b6a1ee (sm_scale and geometry exact;
#   page assignment re-drawn per draw over the captured pool)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 64
NUM_PAGES = 29441
KV_INDPTR = (0, 46, 75, 149, 252, 534, 562, 591, 608, 670, 692, 712, 771, 786, 810, 841, 1193, 1275, 3349, 5940, 6987,
             8811, 10637, 11618, 12695, 13716, 13747, 13956, 13981, 14014, 14041, 14083, 14103, 14131, 14167, 14189,
             14613, 14758, 14951, 14972, 14992, 15018, 15073, 15109, 15150, 17537, 20213, 20539, 20565, 20969, 21115,
             21131, 21158, 21213, 23212, 23247, 24874, 25967, 26278, 26339, 26358, 26710, 27130, 27143, 27222)
KV_LAST_PAGE_LEN = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
SM_SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_indptr: torch.Tensor,
              kv_indices: torch.Tensor, kv_last_page_len: torch.Tensor,
              sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 64

    assert kv_indptr.shape[0] == batch_size + 1
    assert kv_indices.shape[0] == kv_indptr[-1].item()

    device = q.device
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=q.dtype, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    q_wide = q.to(compute_dtype)

    for b in range(batch_size):
        ps = int(kv_indptr[b].item())
        pe = int(kv_indptr[b + 1].item())
        last_len = int(kv_last_page_len[b].item())

        page_ids = kv_indices[ps:pe].to(torch.long)
        num_full_pages = len(page_ids) - 1

        k_full = k_cache[page_ids[:num_full_pages]].to(compute_dtype).reshape(-1, num_kv_heads, head_dim)
        v_full = v_cache[page_ids[:num_full_pages]].to(compute_dtype).reshape(-1, num_kv_heads, head_dim)
        k_tokens = torch.cat([k_full, k_cache[page_ids[-1], :last_len].to(compute_dtype)], dim=0)
        v_tokens = torch.cat([v_full, v_cache[page_ids[-1], :last_len].to(compute_dtype)], dim=0)

        k = k_tokens.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v = v_tokens.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        q_b = q_wide[b].unsqueeze(1)

        logits = torch.bmm(q_b, k.transpose(1, 2)).squeeze(1) * sm_scale
        lse[b] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits, dim=-1)
        output[b] = torch.bmm(attn.unsqueeze(1), v).squeeze(1).to(q.dtype)

    return output, lse

def make_inputs() -> list:
    device = torch.device("cuda")
    batch_size = len(KV_INDPTR) - 1
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv_indptr = torch.tensor(KV_INDPTR, device=device, dtype=torch.int32)
    kv_indices = torch.randperm(NUM_PAGES, device=device)[:KV_INDPTR[-1]].to(torch.int32)
    kv_last_page_len = torch.tensor(KV_LAST_PAGE_LEN, device=device, dtype=torch.int32)
    return [q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, SM_SCALE]
