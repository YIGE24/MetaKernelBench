"""Causal grouped-query attention prefill with a paged KV cache from Qwen3 235B A22B at TP=4 (page size 64)."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gqa_paged/gqa_paged_prefill_causal_h16_kv1_d128_ps64.json
# Workload: flashinfer-trace/workloads/gqa_paged/gqa_paged_prefill_causal_h16_kv1_d128_ps64.jsonl
#   record 820d93c0-2670-4ee2-8b19-864e471836a3 (sm_scale and geometry exact;
#   page assignment re-drawn per draw over the captured pool)
# License: Apache-2.0

import math

import torch

CHUNK_Q = 512

NUM_QO_HEADS = 16
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 64
NUM_PAGES = 4166
QO_INDPTR = (0, 17, 79, 172, 442, 458, 475, 480, 530, 540, 548, 595, 598, 610, 629, 969)
KV_INDPTR = (0, 17, 79, 172, 442, 458, 475, 480, 530, 540, 548, 595, 598, 610, 629, 969)
KV_LAST_PAGE_LEN = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
SM_SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, qo_indptr: torch.Tensor,
              kv_indptr: torch.Tensor, kv_indices: torch.Tensor, kv_last_page_len: torch.Tensor,
              sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    batch_size = qo_indptr.shape[0] - 1

    assert num_qo_heads == 16
    assert num_kv_heads == 1
    assert head_dim == 128
    assert page_size == 64

    assert total_q == qo_indptr[-1].item()
    assert kv_indices.shape[0] == kv_indptr[-1].item()

    device = q.device
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=q.dtype, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    q_wide = q.to(compute_dtype)

    for b in range(batch_size):
        qs = int(qo_indptr[b].item())
        qe = int(qo_indptr[b + 1].item())
        kvs = int(kv_indptr[b].item())
        kve = int(kv_indptr[b + 1].item())
        last_len = int(kv_last_page_len[b].item())

        page_ids = kv_indices[kvs:kve].to(torch.long)
        num_full_pages = len(page_ids) - 1

        k_full = k_cache[page_ids[:num_full_pages]].to(compute_dtype).reshape(-1, num_kv_heads, head_dim)
        v_full = v_cache[page_ids[:num_full_pages]].to(compute_dtype).reshape(-1, num_kv_heads, head_dim)
        k_tokens = torch.cat([k_full, k_cache[page_ids[-1], :last_len].to(compute_dtype)], dim=0)
        v_tokens = torch.cat([v_full, v_cache[page_ids[-1], :last_len].to(compute_dtype)], dim=0)

        num_kv = k_tokens.shape[0]
        num_q = qe - qs
        delta = num_kv - num_q

        k_exp = k_tokens.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v_exp = v_tokens.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        kv_pos = torch.arange(num_kv, device=device)

        for chunk_start in range(0, num_q, CHUNK_Q):
            chunk_end = min(chunk_start + CHUNK_Q, num_q)
            q_chunk = q_wide[qs + chunk_start:qs + chunk_end]

            logits = torch.einsum("qhd,hkd->hqk", q_chunk, k_exp) * sm_scale

            q_pos = torch.arange(chunk_start, chunk_end, device=device).unsqueeze(1)
            mask = kv_pos.unsqueeze(0) > q_pos + delta
            logits.masked_fill_(mask.unsqueeze(0), -float("inf"))

            lse[qs + chunk_start:qs + chunk_end] = (torch.logsumexp(logits, dim=-1) / math.log(2.0)).permute(1, 0)

            attn = torch.softmax(logits, dim=-1)
            output[qs + chunk_start:qs + chunk_end] = torch.einsum("hqk,hkd->qhd", attn, v_exp).to(q.dtype)

    return output, lse

def make_inputs() -> list:
    device = torch.device("cuda")
    total_q = QO_INDPTR[-1]
    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor(QO_INDPTR, device=device, dtype=torch.int32)
    kv_indptr = torch.tensor(KV_INDPTR, device=device, dtype=torch.int32)
    kv_indices = torch.randperm(NUM_PAGES, device=device)[:KV_INDPTR[-1]].to(torch.int32)
    kv_last_page_len = torch.tensor(KV_LAST_PAGE_LEN, device=device, dtype=torch.int32)
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, SM_SCALE]
