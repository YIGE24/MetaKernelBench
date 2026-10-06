"""Causal grouped-query attention prefill with a paged KV cache from Llama 3.2 3B (page size 1)."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gqa_paged/gqa_paged_prefill_causal_h24_kv8_d128_ps1.json
# Workload: flashinfer-trace/workloads/gqa_paged/gqa_paged_prefill_causal_h24_kv8_d128_ps1.jsonl
#   record 0b88caa4-106e-4017-a2a0-fc9840e4b66c (sm_scale and geometry exact;
#   page assignment re-drawn per draw over the captured pool)
# License: Apache-2.0

import math

import torch

CHUNK_Q = 512

NUM_QO_HEADS = 24
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 1
NUM_PAGES = 77646
QO_INDPTR = (0, 145, 311, 484, 886, 1222, 1568, 1908, 2250, 2695, 3049, 3595, 4042, 4555, 5093, 5638, 6413, 6974, 7689,
             8514, 9110, 10050, 11131, 12281, 13236, 14199, 15449, 16372)
KV_INDPTR = (0, 145, 311, 484, 886, 1222, 1568, 1908, 2250, 2695, 3049, 3595, 4042, 4555, 5093, 5638, 6413, 6974, 7689,
             8514, 9110, 10050, 11131, 12281, 13236, 14199, 15449, 16372)
SM_SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, qo_indptr: torch.Tensor,
              kv_indptr: torch.Tensor, kv_indices: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    batch_size = qo_indptr.shape[0] - 1

    assert num_qo_heads == 24
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1

    assert total_q == qo_indptr[-1].item()
    assert kv_indices.shape[0] == kv_indptr[-1].item()

    device = q.device
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=q.dtype, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    q_wide = q.to(compute_dtype)
    k_flat = k_cache.squeeze(1)
    v_flat = v_cache.squeeze(1)

    for b in range(batch_size):
        qs = int(qo_indptr[b].item())
        qe = int(qo_indptr[b + 1].item())
        kvs = int(kv_indptr[b].item())
        kve = int(kv_indptr[b + 1].item())

        page_ids = kv_indices[kvs:kve].to(torch.long)
        k = k_flat[page_ids].to(compute_dtype)
        v = v_flat[page_ids].to(compute_dtype)
        num_kv = k.shape[0]
        num_q = qe - qs
        delta = num_kv - num_q

        k_exp = k.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v_exp = v.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
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
    q = torch.randn(QO_INDPTR[-1], NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor(QO_INDPTR, device=device, dtype=torch.int32)
    kv_indptr = torch.tensor(KV_INDPTR, device=device, dtype=torch.int32)
    kv_indices = torch.randperm(NUM_PAGES, device=device)[:KV_INDPTR[-1]].to(torch.int32)
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, SM_SCALE]
