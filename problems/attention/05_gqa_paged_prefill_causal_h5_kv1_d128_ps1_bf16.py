"""Causal grouped-query attention prefill with a paged KV cache from Llama 4 Scout 17B-16E at TP=8 (page size 1)."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gqa_paged/gqa_paged_prefill_causal_h5_kv1_d128_ps1.json
# Workload: flashinfer-trace/workloads/gqa_paged/gqa_paged_prefill_causal_h5_kv1_d128_ps1.jsonl
#   record b84bca7b-dd0f-4315-a523-b106191c049e (sm_scale and geometry exact;
#   page assignment re-drawn per draw over the captured pool)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 5
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 1
NUM_PAGES = 44264
QO_INDPTR = (0, 5, 73, 74, 94, 113, 124, 153, 156, 162, 182, 216, 229, 338, 347, 353, 380, 389, 412, 432, 735, 1002,
             1250, 1325, 1373, 1394, 1402, 1406, 1840, 1855, 2171, 2191, 2205, 2222, 2270, 2274, 2286, 2292, 2441, 2448,
             2454, 2462, 3211, 5262, 5269, 5364, 5383, 5531, 5537, 5563, 5636, 5645, 5656, 5708, 10847)
KV_INDPTR = (0, 7, 76, 79, 100, 122, 134, 166, 170, 178, 200, 235, 250, 364, 376, 383, 414, 424, 448, 469, 774, 1043,
             1293, 1369, 1418, 1440, 1450, 1455, 1890, 1906, 2225, 2246, 2261, 2279, 2328, 2335, 2349, 2357, 2510, 2519,
             2526, 2536, 3289, 5341, 5350, 5449, 5469, 5619, 5626, 5653, 5727, 5737, 5752, 5805, 10945)
SM_SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, qo_indptr: torch.Tensor,
              kv_indptr: torch.Tensor, kv_indices: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 5
    assert num_kv_heads == 1
    assert head_dim == 128
    assert page_size == 1

    assert total_q == qo_indptr[-1].item()
    assert kv_indices.shape[0] == kv_indptr[-1].item()

    device = q.device
    compute_dtype = torch.promote_types(q.dtype, torch.float32)

    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=q.dtype, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

    q_wide = q.to(compute_dtype)
    k_cache_flat = k_cache.squeeze(1)
    v_cache_flat = v_cache.squeeze(1)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        num_kv_tokens = page_ids.shape[0]
        k_seq = k_cache_flat[page_ids, 0, :].to(compute_dtype)
        v_seq = v_cache_flat[page_ids, 0, :].to(compute_dtype)

        q_seq = q_wide[q_start:q_end]
        num_q_tokens = q_seq.shape[0]
        delta = num_kv_tokens - num_q_tokens

        logits = torch.einsum("qhd,kd->qhk", q_seq, k_seq) * sm_scale

        q_idx = torch.arange(num_q_tokens, device=device, dtype=torch.long).unsqueeze(1)
        kv_idx = torch.arange(num_kv_tokens, device=device, dtype=torch.long).unsqueeze(0)
        causal_mask = (kv_idx <= q_idx + delta).unsqueeze(1)
        logits = logits.masked_fill(~causal_mask, -float("inf"))

        lse[q_start:q_end] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits, dim=-1)
        output[q_start:q_end] = torch.einsum("qhk,kd->qhd", attn, v_seq).to(q.dtype)

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
