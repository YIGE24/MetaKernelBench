"""Grouped-query attention decode with a paged KV cache from Llama 4 Scout 17B-16E at TP=8 (page size 1)."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gqa_paged/gqa_paged_decode_h5_kv1_d128_ps1.json
# Workload: flashinfer-trace/workloads/gqa_paged/gqa_paged_decode_h5_kv1_d128_ps1.jsonl
#   record 69ec7197-791c-419a-ae93-a3d2f0d49fd0 (sm_scale and geometry exact;
#   page assignment re-drawn per draw over the captured pool)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 5
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 1
NUM_PAGES = 53173
KV_INDPTR = (0, 20, 28, 98, 102, 124, 147, 160, 193, 198, 207, 230, 266, 282, 397, 410, 418, 450, 461, 486, 508, 814,
             1084, 1335, 1412, 1462, 1485, 1496, 1502, 1938, 1955, 2275, 2297, 2313, 2332, 2382, 2390, 2405, 2414, 2568,
             2578, 2586, 2597, 4650, 4660, 4760, 4781, 4932, 4940, 4968, 5043, 5054, 5070, 5124, 10265, 15933, 15946,
             16069, 16081, 17894, 19017, 19040, 19118, 19134)
SM_SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_indptr: torch.Tensor,
              kv_indices: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    assert num_qo_heads == 5
    assert num_kv_heads == 1
    assert head_dim == 128
    assert page_size == 1

    assert kv_indptr.shape[0] == batch_size + 1
    assert kv_indices.shape[0] == kv_indptr[-1].item()

    device = q.device
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=q.dtype, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads

    k_flat = k_cache.squeeze(1)
    v_flat = v_cache.squeeze(1)
    q_wide = q.to(compute_dtype)

    for b in range(batch_size):
        ps = int(kv_indptr[b].item())
        pe = int(kv_indptr[b + 1].item())

        idx = kv_indices[ps:pe].to(torch.long)
        k = k_flat[idx].to(compute_dtype).permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v = v_flat[idx].to(compute_dtype).permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
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
    return [q, k_cache, v_cache, kv_indptr, kv_indices, SM_SCALE]
