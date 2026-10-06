"""Grouped-query attention decode with a paged KV cache from Llama 3.1 8B (page size 1)."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gqa_paged/gqa_paged_decode_h32_kv8_d128_ps1.json
# Workload: flashinfer-trace/workloads/gqa_paged/gqa_paged_decode_h32_kv8_d128_ps1.jsonl
#   record 5444fc56-b282-4228-b233-ee170d1bd127 (sm_scale and geometry exact;
#   page assignment re-drawn per draw over the captured pool)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 1
NUM_PAGES = 68237
KV_INDPTR = (0, 542, 1067, 1637, 2236, 3014, 3538, 4063, 4576, 5134, 5652, 6168, 6723, 7234, 7754, 8281, 9129, 9707,
             12277, 15364, 16907, 19227, 21549, 23026, 24599, 26116, 26643, 27348, 27869, 28398, 28921, 29459, 29975,
             30499, 31031, 31549, 32469, 33110, 33799, 34316, 34832, 35354, 35905, 36437, 36974, 39857, 43029, 43851,
             44373, 45273, 45915, 46427, 46950, 47501, 49996, 50527, 52650, 54239, 55046, 55603, 56118, 56966, 57882,
             58391, 58966)
SM_SCALE = 0.0883883461356163

@torch.no_grad()
def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_indptr: torch.Tensor,
              kv_indices: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    assert num_qo_heads == 32
    assert num_kv_heads == 8
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
