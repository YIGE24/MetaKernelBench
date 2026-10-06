"""Multi-head latent attention decode with a paged KV cache from DeepSeek-V3 at TP=8."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps1.json
# Workload: flashinfer-trace/workloads/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps1.jsonl
#   record 5bef8d88-0f74-4ccb-a256-b02842951df3 (sm_scale and geometry exact;
#   page assignment re-drawn per draw over the captured pool)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 1
NUM_PAGES = 989669
KV_INDPTR = (0, 537, 1051, 1689, 2199, 3771, 5311, 6904, 8494, 11155, 13258, 15676, 24873, 25427, 25946, 27825, 28517,
             29331, 30010, 30680, 31220, 31736, 32254, 32866, 33378, 36677, 37310, 37846, 38896, 39415, 39983, 40641,
             41155, 41689, 42417, 43059, 43975, 45419, 47001, 48634, 50171, 51682, 52311, 53407, 54908, 56394, 57906,
             58425, 58975, 59915, 60483, 61003, 64016, 64571, 65094, 66520, 67099, 67743, 68267, 68784, 69317, 70745,
             72061, 73455, 75145)
SM_SCALE = 0.1352337747812271

@torch.no_grad()
def reference(q_nope: torch.Tensor, q_pe: torch.Tensor, ckv_cache: torch.Tensor, kpe_cache: torch.Tensor,
              kv_indptr: torch.Tensor, kv_indices: torch.Tensor,
              sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device
    compute_dtype = torch.promote_types(q_nope.dtype, torch.float32)

    Kc_all = ckv_cache.squeeze(1)
    Kp_all = kpe_cache.squeeze(1)

    output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=q_nope.dtype, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        tok_idx = kv_indices[page_beg:page_end].to(torch.long)

        Kc = Kc_all[tok_idx].to(compute_dtype)
        Kp = Kp_all[tok_idx].to(compute_dtype)
        qn = q_nope[b].to(compute_dtype)
        qp = q_pe[b].to(compute_dtype)

        logits = (qn @ Kc.T) + (qp @ Kp.T)
        logits_scaled = logits * sm_scale

        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)
        out = attn @ Kc
        output[b] = out.to(q_nope.dtype)

    return output, lse

def make_inputs() -> list:
    device = torch.device("cuda")
    batch_size = len(KV_INDPTR) - 1
    q_nope = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_CKV, device=device, dtype=torch.bfloat16)
    q_pe = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM_KPE, device=device, dtype=torch.bfloat16)
    ckv_cache = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV, device=device, dtype=torch.bfloat16)
    kpe_cache = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE, device=device, dtype=torch.bfloat16)
    kv_indptr = torch.tensor(KV_INDPTR, device=device, dtype=torch.int32)
    kv_indices = torch.randperm(NUM_PAGES, device=device)[:KV_INDPTR[-1]].to(torch.int32)
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, SM_SCALE]
