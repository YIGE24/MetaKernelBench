"""Causal multi-head latent attention prefill with a paged KV cache from DeepSeek-V3 at TP=8."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/mla_paged/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1.json
# Workload: flashinfer-trace/workloads/mla_paged/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1.jsonl
#   record 892d1b8f-f6fe-40ad-aa3f-b6d391753bd3 (sm_scale and geometry exact;
#   page assignment re-drawn per draw over the captured pool)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 1
NUM_PAGES = 989669
QO_INDPTR = (0, 20, 31, 114, 128, 164, 173, 190, 327, 423, 456, 460, 499, 520, 602, 603, 613, 1033, 1061, 1081, 1142,
             1532, 1581, 1757, 1823, 1832, 1864, 1943, 1954)
KV_INDPTR = (0, 23, 38, 123, 141, 180, 191, 212, 353, 452, 488, 494, 538, 561, 651, 655, 669, 1092, 1122, 1145, 1209,
             1602, 1655, 1835, 1905, 1917, 1951, 2032, 2044)
SM_SCALE = 0.1352337747812271

@torch.no_grad()
def reference(q_nope: torch.Tensor, q_pe: torch.Tensor, ckv_cache: torch.Tensor, kpe_cache: torch.Tensor,
              qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor,
              sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    num_kv_indices = kv_indices.shape[0]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    assert total_q == qo_indptr[-1].item()
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device
    compute_dtype = torch.promote_types(q_nope.dtype, torch.float32)

    Kc_all = ckv_cache.squeeze(1)
    Kp_all = kpe_cache.squeeze(1)

    output = torch.zeros((total_q, num_qo_heads, head_dim_ckv), dtype=q_nope.dtype, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        kv_len = page_end - page_beg
        tok_idx = kv_indices[page_beg:page_end].to(torch.long)
        Kc = Kc_all[tok_idx].to(compute_dtype)
        Kp = Kp_all[tok_idx].to(compute_dtype)

        q_nope_batch = q_nope[q_start:q_end].to(compute_dtype)
        q_pe_batch = q_pe[q_start:q_end].to(compute_dtype)

        q_len = q_end - q_start

        for i in range(q_len):
            qn = q_nope_batch[i]
            qp = q_pe_batch[i]

            logits = (qn @ Kc.T) + (qp @ Kp.T)
            logits_scaled = logits * sm_scale

            prefix_len = kv_len - q_len
            query_abs_pos = prefix_len + i

            causal_mask = torch.arange(kv_len, device=device) > query_abs_pos
            logits_scaled.masked_fill_(causal_mask.unsqueeze(0), -float("inf"))

            lse[q_start + i] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)
            out = attn @ Kc
            output[q_start + i] = out.to(q_nope.dtype)

    return output, lse

def make_inputs() -> list:
    device = torch.device("cuda")
    total_q = QO_INDPTR[-1]
    q_nope = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM_CKV, device=device, dtype=torch.bfloat16)
    q_pe = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM_KPE, device=device, dtype=torch.bfloat16)
    ckv_cache = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV, device=device, dtype=torch.bfloat16)
    kpe_cache = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor(QO_INDPTR, device=device, dtype=torch.int32)
    kv_indptr = torch.tensor(KV_INDPTR, device=device, dtype=torch.int32)
    kv_indices = torch.randperm(NUM_PAGES, device=device)[:KV_INDPTR[-1]].to(torch.int32)
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, SM_SCALE]
