"""Causal grouped-query attention prefill with ragged inputs from Qwen3 Next 80B A3B at TP=2."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gqa_ragged/gqa_ragged_prefill_causal_h8_kv1_d256.json
# Workload: flashinfer-trace/workloads/gqa_ragged/gqa_ragged_prefill_causal_h8_kv1_d256.jsonl
#   record 96ee50a6-c16e-4231-9253-e4565ed6213d (sm_scale and query lengths exact;
#   authored fixed prefixes, values re-drawn per draw)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
Q_LENS = (35, 71, 76, 69, 62, 52, 61, 39, 102, 94, 64, 44, 63, 37, 49, 56, 53, 58, 47, 18, 61, 83, 49, 63, 64, 19,
          82, 63, 17, 39, 59, 46, 66, 86, 41, 58, 36, 41, 33, 18, 19, 25, 32, 39, 68, 47, 73, 41, 50, 13, 27, 59, 22,
          29, 66, 23, 20, 37, 32, 40, 19, 55, 43)
PREFIX_LENS = (0, 64, 17, 39, 5, 58, 26, 44, 11, 0, 33, 61, 8, 22, 49, 0, 14, 55, 30, 3, 41, 64, 19, 7, 36, 0, 52, 24,
               60, 13, 28, 46, 1, 0, 57, 20, 35, 9, 63, 42, 16, 0, 50, 27, 4, 38, 59, 23, 12, 45, 0, 31, 54, 18, 6, 40,
               62, 29, 15, 48, 0, 34, 21)
SM_SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qo_indptr: torch.Tensor,
              kv_indptr: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 8
    assert num_kv_heads == 1
    assert head_dim == 256

    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()

    device = q.device
    compute_dtype = torch.promote_types(q.dtype, torch.float32)

    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=q.dtype, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads

    q_wide = q.to(compute_dtype)
    k_wide = k.to(compute_dtype)
    v_wide = v.to(compute_dtype)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        q_batch = q_wide[q_start:q_end]
        k_batch = k_wide[kv_start:kv_end]
        v_batch = v_wide[kv_start:kv_end]

        num_q_tokens = q_batch.shape[0]
        num_kv_tokens = k_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens

        k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)
        v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)

        logits = torch.einsum("qhd,khd->qhk", q_batch, k_expanded) * sm_scale

        q_positions = torch.arange(num_q_tokens, device=device)
        kv_positions = torch.arange(num_kv_tokens, device=device)
        causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
        logits = logits.masked_fill(~causal_mask[:, None, :], -float("inf"))

        lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        lse[q_start:q_end] = lse_batch

        attn_weights = torch.softmax(logits, dim=-1)
        output_batch = torch.einsum("qhk,khd->qhd", attn_weights, v_expanded)
        output[q_start:q_end] = output_batch.to(q.dtype)

    return output, lse

def make_inputs() -> list:
    device = torch.device("cuda")
    total_q = sum(Q_LENS)
    total_kv = total_q + sum(PREFIX_LENS)
    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q_lens = torch.tensor(Q_LENS, device=device)
    kv_lens = q_lens + torch.tensor(PREFIX_LENS, device=device)
    qo_indptr = torch.cat([q_lens.new_zeros(1), q_lens.cumsum(0)]).to(torch.int32)
    kv_indptr = torch.cat([kv_lens.new_zeros(1), kv_lens.cumsum(0)]).to(torch.int32)
    return [q, k, v, qo_indptr, kv_indptr, SM_SCALE]
