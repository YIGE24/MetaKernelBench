"""Causal grouped-query attention prefill with ragged inputs from Qwen3 32B at TP=4."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gqa_ragged/gqa_ragged_prefill_causal_h16_kv2_d128.json
# Workload: flashinfer-trace/workloads/gqa_ragged/gqa_ragged_prefill_causal_h16_kv2_d128.jsonl
#   record 560c22ab-7115-40f8-b2d8-63b98563e53f (sm_scale and query lengths exact;
#   authored fixed prefixes, values re-drawn per draw)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128
Q_LENS = (133, 181, 9, 8, 14, 43, 24, 29, 2383, 2678, 314, 14, 420, 134, 4, 15, 43, 2024, 23, 1618, 1086, 300, 49, 7,
          340, 422, 1, 68, 18, 7, 74, 3, 20, 21, 12, 31, 3, 7, 21, 40, 14, 107, 11, 6, 30, 9, 26, 24, 301, 266, 247, 75,
          49, 23, 9, 4, 441, 15, 349, 27, 14, 17, 54, 6, 13, 8, 147, 8, 7, 9, 730)
PREFIX_LENS = (0, 64, 17, 39, 5, 58, 26, 44, 11, 0, 33, 61, 8, 22, 49, 0, 14, 55, 30, 3, 41, 64, 19, 7, 36, 0, 52, 24,
               60, 13, 28, 46, 1, 0, 57, 20, 35, 9, 63, 42, 16, 0, 50, 27, 4, 38, 59, 23, 12, 45, 0, 31, 54, 18, 6, 40,
               62, 29, 15, 48, 0, 34, 21, 51, 10, 56, 2, 43, 25, 37, 0)
SM_SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qo_indptr: torch.Tensor,
              kv_indptr: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 16
    assert num_kv_heads == 2
    assert head_dim == 128

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
