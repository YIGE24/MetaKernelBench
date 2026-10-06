"""Causal multi-head latent attention prefill with ragged inputs from DeepSeek-V3 at TP=8."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/mla_ragged/mla_ragged_prefill_causal_h16_qk192_vo128.json
# Workload: flashinfer-trace/workloads/mla_ragged/mla_ragged_prefill_causal_h16_qk192_vo128.jsonl
#   record 79a808fd-78a2-4cd9-a236-d8c3b018e1e4 (sm_scale and geometry exact;
#   values re-drawn per draw)
# License: Apache-2.0

import math

import torch

NUM_QO_HEADS = 16
NUM_KV_HEADS = 16
QK_DIM = 192
VO_DIM = 128
Q_LENS = (39, 38, 39, 39, 39, 40, 40, 40, 41, 40, 40, 42, 42, 41, 42, 42, 43, 42, 43, 42, 43, 43, 44, 44, 44, 46, 45,
          44, 45, 45, 46, 46, 46, 46, 46, 47, 47, 47, 47, 48, 49, 48, 47, 48, 49, 50, 49, 49, 50, 50, 51, 51, 50, 51,
          51, 52, 50, 52, 53, 53, 54, 53, 52)
PREFIX_LENS = (256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
               256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
               256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256)
SM_SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qo_indptr: torch.Tensor,
              kv_indptr: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_qo_heads, qk_dim = q.shape
    total_kv, num_kv_heads, vo_dim = v.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 16
    assert num_kv_heads == 16
    assert qk_dim == 192
    assert vo_dim == 128

    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()

    device = q.device
    compute_dtype = torch.promote_types(q.dtype, torch.float32)

    output = torch.zeros((total_q, num_qo_heads, vo_dim), dtype=q.dtype, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=compute_dtype, device=device)

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

        logits = torch.einsum("qhd,khd->qhk", q_batch, k_batch) * sm_scale

        q_positions = torch.arange(num_q_tokens, device=device)
        kv_positions = torch.arange(num_kv_tokens, device=device)
        causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
        logits = logits.masked_fill(~causal_mask[:, None, :], -float("inf"))

        lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        lse[q_start:q_end] = lse_batch

        attn_weights = torch.softmax(logits, dim=-1)
        output_batch = torch.einsum("qhk,khd->qhd", attn_weights, v_batch)
        output[q_start:q_end] = output_batch.to(q.dtype)

    return output, lse

def make_inputs() -> list:
    device = torch.device("cuda")
    total_q = sum(Q_LENS)
    total_kv = total_q + sum(PREFIX_LENS)
    q_lens = torch.tensor(Q_LENS, device=device)
    kv_lens = q_lens + torch.tensor(PREFIX_LENS, device=device)
    qo_indptr = torch.cat([q_lens.new_zeros(1), q_lens.cumsum(0)]).to(torch.int32)
    kv_indptr = torch.cat([kv_lens.new_zeros(1), kv_lens.cumsum(0)]).to(torch.int32)
    q = torch.randn(total_q, NUM_QO_HEADS, QK_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(total_kv, NUM_KV_HEADS, QK_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(total_kv, NUM_KV_HEADS, VO_DIM, device=device, dtype=torch.bfloat16)
    return [q, k, v, qo_indptr, kv_indptr, SM_SCALE]
