"""Gated delta rule variable-length prefill over ragged sequences from Qwen3-Next 80B A3B at TP=2."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gdn/gdn_prefill_qk8_v16_d128_k_last.json
# Workload: flashinfer-trace/workloads/gdn/gdn_prefill_qk8_v16_d128_k_last.jsonl
#   record 1a9da4c9-224e-4bfa-b221-fe1939ca5527 (sequence boundaries, scale and the captured A_log/dt_bias
#   exact; query and key drawn L2-normalized as the capture requires, value and gate projections re-drawn at
#   capture-measured scales, non-zero state authored for observability)
# License: Apache-2.0

import torch
import torch.nn.functional as F

TOTAL_SEQ_LEN = 959
NUM_SEQS = 4
NUM_Q_HEADS = 8
NUM_K_HEADS = 8
NUM_V_HEADS = 16
HEAD_SIZE = 128
CU_SEQLENS = (0, 19, 50, 940, 959)
A_LOG = (1.3984375, 1.203125, 2.203125, 0.1474609375, 1.46875, -0.6171875, 1.8125, -1.359375, 0.50390625, 1.03125,
         2.953125, 1.2890625, 1.3203125, 2.109375, 3.34375, 1.78125)
DT_BIAS = (-2.34375, -3.421875, -4.03125, -1.65625, -3.25, 2.046875, -5.625, -2.65625, -2.375, -2.984375, -2.328125,
           -4.09375, -4.25, -3.71875, -4.96875, -3.578125)
A_MEAN = -0.12
A_STD = 1.26
B_MEAN = 1.47
B_STD = 2.01
V_STD = 0.041
STATE_STD = HEAD_SIZE ** -0.5
SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: torch.Tensor, A_log: torch.Tensor,
              a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor, cu_seqlens: torch.Tensor,
              scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 8
    assert num_k_heads == 8
    assert num_v_heads == 16
    assert head_size == 128

    assert cu_seqlens.shape[0] == num_seqs + 1
    assert total_seq_len == cu_seqlens[-1].item()

    compute_dtype = torch.promote_types(q.dtype, torch.float32)

    x = a.to(compute_dtype) + dt_bias.to(compute_dtype)
    g = torch.exp(-torch.exp(A_log.to(compute_dtype)) * F.softplus(x))
    beta = torch.sigmoid(b.to(compute_dtype))

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros((total_seq_len, num_sab_heads, head_size), dtype=q.dtype, device=device)
    new_state = torch.zeros((num_seqs, num_sab_heads, head_size, head_size), dtype=compute_dtype, device=device)

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        state_HKV = state[seq_idx].clone().to(compute_dtype).transpose(-1, -2)

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).to(compute_dtype)
            k_H1K = k_exp[t].unsqueeze(1).to(compute_dtype)
            v_H1V = v[t].unsqueeze(1).to(compute_dtype)
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = k_H1K @ old_state_HKV
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum("hkl,hlv->hkv", k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum("hkl,hlv->hkv", k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * (q_H1K @ state_HKV)
            output[t] = o_H1V.squeeze(1).to(q.dtype)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)

    return output, new_state

def make_inputs() -> list:
    device = torch.device("cuda")
    q = F.normalize(torch.randn(TOTAL_SEQ_LEN, NUM_Q_HEADS, HEAD_SIZE, device=device, dtype=torch.float32),
                    p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(TOTAL_SEQ_LEN, NUM_K_HEADS, HEAD_SIZE, device=device, dtype=torch.float32),
                    p=2, dim=-1).to(torch.bfloat16)
    v = V_STD * torch.randn(TOTAL_SEQ_LEN, NUM_V_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
    state = STATE_STD * torch.randn(NUM_SEQS, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE, device=device, dtype=torch.float32)
    A_log = torch.tensor(A_LOG, device=device, dtype=torch.float32)
    a = (A_MEAN + A_STD * torch.randn(TOTAL_SEQ_LEN, NUM_V_HEADS, device=device,
                                      dtype=torch.float32)).to(torch.bfloat16)
    dt_bias = torch.tensor(DT_BIAS, device=device, dtype=torch.float32)
    b = (B_MEAN + B_STD * torch.randn(TOTAL_SEQ_LEN, NUM_V_HEADS, device=device,
                                      dtype=torch.float32)).to(torch.bfloat16)
    cu_seqlens = torch.tensor(CU_SEQLENS, device=device, dtype=torch.int64)
    return [q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, SCALE]
