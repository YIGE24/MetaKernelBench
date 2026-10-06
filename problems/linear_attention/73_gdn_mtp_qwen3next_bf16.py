"""Gated delta rule multi-token speculative verification over a pooled state from Qwen3-Next 80B A3B at TP=2."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gdn/gdn_mtp_qk8_v16_d128_k_last.json
# Workload: flashinfer-trace/workloads/gdn/gdn_mtp_qk8_v16_d128_k_last.jsonl
#   record 4c266399-6168-447d-b374-cfb1234e296c (batch, draft length, pool size, scale and the captured
#   A_log/dt_bias exact; pool routing re-drawn per draw over the pool, tensors re-drawn per draw at
#   capture-measured scales)
# License: Apache-2.0

import torch
import torch.nn.functional as F

BATCH_SIZE = 32
SEQ_LEN = 4
NUM_Q_HEADS = 8
NUM_K_HEADS = 8
NUM_V_HEADS = 16
HEAD_SIZE = 128
POOL_SIZE = 49
A_LOG = (-1.046875, -0.5234375, -1.828125, -1.15625, -0.0244140625, 0.28515625, -1.9921875, -2.0, 0.11962890625,
         0.3671875, -1.59375, -1.34375, 0.333984375, -1.640625, -0.5234375, -0.2890625)
DT_BIAS = (-5.25, -5.75, -2.015625, -3.3125, -6.25, -6.65625, -0.88671875, -0.6328125, -4.8125, -5.28125, -1.7578125,
           -3.484375, 2.3125, -6.40625, -6.34375, -6.75)
A_MEAN = 0.82
A_STD = 2.24
B_MEAN = 0.45
B_STD = 2.02
Q_STD = 0.63
K_STD = 0.26
V_STD = 0.13
STATE_STD = 0.0215
SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, initial_state: torch.Tensor,
              initial_state_indices: torch.Tensor, A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor,
              b: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, num_q_heads, head_size = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, _ = v.shape
    device = q.device

    assert num_q_heads == 8
    assert num_k_heads == 8
    assert num_v_heads == 16
    assert head_size == 128
    assert T > 1, "MTP requires seq_len > 1"

    assert num_v_heads >= num_q_heads
    assert num_v_heads % num_q_heads == 0
    assert num_k_heads == num_q_heads

    compute_dtype = torch.promote_types(q.dtype, torch.float32)

    x = a.to(compute_dtype) + dt_bias.to(compute_dtype)
    g = torch.exp(-torch.exp(A_log.to(compute_dtype)) * F.softplus(x))
    beta = torch.sigmoid(b.to(compute_dtype))

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=2)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    output = torch.zeros((B, T, num_v_heads, head_size), dtype=q.dtype, device=device)

    for b_idx in range(B):
        state_idx = int(initial_state_indices[b_idx].item())
        state_HVK = initial_state[state_idx].clone().to(compute_dtype).transpose(-1, -2)

        for t in range(T):
            q_HK = q_exp[b_idx, t].to(compute_dtype)
            k_HK = k_exp[b_idx, t].to(compute_dtype)
            v_HV = v[b_idx, t].to(compute_dtype)
            g_H = g[b_idx, t]
            beta_H = beta[b_idx, t]

            for h_idx in range(num_v_heads):
                q_h = q_HK[h_idx]
                k_h = k_HK[h_idx]
                v_h = v_HV[h_idx]
                h_state = state_HVK[h_idx]
                g_val = g_H[h_idx]
                beta_val = beta_H[h_idx]

                old_state = g_val * h_state
                old_v = k_h @ old_state
                new_v = beta_val * v_h + (1 - beta_val) * old_v
                state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
                state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
                h_state = old_state - state_remove + state_update

                output[b_idx, t, h_idx] = (scale * (q_h @ h_state)).to(q.dtype)

                state_HVK[h_idx] = h_state

    final_state = initial_state.clone()

    return output, final_state

def make_inputs() -> list:
    device = torch.device("cuda")
    q = Q_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_Q_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
    k = K_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_K_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
    v = V_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
    initial_state = STATE_STD * torch.randn(POOL_SIZE, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE, device=device,
                                            dtype=torch.float32)
    initial_state_indices = torch.randperm(POOL_SIZE, device=device)[:BATCH_SIZE].to(torch.int32)
    A_log = torch.tensor(A_LOG, device=device, dtype=torch.float32)
    a = (A_MEAN + A_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, device=device,
                                      dtype=torch.float32)).to(torch.bfloat16)
    dt_bias = torch.tensor(DT_BIAS, device=device, dtype=torch.float32)
    b = (B_MEAN + B_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, device=device,
                                      dtype=torch.float32)).to(torch.bfloat16)
    return [q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b, SCALE]
