"""Gated delta rule decode with a recurrent state update from Qwen3-Next 80B A3B at TP=2."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/gdn/gdn_decode_qk8_v16_d128_k_last.json
# Workload: flashinfer-trace/workloads/gdn/gdn_decode_qk8_v16_d128_k_last.jsonl
#   record 35b0abd8-5892-44f2-a152-52cba6b59493 (batch size, scale and the captured A_log/dt_bias exact;
#   query, key, value, state and gate projections re-drawn per draw at capture-measured scales)
# License: Apache-2.0

import torch
import torch.nn.functional as F

BATCH_SIZE = 64
SEQ_LEN = 1
NUM_Q_HEADS = 8
NUM_K_HEADS = 8
NUM_V_HEADS = 16
HEAD_SIZE = 128
A_LOG = (-0.169921875, 2.34375, 1.3359375, 2.1875, -0.9375, -1.09375, 1.765625, -0.443359375, 0.91015625, -0.8125,
         2.328125, 0.205078125, 1.375, 2.796875, -0.88671875, -1.3046875)
DT_BIAS = (-2.078125, -5.625, -4.3125, -3.421875, -3.796875, -3.296875, -4.375, -6.25, -3.703125, -0.2890625,
           -5.21875, -6.1875, -3.046875, -5.0625, -4.875, -5.34375)
A_MEAN = 1.33
A_STD = 1.56
B_MEAN = 0.4
B_STD = 1.38
Q_STD = 0.18
K_STD = 0.13
V_STD = 0.073
STATE_STD = 0.009
SCALE = 0.08838834764831843

@torch.no_grad()
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: torch.Tensor, A_log: torch.Tensor,
              a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor,
              scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    assert num_q_heads == 8
    assert num_k_heads == 8
    assert num_v_heads == 16
    assert K == 128 and V == 128
    assert T == 1

    assert num_v_heads >= num_q_heads
    assert num_v_heads % num_q_heads == 0
    assert num_k_heads == num_q_heads

    compute_dtype = torch.promote_types(q.dtype, torch.float32)

    x = a.to(compute_dtype) + dt_bias.to(compute_dtype)
    g = torch.exp(-torch.exp(A_log.to(compute_dtype)) * F.softplus(x))
    beta = torch.sigmoid(b.to(compute_dtype))

    q_wide = q.squeeze(1).to(compute_dtype)
    k_wide = k.squeeze(1).to(compute_dtype)
    v_wide = v.squeeze(1).to(compute_dtype)
    g_wide = g.squeeze(1)
    beta_wide = beta.squeeze(1)
    state_wide = state.to(compute_dtype)

    q_exp = q_wide.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_wide.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    new_state = torch.zeros_like(state_wide)
    output = torch.zeros(B, num_heads, V, dtype=compute_dtype, device=device)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]
            k_h = k_exp[b_idx, h_idx]
            v_h = v_wide[b_idx, h_idx]
            h_state = state_wide[b_idx, h_idx].clone().transpose(-1, -2)
            g_val = g_wide[b_idx, h_idx]
            beta_val = beta_wide[b_idx, h_idx]

            old_state = g_val * h_state
            old_v = k_h @ old_state
            new_v = beta_val * v_h + (1 - beta_val) * old_v
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
            h_state = old_state - state_remove + state_update

            output[b_idx, h_idx] = scale * (q_h @ h_state)
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)

    output = output.unsqueeze(1).to(q.dtype)
    return output, new_state

def make_inputs() -> list:
    device = torch.device("cuda")
    q = Q_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_Q_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
    k = K_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_K_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
    v = V_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
    state = STATE_STD * torch.randn(BATCH_SIZE, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE, device=device, dtype=torch.float32)
    A_log = torch.tensor(A_LOG, device=device, dtype=torch.float32)
    a = (A_MEAN + A_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, device=device,
                                      dtype=torch.float32)).to(torch.bfloat16)
    dt_bias = torch.tensor(DT_BIAS, device=device, dtype=torch.float32)
    b = (B_MEAN + B_STD * torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, device=device,
                                      dtype=torch.float32)).to(torch.bfloat16)
    return [q, k, v, state, A_log, a, dt_bias, b, SCALE]
