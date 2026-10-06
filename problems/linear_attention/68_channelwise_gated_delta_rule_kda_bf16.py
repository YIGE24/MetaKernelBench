"""Kimi Delta Attention recurrence: per-channel log-decay gating with delta-rule state correction."""

# Source: flash-linear-attention
# Reference: flash-linear-attention/fla/ops/kda/naive.py#naive_recurrent_kda
# Workload: flash-linear-attention/tests/ops/test_kda.py input construction at
#   fla/models/kda/configuration_kda.py head geometry
# License: MIT

import torch
import torch.nn.functional as F

BATCH_SIZE = 8
SEQ_LEN = 1024
NUM_K_HEADS = 16
NUM_V_HEADS = 16
HEAD_DIM = 128
GATE_LOGIT_NORMALIZER = 16.0

@torch.no_grad()
def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = torch.promote_types(v.dtype, torch.float32)
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    scale = K ** -0.5

    q, k, v, g, beta = (x.to(compute_dtype) for x in (q, k, v, g, beta))
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    S = k.new_zeros(B, HV, K, V).to(q)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)

    return o.to(dtype)

def make_inputs() -> list:
    device = torch.device("cuda")

    return [
        torch.randn(BATCH_SIZE, SEQ_LEN, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(BATCH_SIZE, SEQ_LEN, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device),
        (
            F.logsigmoid(torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, HEAD_DIM, dtype=torch.float32, device=device))
            / GATE_LOGIT_NORMALIZER
        ).to(torch.bfloat16),
        torch.randn(BATCH_SIZE, SEQ_LEN, NUM_V_HEADS, dtype=torch.bfloat16, device=device).sigmoid(),
    ]
