"""Gated linear attention recurrence with per-key-channel log-decay over a matrix-valued state."""

# Source: flash-linear-attention
# Reference: flash-linear-attention/fla/ops/gla/naive.py#naive_recurrent_gla
# Workload: flash-linear-attention/tests/ops/test_gla.py input construction at
#   fla/models/gla/configuration_gla.py head geometry
# License: MIT

import torch
import torch.nn.functional as F

BATCH_SIZE = 8
SEQ_LEN = 1024
NUM_HEADS = 4
KEY_DIM = 256
VALUE_DIM = 512
GATE_LOGIT_NORMALIZER = 16.0

@torch.no_grad()
def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    dtype = q.dtype
    q, k, v, gk = (x.transpose(1, 2).to(compute_dtype) for x in (q, k, v, gk))
    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    scale = K ** -0.5

    h = q.new_zeros(B, H, K, V, dtype=compute_dtype)

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        gk_i = gk[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * gk_i[..., None] + kv_i
        o[:, :, i] = (q_i[..., None] * h).sum(-2)

    return o.transpose(1, 2).to(dtype)

def make_inputs() -> list:
    device = torch.device("cuda")

    return [
        torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, KEY_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, KEY_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, VALUE_DIM, dtype=torch.bfloat16, device=device),
        (
            F.logsigmoid(torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, KEY_DIM, dtype=torch.float32, device=device))
            / GATE_LOGIT_NORMALIZER
        ).to(torch.bfloat16),
    ]
