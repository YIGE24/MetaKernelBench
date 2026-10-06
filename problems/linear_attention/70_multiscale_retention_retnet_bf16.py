"""Multi-scale retention with per-head fixed exponential decay applied to a causal query-key score matrix."""

# Source: flash-linear-attention
# Reference: flash-linear-attention/fla/ops/retention/naive.py#naive_retention
# Workload: flash-linear-attention/tests/ops/test_retention.py input construction at
#   fla/models/retnet/configuration_retnet.py head geometry
# License: MIT

import torch

BATCH_SIZE = 8
SEQ_LEN = 1024
NUM_HEADS = 8
KEY_DIM = 256
VALUE_DIM = 512

@torch.no_grad()
def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    orig_type = q.dtype
    q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
    _, n_heads, seq_len, d_head = q.shape

    s = (1 - q.new_tensor(2.0, dtype=compute_dtype)
         .pow(-5.0 - q.new_tensor(range(n_heads), dtype=compute_dtype))).log2()
    n = q.new_tensor(range(seq_len), dtype=compute_dtype)
    n = torch.exp2((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)

    s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
    o = torch.einsum('bhqk,bhkd->bhqd', s, v)

    return o.to(orig_type)

def make_inputs() -> list:
    device = torch.device("cuda")

    return [
        torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, KEY_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, KEY_DIM, dtype=torch.bfloat16, device=device),
        torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, VALUE_DIM, dtype=torch.bfloat16, device=device),
    ]
