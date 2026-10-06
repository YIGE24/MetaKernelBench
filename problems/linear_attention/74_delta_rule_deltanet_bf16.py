"""Delta rule linear attention recurrence applying per-token rank-one state corrections without decay."""

# Source: flash-linear-attention
# Reference: flash-linear-attention/fla/ops/delta_rule/naive.py#delta_rule_recurrence
# Workload: flash-linear-attention/tests/ops/test_delta.py input construction at
#   fla/models/delta_net/configuration_delta_net.py head geometry
# License: MIT

import torch
import torch.nn.functional as F

BATCH_SIZE = 8
SEQ_LEN = 1024
NUM_HEADS = 16
KEY_DIM = 128
VALUE_DIM = 128

@torch.no_grad()
def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
              initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    b, h, l, d_k = q.shape
    q, k, v, beta = (x.to(compute_dtype) for x in (q, k, v, beta))
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)

    beta = beta[..., None]

    S += initial_state.to(compute_dtype)

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", _q, S)
    return o.to(orig_dtype), S

def make_inputs() -> list:
    device = torch.device("cuda")
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, KEY_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, KEY_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, VALUE_DIM, device=device, dtype=torch.bfloat16)
    beta = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, device=device, dtype=torch.bfloat16).sigmoid()
    initial_state = torch.randn(BATCH_SIZE, NUM_HEADS, KEY_DIM, VALUE_DIM, device=device, dtype=torch.float32)
    return [q, k, v, beta, initial_state]
