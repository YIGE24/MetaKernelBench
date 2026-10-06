"""Fused residual-add RMS normalization over the hidden state from Llama 3.1 8B."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/rmsnorm/fused_add_rmsnorm_h4096.json
# Workload: flashinfer-trace/workloads/rmsnorm/fused_add_rmsnorm_h4096.jsonl
#   record 3e460c8c-7cad-4071-959f-e689ca024206 (batch size exact;
#   values re-drawn per draw, norm weight drawn near the trained regime)
# License: Apache-2.0

import torch

BATCH_SIZE = 14509
HIDDEN_SIZE = 4096

@torch.no_grad()
def reference(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _, hidden_size = hidden_states.shape

    assert hidden_size == 4096

    EPS = 1e-5

    compute_dtype = torch.promote_types(hidden_states.dtype, torch.float32)
    x = hidden_states.to(compute_dtype) + residual.to(compute_dtype)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(compute_dtype)
    return y.to(hidden_states.dtype)

def make_inputs() -> list:
    device = torch.device("cuda")
    hidden_states = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    residual = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(HIDDEN_SIZE, device=device, dtype=torch.float32)).to(torch.bfloat16)
    return [hidden_states, residual, weight]
