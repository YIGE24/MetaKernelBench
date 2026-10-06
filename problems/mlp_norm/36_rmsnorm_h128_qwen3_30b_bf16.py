"""RMS normalization of flattened per-head query/key rows from Qwen3 30B A3B."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/rmsnorm/rmsnorm_h128.json
# Workload: flashinfer-trace/workloads/rmsnorm/rmsnorm_h128.jsonl
#   record f2872f89-5d8d-403a-9603-5918541cb9e0 (batch size exact;
#   values re-drawn per draw, norm weight drawn near the trained regime)
# License: Apache-2.0

import torch

BATCH_SIZE = 520128
HIDDEN_SIZE = 128

@torch.no_grad()
def reference(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _, hidden_size = hidden_states.shape

    assert hidden_size == 128

    EPS = 1e-6

    compute_dtype = torch.promote_types(hidden_states.dtype, torch.float32)
    x = hidden_states.to(compute_dtype)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(compute_dtype)
    return y.to(hidden_states.dtype)

def make_inputs() -> list:
    device = torch.device("cuda")
    hidden_states = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(HIDDEN_SIZE, device=device, dtype=torch.float32)).to(torch.bfloat16)
    return [hidden_states, weight]
