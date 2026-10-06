"""DeepSeek-V3 grouped Top-8 MoE on one EP rank with FP8 block scales."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
# Workload: flashinfer-trace/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl
#   record 5e8dc11c-f2a9-42d5-8dce-9419cbf34d5d (dims and bias exact; logits re-synthesized per draw from
#   the captured moments on the bf16 grid the captured gate emits)
# License: Apache-2.0

import torch

NUM_GLOBAL_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
TOP_K = 8
NUM_GROUPS = 8
TOPK_GROUPS = 4
BLOCK_SIZE = 128

SEQ_LEN = 14107
LOCAL_EXPERT_OFFSET = 32
ROUTED_SCALING_FACTOR = 2.5

LOGITS_MEAN = -1.6706
LOGITS_STD = 0.6560
DECISION_MARGIN = 1e-4
REDRAW_LIMIT = 16
BIAS_BASE = 4.9375
BIAS_HIGH = 4.96875
_BIAS_HIGH_EXPERTS = [
    3, 6, 7, 9, 10, 11, 16, 17, 18, 23, 30, 31, 36, 37, 41, 42, 43, 44, 46, 47, 48, 49, 54, 55, 56, 58, 61, 62,
    63, 66, 67, 68, 69, 72, 74, 75, 79, 80, 81, 82, 83, 84, 87, 90, 92, 96, 100, 101, 102, 104, 105, 108, 109,
    110, 112, 114, 119, 120, 123, 125, 126, 127, 131, 136, 147, 151, 152, 153, 154, 157, 159, 167, 170, 171,
    174, 183, 185, 195, 196, 197, 200, 203, 214, 216, 217, 219, 222, 223, 226, 230, 231, 234, 235, 236, 237,
    238, 240, 241, 242, 243, 253
]

@torch.no_grad()
def reference(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    tokens = routing_logits.shape[0]
    assert routing_logits.shape == (tokens, NUM_GLOBAL_EXPERTS)
    assert routing_bias.shape == (NUM_GLOBAL_EXPERTS,)
    assert hidden_states.shape == (tokens, HIDDEN_SIZE)
    assert hidden_states_scale.shape == (HIDDEN_SIZE // BLOCK_SIZE, tokens)
    assert gemm1_weights.shape == (NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert gemm1_weights_scale.shape == (
        NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE // BLOCK_SIZE, HIDDEN_SIZE // BLOCK_SIZE)
    assert gemm2_weights.shape == (NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    assert gemm2_weights_scale.shape == (NUM_LOCAL_EXPERTS, HIDDEN_SIZE // BLOCK_SIZE, INTERMEDIATE_SIZE // BLOCK_SIZE)

    compute_dtype = torch.promote_types(routing_logits.dtype, torch.float32)

    activation_scale = hidden_states_scale.to(compute_dtype).t().unsqueeze(-1)
    activation = (
        hidden_states.to(compute_dtype)
        .view(tokens, HIDDEN_SIZE // BLOCK_SIZE, BLOCK_SIZE)
        .mul(activation_scale)
        .view(tokens, HIDDEN_SIZE)
    )

    scores = 1.0 / (1.0 + torch.exp(-routing_logits.to(compute_dtype)))
    scores_with_bias = scores + routing_bias.to(compute_dtype)
    experts_per_group = NUM_GLOBAL_EXPERTS // NUM_GROUPS
    grouped_scores = scores_with_bias.view(tokens, NUM_GROUPS, experts_per_group)
    group_scores = torch.topk(grouped_scores, k=2, dim=2, sorted=False).values.sum(dim=2)
    selected_groups = torch.topk(group_scores, k=TOPK_GROUPS, dim=1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(2)
        .expand(tokens, NUM_GROUPS, experts_per_group)
        .reshape(tokens, NUM_GLOBAL_EXPERTS)
    )
    pruned_scores = scores_with_bias.masked_fill(~expert_mask, torch.finfo(torch.float32).min)
    selected_experts = torch.topk(pruned_scores, k=TOP_K, dim=1, sorted=False).indices

    selected_mask = torch.zeros_like(scores)
    selected_mask.scatter_(1, selected_experts, 1.0)
    routing_weights = scores * selected_mask
    routing_weights = routing_weights / (routing_weights.sum(dim=1, keepdim=True) + 1e-20)
    routing_weights = routing_weights * routed_scaling_factor

    output = torch.zeros((tokens, HIDDEN_SIZE), dtype=compute_dtype, device=hidden_states.device)

    for local_expert in range(NUM_LOCAL_EXPERTS):
        global_expert = local_expert_offset + local_expert
        token_mask = (selected_experts == global_expert).any(dim=1)
        if not token_mask.any():
            continue
        token_indices = torch.nonzero(token_mask, as_tuple=False).squeeze(1)

        scale13 = gemm1_weights_scale[local_expert].to(compute_dtype)
        weight13 = (
            gemm1_weights[local_expert]
            .to(compute_dtype)
            .view(
                2 * INTERMEDIATE_SIZE // BLOCK_SIZE,
                BLOCK_SIZE,
                HIDDEN_SIZE // BLOCK_SIZE,
                BLOCK_SIZE,
            )
            .mul(scale13.unsqueeze(1).unsqueeze(3))
            .view(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
        )
        projected = activation.index_select(0, token_indices).matmul(weight13.t())
        up, gate = projected.split(INTERMEDIATE_SIZE, dim=1)
        intermediate = gate / (1.0 + torch.exp(-gate)) * up

        scale2 = gemm2_weights_scale[local_expert].to(compute_dtype)
        weight2 = (
            gemm2_weights[local_expert]
            .to(compute_dtype)
            .view(
                HIDDEN_SIZE // BLOCK_SIZE,
                BLOCK_SIZE,
                INTERMEDIATE_SIZE // BLOCK_SIZE,
                BLOCK_SIZE,
            )
            .mul(scale2.unsqueeze(1).unsqueeze(3))
            .view(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        )
        expert_output = intermediate.matmul(weight2.t())
        token_weights = routing_weights[token_indices, global_expert].unsqueeze(1)
        output.index_add_(0, token_indices, expert_output * token_weights)

    return output.to(routing_bias.dtype)

def make_inputs() -> list:
    device = torch.device("cuda")
    routing_logits, routing_bias = _routing_inputs(device)
    hidden_states = _random_fp8((SEQ_LEN, HIDDEN_SIZE), device)
    hidden_states_scale = torch.rand(HIDDEN_SIZE // BLOCK_SIZE, SEQ_LEN, device=device, dtype=torch.float32)
    gemm1_weights = _random_fp8((NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), device)
    gemm1_weights_scale = torch.rand(
        NUM_LOCAL_EXPERTS,
        2 * INTERMEDIATE_SIZE // BLOCK_SIZE,
        HIDDEN_SIZE // BLOCK_SIZE,
        device=device,
        dtype=torch.float32,
    )
    gemm2_weights = _random_fp8((NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), device)
    gemm2_weights_scale = torch.rand(
        NUM_LOCAL_EXPERTS,
        HIDDEN_SIZE // BLOCK_SIZE,
        INTERMEDIATE_SIZE // BLOCK_SIZE,
        device=device,
        dtype=torch.float32,
    )
    return [
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        LOCAL_EXPERT_OFFSET,
        ROUTED_SCALING_FACTOR,
    ]

def _routing_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bias = torch.full((NUM_GLOBAL_EXPERTS,), BIAS_BASE, dtype=torch.bfloat16)
    bias[_BIAS_HIGH_EXPERTS] = BIAS_HIGH
    bias = bias.to(device)
    logits = _draw_logits(SEQ_LEN, device)
    for _ in range(REDRAW_LIMIT):
        unstable = _unstable_rows(logits, bias)
        if not bool(unstable.any()):
            return logits, bias
        logits[unstable] = _draw_logits(int(unstable.sum()), device)
    raise ValueError(f"routing margins under {DECISION_MARGIN} persist after {REDRAW_LIMIT} redraws")

def _draw_logits(rows: int, device: torch.device) -> torch.Tensor:
    logits = torch.randn(rows, NUM_GLOBAL_EXPERTS, device=device) * LOGITS_STD + LOGITS_MEAN
    return logits.to(torch.bfloat16).to(torch.float32)

def _unstable_rows(routing_logits: torch.Tensor, routing_bias: torch.Tensor) -> torch.Tensor:
    scores_with_bias = 1.0 / (1.0 + torch.exp(-routing_logits.float())) + routing_bias.float()
    experts_per_group = NUM_GLOBAL_EXPERTS // NUM_GROUPS
    grouped_scores = scores_with_bias.view(-1, NUM_GROUPS, experts_per_group)
    group_scores = torch.topk(grouped_scores, k=2, dim=2, sorted=False).values.sum(dim=2)
    sorted_group_scores, _ = group_scores.sort(dim=1, descending=True)
    group_gap = sorted_group_scores[:, TOPK_GROUPS - 1] - sorted_group_scores[:, TOPK_GROUPS]
    selected_groups = torch.topk(group_scores, k=TOPK_GROUPS, dim=1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(2)
        .expand(-1, NUM_GROUPS, experts_per_group)
        .reshape(-1, NUM_GLOBAL_EXPERTS)
    )
    pruned_scores = scores_with_bias.masked_fill(~expert_mask, torch.finfo(torch.float32).min)
    sorted_scores, _ = pruned_scores.sort(dim=1, descending=True)
    expert_gap = sorted_scores[:, TOP_K - 1] - sorted_scores[:, TOP_K]
    return (group_gap < DECISION_MARGIN) | (expert_gap < DECISION_MARGIN)

def _random_fp8(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
