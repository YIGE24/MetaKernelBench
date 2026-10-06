"""Native sparse attention over a paged KV cache from DeepSeek-V3.2 at TP=8."""

# Source: FlashInfer-Trace / FlashInfer-Bench
# Reference: flashinfer-trace/definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json
# Workload: flashinfer-trace/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl
#   record 0f8137e0c871466f8b4bdce5bfd57658 (sm_scale, per-token selection counts and the long-context
#   selection pattern exact; page placement re-drawn per draw over the captured pool, rows emitted in
#   ascending token order)
# License: Apache-2.0

import math

import torch

NUM_TOKENS = 62
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 2048
NUM_PAGES = 6465
SM_SCALE = 0.1352337788608801

SELECT_LENS = (18, 11, 2048, 20, 25, 45, 135, 7, 8, 10, 22, 17, 11, 30, 130, 9, 600, 255, 1033, 1010, 58, 2, 21, 954,
               7, 14, 1015, 28, 64, 5, 18, 33, 26, 86, 5, 14, 10, 90, 53, 22, 2, 17, 154, 51, 121, 25, 16, 5, 19, 12,
               44, 124, 33, 1157, 9, 5, 1319, 300, 23, 35, 389, 8)
LONG_ROW = 2
LONG_ROW_CONTEXT = 2161
LONG_ROW_RUNS = ((0, 53), (54, 10), (65, 25), (91, 11), (103, 42), (146, 12), (159, 65), (225, 7), (233, 3),
                 (237, 40), (278, 1), (280, 12), (293, 13), (307, 8), (316, 1), (318, 5), (324, 4), (329, 3),
                 (333, 3), (337, 21), (359, 14), (374, 1), (377, 1), (379, 24), (404, 110), (515, 13), (529, 20),
                 (550, 6), (557, 41), (599, 16), (616, 14), (631, 15), (647, 10), (658, 6), (665, 24), (691, 11),
                 (703, 15), (719, 10), (730, 21), (752, 21), (774, 26), (801, 28), (830, 7), (838, 11), (850, 1),
                 (852, 12), (865, 18), (884, 5), (890, 1), (892, 1), (894, 9), (904, 11), (916, 10), (927, 12),
                 (940, 4), (946, 23), (970, 7), (978, 6), (985, 29), (1015, 5), (1021, 22), (1045, 13), (1059, 37),
                 (1097, 10), (1108, 42), (1151, 1), (1153, 30), (1184, 2), (1187, 31), (1219, 33), (1253, 8),
                 (1262, 102), (1366, 54), (1421, 98), (1520, 19), (1540, 23), (1564, 7), (1572, 31), (1604, 35),
                 (1640, 21), (1662, 4), (1668, 10), (1681, 4), (1688, 15), (1706, 4), (1711, 13), (1725, 3),
                 (1729, 1), (1731, 3), (1736, 12), (1749, 47), (1798, 70), (1869, 108), (1978, 2), (1981, 9),
                 (1992, 6), (1999, 31), (2031, 24), (2056, 105))

@torch.no_grad()
def reference(q_nope: torch.Tensor, q_pe: torch.Tensor, ckv_cache: torch.Tensor, kpe_cache: torch.Tensor,
              sparse_indices: torch.Tensor, sm_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    _, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    compute_dtype = torch.promote_types(q_nope.dtype, torch.float32)

    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)

    invalid_mask = sparse_indices == -1
    safe_indices = sparse_indices.clone()
    safe_indices[invalid_mask] = 0

    Kc = Kc_all[safe_indices.long()].to(compute_dtype)
    Kp = Kp_all[safe_indices.long()].to(compute_dtype)

    qn = q_nope.to(compute_dtype)
    qp = q_pe.to(compute_dtype)

    logits = qn @ Kc.transpose(-1, -2) + qp @ Kp.transpose(-1, -2)
    logits_scaled = logits * sm_scale
    logits_scaled.masked_fill_(invalid_mask.unsqueeze(1), -float("inf"))

    lse = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

    attn = torch.softmax(logits_scaled, dim=-1)
    output = (attn @ Kc).to(q_nope.dtype)

    return output, lse

def make_inputs() -> list:
    device = torch.device("cuda")
    q_nope = torch.randn(NUM_TOKENS, NUM_QO_HEADS, HEAD_DIM_CKV, device=device, dtype=torch.bfloat16)
    q_pe = torch.randn(NUM_TOKENS, NUM_QO_HEADS, HEAD_DIM_KPE, device=device, dtype=torch.bfloat16)
    ckv_cache = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV, device=device, dtype=torch.bfloat16)
    kpe_cache = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE, device=device, dtype=torch.bfloat16)
    sparse_indices = _draw_sparse_indices(device)
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, SM_SCALE]

def _draw_sparse_indices(device: torch.device) -> torch.Tensor:
    context_lens = [LONG_ROW_CONTEXT if b == LONG_ROW else SELECT_LENS[b] for b in range(NUM_TOKENS)]
    pages_per_row = [(length + PAGE_SIZE - 1) // PAGE_SIZE for length in context_lens]
    perm = torch.randperm(NUM_PAGES, device=device)[:sum(pages_per_row)]
    sparse_indices = torch.full((NUM_TOKENS, TOPK), -1, dtype=torch.int32, device=device)
    cursor = 0
    for b in range(NUM_TOKENS):
        pages = perm[cursor:cursor + pages_per_row[b]]
        cursor += pages_per_row[b]
        if b == LONG_ROW:
            logical = torch.cat([torch.arange(start, start + length, device=device)
                                 for start, length in LONG_ROW_RUNS])
        else:
            logical = torch.arange(SELECT_LENS[b], device=device)
        tokens = pages[logical // PAGE_SIZE] * PAGE_SIZE + logical % PAGE_SIZE
        sparse_indices[b, :tokens.numel()] = tokens.sort().values.to(torch.int32)
    return sparse_indices
