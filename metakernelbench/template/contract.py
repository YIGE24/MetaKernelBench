"""The public grading contract shared by the agent and the grader: the accuracy rule and the timing method."""

import contextlib
import statistics
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch

type Output = torch.Tensor | tuple[Output, ...] | list[Output]
type Inputs = list[Any]
type MakeInputs = Callable[[], Inputs]
type Reference = Callable[..., Output]

CORRECTNESS_DRAWS = 5
ALLOWANCE_FACTOR = 3.0
ANCHOR_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
WARMUP_DRAWS = 10
TIMING_DRAWS = 50
EVALUATION_TIMEOUT_SEC = 600

def matches_reference(actual: Any, reference: Reference, make_inputs: MakeInputs, seed: int) -> bool:
    with reference_numerics():
        expected = reference(*drawn_inputs(make_inputs, seed))
        anchor_inputs = [x.to(torch.float64) if type(x) is torch.Tensor and x.dtype in ANCHOR_DTYPES else x
                         for x in drawn_inputs(make_inputs, seed)]
        anchor = reference(*anchor_inputs)
    return matches(actual, expected, anchor)

def matches(actual: Any, expected: Output, anchor: Output) -> bool:
    if type(expected) in (tuple, list):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            return False
        return all(matches(actual_item, expected_item, anchor_item)
                   for actual_item, expected_item, anchor_item in zip(actual, expected, anchor, strict=True))
    if (type(expected) is not torch.Tensor or type(actual) is not torch.Tensor
            or actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual.device != expected.device):
        return False
    if not expected.dtype.is_floating_point:
        return torch.equal(actual, expected)
    if expected.numel() == 0:
        return True
    anchor64 = anchor.to(torch.float64)
    reference_error = float((expected.to(torch.float64) - anchor64).abs().max())
    allowance_floor = torch.finfo(expected.dtype).eps * float(anchor64.abs().max())
    allowance = ALLOWANCE_FACTOR * max(reference_error, allowance_floor)
    return float((actual.to(torch.float64) - anchor64).abs().max()) <= allowance

def baseline_ms(reference: Reference, make_inputs: MakeInputs, warmup: Sequence[int],
                timing: Sequence[int]) -> float:
    with reference_numerics():
        return median_ms(reference, make_inputs, warmup, timing)

def median_ms(fn: Callable[..., Any], make_inputs: MakeInputs, warmup: Sequence[int], timing: Sequence[int],
              check: Callable[[Any, int], None] | None = None) -> float:
    if len(warmup) != WARMUP_DRAWS or len(timing) != TIMING_DRAWS:
        raise ValueError(f"median_ms takes {WARMUP_DRAWS} warmup and {TIMING_DRAWS} timing seeds, "
                         f"got {len(warmup)} and {len(timing)}")
    for seed in warmup:
        fn(*drawn_inputs(make_inputs, seed))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for seed in timing:
        inputs = drawn_inputs(make_inputs, seed)
        torch.cuda.synchronize()
        start.record()
        out = fn(*inputs)
        torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        if check is not None:
            check(out, seed)
    return statistics.median(times)

def drawn_inputs(make_inputs: MakeInputs, seed: int) -> Inputs:
    with reference_numerics(), torch.random.fork_rng():
        torch.manual_seed(seed)
        return make_inputs()

@contextlib.contextmanager
def reference_numerics() -> Iterator[None]:
    matmul = torch.backends.cuda.matmul
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    bf16_reduction = matmul.allow_bf16_reduced_precision_reduction
    fp16_reduction = matmul.allow_fp16_reduced_precision_reduction
    precision = torch.get_float32_matmul_precision()
    dtype = torch.get_default_dtype()
    torch.backends.cudnn.allow_tf32 = False
    matmul.allow_bf16_reduced_precision_reduction = False
    matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        matmul.allow_bf16_reduced_precision_reduction = bf16_reduction
        matmul.allow_fp16_reduced_precision_reduction = fp16_reduction
        torch.set_float32_matmul_precision(precision)
        torch.set_default_dtype(dtype)
