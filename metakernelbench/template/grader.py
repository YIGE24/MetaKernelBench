"""The private grader: runs a submitted kernel in a bounded child process and scores it against the eager baseline."""

import contextlib
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO

import torch

import contract

GRADER_PATH = Path(__file__).resolve()
TASK_PATH = GRADER_PATH.parent / "task.py"
KERNEL_PATH = Path("/app/kernel.py")
VERIFIER_DIR = Path("/logs/verifier")
TASK_MODULE = "mkbench_task"
KERNEL_MODULE = "mkbench_kernel"
SEED_BITS = 63
ERROR_TAIL_CHARS = 2_000
CHILD_TAIL_BYTES = 5_000_000
REWARD_KEYS = ("imported", "correct", "speedup_vs_eager", "passed")
TIMING_CHECK_DRAWS = 3

SYSTEM_RNG = random.SystemRandom()

@dataclass(frozen=True)
class Assignment:
    correctness: list[int]
    warmup: list[int]
    timing: list[int]
    sentinel: str

def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        assignment = Assignment(**json.loads(sys.stdin.readline()))
        result = _evaluate_kernel(assignment)
        print(f"\n{assignment.sentinel}{json.dumps(result)}", flush=True)
        os._exit(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable, the container cannot grade a kernel")
    task = _load_module(TASK_PATH, TASK_MODULE)
    assignment = _draw_assignment()
    rewards, details = _score(_child_result(assignment), task, assignment)
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFIER_DIR / "reward.json").write_text(json.dumps(rewards, indent=2))
    (VERIFIER_DIR / "reward-details.json").write_text(json.dumps(details, indent=2))
    _print_report(rewards, details)

def _draw_assignment() -> Assignment:
    def draw() -> int:
        return SYSTEM_RNG.getrandbits(SEED_BITS)

    return Assignment(correctness=[draw() for _ in range(contract.CORRECTNESS_DRAWS)],
                      warmup=[draw() for _ in range(contract.WARMUP_DRAWS)],
                      timing=[draw() for _ in range(contract.TIMING_DRAWS)],
                      sentinel=f"MKBENCH_RESULT_{draw():016x} ")

def _child_result(assignment: Assignment) -> dict[str, Any]:
    stdout, stderr, returncode = _child_output(assignment)
    payloads = [line.removeprefix(assignment.sentinel)
                for line in stdout.splitlines() if line.startswith(assignment.sentinel)]
    if returncode == 0:
        if not payloads:
            return _normalized_result({"error": "child evaluator exited without reporting a result"})
        try:
            final = json.loads(payloads[-1])
            if type(final) is dict and "stage" in final:
                error = f"child evaluator exited during {final['stage']} without completing the evaluation"
                final = {**final, "error": error}
            return _normalized_result(final)
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            return _normalized_result({"error": f"child evaluator printed an unusable result: {exc}"})
    snapshot = _last_snapshot(payloads)
    stage = "startup" if snapshot is None else snapshot.get("stage", "final reporting")
    if returncode is None:
        error = (f"kernel evaluation hung during {stage}, killed after {contract.EVALUATION_TIMEOUT_SEC}s\n"
                 f"{stdout[-ERROR_TAIL_CHARS:]}")
    else:
        error = f"child evaluator {_death_text(returncode)} during {stage}"
        if returncode < 0:
            error += "; native crashes leave no Python traceback"
        if stderr:
            error += f"\n{stderr[-ERROR_TAIL_CHARS:]}"
    return _normalized_result({**(snapshot or {}), "error": error})

def _child_output(assignment: Assignment) -> tuple[str, str, int | None]:
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        with subprocess.Popen([sys.executable, str(GRADER_PATH), "--child"], stdin=subprocess.PIPE,
                              stdout=out, stderr=err, start_new_session=True) as child:
            try:
                with contextlib.suppress(BrokenPipeError):
                    child.stdin.write((json.dumps(asdict(assignment)) + "\n").encode())
                    child.stdin.close()
                returncode = child.wait(timeout=contract.EVALUATION_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                returncode = None
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
        return _tail(out), _tail(err), returncode

def _tail(stream: BinaryIO) -> str:
    size = stream.seek(0, os.SEEK_END)
    stream.seek(max(size - CHILD_TAIL_BYTES, 0))
    return stream.read().decode(errors="replace")

def _last_snapshot(payloads: list[str]) -> dict[str, Any] | None:
    for payload in reversed(payloads):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if type(parsed) is dict:
            return parsed
    return None

def _death_text(returncode: int) -> str:
    if returncode >= 0:
        return f"died with exit code {returncode}"
    number = -returncode
    try:
        return f"was killed by signal {number} ({signal.Signals(number).name})"
    except ValueError:
        return f"was killed by signal {number}"

def _normalized_result(raw: dict[str, Any]) -> dict[str, Any]:
    kernel_ms = raw.get("kernel_ms")
    if kernel_ms is not None:
        kernel_ms = float(kernel_ms)
        if not (math.isfinite(kernel_ms) and kernel_ms > 0):
            raise ValueError(f"kernel_ms {kernel_ms} is not a positive finite time")
    error = raw.get("error")
    mismatch = raw.get("mismatch")
    return {"error": None if error is None else str(error),
            "mismatch": None if mismatch is None else str(mismatch),
            "imported": bool(raw.get("imported", False)),
            "n_correct_draws": int(raw.get("n_correct_draws", 0)),
            "kernel_ms": kernel_ms}

def _score(result: dict[str, Any], task: ModuleType,
           assignment: Assignment) -> tuple[dict[str, float], dict[str, Any]]:
    rewards = dict.fromkeys(REWARD_KEYS, 0.0)
    details: dict[str, Any] = {"error": result["error"], "mismatch": result["mismatch"],
                               "assignment": asdict(assignment), "n_correct_draws": result["n_correct_draws"]}
    correct = result["n_correct_draws"] == contract.CORRECTNESS_DRAWS
    rewards["imported"] = 1.0 if result["imported"] else 0.0
    rewards["correct"] = 1.0 if correct else 0.0
    kernel_ms = result["kernel_ms"]
    if correct and kernel_ms is not None:
        eager_ms = contract.baseline_ms(task.reference, task.make_inputs, assignment.warmup, assignment.timing)
        details["median_ms"] = {"kernel": kernel_ms, "eager": eager_ms}
        rewards["speedup_vs_eager"] = eager_ms / kernel_ms
        rewards["passed"] = 1.0
    return rewards, details

def _print_report(rewards: dict[str, float], details: dict[str, Any]) -> None:
    if details["error"]:
        print(f"run failed:\n{details['error']}")
    print(f"imported: {_yes_no(rewards['imported'])}")
    print(f"correct:  {_yes_no(rewards['correct'])} ({details['n_correct_draws']}/{contract.CORRECTNESS_DRAWS} draws)")
    if details["mismatch"]:
        print(f"mismatch: {details['mismatch']}")
    if "median_ms" in details:
        ms = details["median_ms"]
        print(f"median ms over {contract.TIMING_DRAWS} draws: kernel={ms['kernel']:.4f} eager={ms['eager']:.4f}")
        print(f"speedup vs eager: {rewards['speedup_vs_eager']:.3f}x")
    print(f"passed:   {_yes_no(rewards['passed'])}")

def _yes_no(value: float) -> str:
    return "yes" if value else "no"

def _evaluate_kernel(assignment: Assignment) -> dict[str, Any]:
    torch._dynamo.config.disable = True
    result: dict[str, Any] = {"error": None}

    def enter_stage(stage: str) -> None:
        result["stage"] = stage
        print(f"\n{assignment.sentinel}{json.dumps(result)}", flush=True)

    try:
        task = _load_module(TASK_PATH, TASK_MODULE)
        reference, make_inputs = task.reference, task.make_inputs
        enter_stage("the kernel import")
        kernel = _load_module(KERNEL_PATH, KERNEL_MODULE)
        result["imported"] = True
        result["n_correct_draws"] = 0
        for n_draw, seed in enumerate(assignment.correctness, start=1):
            enter_stage(f"correctness draw {n_draw} of {contract.CORRECTNESS_DRAWS}")
            actual = kernel.solve(*contract.drawn_inputs(make_inputs, seed))
            torch.cuda.synchronize()
            matched = contract.matches_reference(actual, reference, make_inputs, seed)
            result["n_correct_draws"] += matched
            if not matched and "mismatch" not in result:
                result["mismatch"] = _mismatch_note(actual, reference, make_inputs, seed)
        if result["n_correct_draws"] == contract.CORRECTNESS_DRAWS:
            enter_stage("timing")
            result["kernel_ms"] = contract.median_ms(kernel.solve, make_inputs, assignment.warmup, assignment.timing,
                                                     check=_timing_check(reference, make_inputs, assignment.timing))
    except (Exception, SystemExit) as exc:
        result["error"] = _kernel_traceback(exc)
    result.pop("stage", None)
    return result

def _timing_check(reference: contract.Reference, make_inputs: contract.MakeInputs,
                  timing: list[int]) -> Callable[[Any, int], None]:
    check_seeds = set(SYSTEM_RNG.sample(timing, TIMING_CHECK_DRAWS))

    def check(out: Any, seed: int) -> None:
        if seed in check_seeds and not contract.matches_reference(out, reference, make_inputs, seed):
            raise ValueError(f"kernel output diverged from the reference during timing: "
                             f"{_mismatch_note(out, reference, make_inputs, seed)}")

    return check

def _mismatch_note(actual: Any, reference: contract.Reference, make_inputs: contract.MakeInputs, seed: int) -> str:
    with contract.reference_numerics():
        expected = reference(*contract.drawn_inputs(make_inputs, seed))
        anchor_inputs = [x.to(torch.float64) if type(x) is torch.Tensor and x.dtype in contract.ANCHOR_DTYPES else x
                         for x in contract.drawn_inputs(make_inputs, seed)]
        anchor = reference(*anchor_inputs)
    note = _describe_mismatch(actual, expected, anchor)
    return note if note is not None else "the mismatch did not reproduce when recomputed for this diagnosis"

def _describe_mismatch(actual: Any, expected: contract.Output, anchor: contract.Output,
                       path: str = "output") -> str | None:
    if type(expected) in (tuple, list):
        if type(actual) is not type(expected):
            return f"{path} is {type(actual).__name__}, expected {type(expected).__name__}"
        if len(actual) != len(expected):
            return f"{path} has length {len(actual)}, expected {len(expected)}"
        for n, (actual_item, expected_item, anchor_item) in enumerate(zip(actual, expected, anchor, strict=True)):
            note = _describe_mismatch(actual_item, expected_item, anchor_item, f"{path}[{n}]")
            if note is not None:
                return note
        return None
    if type(actual) is not torch.Tensor:
        return f"{path} is {type(actual).__name__}, expected a tensor"
    if actual.shape != expected.shape:
        return f"{path} has shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    if actual.dtype != expected.dtype:
        return f"{path} has dtype {actual.dtype}, expected {expected.dtype}"
    if actual.device != expected.device:
        return f"{path} is on device {actual.device}, expected {expected.device}"
    if not expected.dtype.is_floating_point:
        n_wrong = int((actual != expected).sum())
        return f"{path} differs from the reference in {n_wrong} of {expected.numel()} elements" if n_wrong else None
    if expected.numel() == 0:
        return None
    anchor64 = anchor.to(torch.float64)
    reference_error = float((expected.to(torch.float64) - anchor64).abs().max())
    allowance_floor = torch.finfo(expected.dtype).eps * float(anchor64.abs().max())
    allowance = contract.ALLOWANCE_FACTOR * max(reference_error, allowance_floor)
    errors = (actual.to(torch.float64) - anchor64).abs()
    error = float(errors.max())
    if not math.isfinite(error):
        return f"{path} contains non-finite values"
    if error <= allowance:
        return None
    n_over = int((errors > allowance).sum())
    ratio_text = f" ({error / allowance:.1f}x over)" if allowance > 0 else ""
    return (f"{path} max abs error {error:.3e} exceeds the allowance {allowance:.3e}{ratio_text}, "
            f"{n_over} of {expected.numel()} elements over")

def _kernel_traceback(exc: BaseException) -> str:
    formatted = traceback.TracebackException.from_exception(exc)
    pending = [formatted]
    while pending:
        part = pending.pop()
        part.stack = traceback.StackSummary.from_list([frame for frame in part.stack if not _hidden_frame(frame)])
        pending.extend(chained for chained in (part.__cause__, part.__context__) if chained is not None)
    return "".join(formatted.format())

def _hidden_frame(frame: traceback.FrameSummary) -> bool:
    return frame.filename.startswith("<frozen importlib") or Path(frame.filename).resolve() == GRADER_PATH

def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module

if __name__ == "__main__":
    main()
