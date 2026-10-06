"""The GPU judge: every evaluation runs the private grader in a fresh, network-less B200 container."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import modal

from metakernelbench.build import DOCKERFILE_NAME, GRADER_NAME, environment_dir, grader_path
from metakernelbench.catalog import KERNEL_BYTE_LIMIT, KERNEL_PATH, TARGET_GPU
from metakernelbench.results import error_text

GRADER_TIMEOUT_SEC = 900
FUNCTION_TIMEOUT_SEC = GRADER_TIMEOUT_SEC + 120
MAX_EVALUATIONS_IN_FLIGHT = 10
CONTAINER_CPUS = 4.0
CONTAINER_MEMORY_MB = 16384
GRADER_REMOTE_DIR = "/tests"
VERIFIER_DIR = "/logs/verifier"

@dataclass(frozen=True)
class Verdict:
    rewards: dict[str, float]
    details: dict[str, Any]
    report: str
    stderr: str
    eval_sec: float

    @classmethod
    def unevaluated(cls, report: str) -> Self:
        return cls(rewards={}, details={}, report=report, stderr="", eval_sec=0.0)

class JudgeFailure(Exception):
    pass

class Judge:
    def __init__(self, tasks: list[str]) -> None:
        self._app: modal.App = modal.App("mkbench-judge")
        self._functions: dict[str, modal.Function] = {}
        self._run_context: Any = None
        self._gpu_slots: asyncio.Semaphore = asyncio.Semaphore(MAX_EVALUATIONS_IN_FLIGHT)
        evaluate = _make_evaluate(f"{GRADER_REMOTE_DIR}/{GRADER_NAME}", KERNEL_PATH, VERIFIER_DIR, GRADER_TIMEOUT_SEC)
        for task in tasks:
            environment = environment_dir(task)
            image = modal.Image.from_dockerfile(environment / DOCKERFILE_NAME, context_dir=environment)
            image = image.add_local_file(grader_path(task), f"{GRADER_REMOTE_DIR}/{GRADER_NAME}")
            self._functions[task] = self._app.function(
                name=f"evaluate_{task}", image=image, gpu=TARGET_GPU, cpu=CONTAINER_CPUS, memory=CONTAINER_MEMORY_MB,
                timeout=FUNCTION_TIMEOUT_SEC, single_use_containers=True, block_network=True,
                restrict_modal_access=True, serialized=True)(evaluate)

    async def __aenter__(self) -> Self:
        self._run_context = self._app.run()
        await self._run_context.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> bool | None:
        return await self._run_context.__aexit__(*exc_info)

    async def evaluate(self, task: str, kernel_source: str) -> Verdict:
        n_bytes = len(kernel_source.encode())
        if n_bytes > KERNEL_BYTE_LIMIT:
            return Verdict.unevaluated(
                f"kernel.py is {n_bytes} bytes, over the {KERNEL_BYTE_LIMIT} byte limit, not evaluated")
        async with self._gpu_slots:
            try:
                outcome = await self._functions[task].remote.aio(kernel_source)
            except Exception as exc:
                raise JudgeFailure(error_text(type(exc).__name__, exc)) from exc
        return Verdict(**outcome)

def _make_evaluate(remote_grader_path: str, remote_kernel_path: str, remote_verifier_dir: str,
                   timeout_sec: int) -> Callable[[str], dict[str, Any]]:
    def evaluate(kernel_source: str) -> dict[str, Any]:
        import json
        import subprocess
        import sys
        import time
        from pathlib import Path

        report_limit = 20_000
        stderr_limit = 2_000
        started = time.monotonic()
        Path(remote_kernel_path).write_text(kernel_source)
        try:
            process = subprocess.run([sys.executable, remote_grader_path], capture_output=True, text=True,
                                     timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"grader ran past {timeout_sec}s, the kernel itself is bounded inside it") from exc
        reward_path = Path(remote_verifier_dir) / "reward.json"
        if not reward_path.exists():
            raise RuntimeError(f"grader died with exit {process.returncode}: {process.stderr[-stderr_limit:]}")
        return {"rewards": json.loads(reward_path.read_text()),
                "details": json.loads((Path(remote_verifier_dir) / "reward-details.json").read_text()),
                "report": process.stdout[-report_limit:],
                "stderr": process.stderr[-stderr_limit:],
                "eval_sec": round(time.monotonic() - started, 1)}
    return evaluate
