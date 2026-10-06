"""One Harbor trial's lifecycle: the sandbox phase under a watchdog, the final verdict, and the retries of each."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.trial import Trial

from metakernelbench.build import task_config, task_dir
from metakernelbench.catalog import KERNEL_PATH, TARGET_GPU
from metakernelbench.host import log_progress, run_hosted_trial
from metakernelbench.judge import Judge, JudgeFailure, Verdict
from metakernelbench.results import TrialRecord, Usage, error_text
from metakernelbench.store import VERDICT_NAME, Store, archive_trial, artifact_path, write_json

AGENT_NAME = AgentName.TERMINUS_2.value
AGENT_SETUP_TIMEOUT_SEC = 360
SANDBOX_PHASE_SLACK_SEC = 1200
MAX_ATTEMPTS = 3
RETRY_WAIT_SEC = 30
TRANSIENT_FAILURES = frozenset((
    "EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "RateLimitError", "APIConnectionError",
    "ServiceUnavailableError", "BadGatewayError", "InternalServerError", "Timeout", "APIError",
))
OPENROUTER_PROVIDER_ORDER = {"openrouter/openai/gpt-5.6-sol": ("openai/flex", "openai")}

@dataclass(frozen=True)
class Resources:
    store: Store
    semaphore: asyncio.Semaphore
    judge: Judge

async def run_trial(resources: Resources, task: str, attachment: Path | None, trial_dir: Path) -> TrialRecord:
    trial_name = trial_dir.name
    try:
        result = await _attempt(task, trial_name, "sandbox phase",
                                lambda: _run_sandbox_phase(resources, task, attachment, trial_dir))
        verdict = await _attempt(task, trial_name, "final verdict",
                                 lambda: _grade_final_kernel(resources.judge, task, trial_dir))
    except _TrialFailure as failure:
        record = TrialRecord(error=str(failure))
    else:
        record = TrialRecord(rewards=verdict.rewards, timed_out=_agent_timed_out(result), usage=_agent_usage(result))
    _log_final_result(task, trial_name, record)
    return record

class _TrialFailure(Exception):
    def __init__(self, exception_type: str, message: str, transient: bool) -> None:
        super().__init__(error_text(exception_type, message))
        self.transient: bool = transient

async def _attempt[T](task: str, trial_name: str, phase: str, action: Callable[[], Awaitable[T]]) -> T:
    n_attempt = 1
    while True:
        try:
            return await action()
        except _TrialFailure as failure:
            if not failure.transient or n_attempt == MAX_ATTEMPTS:
                raise
            log_progress(task, trial_name,
                         f"{phase} attempt {n_attempt}/{MAX_ATTEMPTS} hit {failure}; retrying in {RETRY_WAIT_SEC}s")
            await asyncio.sleep(RETRY_WAIT_SEC)
            n_attempt += 1

async def _run_sandbox_phase(resources: Resources, task: str, attachment: Path | None,
                             trial_dir: Path) -> TrialResult:
    trial_name = trial_dir.name
    if trial_dir.exists():
        archive_trial(resources.store, trial_dir)
    limit_sec = _sandbox_phase_limit_sec(task)
    config = _trial_config(resources.store.model, task, attachment, trial_dir, limit_sec)
    log_progress(task, trial_name, "waiting for a sandbox slot")
    async with resources.semaphore:
        log_progress(task, trial_name, "sandbox slot acquired; creating the Harbor trial")
        try:
            async with asyncio.timeout(limit_sec):
                trial = await Trial.create(config)
                log_progress(task, trial_name, "Harbor trial created; starting the sandbox and the agent")
                result = await run_hosted_trial(resources.judge, trial, task, trial_dir)
        except TimeoutError:
            raise _TrialFailure(TimeoutError.__name__, f"sandbox phase exceeded {limit_sec}s",
                                transient=True) from None
    log_progress(task, trial_name, "sandbox released")
    failure = result.exception_info
    if failure is not None and not _agent_timed_out(result):
        raise _TrialFailure(failure.exception_type, failure.exception_message,
                            transient=failure.exception_type in TRANSIENT_FAILURES)
    return result

def _sandbox_phase_limit_sec(task: str) -> int:
    config = task_config(task)
    return int(config.environment.build_timeout_sec + AGENT_SETUP_TIMEOUT_SEC + config.agent.timeout_sec
               + SANDBOX_PHASE_SLACK_SEC)

def _agent_config(model: str) -> AgentConfig:
    order = OPENROUTER_PROVIDER_ORDER.get(model)
    kwargs = {} if order is None else {
        "llm_call_kwargs": {"extra_body": {"provider": {"order": list(order), "allow_fallbacks": False}}},
    }
    return AgentConfig(name=AGENT_NAME, model_name=model, override_setup_timeout_sec=AGENT_SETUP_TIMEOUT_SEC,
                       kwargs=kwargs)

def _trial_config(model: str, task: str, attachment: Path | None, trial_dir: Path, limit_sec: int) -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(path=task_dir(task)),
        trial_name=trial_dir.name,
        trials_dir=trial_dir.parent,
        agent=_agent_config(model),
        environment=EnvironmentConfig(type=EnvironmentType.MODAL, kwargs={"sandbox_timeout_secs": limit_sec}),
        verifier=VerifierConfig(disable=True),
        extra_instruction_paths=[] if attachment is None else [attachment],
    )

async def _grade_final_kernel(judge: Judge, task: str, trial_dir: Path) -> Verdict:
    trial_name = trial_dir.name
    kernel_path = artifact_path(trial_dir, KERNEL_PATH)
    if kernel_path.is_file():
        log_progress(task, trial_name, f"starting the final {TARGET_GPU} verdict")
        started = time.monotonic()
        try:
            verdict = await judge.evaluate(task, kernel_path.read_text(errors="replace"))
        except JudgeFailure as failure:
            raise _TrialFailure(JudgeFailure.__name__, str(failure), transient=True) from failure
        log_progress(task, trial_name, f"final {TARGET_GPU} verdict returned in {time.monotonic() - started:.0f}s")
    else:
        log_progress(task, trial_name, "final kernel is missing; final verdict cannot run")
        verdict = Verdict.unevaluated(f"no {KERNEL_PATH} was collected")
    write_json(trial_dir / VERDICT_NAME, asdict(verdict))
    return verdict

def _agent_timed_out(result: TrialResult) -> bool:
    failure = result.exception_info
    return failure is not None and failure.exception_type == AgentTimeoutError.__name__

def _agent_usage(result: TrialResult) -> Usage:
    n_input, n_cache, n_output, cost = result.compute_token_cost_totals()
    timing = result.agent_execution
    return Usage(n_input_tokens=n_input, n_cache_tokens=n_cache, n_output_tokens=n_output, cost_usd=cost,
                 agent_sec=(timing.finished_at - timing.started_at).total_seconds())

def _log_final_result(task: str, trial_name: str, record: TrialRecord) -> None:
    if record.error is not None:
        log_progress(task, trial_name, f"final result: {record.error}")
        return
    timeout_note = " (timed out)" if record.timed_out else ""
    log_progress(task, trial_name, f"final result: passed={record.rewards.get('passed', 0.0):g} "
                                   f"vs_eager={record.rewards.get('speedup_vs_eager', 0.0):.3f}x{timeout_note}")
