"""The host side of an agent run: the submission mailbox paired with test.sh, the heartbeat, the narration."""

import asyncio
import contextlib
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial

from metakernelbench.catalog import KERNEL_BYTE_LIMIT, KERNEL_PATH, SUBMISSION_LIMIT, TARGET_GPU
from metakernelbench.judge import Judge, JudgeFailure
from metakernelbench.results import error_text
from metakernelbench.store import SUBMISSIONS_NAME, append_jsonl

SUBMISSIONS_DIR = "/app/.submissions"
CLOCK_PATH = "/app/.started"
POLL_INTERVAL_SEC = 2.0
HEARTBEAT_SEC = 60

def log_progress(task: str, trial_name: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{task}/{trial_name}] {message}", flush=True)

async def run_hosted_trial(judge: Judge, trial: Trial, task: str, trial_dir: Path) -> TrialResult:
    host = _Host(judge, trial, task, trial_dir)

    async def mark_environment_start(_event: TrialHookEvent) -> None:
        host.log("sandbox environment startup began")

    async def mark_active(_event: TrialHookEvent) -> None:
        host.started_at = time.monotonic()
        host.budget_sec = trial.task.config.agent.timeout_sec
        host.agent_active.set()
        budget = _duration_text(host.budget_sec) if host.budget_sec is not None else "unbounded"
        host.log(f"agent started; time budget={budget}")
        if host.budget_sec is not None:
            await _start_clock(host, int(host.budget_sec))

    async def mark_done(_event: TrialHookEvent) -> None:
        host.agent_done.set()
        if host.started_at is None:
            host.log("agent ended before becoming active")
            return
        host.log(f"agent ended after {_duration_text(host.elapsed_sec())}; "
                 f"submissions={host.submissions_used}/{SUBMISSION_LIMIT}")

    trial.add_hook(TrialEvent.ENVIRONMENT_START, mark_environment_start)
    trial.add_hook(TrialEvent.AGENT_START, mark_active)
    trial.add_hook(TrialEvent.AGENT_END, mark_done)
    mailbox = asyncio.create_task(_serve_mailbox(host))
    heartbeat = asyncio.create_task(_log_heartbeats(host))
    try:
        return await trial.run()
    finally:
        host.agent_done.set()
        mailbox.cancel()
        heartbeat.cancel()
        await asyncio.gather(mailbox, heartbeat, return_exceptions=True)

@dataclass
class _Host:
    judge: Judge
    trial: Trial
    task: str
    trial_dir: Path
    agent_active: asyncio.Event = field(default_factory=asyncio.Event)
    agent_done: asyncio.Event = field(default_factory=asyncio.Event)
    started_at: float | None = None
    budget_sec: float | None = None
    submissions_used: int = 0
    gpu_state: str = "no submissions yet"

    def log(self, message: str) -> None:
        log_progress(self.task, self.trial.config.trial_name, message)

    def elapsed_sec(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def remaining_sec(self) -> float | None:
        return None if self.budget_sec is None else max(0.0, self.budget_sec - self.elapsed_sec())

async def _start_clock(host: _Host, budget_sec: int) -> None:
    try:
        started = await host.trial.agent_environment.exec(f"echo $(date +%s) {budget_sec} > {CLOCK_PATH}")
    except Exception as exc:
        host.log(f"shell clock not started: {error_text(type(exc).__name__, exc)}")
        return
    if started.return_code != 0:
        host.log(f"shell clock not started: exit {started.return_code}")

async def _log_heartbeats(host: _Host) -> None:
    while not await _agent_ended_within(host, HEARTBEAT_SEC):
        if host.started_at is None:
            host.log("heartbeat | sandbox setup still in progress; agent has not started")
            continue
        remaining = host.remaining_sec()
        remaining_text = "unbounded" if remaining is None else _duration_text(remaining)
        host.log(f"heartbeat | elapsed={_duration_text(host.elapsed_sec())} | remaining={remaining_text} | "
                 f"submissions={host.submissions_used}/{SUBMISSION_LIMIT} | gpu={host.gpu_state}")

async def _serve_mailbox(host: _Host) -> None:
    served: set[str] = set()
    undelivered: dict[str, str] = {}
    await _wait_for_first(host.agent_active, host.agent_done)
    while not await _agent_ended_within(host, POLL_INTERVAL_SEC):
        try:
            for request_id in await _pending_request_ids(host, served):
                host.log(f"submission request detected; {host.submissions_used}/{SUBMISSION_LIMIT} used")
                undelivered[request_id] = await _serve_submission(host, request_id)
                served.add(request_id)
            for request_id, text in list(undelivered.items()):
                await _deliver_result(host, request_id, text)
                del undelivered[request_id]
        except Exception as exc:
            host.log(f"submission mailbox failed: {error_text(type(exc).__name__, exc)}; polling continues")

async def _agent_ended_within(host: _Host, seconds: float) -> bool:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(host.agent_done.wait(), seconds)
    return host.agent_done.is_set()

async def _wait_for_first(*events: asyncio.Event) -> None:
    waits = [asyncio.create_task(event.wait()) for event in events]
    try:
        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for wait in waits:
            wait.cancel()

async def _pending_request_ids(host: _Host, served: set[str]) -> list[str]:
    listing = await host.trial.agent_environment.exec(f"ls -1 {SUBMISSIONS_DIR} 2>/dev/null || true")
    names = (listing.stdout or "").split()
    requests = {name.removesuffix(".request") for name in names if name.endswith(".request")}
    results = {name.removesuffix(".result") for name in names if name.endswith(".result")}
    return sorted(requests - results - served)

async def _serve_submission(host: _Host, request_id: str) -> str:
    elapsed_sec = round(host.elapsed_sec(), 1)
    if host.submissions_used >= SUBMISSION_LIMIT:
        host.gpu_state = "submission limit reached"
        return _refuse(host, request_id, elapsed_sec, "limit reached",
                       f"submission limit reached ({SUBMISSION_LIMIT} used), no evaluation was run; "
                       f"the final {KERNEL_PATH} is still graded when the budget ends")
    snapshot = await host.trial.agent_environment.exec(
        f"head -c {KERNEL_BYTE_LIMIT + 1} {SUBMISSIONS_DIR}/{request_id}.request")
    if snapshot.return_code != 0:
        return _refuse(host, request_id, elapsed_sec, "unreadable request",
                       "your submission could not be read, not counted against your limit")
    if len((snapshot.stdout or "").encode()) > KERNEL_BYTE_LIMIT:
        return _refuse(host, request_id, elapsed_sec, "over size limit",
                       f"kernel.py is over {KERNEL_BYTE_LIMIT} bytes, not evaluated and not counted against your limit")
    submission_number = host.submissions_used + 1
    host.gpu_state = f"submission {submission_number}/{SUBMISSION_LIMIT} evaluating on {TARGET_GPU}"
    host.log(f"submission {submission_number}/{SUBMISSION_LIMIT} queued for {TARGET_GPU}")
    started = time.monotonic()
    try:
        outcome = await host.judge.evaluate(host.task, snapshot.stdout or "")
    except asyncio.CancelledError:
        _log_submission(host, request_id, elapsed_sec, counted=False, note="cancelled at agent end")
        raise
    except JudgeFailure as failure:
        host.gpu_state = f"submission {submission_number} infrastructure error"
        return _refuse(host, request_id, elapsed_sec, str(failure),
                       "evaluation infrastructure error, not counted against your limit")
    wall_sec = round(time.monotonic() - started, 1)
    host.submissions_used = submission_number
    host.gpu_state = f"submission {submission_number} completed"
    host.log(f"submission {submission_number}/{SUBMISSION_LIMIT} returned in {wall_sec:.0f}s | "
             f"correct={outcome.rewards.get('correct', 0):g} | passed={outcome.rewards.get('passed', 0):g} | "
             f"vs_eager={outcome.rewards.get('speedup_vs_eager', 0):.3f}x")
    _log_submission(host, request_id, elapsed_sec, counted=True, rewards=outcome.rewards, wall_sec=wall_sec,
                    eval_sec=outcome.eval_sec)
    return (f"submission {submission_number}/{SUBMISSION_LIMIT}, evaluated on {TARGET_GPU} in {wall_sec:.0f}s\n"
            f"{_budget_status(host)}\n\n{outcome.report}\n")

def _refuse(host: _Host, request_id: str, elapsed_sec: float, note: str, reason: str) -> str:
    host.log(f"submission not counted: {note}")
    _log_submission(host, request_id, elapsed_sec, counted=False, note=note)
    return f"{reason}\n{_budget_status(host)}\n"

def _budget_status(host: _Host) -> str:
    remaining = host.remaining_sec()
    minutes_left = "unbounded" if remaining is None else int(remaining / 60)
    return (f"{SUBMISSION_LIMIT - host.submissions_used} of {SUBMISSION_LIMIT} submissions and "
            f"{minutes_left} minutes of budget left")

def _log_submission(host: _Host, request_id: str, elapsed_sec: float, counted: bool,
                    rewards: dict[str, float] | None = None, note: str | None = None,
                    wall_sec: float | None = None, eval_sec: float | None = None) -> None:
    entry = {"id": request_id, "elapsed_sec": elapsed_sec, "counted": counted, "rewards": rewards, "note": note,
             "wall_sec": wall_sec, "eval_sec": eval_sec}
    append_jsonl(host.trial_dir / SUBMISSIONS_NAME, entry)

async def _deliver_result(host: _Host, request_id: str, text: str) -> None:
    if host.agent_done.is_set():
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        host_path = Path(tmp_dir) / "result"
        host_path.write_text(text)
        await host.trial.agent_environment.upload_file(host_path, f"{SUBMISSIONS_DIR}/{request_id}.result.tmp")
    await host.trial.agent_environment.exec(
        f"mv {SUBMISSIONS_DIR}/{request_id}.result.tmp {SUBMISSIONS_DIR}/{request_id}.result")

def _duration_text(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    minutes, seconds = divmod(whole_seconds, 60)
    return f"{minutes}m{seconds:02d}s"
