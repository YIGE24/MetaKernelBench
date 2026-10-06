"""The transfer protocol as convergence: run what the plan lacks, solos before skills, each trial failing alone."""

import asyncio
from collections.abc import Awaitable
from dataclasses import replace
from typing import Any

import harbor

from metakernelbench.host import log_progress
from metakernelbench.judge import Judge
from metakernelbench.plan import Invocation, PlannedTrial, TrialStatus, grouped, plan, print_plan
from metakernelbench.results import (
    CellResult, TrialRecord, collect_results, error_text, write_results, write_trial_record)
from metakernelbench.skill import skill_attachment_text, skill_digest
from metakernelbench.store import append_invocation, archive_trial, repo_commit, timestamp
from metakernelbench.trial import Resources, run_trial

async def run_experiment(invocation: Invocation) -> None:
    planned = plan(invocation)
    if not planned:
        raise SystemExit(f"no trial in the selected cells matches --trials {invocation.trials_pattern!r}")
    counts = {status: sum(trial.status is status for trial in planned) for status in TrialStatus}
    print_plan(planned, counts, invocation.dry_run)
    if invocation.dry_run:
        return
    if counts[TrialStatus.STALE] and not invocation.rerun_stale:
        raise SystemExit("stale trials block this invocation, pass --rerun-stale to archive and redo them")
    runnable = [trial for trial in planned if trial.status is not TrialStatus.REUSE]
    if runnable:
        append_invocation(invocation.store, _invocation_entry(invocation, counts))
        await _run_trials(invocation, runnable)
    _print_store_state(invocation, _write_snapshot(invocation))

async def _run_trials(invocation: Invocation, runnable: list[PlannedTrial]) -> None:
    snapshot_lock = asyncio.Lock()
    async with Judge(sorted({trial.task for trial in runnable})) as judge:
        resources = Resources(store=invocation.store, semaphore=asyncio.Semaphore(invocation.n_concurrent), judge=judge)
        async with asyncio.TaskGroup() as group:
            for problem, trials in grouped(runnable, key=lambda trial: trial.problem):
                group.create_task(_run_problem(invocation, resources, problem, trials, snapshot_lock))

async def _run_problem(invocation: Invocation, resources: Resources, problem: str, trials: list[PlannedTrial],
                       snapshot_lock: asyncio.Lock) -> None:
    try:
        async with asyncio.TaskGroup() as group:
            for _, replicate_trials in grouped(trials, key=lambda trial: trial.n_replicate):
                group.create_task(_run_replicate(invocation, resources, replicate_trials))
    except Exception as exc:
        error = "; ".join(dict.fromkeys(error_text(type(leaf).__name__, leaf) for leaf in _leaf_exceptions(exc)))
        log_progress(problem, "convergence", error)
    async with snapshot_lock:
        _write_snapshot(invocation)

async def _run_replicate(invocation: Invocation, resources: Resources, trials: list[PlannedTrial]) -> None:
    for trial in trials:
        if trial.trial_dir.exists():
            archive_trial(invocation.store, trial.trial_dir)
    solos = [trial for trial in trials if trial.source is None]
    skills = [trial for trial in trials if trial.source is not None]
    async with asyncio.TaskGroup() as group:
        handles = {trial.arm: group.create_task(_guarded(trial, _run_solo(resources, trial))) for trial in solos}
        for trial in skills:
            group.create_task(_guarded(trial, _run_skill(invocation, resources, trial, handles.get(trial.source.arm))))

async def _guarded(trial: PlannedTrial, attempt: Awaitable[TrialRecord | None]) -> TrialRecord | None:
    try:
        return await attempt
    except Exception as exc:
        error = error_text(type(exc).__name__, exc)
        log_progress(trial.task, trial.coordinate(), f"host-side failure, no record written: {error}")
        return TrialRecord(error=error)

async def _run_solo(resources: Resources, trial: PlannedTrial) -> TrialRecord:
    record = await run_trial(resources, trial.task, None, trial.trial_dir)
    write_trial_record(trial.trial_dir, record, trial.stamps)
    return record

async def _run_skill(invocation: Invocation, resources: Resources, trial: PlannedTrial,
                     source_handle: asyncio.Task[TrialRecord | None] | None) -> None:
    if source_handle is not None and (await source_handle).error is not None:
        log_progress(trial.task, trial.coordinate(), "skipped, its source solo left no usable record")
        return
    skill_text = skill_attachment_text(trial.source.trial_dir)
    if skill_text is None:
        log_progress(trial.task, trial.coordinate(), "skipped and recorded, its source solo wrote no SKILL.md")
        write_trial_record(trial.trial_dir, TrialRecord(skipped="the source solo wrote no SKILL.md"), trial.stamps)
        return
    attachment = invocation.store.skill_attachment_path(trial.problem, trial.n_replicate, trial.cell.direction())
    attachment.parent.mkdir(parents=True, exist_ok=True)
    attachment.write_text(skill_text)
    record = await run_trial(resources, trial.task, attachment, trial.trial_dir)
    write_trial_record(trial.trial_dir, record, replace(trial.stamps, skill_sha256=skill_digest(skill_text)))

def _invocation_entry(invocation: Invocation, counts: dict[TrialStatus, int]) -> dict[str, Any]:
    return {"started_at": timestamp(),
            "model": invocation.store.model,
            "categories": sorted(invocation.categories),
            "problems": invocation.problems_pattern,
            "trials": invocation.trials_pattern,
            "n_replicates": invocation.n_replicates,
            "n_concurrent": invocation.n_concurrent,
            "rerun_stale": invocation.rerun_stale,
            "harbor_version": harbor.__version__,
            "commit": repo_commit(),
            "plan": {str(status): count for status, count in counts.items()}}

def _write_snapshot(invocation: Invocation) -> list[CellResult]:
    results = collect_results(invocation.store)
    write_results(invocation.store, results)
    return results

def _print_store_state(invocation: Invocation, results: list[CellResult]) -> None:
    resolved = sum(rep.resolved() for result in results for rep in result.replicates)
    total = sum(len(result.replicates) for result in results)
    print(f"store holds {resolved}/{total} resolved cell-replicates, snapshot at {invocation.store.snapshot_path()}")

def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaf_exceptions(sub)]
    return [exc]
