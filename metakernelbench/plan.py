"""Convergent planning: a status for every selected trial and its dependencies, reading the disk only."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path

from metakernelbench.build import task_fingerprint
from metakernelbench.catalog import CellSpec, cell_tasks, skill_arm, solo_arm, task_name, trial_coordinate
from metakernelbench.results import StoredTrial, TrialStamps, load_trial_record
from metakernelbench.skill import skill_attachment_text, skill_digest
from metakernelbench.store import Store

@dataclass(frozen=True)
class Invocation:
    store: Store
    cells: list[CellSpec]
    categories: frozenset[str]
    problems_pattern: str
    trials_pattern: str
    n_concurrent: int
    n_replicates: int
    rerun_stale: bool
    dry_run: bool

class TrialStatus(StrEnum):
    REUSE = "reuse"
    RUN = "run"
    RERUN = "rerun"
    STALE = "stale"

@dataclass(frozen=True)
class PlannedTrial:
    problem: str
    n_replicate: int
    arm: str
    task: str
    status: TrialStatus
    stamps: TrialStamps
    trial_dir: Path
    cell: CellSpec | None = None
    source: "PlannedTrial | None" = None

    def coordinate(self) -> str:
        return trial_coordinate(self.problem, self.n_replicate, self.arm)

def plan(invocation: Invocation) -> list[PlannedTrial]:
    fresh = {task: TrialStamps(task_fingerprint(task), None) for task in cell_tasks(invocation.cells)}
    planned: list[PlannedTrial] = []
    for _, cells in grouped(invocation.cells, key=lambda cell: cell.problem):
        for n_replicate in range(1, invocation.n_replicates + 1):
            planned.extend(_plan_replicate(invocation, cells, n_replicate, fresh))
    return planned

def print_plan(planned: list[PlannedTrial], counts: dict[TrialStatus, int], dry_run: bool) -> None:
    print("plan: " + ", ".join(f"{count} {status}" for status, count in counts.items()))
    shown = (TrialStatus.RUN, TrialStatus.RERUN, TrialStatus.STALE) if dry_run else (TrialStatus.STALE,)
    for trial in planned:
        if trial.status in shown:
            print(f"  {trial.status}: {trial.coordinate()} [{trial.task}]")

def grouped[T, K](items: list[T], key: Callable[[T], K]) -> list[tuple[K, list[T]]]:
    groups: dict[K, list[T]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return sorted(groups.items())

def _plan_replicate(invocation: Invocation, cells: list[CellSpec], n_replicate: int,
                    fresh: dict[str, TrialStamps]) -> list[PlannedTrial]:
    problem = cells[0].problem

    def selected(arm: str) -> bool:
        return fnmatch(trial_coordinate(problem, n_replicate, arm), invocation.trials_pattern)

    skill_cells = [cell for cell in cells if selected(skill_arm(cell.direction()))]
    solo_tasks = {solo_arm(dsl): task_name(problem, dsl)
                  for cell in cells for dsl in (cell.source_dsl, cell.target_dsl)
                  if selected(solo_arm(dsl))}
    for cell in skill_cells:
        solo_tasks.setdefault(solo_arm(cell.source_dsl), cell.source_task())
    solos: dict[str, PlannedTrial] = {}
    for arm, task in sorted(solo_tasks.items()):
        trial_dir = invocation.store.trial_dir(problem, n_replicate, arm)
        status = _record_status(load_trial_record(trial_dir), fresh[task])
        solos[arm] = PlannedTrial(problem, n_replicate, arm, task, status, fresh[task], trial_dir)
    skills = [_plan_skill(invocation, cell, n_replicate, solos, fresh) for cell in skill_cells]
    return [*solos.values(), *skills]

def _plan_skill(invocation: Invocation, cell: CellSpec, n_replicate: int, solos: dict[str, PlannedTrial],
                fresh: dict[str, TrialStamps]) -> PlannedTrial:
    arm = skill_arm(cell.direction())
    task = cell.target_task()
    source = solos[solo_arm(cell.source_dsl)]
    trial_dir = invocation.store.trial_dir(cell.problem, n_replicate, arm)
    status = _skill_status(trial_dir, fresh[task], source)
    return PlannedTrial(cell.problem, n_replicate, arm, task, status, fresh[task], trial_dir, cell=cell, source=source)

def _record_status(stored: StoredTrial | None, fresh: TrialStamps) -> TrialStatus:
    if stored is None:
        return TrialStatus.RUN
    if stored.record.error is not None:
        return TrialStatus.RERUN
    if stored.stamps.task_fingerprint != fresh.task_fingerprint:
        return TrialStatus.STALE
    return TrialStatus.REUSE

def _skill_status(trial_dir: Path, fresh: TrialStamps, source: PlannedTrial) -> TrialStatus:
    stored = load_trial_record(trial_dir)
    status = _record_status(stored, fresh)
    if status is not TrialStatus.REUSE:
        return status
    if source.status is not TrialStatus.REUSE:
        return TrialStatus.STALE
    if stored.stamps.skill_sha256 != skill_digest(skill_attachment_text(source.trial_dir)):
        return TrialStatus.STALE
    return TrialStatus.REUSE
