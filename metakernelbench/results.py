"""The record of one trial's verdict on disk, and the analysis of cells, replicates, rates, lifts, and usage."""

import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import harbor

from metakernelbench.catalog import CellSpec, cell_specs, skill_arm, solo_arm
from metakernelbench.store import RECORD_NAME, Store, write_json

SPEED_THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 4.0)

LEGEND = {
    "arms": "solo_<dsl> attempts the problem in that DSL from scratch and distills SKILL.md as it goes; "
            "<a>2<b>_skill attempts it in DSL b from scratch with solo_<a>'s SKILL.md attached to the instruction",
    "records": "per cell and replicate: source is the solo in the source DSL, base the solo in the target DSL (the "
               "no-skill baseline), skill the attached attempt; resolved means all three hold a verdict; a skill "
               "record marked skipped was never run because its source wrote no SKILL.md",
    "rates": "R@t is the share of resolved cell-replicates that passed with a speedup over torch eager of at "
             "least t (R@0 is the pass rate, R@1 the eager-matching rate), and S is 100 times the mean over the five "
             "thresholds; lift is skill minus base, where a skipped skill arm counts as its base since nothing was "
             "transferred; with_skill repeats the contrast on the cell-replicates whose source wrote a SKILL.md; "
             "by_source_verdict splits the contrast by whether the source solo passed",
    "source_coverage": "the share of resolved cell-replicates whose source solo passed",
    "skill_coverage": "the share of resolved cell-replicates whose source solo wrote a SKILL.md",
    "usage": "per arm over every trial holding a measured record, each trial counted once: trials counted, summed "
             "cost in USD and tokens (input including cache, cache, output) as the LLM gateway reported them through "
             "Harbor, and the mean agent wall time in seconds; a sum is null when any counted record lacks that number",
}

@dataclass(frozen=True)
class Usage:
    n_input_tokens: int | None
    n_cache_tokens: int | None
    n_output_tokens: int | None
    cost_usd: float | None
    agent_sec: float

@dataclass(frozen=True)
class TrialRecord:
    rewards: dict[str, float] = field(default_factory=dict)
    timed_out: bool = False
    error: str | None = None
    skipped: str | None = None
    usage: Usage | None = None

    def passed(self) -> bool:
        return self.rewards.get("passed", 0.0) >= 1.0

    def attains(self, threshold: float) -> bool:
        return self.passed() and self.rewards.get("speedup_vs_eager", 0.0) >= threshold

@dataclass(frozen=True)
class TrialStamps:
    task_fingerprint: str
    skill_sha256: str | None

@dataclass(frozen=True)
class StoredTrial:
    record: TrialRecord
    stamps: TrialStamps

def error_text(exception_type: str, message: object) -> str:
    return f"{exception_type}: {message}"

def write_trial_record(trial_dir: Path, record: TrialRecord, stamps: TrialStamps) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_json(trial_dir / RECORD_NAME, {**asdict(record), "stamps": asdict(stamps)})

def load_trial_record(trial_dir: Path) -> StoredTrial | None:
    path = trial_dir / RECORD_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    usage = data.get("usage")
    return StoredTrial(
        record=TrialRecord(rewards={key: float(value) for key, value in data["rewards"].items()},
                           timed_out=bool(data["timed_out"]), error=data["error"], skipped=data.get("skipped"),
                           usage=None if usage is None else Usage(**usage)),
        stamps=TrialStamps(task_fingerprint=data["stamps"]["task_fingerprint"],
                           skill_sha256=data["stamps"]["skill_sha256"]))

@dataclass(frozen=True)
class ReplicateResult:
    n_replicate: int
    source: TrialRecord | None
    base: TrialRecord | None
    skill: TrialRecord | None

    def resolved(self) -> bool:
        return all(record is not None and record.error is None for record in (self.source, self.base, self.skill))

    def skill_present(self) -> bool:
        return self.skill.skipped is None

    def skill_outcome(self) -> TrialRecord:
        return self.skill if self.skill_present() else self.base

@dataclass(frozen=True)
class CellResult:
    cell: CellSpec
    replicates: list[ReplicateResult]

def collect_results(store: Store) -> list[CellResult]:
    results: list[CellResult] = []
    by_problem: dict[str, list[CellSpec]] = {}
    for cell in cell_specs():
        by_problem.setdefault(cell.problem, []).append(cell)
    for problem, cells in by_problem.items():
        numbers = _replicate_numbers(store, problem)
        if numbers:
            results.extend(CellResult(cell, [_replicate_from_disk(store, cell, n) for n in numbers]) for cell in cells)
    return results

def write_results(store: Store, results: list[CellResult]) -> None:
    ordered = sorted(results, key=lambda result: (result.cell.problem, result.cell.direction()))
    write_json(store.snapshot_path(), {
        "legend": LEGEND,
        "model": store.model,
        "harbor_version": harbor.__version__,
        "overall": _overall(ordered),
        "cells": [_cell_dict(result) for result in ordered],
    })

def _replicate_numbers(store: Store, problem: str) -> list[int]:
    problem_dir = store.problem_dir(problem)
    if not problem_dir.is_dir():
        return []
    return sorted({number for path in problem_dir.iterdir()
                   if (number := store.trial_replicate(path.name)) is not None})

def _replicate_from_disk(store: Store, cell: CellSpec, n_replicate: int) -> ReplicateResult:
    def record(arm: str) -> TrialRecord | None:
        stored = load_trial_record(store.trial_dir(cell.problem, n_replicate, arm))
        return stored.record if stored is not None else None

    return ReplicateResult(n_replicate=n_replicate, source=record(solo_arm(cell.source_dsl)),
                           base=record(solo_arm(cell.target_dsl)), skill=record(skill_arm(cell.direction())))

def _overall(results: list[CellResult]) -> dict[str, Any]:
    directions = sorted({result.cell.direction() for result in results})
    categories = sorted({result.cell.category for result in results})
    return {"n_cells": len(results),
            **_group_metrics(results),
            "by_direction": {name: _group_metrics([result for result in results if result.cell.direction() == name])
                             for name in directions},
            "by_category": {name: _group_metrics([result for result in results if result.cell.category == name])
                            for name in categories}}

def _cell_dict(result: CellResult) -> dict[str, Any]:
    cell = result.cell
    return {"cell": {"problem": cell.problem, "category": cell.category, "direction": cell.direction(),
                     "source_task": cell.source_task(), "target_task": cell.target_task()},
            **_group_metrics([result]),
            "replicates": [{"resolved": rep.resolved(), **asdict(rep)} for rep in result.replicates]}

def _group_metrics(results: list[CellResult]) -> dict[str, Any]:
    replicates = [rep for result in results for rep in result.replicates]
    resolved = [rep for rep in replicates if rep.resolved()]
    return {"n_cell_replicates": len(replicates),
            "n_resolved": len(resolved),
            "source_coverage": _mean(rep.source.passed() for rep in resolved),
            "skill_coverage": _mean(rep.skill_present() for rep in resolved),
            **_contrast(resolved),
            "with_skill": _contrast([rep for rep in resolved if rep.skill_present()]),
            "by_source_verdict": {"passed": _contrast([rep for rep in resolved if rep.source.passed()]),
                                  "failed": _contrast([rep for rep in resolved if not rep.source.passed()])},
            "usage": _usage_by_arm(results)}

def _usage_by_arm(results: list[CellResult]) -> dict[str, dict[str, Any]]:
    records: dict[tuple[str, int, str], TrialRecord | None] = {}
    for result in results:
        cell = result.cell
        for rep in result.replicates:
            records[cell.problem, rep.n_replicate, solo_arm(cell.source_dsl)] = rep.source
            records[cell.problem, rep.n_replicate, solo_arm(cell.target_dsl)] = rep.base
            records[cell.problem, rep.n_replicate, skill_arm(cell.direction())] = rep.skill
    arms = sorted({arm for _, _, arm in records})
    return {arm: _usage([record for (_, _, name), record in records.items() if name == arm]) for arm in arms}

def _usage(records: list[TrialRecord | None]) -> dict[str, Any]:
    usages = [record.usage for record in records if record is not None and record.usage is not None]
    return {"n_trials": len(usages),
            "cost_usd": _total(usage.cost_usd for usage in usages),
            "n_input_tokens": _total(usage.n_input_tokens for usage in usages),
            "n_cache_tokens": _total(usage.n_cache_tokens for usage in usages),
            "n_output_tokens": _total(usage.n_output_tokens for usage in usages),
            "agent_sec": _mean(usage.agent_sec for usage in usages)}

def _mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    return statistics.mean(numbers) if numbers else None

def _total(values: Iterable[float | None]) -> float | None:
    numbers = list(values)
    return sum(numbers) if numbers and None not in numbers else None

def _contrast(resolved: list[ReplicateResult]) -> dict[str, Any]:
    if not resolved:
        return {"base": None, "skill": None, "lift": None}
    base = _ladder([rep.base for rep in resolved])
    skill = _ladder([rep.skill_outcome() for rep in resolved])
    return {"base": base, "skill": skill, "lift": {name: skill[name] - base[name] for name in base}}

def _ladder(records: list[TrialRecord]) -> dict[str, float]:
    rates = {f"R@{threshold:g}": statistics.mean(record.attains(threshold) for record in records)
             for threshold in SPEED_THRESHOLDS}
    return {**rates, "S": 100 * statistics.mean(rates.values())}
