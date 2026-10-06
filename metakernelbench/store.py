"""One model's result store: layout, identity, atomic writes, the single-writer lock, archiving, and provenance."""

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from metakernelbench.catalog import ROOT_DIR, coordinate_replicate, trial_coordinate

RESULTS_DIR = ROOT_DIR / "results"

IDENTITY_NAME = "store.json"
INVOCATIONS_NAME = "invocations.jsonl"
SNAPSHOT_NAME = "results.json"
LOCK_NAME = ".lock"
ARCHIVE_NAME = "archive"
PROBLEMS_NAME = "problems"
SKILLS_NAME = "skills"

RECORD_NAME = "record.json"
VERDICT_NAME = "verdict.json"
SUBMISSIONS_NAME = "submissions.jsonl"
ARTIFACTS_NAME = "artifacts"

@dataclass(frozen=True)
class Store:
    dir: Path
    model: str
    prefix: str

    def snapshot_path(self) -> Path:
        return self.dir / SNAPSHOT_NAME

    def problem_dir(self, problem: str) -> Path:
        return self.dir / PROBLEMS_NAME / problem

    def trial_dir(self, problem: str, n_replicate: int, arm: str) -> Path:
        return self.problem_dir(problem) / f"{self.prefix}_{trial_coordinate(problem, n_replicate, arm)}"

    def trial_replicate(self, dir_name: str) -> int | None:
        head = f"{self.prefix}_"
        return coordinate_replicate(dir_name.removeprefix(head)) if dir_name.startswith(head) else None

    def skill_attachment_path(self, problem: str, n_replicate: int, direction: str) -> Path:
        return self.problem_dir(problem) / SKILLS_NAME / f"r{n_replicate}_{direction}.md"

def artifact_path(trial_dir: Path, sandbox_path: str) -> Path:
    return trial_dir / ARTIFACTS_NAME / sandbox_path.removeprefix("/")

def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)

def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")

def open_store(model: str) -> Store:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", model.split("/")[-1])
    if not re.search(r"[0-9A-Za-z]", slug):
        raise SystemExit(f"model id {model!r} yields no usable store directory name")
    directory = RESULTS_DIR / slug
    identity_path = directory / IDENTITY_NAME
    if identity_path.is_file():
        stored_model = json.loads(identity_path.read_text())["model"]
        if stored_model != model:
            raise SystemExit(f"{directory} belongs to model {stored_model!r}, not {model!r}")
    return Store(dir=directory, model=model, prefix=hashlib.sha256(model.encode()).hexdigest()[:6])

def claim_store(store: Store) -> None:
    store.dir.mkdir(parents=True, exist_ok=True)
    identity_path = store.dir / IDENTITY_NAME
    if identity_path.is_file():
        return
    write_json(identity_path, {"model": store.model, "prefix": store.prefix, "created_at": timestamp()})

@contextlib.contextmanager
def hold_store_lock(store: Store) -> Iterator[None]:
    lock_path = store.dir / LOCK_NAME
    _clear_dead_lock(lock_path)
    try:
        with lock_path.open("x") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise SystemExit(f"{store.dir} is locked by a running invocation (pid {_lock_holder(lock_path)}), "
                         f"delete {lock_path} if that process is gone") from None
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)

def archive_trial(store: Store, trial_dir: Path) -> None:
    archive_dir = store.dir / ARCHIVE_NAME
    archive_dir.mkdir(exist_ok=True)
    shutil.move(trial_dir, archive_dir / f"{datetime.now():%Y-%m-%d__%H-%M-%S}__{trial_dir.name}")

def append_invocation(store: Store, entry: dict[str, Any]) -> None:
    append_jsonl(store.dir / INVOCATIONS_NAME, entry)

def repo_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(ROOT_DIR), "status", "--porcelain", "--", ".", ":!results"],
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return f"{commit}+dirty" if status else commit

def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def _clear_dead_lock(lock_path: Path) -> None:
    try:
        pid = int(lock_path.read_text())
    except (FileNotFoundError, ValueError):
        return
    if not _process_alive(pid):
        lock_path.unlink(missing_ok=True)

def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def _lock_holder(lock_path: Path) -> str:
    try:
        return lock_path.read_text().strip() or "unknown"
    except FileNotFoundError:
        return "unknown"
