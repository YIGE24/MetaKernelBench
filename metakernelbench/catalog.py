"""The static world: problems, DSLs, directed cells, trial coordinates, and the conditions every trial runs under."""

import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
PROBLEMS_DIR = ROOT_DIR / "problems"

KERNEL_PATH = "/app/kernel.py"
SKILL_PATH = "/app/SKILL.md"

TARGET_GPU = "B200"
SUBMISSION_LIMIT = 10
KERNEL_BYTE_LIMIT = 300_000

CATEGORY_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
PROBLEM_PATTERN = re.compile(r"[0-9]{2}_[a-z0-9]+(?:_[a-z0-9]+)*")
COORDINATE_PATTERN = re.compile(r"p[0-9]{2}_r([0-9]+)_[a-z0-9_]+")

@dataclass(frozen=True)
class DslSpec:
    display: str
    import_name: str
    pip_requirements: tuple[str, ...]

DSL_SPECS = {
    "cutedsl": DslSpec(display="CuTe DSL", import_name="cutlass",
                       pip_requirements=("nvidia-cutlass-dsl[cu13]==4.7.0", "apache-tvm-ffi==0.1.12")),
    "tirx": DslSpec(display="TIRx", import_name="tvm",
                    pip_requirements=("apache-tvm[cuda]==0.25.0.post1", "apache-tvm-ffi==0.1.12")),
}

def problem_files() -> list[Path]:
    numbers: dict[str, Path] = {}
    for path in PROBLEMS_DIR.rglob("*.py"):
        relative = path.relative_to(PROBLEMS_DIR)
        if any(part.startswith((".", "_")) for part in relative.parts):
            continue
        if len(relative.parts) != 2:
            raise ValueError(f"{relative.as_posix()} must live at problems/<category>/<problem>.py")
        category = relative.parts[0]
        if CATEGORY_PATTERN.fullmatch(category) is None:
            raise ValueError(f"category {category!r} must match {CATEGORY_PATTERN.pattern}")
        if PROBLEM_PATTERN.fullmatch(path.stem) is None:
            raise ValueError(f"{relative.as_posix()} must be named NN_<problem>.py "
                             "using lowercase ASCII letters, digits, and single underscores")
        number = path.stem[:2]
        if number in numbers:
            previous = numbers[number].relative_to(PROBLEMS_DIR).as_posix()
            raise ValueError(f"catalog id {number} used by both {previous} and {relative.as_posix()}")
        numbers[number] = path
    return [numbers[number] for number in sorted(numbers)]

def task_name(problem: str, dsl: str) -> str:
    return f"{problem}_{dsl}"

def solo_arm(dsl: str) -> str:
    return f"solo_{dsl}"

def skill_arm(direction: str) -> str:
    return f"{direction}_skill"

def trial_coordinate(problem: str, n_replicate: int, arm: str) -> str:
    return f"p{problem[:2]}_r{n_replicate}_{arm}"

def coordinate_replicate(coordinate: str) -> int | None:
    match = COORDINATE_PATTERN.fullmatch(coordinate)
    return int(match.group(1)) if match is not None else None

@dataclass(frozen=True)
class CellSpec:
    problem: str
    category: str
    source_dsl: str
    target_dsl: str

    def direction(self) -> str:
        return f"{self.source_dsl}2{self.target_dsl}"

    def source_task(self) -> str:
        return task_name(self.problem, self.source_dsl)

    def target_task(self) -> str:
        return task_name(self.problem, self.target_dsl)

def cell_specs() -> list[CellSpec]:
    return [CellSpec(problem=path.stem, category=path.parent.name, source_dsl=source, target_dsl=target)
            for path in problem_files()
            for source, target in permutations(DSL_SPECS, 2)]

def cell_tasks(cells: list[CellSpec]) -> list[str]:
    return sorted({task for cell in cells for task in (cell.source_task(), cell.target_task())})
