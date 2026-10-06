"""Assembles build/<task>/ from the problems and template/: layout, rendering, and fingerprints."""

import ast
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from harbor.models.task.config import TaskConfig

from metakernelbench.catalog import (
    DSL_SPECS, KERNEL_BYTE_LIMIT, PACKAGE_DIR, ROOT_DIR, SUBMISSION_LIMIT, TARGET_GPU, DslSpec, problem_files,
    task_name)

TEMPLATE_DIR = PACKAGE_DIR / "template"
DSL_DOCS_DIR = PACKAGE_DIR / "dsl_docs"
BUILD_DIR = ROOT_DIR / "build"

ENVIRONMENT_NAME = "environment"
DOCKERFILE_NAME = "Dockerfile"
TASK_TOML_NAME = "task.toml"
INSTRUCTION_NAME = "instruction.md"
FINGERPRINT_NAME = "fingerprint"
BUILD_FINGERPRINT_PATH = BUILD_DIR / FINGERPRINT_NAME

SANDBOX_FILES = ("contract.py", "test.sh", "clock.sh")
GRADER_NAME = "grader.py"
RENDERED_FILES = (DOCKERFILE_NAME, TASK_TOML_NAME, INSTRUCTION_NAME)
PROBLEM_INTERFACE = ("reference", "make_inputs")

def task_dir(task: str) -> Path:
    return BUILD_DIR / task

def environment_dir(task: str) -> Path:
    return task_dir(task) / ENVIRONMENT_NAME

def grader_path(task: str) -> Path:
    return task_dir(task) / GRADER_NAME

def task_config(task: str) -> TaskConfig:
    return TaskConfig.model_validate_toml((task_dir(task) / TASK_TOML_NAME).read_text())

def task_fingerprint(task: str) -> str:
    return (task_dir(task) / FINGERPRINT_NAME).read_text()

def build_is_fresh() -> bool:
    return BUILD_FINGERPRINT_PATH.is_file() and BUILD_FINGERPRINT_PATH.read_text() == _inputs_fingerprint()

def main() -> None:
    variants = _task_variants()
    _check_template()
    _check_docs()
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    for variant in variants:
        _assemble_task(variant)
    BUILD_FINGERPRINT_PATH.write_text(_inputs_fingerprint())
    print(f"built {len(variants)} tasks into {BUILD_DIR}")

@dataclass(frozen=True)
class _TaskVariant:
    task: str
    problem_path: Path
    dsl: str
    description: str

def _task_variants() -> list[_TaskVariant]:
    variants: list[_TaskVariant] = []
    for path in problem_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        _check_problem_interface(path, tree)
        description = _problem_description(path, tree)
        variants.extend(_TaskVariant(task_name(path.stem, dsl), path, dsl, description) for dsl in DSL_SPECS)
    return variants

def _check_problem_interface(path: Path, tree: ast.Module) -> None:
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in PROBLEM_INTERFACE if name not in defined]
    if missing:
        raise ValueError(f"{path.name} defines no {', '.join(missing)}, the grader cannot score it")

def _problem_description(path: Path, tree: ast.Module) -> str:
    description = (ast.get_docstring(tree) or "").strip()
    if not description or "\n" in description:
        raise ValueError(f"{path.name} needs a one-line module docstring, the task description")
    return description

def _check_template() -> None:
    present = {path.relative_to(TEMPLATE_DIR).as_posix() for path in _visible_files(TEMPLATE_DIR)}
    placed = {*SANDBOX_FILES, GRADER_NAME, *RENDERED_FILES}
    if present != placed:
        raise ValueError(f"template/ and build.py disagree on {sorted(present ^ placed)}")

def _check_docs() -> None:
    for dsl, spec in DSL_SPECS.items():
        docs_dir = DSL_DOCS_DIR / dsl
        if not (docs_dir.is_dir() and _visible_files(docs_dir)):
            raise ValueError(f"metakernelbench/dsl_docs/{dsl} holds no files, {spec.display} would ship an empty /docs")

def _assemble_task(variant: _TaskVariant) -> None:
    directory = task_dir(variant.task)
    environment = environment_dir(variant.task)
    environment.mkdir(parents=True)
    shutil.copy(variant.problem_path, environment / "task.py")
    for name in SANDBOX_FILES:
        shutil.copy(TEMPLATE_DIR / name, environment / name)
    shutil.copy(TEMPLATE_DIR / GRADER_NAME, grader_path(variant.task))
    shutil.copytree(DSL_DOCS_DIR / variant.dsl, environment / "docs",
                    ignore=shutil.ignore_patterns(".*", "__pycache__"))
    spec = DSL_SPECS[variant.dsl]
    _render_dockerfile(spec, environment)
    config = _render_task_toml(variant, directory)
    _render_instruction(variant, spec, config, directory)
    (directory / FINGERPRINT_NAME).write_text(_files_digest(_visible_files(directory), directory))

def _render_dockerfile(spec: DslSpec, environment: Path) -> None:
    pip_requirements = " ".join(f'"{requirement}"' for requirement in spec.pip_requirements)
    rendered = (TEMPLATE_DIR / DOCKERFILE_NAME).read_text().format(pip_requirements=pip_requirements,
                                                                    import_name=spec.import_name)
    (environment / DOCKERFILE_NAME).write_text(rendered)

def _render_task_toml(variant: _TaskVariant, directory: Path) -> TaskConfig:
    rendered = "\n".join([
        (TEMPLATE_DIR / TASK_TOML_NAME).read_text(),
        "[task]",
        f'name = "metakernelbench/{variant.task}"',
        f"description = {json.dumps(variant.description)}",
    ]) + "\n"
    config = TaskConfig.model_validate_toml(rendered)
    (directory / TASK_TOML_NAME).write_text(rendered)
    return config

def _render_instruction(variant: _TaskVariant, spec: DslSpec, config: TaskConfig, directory: Path) -> None:
    if config.agent.timeout_sec is None or config.environment.cpus is None or config.environment.memory_mb is None:
        raise ValueError("template/task.toml must declare [agent] timeout_sec and [environment] cpus and memory_mb")
    rendered = (TEMPLATE_DIR / INSTRUCTION_NAME).read_text().format(
        name=variant.task, description=variant.description, gpu=TARGET_GPU,
        budget_text=f"{config.agent.timeout_sec / 60:g} minutes",
        sandbox_cpus=config.environment.cpus, sandbox_memory=f"{config.environment.memory_mb / 1024:g} GB",
        dsl_display=spec.display, n_submissions=SUBMISSION_LIMIT, kernel_byte_limit=KERNEL_BYTE_LIMIT)
    (directory / INSTRUCTION_NAME).write_text(rendered)

def _inputs_fingerprint() -> str:
    paths = [PACKAGE_DIR / "build.py", PACKAGE_DIR / "catalog.py",
             *_visible_files(TEMPLATE_DIR), *_visible_files(DSL_DOCS_DIR), *problem_files()]
    return _files_digest(paths, ROOT_DIR)

def _files_digest(paths: list[Path], root: Path) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()

def _visible_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and not any(part.startswith(".") or part == "__pycache__"
                                                for part in path.relative_to(root).parts))

if __name__ == "__main__":
    main()
