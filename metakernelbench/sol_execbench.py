"""Assembles the SOL-ExecBench-derived problems from the user's own copy of the nvidia/SOL-ExecBench dataset."""

import argparse
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from metakernelbench.catalog import PROBLEMS_DIR

RECIPES_DIR = PROBLEMS_DIR / "_sol_execbench"
CACHE_DIR = Path.home() / ".cache" / "metakernelbench" / "sol-execbench"
SUBSETS = ("L1", "L2", "Quant")
SOURCES = {
    "huggingface": "https://huggingface.co/datasets/nvidia/SOL-ExecBench/resolve/main/data/{subset}.parquet",
    "modelscope": "https://www.modelscope.cn/datasets/nv-community/SOL-ExecBench/resolve/master/data/"
                  "{subset}.parquet",
}
PARQUET_SHA256 = {
    "L1": "dbaf8476812194920e405118cc5f4abc65035b79a4307b5ee88fb07666ceed48",
    "L2": "4a55a4bfbe9a6f9b2ad4c5c5c680977e66926175643739530bf5a8ca3e87cd54",
    "Quant": "aa30e9a66ac2292499eacb1cc73ad789ddc7c5f5808ddb10d9dc211e94560036",
}
HOLE = "\ue000"
ENTRY = "\ue001"
WORKLOAD = "\ue002"
UNIT_PATTERN = re.compile(r"\w+|[^\w\s]")

@dataclass(frozen=True)
class UpstreamEntry:
    name: str
    reference: str
    workload_ids: tuple[str, ...]

def main() -> None:
    source = _parse_source()
    entries = upstream_entries(_dataset_dir(source))
    recipes = sorted(RECIPES_DIR.glob("*.json"))
    for recipe_path in recipes:
        recipe = json.loads(recipe_path.read_bytes().decode("utf-8"))
        text = assemble(recipe, entries[(recipe["subset"], recipe["row"])])
        if file_hash(text) != recipe["sha256"]:
            raise ValueError(f"{recipe['problem']} assembled from {source} does not match the recorded problem; "
                             "the dataset copy differs from the pinned revision")
        target = PROBLEMS_DIR / recipe["category"] / f"{recipe['problem']}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    print(f"assembled {len(recipes)} problems into {PROBLEMS_DIR}")

def assemble(recipe: dict[str, Any], entry: UpstreamEntry) -> str:
    upstream = units(entry.reference)
    filling = [unit for position, length in recipe["runs"] for unit in upstream[position:position + length]]
    layout: str = recipe["layout"]
    if len(filling) != layout.count(HOLE):
        raise ValueError(f"{recipe['problem']}: the recipe's runs do not cover its holes")
    filled = iter(filling)
    substitutes = {ENTRY: entry.name, WORKLOAD: entry.workload_ids[recipe["workload"]]}
    return "".join(next(filled) if character == HOLE else substitutes.get(character, character) for character in layout)

def units(text: str) -> list[str]:
    return UNIT_PATTERN.findall(text)

def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def upstream_entries(dataset_dir: Path) -> dict[tuple[str, int], UpstreamEntry]:
    entries: dict[tuple[str, int], UpstreamEntry] = {}
    for subset in SUBSETS:
        path = dataset_dir / f"{subset}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != PARQUET_SHA256[subset]:
            print(f"{path} differs from the pinned dataset files; every assembled problem is still verified")
        table = pq.read_table(path, columns=["name", "reference", "workloads"])
        columns = (table.column(name).to_pylist() for name in ("name", "reference", "workloads"))
        for row, (name, reference, workloads) in enumerate(zip(*columns)):
            records = json.loads(workloads)
            entries[(subset, row)] = UpstreamEntry(name, reference, tuple(record["uuid"] for record in records))
    return entries

def _parse_source() -> str:
    parser = argparse.ArgumentParser(description="Assemble the SOL-ExecBench-derived problems from the dataset")
    parser.add_argument("--source", default="huggingface",
                        help="huggingface, modelscope, or a directory holding L1.parquet, L2.parquet, Quant.parquet")
    return parser.parse_args().source

def _dataset_dir(source: str) -> Path:
    if source not in SOURCES:
        return Path(source)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        target = CACHE_DIR / f"{subset}.parquet"
        if not target.is_file():
            urllib.request.urlretrieve(SOURCES[source].format(subset=subset), target)
    return CACHE_DIR

if __name__ == "__main__":
    main()
