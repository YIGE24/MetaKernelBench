"""The command-line entry point: converge one model's result store toward the selected trials."""

import argparse
import asyncio
from fnmatch import fnmatch

from dotenv import load_dotenv

from metakernelbench.build import build_is_fresh
from metakernelbench.catalog import ROOT_DIR, cell_specs
from metakernelbench.plan import Invocation
from metakernelbench.protocol import run_experiment
from metakernelbench.store import claim_store, hold_store_lock, open_store

def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    invocation = _parse_invocation()
    if invocation.dry_run:
        asyncio.run(run_experiment(invocation))
        return
    claim_store(invocation.store)
    with hold_store_lock(invocation.store):
        asyncio.run(run_experiment(invocation))

def _parse_invocation() -> Invocation:
    parser = argparse.ArgumentParser(description="Converge one model's result store toward the selected trials")
    parser.add_argument("--model", required=True, help="litellm model id, every model converges its own store")
    parser.add_argument("--problems", default="*", help="glob over problem names, e.g. '32_*'")
    parser.add_argument("--categories", action="append", default=[],
                        help="problem category to include, repeatable, e.g. --categories moe")
    parser.add_argument("--trials", default="*", help="glob over trial coordinates, e.g. 'p32_r1_solo_cutedsl'")
    parser.add_argument("--n-concurrent", type=_positive_int, default=10,
                        help="trials in flight at once, each holds one sandbox and at most one B200")
    parser.add_argument("--n-replicates", type=_positive_int, default=3,
                        help="convergence target, replicates 1..n of every cell")
    parser.add_argument("--rerun-stale", action="store_true",
                        help="archive and redo stale trials, whose task build or skill attachment changed")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit without running anything")
    args = parser.parse_args()
    categories = frozenset(args.categories)
    cells = [cell for cell in cell_specs()
             if (not categories or cell.category in categories) and fnmatch(cell.problem, args.problems)]
    if not cells:
        scope = f"categories {sorted(categories)}" if categories else "problems/"
        raise SystemExit(f"no problem in {scope} matches {args.problems!r}")
    if not build_is_fresh():
        raise SystemExit("build/ is missing or stale, run python -m metakernelbench.build first")
    return Invocation(store=open_store(args.model), cells=cells, categories=categories,
                      problems_pattern=args.problems, trials_pattern=args.trials,
                      n_concurrent=args.n_concurrent, n_replicates=args.n_replicates,
                      rerun_stale=args.rerun_stale, dry_run=args.dry_run)

def _positive_int(text: str) -> int:
    if not text.isdecimal() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer")
    return int(text)

if __name__ == "__main__":
    main()
