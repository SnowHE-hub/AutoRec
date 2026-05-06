"""One-click pipeline: steps 1-6 (data → ETL → features → evaluation).

Usage
-----
    python scripts/run_all.py            # run all steps
    python scripts/run_all.py --from 3  # resume from step 3
    python scripts/run_all.py --only 5  # run step 5 only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _hdr(n: int, title: str) -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  Step {n}: {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")

def _ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}  {msg}")

def _err(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")

def _run_module(module: str) -> bool:
    """Run `python -m <module>` as a subprocess; stream output; return success."""
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
    )
    return result.returncode == 0


# ── individual step functions ─────────────────────────────────────────────────

def step1_install() -> bool:
    """pip install -r requirements.txt"""
    req = ROOT / "requirements.txt"
    if not req.exists():
        _err("requirements.txt not found")
        return False
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
        cwd=ROOT,
    )
    return result.returncode == 0


def step2_generate() -> bool:
    """Generate synthetic data → data/raw/"""
    return _run_module("autorec.etl.synthetic_generator")


def step3_etl() -> bool:
    """Clean + quality-check → data/processed/"""
    return _run_module("autorec.etl.quality_checks")


def step4_load() -> bool:
    """Load processed data into SQLite (or PostgreSQL)"""
    return _run_module("autorec.etl.load")


def step5_user_features() -> bool:
    """Build user feature matrix → data/features/user_features.parquet"""
    return _run_module("autorec.features.user_features")


def step5b_item_features() -> bool:
    """Build item feature matrix → data/features/item_features.parquet"""
    return _run_module("autorec.features.item_features")


def step6_evaluate() -> bool:
    """Offline evaluation of all three models → data/processed/eval_results.csv"""
    return _run_module("autorec.eval.evaluator")


# ── step registry ─────────────────────────────────────────────────────────────

STEPS: list[tuple[int, str, Callable[[], bool]]] = [
    (1, "Install dependencies",                   step1_install),
    (2, "Generate synthetic data",                step2_generate),
    (3, "ETL: clean + quality check",             step3_etl),
    (4, "Load database (SQLite / PostgreSQL)",     step4_load),
    (5, "Feature engineering: user features",     step5_user_features),
    (6, "Feature engineering: item features",     step5b_item_features),
    (7, "Offline evaluation",                     step6_evaluate),
]


# ── runner ────────────────────────────────────────────────────────────────────

def run(from_step: int = 1, only_step: int | None = None) -> None:
    results: list[tuple[int, str, bool, float]] = []

    for num, title, fn in STEPS:
        if only_step is not None and num != only_step:
            continue
        if num < from_step:
            continue

        _hdr(num, title)
        t0 = time.perf_counter()
        ok = fn()
        elapsed = time.perf_counter() - t0
        results.append((num, title, ok, elapsed))

        if ok:
            _ok(f"finished in {elapsed:.1f}s")
        else:
            _err(f"failed after {elapsed:.1f}s")
            print(f"\n{RED}Pipeline aborted at step {num}.{RESET}")
            print("Fix the error above and re-run with:")
            print(f"  python scripts/run_all.py --from {num}\n")
            _print_summary(results)
            sys.exit(1)

    _print_summary(results)
    print(f"\n{GREEN}{BOLD}All steps completed successfully!{RESET}")
    print("\nNext steps:")
    print("  python autorec/api/main.py          # start API on :8000")
    print("  streamlit run ui/dashboard.py       # start Dashboard on :8501\n")


def _print_summary(results: list[tuple[int, str, bool, float]]) -> None:
    if not results:
        return
    print(f"\n{BOLD}{'─'*60}")
    print("  Pipeline summary")
    print(f"{'─'*60}{RESET}")
    total = 0.0
    for num, title, ok, elapsed in results:
        icon  = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        total += elapsed
        print(f"  [{icon}]  Step {num}: {title:<40} {elapsed:5.1f}s")
    print(f"{BOLD}{'─'*60}{RESET}")
    print(f"  Total elapsed: {total:.1f}s\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoRec one-click pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_step", type=int, default=1, metavar="N",
        help="Start from step N (default: 1)",
    )
    group.add_argument(
        "--only", dest="only_step", type=int, default=None, metavar="N",
        help="Run only step N",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}AutoRec — One-click Pipeline{RESET}")
    print(f"Python: {sys.executable}")
    print(f"Root  : {ROOT}")

    run(from_step=args.from_step, only_step=args.only_step)
