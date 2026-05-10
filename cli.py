"""CLI for the LHC resume-state parser.

Usage:
    python cli.py path/to/task.yaml
    python cli.py examples/resumption_v2_004.yaml

Reads a single LHC v0.2 task YAML and prints the parser's response.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from parser import parse_task


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("task_yaml", type=Path, help="path to an LHC v0.2 task YAML")
    p.add_argument(
        "--show-task",
        action="store_true",
        help="also print the task setup + probe before the response",
    )
    args = p.parse_args(argv)

    if not args.task_yaml.exists():
        print(f"[fatal] task file not found: {args.task_yaml}", file=sys.stderr)
        return 2

    task = yaml.safe_load(args.task_yaml.read_text())

    if args.show_task:
        print(f"=== task: {task.get('id', args.task_yaml.stem)} ===")
        print(f"category: {task.get('category')}")
        print(f"difficulty: {task.get('difficulty')}")
        if task.get("description"):
            print(f"\ndescription: {task['description'].strip()}")
        print()
        print("--- setup ---")
        for m in task.get("setup", []):
            print(f"[{m.get('role')}]")
            print(m.get("content", "").rstrip())
            print()
        print("--- probe ---")
        for m in task.get("probe", []):
            print(f"[{m.get('role')}]")
            print(m.get("content", "").rstrip())
            print()
        print("=== parser response ===")

    print(parse_task(task))
    return 0


if __name__ == "__main__":
    sys.exit(main())
