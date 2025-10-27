#!/usr/bin/env python3
"""Run pytest under trace module and report package coverage."""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from trace import Trace

import pytest


def statement_lines(path: pathlib.Path) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="groundkg", help="Package directory to measure")
    parser.add_argument("--min", type=float, default=0.0, help="Minimum required coverage percentage")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Arguments forwarded to pytest")
    args = parser.parse_args()

    pkg_path = pathlib.Path(args.package)
    if not pkg_path.exists():
        parser.error(f"Package path {pkg_path} not found")

    tracer = Trace(count=True, trace=False, ignoredirs=[sys.prefix, sys.exec_prefix])
    exit_code = tracer.runfunc(pytest.main, args.pytest_args or [])
    if exit_code != 0:
        return exit_code

    results = tracer.results()
    counts = results.counts

    total_statements = 0
    total_executed = 0
    report_lines = []
    for path in sorted(pkg_path.rglob("*.py")):
        if path.name == "__init__.py" and not path.read_text(encoding="utf-8").strip():
            continue
        stmts = statement_lines(path)
        if not stmts:
            continue
        executed = sum(1 for line in stmts if counts.get((str(path.resolve()), line), 0))
        total_statements += len(stmts)
        total_executed += executed
        pct = 100.0 * executed / len(stmts)
        report_lines.append(f"{path}: {pct:.1f}% ({executed}/{len(stmts)})")

    overall = 100.0 * total_executed / total_statements if total_statements else 100.0
    print("\n".join(report_lines))
    print(f"Overall coverage for {pkg_path}: {overall:.1f}% ({total_executed}/{total_statements})")

    if overall < args.min:
        print(f"Coverage {overall:.1f}% is below required minimum {args.min}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
