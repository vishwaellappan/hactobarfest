"""Command-line interface for RepoScout.

Every command is read-only with respect to the target repository except for
``index`` (which writes ``.reposcout/index.sqlite3``) and ``plan`` (which writes
a report when an output path is requested).  Declared checks run only with
``checks --run --yes`` or ``plan --run-checks --yes``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .agent import AgentOptions, RepoAgent
from .checks import CheckRun, CheckSet, discover_checks, run_check
from .indexing import Index, ensure_index, index_overview
from .models import Mapping, RepoProfile
from .rendering import (
    render_json,
    render_markdown,
    suggest_output_path,
    write_report,
)
from .retrieval import RetrievalOptions, map_task
from .scanning import ScanResult, detect_repo_root, discover
from .util import human_bytes, now_iso, truncate

FORMATS = ("md", "html", "json")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _out(text: str = "") -> None:
    print(text)


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _rule(title: str = "") -> None:
    if title:
        _out(f"== {title} " + "=" * max(0, 72 - len(title)))
    else:
        _out("=" * 76)


def _read_path(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_root(raw: str | None, *, nearest_git: bool = False) -> Path:
    root = _read_path(raw)
    if not root.is_dir():
        raise SystemExit(f"error: not a directory: {root}")
    return detect_repo_root(root) if nearest_git else root


def _require_task(task: list[str]) -> str:
    text = " ".join(task).strip()
    if not text:
        raise SystemExit("error: a task description is required")
    return text


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    root = _resolve_root(getattr(args, "path", None) or args.root)
    scan = discover(root)
    index, built = ensure_index(
        root, refresh=args.refresh, db_path=_db_path(args.db), scanned=scan
    )
    try:
        stats = index_overview(index)
        if args.json:
            _out(json.dumps(stats, indent=2))
            return 0
        _out(f"Repository: {scan.profile.name}")
        _out(f"Root:       {root}")
        _out(f"Index:      {'built' if built else 'reused'} at {index.db_path}")
        _out(f"Files:      {stats['files']:,} ({human_bytes(sum(f.bytes for f in scan.files))})")
        _out(f"Lines:      {scan.profile.total_lines:,}")
        _out(f"Symbols:    {stats['symbols']:,}")
        _out(f"Imports:    {stats['imports']:,} ({index.conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]:,} resolved edges)")
        _out(f"Tests:      {scan.profile.test_files:,} flagged test files")
        if stats.get("languages"):
            rendered = ", ".join(
                f"{row['lang']} ({row['lines']:,})" for row in stats["languages"][:6]  # type: ignore[index]
            )
            _out(f"Languages:  {rendered}")
        if scan.skipped:
            _out(
                "Skipped:    "
                + ", ".join(f"{key}={value}" for key, value in scan.skipped.items() if value)
            )
        _out("")
        _out("Next: reposcout map \"<task>\" or reposcout plan \"<task>\" -o report.html")
        return 0
    finally:
        index.close()


def cmd_overview(args: argparse.Namespace) -> int:
    root = _resolve_root(getattr(args, "path", None) or args.root)
    scan = discover(root)
    profile = scan.profile
    index, built = ensure_index(
        root, refresh=args.refresh, db_path=_db_path(args.db), scanned=scan
    )
    try:
        underlying = Index.stats(index)
        data = {
            "profile": profile.to_dict(),
            "index": underlying,
            "checks": _checks_summary(root, profile, scan),
            "skipped": scan.skipped,
        }
        if args.json:
            _out(json.dumps(data, indent=2))
            return 0

        _rule(profile.name or "repository")
        if profile.description:
            _out(profile.description)
            _out("")
        _out(f"Root:       {root}")
        _out(f"Index:      {'built' if built else 'reused'} at {index.db_path}")
        _out(f"Files:      {profile.files_indexed:,} indexed, {profile.test_files:,} tests")
        _out(f"Lines:      {profile.total_lines:,}")
        _out(f"Git:        {'yes' if profile.is_git else 'no'}"
             + (f" ({profile.branch} @ {profile.commit})" if profile.commit else ""))
        if profile.languages:
            _out("Languages:  " + ", ".join(f"{k} ({v:,})" for k, v in profile.languages.items()))
        if profile.manifests:
            _out("Manifests:  " + ", ".join(
                f"{m.path}" + (f" [{m.name}]" if m.name else "") for m in profile.manifests[:8]
            ))
        if profile.entry_points:
            _out("Entry:      " + ", ".join(profile.entry_points[:6]))
        if profile.docs:
            _out("Docs:       " + ", ".join(profile.docs[:6]))
        if profile.guidelines:
            _out("Guides:     " + ", ".join(profile.guidelines[:6]))
        if profile.ci_workflows:
            _out("CI:         " + ", ".join(profile.ci_workflows[:6]))
        checks = data["checks"]
        _out(
            f"Checks:     {checks['verified']} declared, {checks['suggested']} suggested, "
            f"{checks['runnable']} runnable here"
        )
        if profile.top_level:
            _out("Top level:  " + ", ".join(profile.top_level[:12]))
        _out("")
        _out(f"Indexed at {underlying['created_at']} ({underlying['files']:,} files, "
             f"{underlying['symbols']:,} symbols).")
        return 0
    finally:
        index.close()


def cmd_map(args: argparse.Namespace) -> int:
    task = _require_task(args.task)
    root = _resolve_root(args.root, nearest_git=True)
    scan = discover(root)
    index, _built = ensure_index(
        root, refresh=args.refresh, db_path=_db_path(args.db), scanned=scan
    )
    try:
        mapping = map_task(
            index,
            task,
            options=RetrievalOptions(
                max_primary=args.limit,
                expand_graph=not args.no_graph,
            ),
        )
        if args.json:
            payload = {
                "task": task,
                "repo_root": str(root),
                "generated_at": now_iso(),
                "mapping": mapping.to_dict(),
            }
            _out(json.dumps(payload, indent=2))
            return 0
        _print_mapping(mapping)
        return 0
    finally:
        index.close()


def cmd_plan(args: argparse.Namespace) -> int:
    task = _require_task(args.task)
    root = _resolve_root(args.root, nearest_git=True)
    if args.run_checks and not args.yes:
        raise SystemExit("error: --run-checks executes declared commands; pass --yes to confirm")
    options = AgentOptions(
        refresh=args.refresh,
        use_llm=args.llm,
        expand_graph=not args.no_graph,
        db_path=_db_path(args.db),
        max_primary=args.limit,
        run_checks=args.run_checks,
        timeout=args.timeout,
    )
    agent = RepoAgent(root, options)
    run = agent.run(task, run_checks=args.run_checks)
    try:
        fmt = args.format
        if args.output:
            target = Path(args.output).expanduser().resolve()
            if fmt == "md" and target.suffix.lower() in {".html", ".json"}:
                fmt = "html" if target.suffix.lower() == ".html" else "json"
            path = write_report(run.report, target, fmt)
            _err(f"Wrote {fmt} report: {path}")
            if run.check_runs:
                _print_check_runs(run.check_runs)
            return 0

        if fmt == "html":
            target = suggest_output_path(root, task)
            path = write_report(run.report, target, "html")
            _err(f"Wrote HTML report: {path}")
            _out(str(path))
            if run.check_runs:
                _print_check_runs(run.check_runs)
            return 0

        if fmt == "json":
            _out(render_json(run.report))
            return 0

        _out(render_markdown(run.report))
        if run.check_runs:
            _print_check_runs(run.check_runs)
        return 0
    finally:
        run.close()


def cmd_checks(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root, nearest_git=True)
    scan = discover(root)
    check_set = discover_checks(root, scan.profile, [item.path for item in scan.files])
    runs: list[CheckRun] = []
    if args.run:
        if not args.yes:
            raise SystemExit(
                "error: --run executes declared commands in this repository; pass --yes to confirm"
            )
        for check in check_set.verified:
            if not check.runnable:
                continue
            try:
                runs.append(run_check(check, root, timeout=args.timeout))
            except (ValueError, OSError) as exc:
                check.tool_notes.append(f"execution failed: {exc}")

    if args.json:
        payload = {
            "repo_root": str(root),
            "verified": [
                {
                    "id": check.id,
                    "command": check.command,
                    "kind": check.kind,
                    "declared_in": check.declared_in,
                    "sources": [s.__dict__ for s in check.sources],
                    "runnable": check.runnable,
                    "notes": check.tool_notes,
                }
                for check in check_set.verified
            ],
            "suggested": [
                {
                    "id": check.id,
                    "command": check.command,
                    "kind": check.kind,
                    "basis": check.basis,
                    "scope": check.scope,
                }
                for check in check_set.suggested
            ],
            "runs": [
                {
                    "id": run.check_id,
                    "command": run.command,
                    "returncode": run.returncode,
                    "ok": run.ok,
                    "timed_out": run.timed_out,
                    "duration_ms": run.duration_ms,
                    "output": run.output,
                }
                for run in runs
            ],
            "notes": check_set.notes,
        }
        _out(json.dumps(payload, indent=2))
        return 0 if all(run.ok for run in runs) else 1

    _print_checks(check_set, show_run_hint=not args.run)
    if runs:
        _print_check_runs(runs)
    return 0 if all(run.ok for run in runs) else 1


def cmd_version(_args: argparse.Namespace) -> int:
    _out(f"RepoScout {__version__}")
    return 0


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _print_mapping(mapping: Mapping) -> None:
    _rule("Task")
    _out(mapping.task)
    _out("")
    _out(
        f"Confidence: {mapping.confidence_label} ({mapping.confidence:.2f})"
        + (f"  identifiers: {', '.join(mapping.identifiers[:6])}" if mapping.identifiers else "")
    )
    for item in mapping.confidence_rationale:
        _out(f"  - {item}")
    if mapping.query_terms:
        terms = ", ".join(term.term for term in mapping.query_terms[:16])
        _out(f"Terms: {terms}")
    _out("")

    for title, hits in (
        ("PRIMARY TARGETS", mapping.primaries),
        ("SUPPORTING FILES", mapping.supporting),
        ("EXISTING TESTS", mapping.tests),
        ("DOCS / CONFIG", mapping.docs),
    ):
        if not hits:
            continue
        _rule(title)
        for position, hit in enumerate(hits, start=1):
            _out(f"{position}. {hit.path}  [score {hit.score:.1f}, {hit.role}, {hit.lines} lines]")
            for reason in hit.reasons[:4]:
                suffix = " (inferred)" if reason.inferred else ""
                _out(f"   - {reason.text}{suffix}")
            for snippet in hit.snippets[:3]:
                _out(f"   {hit.path}:{snippet.line}: {truncate(snippet.text, 160)}")
            if hit.symbols:
                rendered = ", ".join(f"{symbol.qualified} (L{symbol.line})" for symbol in hit.symbols[:4])
                _out(f"   symbols: {rendered}")
            if hit.imported_by:
                _out(f"   imported by: {', '.join(hit.imported_by[:3])}")
            _out("")
    if mapping.notes:
        _rule("NOTES")
        for note in mapping.notes:
            _out(f"- {note}")
        _out("")


def _print_checks(check_set: CheckSet, *, show_run_hint: bool = True) -> None:
    _rule("VERIFIED (declared in this repository)")
    if check_set.verified:
        for check in check_set.verified:
            where = check.declared_in or (check.sources[0].label() if check.sources else "?")
            flags = []
            flags.append("runnable here" if check.runnable else "not runnable here")
            if check.tool_notes:
                flags.append("; ".join(check.tool_notes))
            _out(f"[{check.kind}] {check.command}")
            _out(f"    declared at {where} ({', '.join(flags)})")
    else:
        _out("None. This repository does not declare how to verify changes.")
    _out("")

    _rule("SUGGESTED (inferred, not declared)")
    if check_set.suggested:
        for check in check_set.suggested:
            _out(f"[{check.kind}] {check.command}")
            _out(f"    basis: {check.basis}")
            if check.scope:
                _out(f"    scope: {', '.join(check.scope[:4])}")
    else:
        _out("None.")
    _out("")
    for note in check_set.notes:
        _out(f"- {note}")
    if show_run_hint and any(check.runnable for check in check_set.verified):
        _out("")
        _out("Run declared+runnable checks with: reposcout checks --run --yes")


def _print_check_runs(runs: list[CheckRun]) -> None:
    _rule("CHECK RESULTS")
    for run in runs:
        status = "PASS" if run.ok else "FAIL"
        _out(f"[{status}] {run.command}  ({run.duration_ms} ms, exit {run.returncode})")
        if run.output:
            output_lines = run.output.splitlines()
            for line in output_lines[:40]:
                _out(f"    {truncate(line, 300)}")
            if len(output_lines) > 40:
                _out("    ... output truncated")
        _out("")
    passed = sum(1 for run in runs if run.ok)
    _out(f"{passed}/{len(runs)} executed checks passed. These are real results, not declarations.")


def _checks_summary(root: Path, profile: RepoProfile, scan: ScanResult) -> dict[str, int]:
    try:
        check_set = discover_checks(root, profile, [item.path for item in scan.files])
    except (OSError, ValueError):
        return {"verified": 0, "suggested": 0, "runnable": 0}
    return {
        "verified": len(check_set.verified),
        "suggested": len(check_set.suggested),
        "runnable": sum(1 for check in check_set.checks if check.runnable),
    }


def _db_path(raw: str | None) -> Path | None:
    return Path(raw).expanduser().resolve() if raw else None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-C",
        "--root",
        metavar="PATH",
        help="repository directory (default: current directory)",
    )
    common.add_argument("--db", metavar="PATH", help="alternate SQLite index path")
    common.add_argument(
        "--refresh",
        action="store_true",
        help="rebuild the index before running",
    )

    parser = argparse.ArgumentParser(
        prog="reposcout",
        description=(
            "Map a task to repository code with evidence, and separate declared checks "
            "from unverified suggestions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  reposcout index .\n"
            "  reposcout map \"fix crash in config loader\"\n"
            "  reposcout plan \"add timeout option\" -o brief.html\n"
            "  reposcout checks --run --yes\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"RepoScout {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    index_parser = subparsers.add_parser(
        "index", parents=[common], help="build or refresh the SQLite index"
    )
    index_parser.add_argument(
        "path", nargs="?", help="repository directory (default: current directory)"
    )
    index_parser.add_argument("--json", action="store_true", help="emit JSON statistics")
    index_parser.set_defaults(func=cmd_index)

    overview_parser = subparsers.add_parser(
        "overview", parents=[common], help="describe the repository and its verification setup"
    )
    overview_parser.add_argument(
        "path", nargs="?", help="repository directory (default: current directory)"
    )
    overview_parser.add_argument("--json", action="store_true", help="emit JSON")
    overview_parser.set_defaults(func=cmd_overview)

    map_parser = subparsers.add_parser(
        "map", parents=[common], help="rank files for a task and show the evidence"
    )
    map_parser.add_argument("task", nargs="+", help="task or issue description")
    map_parser.add_argument("--limit", type=int, default=5, help="maximum primary targets (default: 5)")
    map_parser.add_argument("--json", action="store_true", help="emit JSON")
    map_parser.add_argument("--no-graph", action="store_true", help="disable import-graph expansion")
    map_parser.set_defaults(func=cmd_map)

    plan_parser = subparsers.add_parser(
        "plan", parents=[common], help="produce a contribution plan and report"
    )
    plan_parser.add_argument("task", nargs="+", help="task or issue description")
    plan_parser.add_argument(
        "--format", choices=FORMATS, default="md", help="report format (default: md)"
    )
    plan_parser.add_argument("-o", "--output", help="write the report to this path")
    plan_parser.add_argument("--llm", action="store_true", help="opt in to an OpenAI-compatible model")
    plan_parser.add_argument("--limit", type=int, default=5, help="maximum primary targets (default: 5)")
    plan_parser.add_argument("--no-graph", action="store_true", help="disable import-graph expansion")
    plan_parser.add_argument(
        "--run-checks",
        action="store_true",
        help="execute declared+runnable checks (requires --yes)",
    )
    plan_parser.add_argument("--yes", action="store_true", help="confirm executing checks")
    plan_parser.add_argument(
        "--timeout", type=int, default=180, help="per-check timeout in seconds (default: 180)"
    )
    plan_parser.set_defaults(func=cmd_plan)

    checks_parser = subparsers.add_parser(
        "checks", parents=[common], help="list declared and suggested verification commands"
    )
    checks_parser.add_argument("--json", action="store_true", help="emit JSON")
    checks_parser.add_argument(
        "--run", action="store_true", help="execute declared+runnable checks (requires --yes)"
    )
    checks_parser.add_argument("--yes", action="store_true", help="confirm executing checks")
    checks_parser.add_argument(
        "--timeout", type=int, default=180, help="per-check timeout in seconds (default: 180)"
    )
    checks_parser.set_defaults(func=cmd_checks)

    version_parser = subparsers.add_parser("version", help="print the version")
    version_parser.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _err("interrupted")
        return 130
    except BrokenPipeError:
        return 0
    except SystemExit:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        _err(f"error: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
