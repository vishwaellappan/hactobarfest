"""Discovery and auditing of verification commands.

The contract:

``verified``
    The command is written down in this repository (config, CI workflow,
    Makefile, package script, or contributor docs).  ``sources`` points at the
    file and line.  Verified does **not** mean "ran successfully"; runnability
    is tracked separately.

``suggested``
    The command is inferred from ecosystem conventions or file names.  It has
    an explicit ``basis`` and is never presented as a repository fact.
"""

from __future__ import annotations

import configparser
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import tokens as tok
from .models import Check, RepoProfile, Source
from .util import find_line, read_text, split_lines, truncate, which

KIND_ORDER = ["test", "lint", "typecheck", "format", "build", "docs", "security"]
COMMAND_PREFIXES = (
    "pytest",
    "py.test",
    "tox",
    "nox",
    "python -m pytest",
    "python -m unittest",
    "python3 -m pytest",
    "python3 -m unittest",
    "uv run pytest",
    "poetry run pytest",
    "hatch run test",
    "make ",
    "npm ",
    "npm run ",
    "yarn ",
    "pnpm ",
    "bun ",
    "cargo ",
    "go test",
    "go vet",
    "mvn ",
    "./mvnw ",
    "gradle ",
    "./gradlew ",
    "ruff ",
    "black ",
    "mypy ",
    "pyright",
    "flake8",
    "isort ",
    "pre-commit ",
    "eslint ",
    "prettier ",
    "tsc ",
    "npx ",
    "bundle exec ",
    "rake ",
    "composer ",
    "php ",
    "dotnet ",
    "swift test",
    "mix test",
    "stack test",
    "cabal test",
    "zig build",
    "mkdocs ",
    "sphinx-build ",
    "doc8 ",
    "vale ",
    "semgrep ",
    "bandit ",
    "gitleaks ",
)


@dataclass
class CheckSet:
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verified(self) -> list[Check]:
        return [check for check in self.checks if check.status == "verified"]

    @property
    def suggested(self) -> list[Check]:
        return [check for check in self.checks if check.status == "suggested"]

    def by_kind(self) -> dict[str, list[Check]]:
        grouped: dict[str, list[Check]] = {}
        for check in self.checks:
            grouped.setdefault(check.kind, []).append(check)
        return grouped

    def get(self, check_id: str) -> Check | None:
        return next((check for check in self.checks if check.id == check_id), None)


@dataclass
class CheckRun:
    check_id: str
    command: str
    returncode: int
    output: str
    duration_ms: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def discover_checks(root: Path, profile: RepoProfile, paths: Iterable[str]) -> CheckSet:
    root = root.resolve()
    path_list = [p.replace("\\", "/") for p in paths]
    result = CheckSet()
    lookup = {path.lower(): path for path in path_list}
    lines_cache: dict[str, list[str]] = {}

    def lines_of(rel_path: str) -> list[str]:
        if rel_path not in lines_cache:
            text = read_text(root / rel_path, limit=800_000) or ""
            lines_cache[rel_path] = split_lines(text)
        return lines_cache[rel_path]

    def add(
        command: str,
        kind: str,
        status: str,
        *,
        basis: str = "",
        sources: list[Source] | None = None,
        scope: list[str] | None = None,
        declared_in: str | None = None,
    ) -> None:
        command = _clean_command(command)
        if not command:
            return
        actual_kind = kind if kind in KIND_ORDER else classify_command(command) or kind
        if status == "verified" and not sources:
            status = "suggested"
            basis = basis or "command was inferred without a repository source"
        if status == "suggested" and not basis:
            basis = "inferred from repository contents; not declared as a command"
        result.checks.append(
            Check(
                id="pending",
                command=command,
                kind=actual_kind,
                status=status,  # type: ignore[arg-type]
                sources=sources or [],
                basis=basis,
                scope=scope or [],
                declared_in=declared_in,
            )
        )

    # --- Python / polyglot manifests -------------------------------------
    for manifest in profile.manifests:
        text = read_text(root / manifest.path, limit=500_000) or ""
        if not text:
            continue
        lines = lines_of(manifest.path)
        if manifest.kind == "pyproject":
            _pyproject_checks(add, lines, manifest.path)
        elif manifest.kind == "setup.cfg":
            _setup_cfg_checks(add, lines, manifest.path)
        elif manifest.kind == "tox.ini":
            _tox_checks(add, lines, manifest.path)
        elif manifest.kind == "noxfile.py":
            _nox_checks(add, lines, manifest.path)
        elif manifest.kind == "makefile":
            _make_checks(add, lines, manifest.path)
        elif manifest.kind == "package.json":
            _package_json_checks(add, root, lines, manifest.path, lookup)
        elif manifest.kind == "cargo.toml":
            for command, kind in (
                ("cargo test", "test"),
                ("cargo clippy --all-targets --all-features", "lint"),
                ("cargo fmt --check", "format"),
                ("cargo build", "build"),
            ):
                add(
                    command,
                    kind,
                    "suggested",
                    basis="Cargo.toml declares a Rust package; this is the cargo default, not a declared task.",
                )
        elif manifest.kind == "go.mod":
            add(
                "go test ./...",
                "test",
                "suggested",
                basis="go.mod declares a Go module; `go test ./...` is the ecosystem default, not declared.",
            )
            add(
                "go vet ./...",
                "lint",
                "suggested",
                basis="go.mod declares a Go module; `go vet` is a standard check, not declared.",
            )
        elif manifest.kind == "pom.xml":
            add("mvn test", "test", "suggested", basis="pom.xml declares a Maven build; no test command is configured here.")
        elif manifest.kind == "gradle":
            add("./gradlew test", "test", "suggested", basis="A Gradle build file is present; no test task is declared.")
        elif manifest.kind == "composer.json":
            _composer_checks(add, text, lines, manifest.path)
        elif manifest.kind == "gemfile":
            add(
                "bundle exec rake test",
                "test",
                "suggested",
                basis="A Gemfile is present; no Rake test task was found to declare this command.",
            )

    # --- CI workflows and contributor docs --------------------------------
    for workflow in profile.ci_workflows:
        _ci_checks(add, lines_of(workflow), workflow)
    for doc in profile.docs + profile.guidelines:
        _documented_checks(add, lines_of(doc), doc)

    # --- Test-file hints --------------------------------------------------
    test_paths = [path for path in path_list if tok.looks_like_test_path(path)]
    hints = _test_hints(root, test_paths, profile)
    for check in hints:
        add(
            check["command"],
            check["kind"],
            check["status"],
            basis=check["basis"],
            sources=check.get("sources", []),
            scope=check.get("scope", []),
            declared_in=check.get("declared_in"),
        )

    # --- Merge duplicates, audit, assign stable identifiers --------------
    result.checks = _merge(result.checks)
    for check in result.checks:
        audit_check(check, root)
    result.checks.sort(key=lambda check: (check.status != "verified", KIND_ORDER.index(check.kind) if check.kind in KIND_ORDER else 99, check.command))
    counts: dict[str, int] = {}
    for check in result.checks:
        counts[check.kind] = counts.get(check.kind, 0) + 1
        check.id = f"{check.kind}-{counts[check.kind]}"

    if not result.checks:
        result.notes.append(
            "No test, lint or build command is declared in this repository, and no test files were found. "
            "Ask the maintainers how they validate changes before proposing a patch."
        )
    elif not any(check.status == "verified" for check in result.checks):
        result.notes.append(
            "Every command below is a suggestion. Nothing in the repository declares how this project is checked."
        )
    if not test_paths:
        result.notes.append("No test files were detected, so the plan cannot point at an existing test to extend.")
    return result


# ---------------------------------------------------------------------------
# Config parsers
# ---------------------------------------------------------------------------


def _pyproject_checks(add, lines: list[str], path: str) -> None:
    def section(name: str) -> int | None:
        pattern = re.compile(r"^\s*\[" + re.escape(name) + r"\]\s*$")
        for index, line in enumerate(lines):
            if pattern.match(line):
                return index + 1
        return None

    specs = [
        ("tool.pytest.ini_options", "python -m pytest", "test", "pytest is configured in pyproject.toml"),
        ("tool.ruff", "ruff check .", "lint", "ruff is configured in pyproject.toml"),
        ("tool.black", "black --check .", "format", "black is configured in pyproject.toml"),
        ("tool.mypy", "mypy .", "typecheck", "mypy is configured in pyproject.toml"),
        ("tool.isort", "isort --check-only .", "format", "isort is configured in pyproject.toml"),
        ("tool.pyright", "pyright", "typecheck", "pyright is configured in pyproject.toml"),
        ("tool.tox", "tox", "test", "tox is configured in pyproject.toml"),
    ]
    for section_name, command, kind, label in specs:
        line = section(section_name)
        if line:
            add(
                command,
                kind,
                "verified",
                sources=[Source(path, line, f"[{section_name}]")],
                declared_in=f"{path}:{line}",
                basis=label,
            )


def _setup_cfg_checks(add, lines: list[str], path: str) -> None:
    parser = configparser.ConfigParser()
    try:
        parser.read_string("\n".join(lines))
    except configparser.Error:
        return
    specs = [
        ("tool:pytest", "python -m pytest", "test"),
        ("flake8", "flake8", "lint"),
        ("mypy", "mypy .", "typecheck"),
        ("isort", "isort --check-only .", "format"),
    ]
    for section, command, kind in specs:
        if parser.has_section(section):
            line = find_line(lines, f"[{section}]")
            add(
                command,
                kind,
                "verified",
                sources=[Source(path, line or 1, f"[{section}]")],
                declared_in=f"{path}:{line or 1}",
            )


def _tox_checks(add, lines: list[str], path: str) -> None:
    env = ""
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            env = line[1:-1]
            continue
        if not env.startswith("testenv"):
            continue
        if line.startswith("commands"):
            remainder = line.split("=", 1)[1].strip() if "=" in line else ""
            if remainder:
                _add_inline_commands(add, remainder, path, index + 1, "tox.ini commands")
                continue
            cursor = index + 1
            while cursor < len(lines):
                command_line = lines[cursor]
                if command_line.strip().startswith("["):
                    break
                if command_line.startswith((" ", "\t")) and command_line.strip():
                    command = command_line.strip()
                    if not command.startswith(("#", "-")):
                        kind = classify_command(command)
                        if kind:
                            add(
                                command,
                                kind,
                                "verified",
                                sources=[Source(path, cursor + 1, command)],
                                declared_in=f"{path}:{cursor + 1}",
                            )
                cursor += 1


def _nox_checks(add, lines: list[str], path: str) -> None:
    session_re = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(\s*session")
    for index, line in enumerate(lines):
        match = session_re.match(line)
        if not match:
            continue
        name = match.group(1)
        add(
            f"nox -s {name}",
            "test" if "test" in name else "lint" if "lint" in name else "build",
            "verified",
            sources=[Source(path, index + 1, line.strip())],
            declared_in=f"{path}:{index + 1}",
            basis=f"nox session `{name}` is declared",
        )


def _make_checks(add, lines: list[str], path: str) -> None:
    target_re = re.compile(r"^([A-Za-z0-9][\w.-]*)\s*:(?!=)")
    interesting = re.compile(r"(test|check|lint|format|fmt|type|build|doc|style|cover|audit)")
    for index, line in enumerate(lines):
        match = target_re.match(line)
        if not match:
            continue
        target = match.group(1)
        if target.startswith(".") or not interesting.search(target.lower()):
            continue
        add(
            f"make {target}",
            classify_command(target) or "test",
            "verified",
            sources=[Source(path, index + 1, line.strip())],
            declared_in=f"{path}:{index + 1}",
            basis=f"Makefile target `{target}`",
        )


def _package_json_checks(add, root: Path, lines: list[str], path: str, lookup: dict[str, str]) -> None:
    text = "\n".join(lines)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return
    scripts = data.get("scripts", {})
    if not isinstance(scripts, dict):
        return
    # Lockfiles are scanned but deliberately not indexed, so they may be absent
    # from ``lookup``.  Check the working tree as well.
    runner = "npm"
    for candidate, name in (
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
        ("bun.lockb", "bun"),
        ("bun.lock", "bun"),
    ):
        if candidate in lookup or (root / candidate).is_file():
            runner = name
            break
    for name, value in scripts.items():
        if not isinstance(value, str):
            continue
        kind = classify_command(f"{name} {value}", declared_name=name)
        if kind is None:
            continue
        line = find_line(lines, f'"{name}"')
        add(
            f"{runner} run {name}",
            kind,
            "verified",
            sources=[Source(path, line or 1, truncate(f'"{name}": "{value}"', 160))],
            declared_in=f"{path}:{line or 1}",
            basis=f"package.json script `{name}` (runner inferred from lockfile)",
        )


def _composer_checks(add, text: str, lines: list[str], path: str) -> None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return
    scripts = data.get("scripts", {})
    if not isinstance(scripts, dict):
        return
    for name in scripts:
        kind = classify_command(str(name))
        if not kind:
            continue
        line = find_line(lines, f'"{name}"')
        add(
            f"composer run {name}",
            kind,
            "verified",
            sources=[Source(path, line or 1, f'"{name}"')],
            declared_in=f"{path}:{line or 1}",
            basis=f"Composer script `{name}`",
        )


_CI_RUN_RE = re.compile(r"^(\s*)(?:-\s+)?run:\s*(.+?)\s*$")
_CI_SCRIPT_ITEM_RE = re.compile(r"^\s*-\s+(.+?)\s*$")


def _ci_checks(add, lines: list[str], path: str) -> None:
    is_gitlab = path.lower().endswith((".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml"))
    for index, raw in enumerate(lines):
        match = _CI_RUN_RE.match(raw)
        if match:
            value = match.group(2).strip().strip("'\"")
            if value in {"|", ">", "|-", ">-", "|+", ">+"}:
                parent_indent = len(match.group(1))
                cursor = index + 1
                while cursor < len(lines):
                    candidate = lines[cursor]
                    if not candidate.strip():
                        cursor += 1
                        continue
                    indent = len(candidate) - len(candidate.lstrip())
                    if indent <= parent_indent:
                        break
                    command = candidate.strip()
                    if not command.startswith(("#", "echo ", "export ", "cd ", "set ", "if ", "fi", "then", "do", "done", "}")):
                        kind = classify_command(command)
                        if kind:
                            add(
                                command,
                                kind,
                                "verified",
                                sources=[Source(path, cursor + 1, truncate(command, 200))],
                                declared_in=f"{path}:{cursor + 1}",
                                basis="GitHub Actions step",
                            )
                    cursor += 1
            else:
                kind = classify_command(value)
                if kind:
                    add(
                        value,
                        kind,
                        "verified",
                        sources=[Source(path, index + 1, truncate(value, 200))],
                        declared_in=f"{path}:{index + 1}",
                        basis="CI workflow step",
                    )
        elif is_gitlab:
            match = _CI_SCRIPT_ITEM_RE.match(raw)
            if match:
                command = match.group(1).strip().strip("'\"")
                kind = classify_command(command)
                if kind:
                    add(
                        command,
                        kind,
                        "verified",
                        sources=[Source(path, index + 1, truncate(command, 200))],
                        declared_in=f"{path}:{index + 1}",
                        basis="CI script step",
                    )


# A fenced block in a document is usually a directory tree, an example, or
# program output.  Only lines that begin with a real command runner are treated
# as documented checks.
_COMMAND_PREFIXES = frozenset(
    {
        "python", "python3", "py", "pip", "pip3", "pipx", "poetry", "uv", "pdm",
        "hatch", "pipenv", "conda", "pytest", "tox", "nox", "coverage", "coverage3",
        "ruff", "mypy", "pyright", "black", "isort", "flake8", "pylint", "bandit",
        "semgrep", "pip-audit", "safety", "npm", "pnpm", "yarn", "bun", "npx",
        "node", "deno", "eslint", "prettier", "tsc", "jest", "vitest", "mocha",
        "cargo", "rustc", "rustup", "go", "gofmt", "golangci-lint", "staticcheck",
        "mvn", "mvnw", "gradle", "gradlew", "dotnet", "msbuild", "ant", "sbt",
        "composer", "phpunit", "bundle", "rake", "rspec", "ruby", "make", "cmake",
        "ctest", "bazel", "bazelisk", "docker", "docker-compose",
        "gitleaks", "trivy", "swift", "mix", "rebar3", "stack", "cabal", "ghc",
        "perl", "prove", "Rscript", "julia", "zig", "nimble", "dune", "opam",
        "lein", "clojure", "elm", "task", "just", "mage",
    }
)

_VERIFICATION_HEADING_RE = re.compile(
    r"\b(test\w*|lint\w*|check\w*|verif\w*|validat\w*|build\w*|develop\w*|"
    r"contribut\w*|quality|format\w*)\b",
    re.IGNORECASE,
)

_CONTRIBUTION_DOC_RE = re.compile(
    r"(contribut|hacking|develop|testing|test-?guide|maintainer)", re.IGNORECASE
)


def _looks_like_command(line: str) -> bool:
    stripped = line.lstrip("$ ").strip()
    if not stripped or stripped.startswith(("#", "//")):
        return False
    if any(character in stripped for character in "├└│┌┐┘┬┴┼"):
        return False
    first = stripped.split()[0]
    if first.endswith(("/", "\\", ":", "=")) or first.startswith(("-", "=")):
        return False
    return first in _COMMAND_PREFIXES


def _documented_checks(add, lines: list[str], path: str) -> None:
    in_fence = False
    heading = ""
    allow_any_fence = bool(_CONTRIBUTION_DOC_RE.search(path))
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("#"):
            if not in_fence:
                heading = stripped.lstrip("#").strip()
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            continue
        if not stripped or stripped.startswith(("#", "//")):
            continue
        if not allow_any_fence and not _VERIFICATION_HEADING_RE.search(heading):
            continue
        if not _looks_like_command(stripped):
            continue
        command = stripped.lstrip("$ ").strip()
        if any(placeholder in command for placeholder in ("<", ">", "...", "your_", "YOUR_")):
            continue
        kind = classify_command(command)
        if not kind:
            continue
        add(
            command,
            kind,
            "verified",
            sources=[Source(path, index + 1, truncate(command, 200))],
            declared_in=f"{path}:{index + 1}",
            basis=f"documented command in {path} (may be stale)",
        )


def _add_inline_commands(add, text: str, path: str, line: int, basis: str) -> None:
    for piece in text.split("&&"):
        command = piece.strip()
        kind = classify_command(command)
        if kind:
            add(
                command,
                kind,
                "verified",
                sources=[Source(path, line, command)],
                declared_in=f"{path}:{line}",
                basis=basis,
            )


# ---------------------------------------------------------------------------
# Suggestions based on files
# ---------------------------------------------------------------------------


def _test_hints(root: Path, test_paths: list[str], profile: RepoProfile) -> list[dict]:
    hints: list[dict] = []
    if not test_paths:
        return hints

    sample_text = ""
    for path in test_paths[:15]:
        sample_text += read_text(root / path, limit=80_000) or ""
    python_tests = [path for path in test_paths if path.endswith((".py", ".pyi"))]
    js_tests = [path for path in test_paths if path.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"))]

    declares_pytest = any(
        "tool.pytest" in (read_text(root / manifest.path, limit=100_000) or "")
        for manifest in profile.manifests
    )
    if python_tests:
        pytest_basis = "pytest imports appear in test files; no pytest configuration declares the command"
        if "import pytest" in sample_text or "from pytest" in sample_text or "pytest.mark" in sample_text:
            hints.append(
                {
                    "command": "python -m pytest",
                    "kind": "test",
                    "status": "suggested",
                    "basis": pytest_basis,
                    "scope": python_tests[:8],
                }
            )
        elif "unittest" in sample_text:
            test_dir = _common_test_dir(python_tests)
            hints.append(
                {
                    "command": f"python -m unittest discover -s {test_dir} -v",
                    "kind": "test",
                    "status": "suggested",
                    "basis": "test files import unittest; this is the stdlib default, not a declared command",
                    "scope": python_tests[:8],
                }
            )
        elif not declares_pytest:
            hints.append(
                {
                    "command": "python -m pytest",
                    "kind": "test",
                    "status": "suggested",
                    "basis": "Python test files exist; pytest is the common runner but is not configured here",
                    "scope": python_tests[:8],
                }
            )

    if js_tests:
        package_text = ""
        for manifest in profile.manifests:
            if manifest.kind == "package.json":
                package_text = read_text(root / manifest.path, limit=200_000) or ""
                break
        runner = None
        if '"vitest"' in package_text:
            runner = "npx vitest run"
        elif '"jest"' in package_text:
            runner = "npx jest"
        elif '"mocha"' in package_text:
            runner = "npx mocha"
        if runner:
            hints.append(
                {
                    "command": runner,
                    "kind": "test",
                    "status": "suggested",
                    "basis": "the runner is listed as a dependency but no test script declares it",
                    "scope": js_tests[:8],
                }
            )

    if any(path.endswith(".go") for path in test_paths):
        pass  # go test is already suggested from go.mod
    return hints


def _common_test_dir(paths: list[str]) -> str:
    directories = {str(Path(path).parent).replace("\\", "/") for path in paths}
    if len(directories) == 1:
        directory = directories.pop()
        return directory if directory != "." else "."
    return "tests" if any(path.startswith("tests/") for path in paths) else "."


# ---------------------------------------------------------------------------
# Auditing and running
# ---------------------------------------------------------------------------

_COMPOUND_RE = re.compile(r"[|;&><`]|\$\(")


def audit_check(check: Check, root: Path) -> None:
    check.tool_notes = []
    check.runnable = False
    command = check.command.strip()
    if not command:
        check.tool_notes.append("empty command")
        return
    if "${{" in command or _COMPOUND_RE.search(command):
        check.tool_notes.append("compound shell command; run it manually")
        return
    try:
        argv = shlex.split(command)
    except ValueError:
        check.tool_notes.append("command could not be parsed; run it manually")
        return
    if not argv:
        return
    executable = argv[0]
    if executable in {"python", "python3", "py"}:
        module = None
        if "-m" in argv:
            position = argv.index("-m")
            if position + 1 < len(argv):
                module = argv[position + 1]
        if module:
            try:
                available = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                available = False
            if not available:
                check.tool_notes.append(
                    f"Python module `{module}` is not installed in the current interpreter; the command is still declared."
                )
                return
            check.tool_notes.append("will run with this Python interpreter")
            check.runnable = True
            return
        check.tool_notes.append("will run with this Python interpreter")
        check.runnable = True
        return
    if executable.startswith("./") or executable.startswith(".\\"):
        target = root / executable[2:]
        if target.exists():
            check.runnable = True
        else:
            check.tool_notes.append(f"`{executable}` does not exist")
        return
    located = which(executable)
    if located:
        check.runnable = True
    else:
        check.tool_notes.append(f"`{executable}` is not installed on this machine; the command is still declared.")


def run_check(check: Check, root: Path, timeout: int = 180) -> CheckRun:
    if check.status != "verified":
        raise ValueError("Refusing to run an unverified suggestion. Confirm it manually first.")
    if not check.runnable:
        raise ValueError("This command is declared but is not runnable in the current environment.")
    argv = shlex.split(check.command)
    if argv and argv[0] in {"python", "python3", "py"}:
        argv[0] = sys.executable
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        duration = int((time.perf_counter() - started) * 1000)
        output = (completed.stdout or "") + (completed.stderr or "")
        return CheckRun(
            check_id=check.id,
            command=check.command,
            returncode=completed.returncode,
            output=truncate(output.strip(), 8000),
            duration_ms=duration,
        )
    except subprocess.TimeoutExpired as exc:
        duration = int((time.perf_counter() - started) * 1000)
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return CheckRun(
            check_id=check.id,
            command=check.command,
            returncode=124,
            output=truncate(output + "\n[timed out]", 8000),
            duration_ms=duration,
            timed_out=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def classify_command(command: str, declared_name: str = "") -> str | None:
    text = f"{declared_name} {command}".lower()
    if not text.strip():
        return None
    if re.search(r"\b(gitleaks|trivy|bandit|semgrep|npm audit|yarn audit|pnpm audit|safety check|pip-audit)\b", text):
        return "security"
    if re.search(r"\b(test|tests|pytest|unittest|jest|vitest|mocha|tox|nox|spec|rspec|phpunit|ctest)\b", text):
        if re.search(r"\b(install|setup|download)\b", text):
            return None
        return "test"
    if re.search(r"\b(lint|eslint|ruff|flake8|pylint|rubocop|stylelint|clippy|golangci|staticcheck|checkstyle)\b", text):
        return "lint"
    if re.search(r"\b(typecheck|type-check|mypy|pyright|tsc|typescript --noemit)\b", text):
        return "typecheck"
    if re.search(r"\b(format|fmt|prettier|black|isort|rustfmt|gofmt|dprint)\b", text):
        return "format"
    if re.search(r"\b(build|compile|bundle|package|dist|wheel|sdist)\b", text):
        return "build"
    if re.search(r"\b(docs|documentation|mkdocs|sphinx|docusaurus|doc8|vale)\b", text):
        return "docs"
    return None


def _clean_command(command: str) -> str:
    command = command.strip().strip("`")
    command = re.sub(r"^\$\s+", "", command)
    return truncate(command, 240)


def _merge(checks: list[Check]) -> list[Check]:
    merged: dict[str, Check] = {}
    order: list[str] = []
    for check in checks:
        key = re.sub(r"\s+", " ", check.command.strip()).lower()
        if key not in merged:
            merged[key] = check
            order.append(key)
            continue
        existing = merged[key]
        if check.status == "verified" and existing.status == "suggested":
            check.id = existing.id
            merged[key] = check
            existing = check
        see = {(source.path, source.line) for source in existing.sources}
        for source in check.sources:
            if (source.path, source.line) not in see:
                existing.sources.append(source)
                see.add((source.path, source.line))
        existing.scope = sorted(set(existing.scope) | set(check.scope))
        if not existing.basis and check.basis:
            existing.basis = check.basis
    return [merged[key] for key in order]
