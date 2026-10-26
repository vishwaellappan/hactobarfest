"""Repository discovery: walk the tree, honour ignore rules, build a profile."""

from __future__ import annotations

import configparser
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import tokens as tok
from .models import Manifest, RepoProfile
from .util import (
    PROGRAMMING_LANGS,
    detect_language,
    find_line,
    first_sentence,
    now_iso,
    read_text,
    relative,
    run_git,
    sha,
    split_lines,
    truncate,
)

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - handled in the CLI
    tomllib = None  # type: ignore[assignment]

MAX_INDEX_BYTES = 1_500_000
MAX_FILES = 30_000

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "bower_components",
    "vendor",
    "third_party",
    "site-packages",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".gradle",
    ".idea",
    ".vscode",
    ".next",
    ".nuxt",
    ".cache",
    ".parcel-cache",
    "dist",
    "build",
    "target",
    "coverage",
    "htmlcov",
    ".coverage",
    "eggs",
    ".eggs",
    "bin",
    "obj",
    "out",
    ".terraform",
    ".serverless",
    ".yarn",
    ".pnpm-store",
    "Pods",
    "DerivedData",
    "tmp",
    "temp",
}

SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
    "Gemfile.lock",
}

SKIP_SUFFIXES = (".min.js", ".min.css", ".bundle.js", ".map", ".chunk.js")

# Never index credential material.  Documentation variants (``.env.example``,
# ``.env.sample``) are safe and stay indexable.
SECRET_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "secrets.yml",
    "secrets.yaml",
}
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore")


def is_secret_filename(name: str) -> bool:
    lower = name.lower()
    if lower.endswith((".example", ".sample", ".template", ".dist", ".md")):
        return False
    if lower in SECRET_FILENAMES:
        return True
    if lower.startswith(".env.") and lower.count(".") >= 2:
        return True
    return lower.endswith(SECRET_SUFFIXES)

CONFIG_FILENAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "noxfile.py",
    "package.json",
    "tsconfig.json",
    "cargo.toml",
    "go.mod",
    "makefile",
    "justfile",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "requirements.txt",
    "requirements-dev.txt",
    "gemfile",
    "composer.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    ".pre-commit-config.yaml",
    ".editorconfig",
    "mkdocs.yml",
    "docusaurus.config.js",
    "readthedocs.yml",
    ".readthedocs.yaml",
}


def _translate_glob(pattern: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if pattern[index : index + 2] == "**":
                index += 2
                if pattern[index : index + 1] == "/":
                    out.append("(?:.*/)?")
                    index += 1
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                out.append(r"\[")
            else:
                body = pattern[index + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body + "]")
                index = end
        else:
            out.append(re.escape(char))
        index += 1
    return "".join(out)


class GitIgnore:
    """A pragmatic subset of gitignore semantics (last matching rule wins)."""

    def __init__(self, root: Path) -> None:
        self.rules: list[tuple[re.Pattern[str], bool]] = []
        for candidate in (root / ".gitignore", root / ".git" / "info" / "exclude"):
            text = read_text(candidate, limit=200_000) if candidate.exists() else None
            if text:
                self._load(text)

    def _load(self, text: str) -> None:
        for raw in split_lines(text):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:].strip()
            if not line:
                continue
            directory_only = line.endswith("/")
            line = line.rstrip("/")
            anchored = line.startswith("/")
            line = line.lstrip("/")
            if not line:
                continue
            prefix = "" if (anchored or "/" in line) else r"(?:.*/)?"
            regex = "^" + prefix + _translate_glob(line)
            if directory_only:
                regex += r"(?:/.*)?"
            regex += "$"
            try:
                self.rules.append((re.compile(regex), negated))
            except re.error:
                continue

    def ignored(self, rel_path: str, is_dir: bool = False) -> bool:
        target = rel_path.replace("\\", "/")
        if is_dir:
            target = target.rstrip("/") + "/"
        state = False
        for regex, negated in self.rules:
            if regex.match(target.rstrip("/")) or (is_dir and regex.match(target)):
                state = not negated
        return state


@dataclass
class ScannedFile:
    path: str
    abs_path: Path
    lang: str
    lines: int
    bytes: int
    is_test: bool = False
    is_doc: bool = False
    is_config: bool = False
    digest: str = ""

    @property
    def stem(self) -> str:
        return Path(self.path).stem


@dataclass
class ScanResult:
    root: Path
    files: list[ScannedFile]
    profile: RepoProfile
    skipped: dict[str, int] = field(default_factory=dict)

    def programming_files(self) -> list[ScannedFile]:
        return [f for f in self.files if f.lang in PROGRAMMING_LANGS]


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name == ".git"


def discover(root: Path, *, max_files: int = MAX_FILES) -> ScanResult:
    root = root.resolve()
    ignore = GitIgnore(root)
    scanned: list[ScannedFile] = []
    skipped: dict[str, int] = {"ignored": 0, "binary_or_too_large": 0, "generated": 0, "limit": 0}

    for current, dirs, names in os.walk(root):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            child = current_path / name
            rel = relative(root, child)
            if _should_skip_dir(name):
                skipped["ignored"] += 1
                continue
            if ignore.ignored(rel, is_dir=True):
                skipped["ignored"] += 1
                continue
            if child.is_symlink():
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs

        for name in sorted(names):
            if len(scanned) >= max_files:
                skipped["limit"] += 1
                continue
            if name in SKIP_FILES or name.endswith(SKIP_SUFFIXES):
                skipped["generated"] += 1
                continue
            if is_secret_filename(name):
                skipped["ignored"] += 1
                continue
            abs_path = current_path / name
            if abs_path.is_symlink() or not abs_path.is_file():
                continue
            rel = relative(root, abs_path)
            if ignore.ignored(rel, is_dir=False):
                skipped["ignored"] += 1
                continue
            try:
                size = abs_path.stat().st_size
            except OSError:
                continue
            if size > MAX_INDEX_BYTES:
                skipped["binary_or_too_large"] += 1
                continue
            text = read_text(abs_path)
            if text is None:
                skipped["binary_or_too_large"] += 1
                continue
            lang = detect_language(rel, name)
            lower_name = name.lower()
            is_test = tok.looks_like_test_path(rel)
            is_doc = tok.looks_like_doc_path(rel)
            is_config = lower_name in CONFIG_FILENAMES or tok.looks_like_config_path(rel)
            line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
            scanned.append(
                ScannedFile(
                    path=rel,
                    abs_path=abs_path,
                    lang=lang,
                    lines=line_count,
                    bytes=size,
                    is_test=is_test,
                    is_doc=is_doc,
                    is_config=is_config,
                    digest=sha(text),
                )
            )

    profile = build_profile(root, scanned)
    return ScanResult(root=root, files=scanned, profile=profile, skipped=skipped)


def detect_repo_root(start: Path) -> Path:
    """Prefer the nearest enclosing git work tree, else the start directory."""
    start = start.resolve()
    if start.is_file():
        start = start.parent
    current = start
    for _ in range(40):
        if (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return start


def build_profile(root: Path, files: list[ScannedFile]) -> RepoProfile:
    language_counts: dict[str, int] = {}
    total_lines = 0
    docs: list[str] = []
    guidelines: list[str] = []
    issue_templates: list[str] = []
    workflows: list[str] = []
    manifests: list[Manifest] = []
    test_files = 0

    for item in files:
        total_lines += item.lines
        if item.lang in PROGRAMMING_LANGS:
            language_counts[item.lang] = language_counts.get(item.lang, 0) + item.lines
        if item.is_test:
            test_files += 1
        lower = item.path.lower()
        name = lower.rsplit("/", 1)[-1]
        if item.is_doc and not item.is_test:
            docs.append(item.path)
        if (
            (
                name.startswith(("contributing", "code_of_conduct", "code-of-conduct", "security", "support",
                                 "pull_request_template", "pull-request-template", "governance"))
                and (item.is_doc or "." not in name)
            )
            or lower.startswith((".github/pull_request_template", ".github/pull-request-template",
                                 ".github/issue_template/", ".github/issue-templates/"))
            or "contributing" in lower
        ):
            guidelines.append(item.path)
            if "issue_template" in lower:
                issue_templates.append(item.path)
        if (
            lower.startswith(".github/workflows/")
            or lower in {".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml", ".circleci/config.yml"}
            or lower.startswith(".buildkite/")
        ):
            workflows.append(item.path)
        manifest = _manifest_for(item.abs_path, item.path)
        if manifest:
            manifests.append(manifest)

    docs.sort(key=lambda path: (path.count("/"), path.lower()))
    guidelines.sort(key=lambda path: (path.count("/"), path.lower()))
    workflows.sort()
    manifests.sort(key=lambda m: (m.path.count("/"), m.path))
    languages = dict(sorted(language_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10])

    name, description, package_name, entry_points, license_name = _project_identity(
        root, files, manifests
    )

    if not description:
        description = _readme_summary(root, docs)

    top_level = sorted(
        {
            path.split("/")[0] + ("/" if "/" in path else "")
            for path in (item.path for item in files)
        }
    )[:40]

    profile = RepoProfile(
        root=str(root),
        name=name or root.name,
        files_indexed=len(files),
        total_lines=total_lines,
        languages=languages,
        manifests=manifests,
        docs=docs[:30],
        guidelines=guidelines[:20],
        issue_templates=issue_templates[:10],
        ci_workflows=workflows[:10],
        test_files=test_files,
        entry_points=entry_points,
        package_name=package_name,
        description=description,
        license=license_name,
        top_level=top_level,
    )
    _attach_git(root, profile)
    return profile


def _attach_git(root: Path, profile: RepoProfile) -> None:
    if not (root / ".git").exists():
        return
    profile.is_git = True
    profile.branch = run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    profile.commit = run_git(root, "rev-parse", "--short", "HEAD")
    profile.remote = run_git(root, "config", "--get", "remote.origin.url")


def _manifest_for(abs_path: Path, rel_path: str) -> Manifest | None:
    name = abs_path.name.lower()
    kind = {
        "pyproject.toml": "pyproject",
        "setup.py": "setup.py",
        "setup.cfg": "setup.cfg",
        "package.json": "package.json",
        "cargo.toml": "cargo.toml",
        "go.mod": "go.mod",
        "composer.json": "composer.json",
        "pom.xml": "pom.xml",
        "gemfile": "gemfile",
        "build.gradle": "gradle",
        "build.gradle.kts": "gradle",
        "makefile": "makefile",
        "justfile": "justfile",
        "tox.ini": "tox.ini",
        "noxfile.py": "noxfile.py",
        "requirements.txt": "requirements.txt",
        "environment.yml": "environment.yml",
        "dockerfile": "dockerfile",
        "pubspec.yaml": "pubspec.yaml",
        "mix.exs": "mix.exs",
    }.get(name)
    if not kind and rel_path.count("/") <= 1 and (name.endswith(".csproj") or name.endswith(".sln")):
        kind = "dotnet"
    if not kind:
        return None
    text = read_text(abs_path, limit=500_000) or ""
    parsed = _parse_manifest(abs_path, rel_path, kind, text)
    return Manifest(path=rel_path, kind=kind, name=parsed.get("name", ""), description=parsed.get("description", ""))


def _parse_toml(text: str) -> dict[str, Any]:
    if tomllib is None:
        return {}
    try:
        return tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return {}


def _parse_manifest(abs_path: Path, rel_path: str, kind: str, text: str) -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        if kind == "pyproject":
            data = _parse_toml(text)
            project = data.get("project", {}) if isinstance(data, dict) else {}
            poetry = data.get("tool", {}).get("poetry", {}) if isinstance(data, dict) else {}
            source = project or poetry
            info["name"] = str(source.get("name", "") or "")
            info["description"] = str(source.get("description", "") or "")
            info["license"] = _license_value(source.get("license"))
        elif kind == "package.json":
            data = json.loads(text)
            info["name"] = str(data.get("name", "") or "")
            info["description"] = str(data.get("description", "") or "")
            info["license"] = str(data.get("license", "") or "")
        elif kind == "cargo.toml":
            data = _parse_toml(text)
            package = data.get("package", {}) if isinstance(data, dict) else {}
            info["name"] = str(package.get("name", "") or "")
            info["description"] = str(package.get("description", "") or "")
            info["license"] = str(package.get("license", "") or "")
        elif kind == "go.mod":
            match = re.search(r"^module\s+(\S+)", text, re.MULTILINE)
            info["name"] = match.group(1) if match else ""
        elif kind == "setup.cfg":
            parser = configparser.ConfigParser()
            parser.read_string(text)
            if parser.has_section("metadata"):
                info["name"] = parser.get("metadata", "name", fallback="")
                info["description"] = parser.get("metadata", "description", fallback="")
                info["license"] = parser.get("metadata", "license", fallback="")
        elif kind == "setup.py":
            info["name"] = _regex_value(text, r"name\s*=\s*[\"']([^\"']+)")
            info["description"] = _regex_value(text, r"description\s*=\s*[\"']([^\"']+)")
            info["license"] = _regex_value(text, r"license\s*=\s*[\"']([^\"']+)")
        elif kind == "composer.json":
            data = json.loads(text)
            info["name"] = str(data.get("name", "") or "")
            info["description"] = str(data.get("description", "") or "")
            info["license"] = str(data.get("license", "") or "")
        elif kind == "pom.xml":
            match = re.search(r"<artifactId>([^<]+)</artifactId>", text)
            info["name"] = match.group(1) if match else ""
            info["description"] = _regex_value(text, r"<description>([^<]+)</description>")
        elif kind == "gemfile":
            match = re.search(r"gemspec", text)
            info["name"] = "" if match else ""
        else:
            info["description"] = ""
    except (ValueError, KeyError, configparser.Error, json.JSONDecodeError):
        pass
    if not info.get("name"):
        info["name"] = Path(rel_path).stem
    info["description"] = truncate(info.get("description", ""), 300)
    return info


def _license_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("text", "") or value.get("file", "") or "")
    return ""


def _regex_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _project_identity(
    root: Path, files: list[ScannedFile], manifests: list[Manifest]
) -> tuple[str, str, str, list[str], str]:
    name = ""
    description = ""
    package_name = ""
    entry_points: list[str] = []
    license_name = ""

    for manifest in manifests:
        if manifest.kind in {"pyproject", "package.json", "cargo.toml", "go.mod", "setup.cfg", "setup.py", "composer.json"}:
            if not name and manifest.name:
                name = manifest.name
            if not description and manifest.description:
                description = manifest.description
            if not package_name and manifest.name:
                package_name = manifest.name
        text = read_text(root / manifest.path, limit=500_000) or ""
        license_name = license_name or _manifest_license(text, manifest.kind)
        entry_points.extend(_manifest_entry_points(text, manifest.kind))

    if not name:
        for item in files:
            if item.path.lower().startswith("readme"):
                text = read_text(item.abs_path) or ""
                match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                if match:
                    name = truncate(match.group(1), 80)
                break

    if not license_name:
        for item in files:
            lower = item.path.lower().rsplit("/", 1)[-1]
            if lower.startswith(("license", "licence", "copying")):
                text = read_text(item.abs_path, limit=20_000) or ""
                first = next((line.strip() for line in split_lines(text) if line.strip()), "")
                license_name = truncate(first, 60)
                break

    return name, description, package_name, sorted(set(entry_points))[:15], license_name


def _manifest_entry_points(text: str, kind: str) -> list[str]:
    found: list[str] = []
    if kind == "pyproject":
        data = _parse_toml(text)
        scripts = {}
        if isinstance(data, dict):
            project = data.get("project", {})
            scripts = project.get("scripts", {}) if isinstance(project, dict) else {}
            if not scripts:
                poetry = data.get("tool", {}).get("poetry", {})
                scripts = poetry.get("scripts", {}) if isinstance(poetry, dict) else {}
        if isinstance(scripts, dict):
            for key, value in list(scripts.items())[:20]:
                found.append(f"python -m {key}" if isinstance(value, str) else str(key))
    elif kind == "package.json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        bins = data.get("bin", {})
        if isinstance(bins, str):
            found.append(bins)
        elif isinstance(bins, dict):
            found.extend(str(value) for value in list(bins.values())[:10])
    elif kind == "setup.cfg":
        section = re.search(r"\[options\.entry_points\](.*?)(?:\n\[|\Z)", text, re.S)
        if section:
            for match in re.finditer(r"^\s*([\w.-]+)\s*=", section.group(1), re.MULTILINE):
                found.append(match.group(1))
    return found


def _manifest_license(text: str, kind: str) -> str:
    if kind == "pyproject":
        data = _parse_toml(text)
        project = data.get("project", {}) if isinstance(data, dict) else {}
        value = project.get("license") if isinstance(project, dict) else None
        if isinstance(value, dict):
            return str(value.get("text", "") or value.get("file", ""))
        return str(value or "")
    patterns = (
        r"license\s*=\s*[\"']([^\"']+)",
        r"^license\s*[:=]\s*(.+)$",
        r"\"license\"\s*:\s*\"([^\"]+)\"",
        r"<license[^>]*>([^<]+)</license>",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return truncate(match.group(1), 60)
    return ""


def _readme_summary(root: Path, docs: Iterable[str]) -> str:
    for path in docs:
        name = path.lower().rsplit("/", 1)[-1]
        if name.startswith("readme"):
            text = read_text(root / path, limit=200_000) or ""
            lines = split_lines(text)
            start = find_line(lines, "## ") or len(lines)
            body = "\n".join(lines[: max(1, start - 1)])
            return first_sentence(body)
    return ""


def collect_profile(root: Path) -> RepoProfile:
    """Convenience wrapper used by ``overview`` when an index already exists."""
    return build_profile(root, list(discover(root).files))


STAMP = now_iso
