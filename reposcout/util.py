"""Small, dependency-free helpers used across RepoScout."""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import re
import subprocess
from pathlib import Path

TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")
MAX_FILE_BYTES = 2_000_000


def posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def relative(root: Path, path: Path) -> str:
    try:
        return posix(path.relative_to(root))
    except ValueError:
        return posix(path)


def read_bytes(path: Path, limit: int = MAX_FILE_BYTES) -> bytes | None:
    try:
        with path.open("rb") as handle:
            return handle.read(limit)
    except (OSError, PermissionError):
        return None


def is_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data[:8192]:
        return True
    # Heuristic: large share of non-text control bytes.
    sample = data[:8192]
    control = sum(1 for byte in sample if byte < 9 or 13 < byte < 32)
    return control / max(1, len(sample)) > 0.30


def read_text(path: Path, limit: int = MAX_FILE_BYTES) -> str | None:
    data = read_bytes(path, limit)
    if data is None or is_binary(data):
        return None
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def split_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def truncate(text: str, width: int, suffix: str = "...") -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: max(0, width - len(suffix))].rstrip() + suffix


def first_sentence(text: str, width: int = 240) -> str:
    """Return the first meaningful sentence of prose, truncated."""
    cleaned_lines: list[str] = []
    for line in split_lines(text):
        stripped = line.strip()
        if not stripped:
            if cleaned_lines:
                break
            continue
        if stripped.startswith(("#", "```", "---", "===", "<!--", "|")):
            if cleaned_lines:
                break
            continue
        cleaned_lines.append(stripped)
        if len(" ".join(cleaned_lines)) > width:
            break
    sentence = " ".join(cleaned_lines)
    match = re.search(r"(?<=[.!?])\s+", sentence)
    if match:
        sentence = sentence[: match.start()]
    return truncate(sentence, width)


def find_line(lines: list[str], needle: str, start: int = 0) -> int | None:
    """1-based line number of the first line containing ``needle``."""
    lowered = needle.lower()
    for index in range(start, len(lines)):
        if lowered in lines[index].lower():
            return index + 1
    return None


def line_text(lines: list[str], line: int | None) -> str:
    if line is None or line < 1 or line > len(lines):
        return ""
    return lines[line - 1].strip()


def statement_at(text: str, line: int, width: int = 200) -> str:
    lines = split_lines(text)
    return truncate(line_text(lines, line), width)


def detect_language(path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    name = filename.lower()
    special = {
        "dockerfile": "docker",
        "makefile": "make",
        "cmakelists.txt": "cmake",
        "justfile": "just",
        "procfile": "procfile",
        "gemfile": "ruby",
        "rakefile": "ruby",
        "requirements.txt": "text",
        "license": "text",
        "notice": "text",
        "codeowners": "text",
    }
    if name in special:
        return special[name]
    return {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mts": "typescript",
        ".cts": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hh": "cpp",
        ".m": "objective-c",
        ".mm": "objective-c",
        ".swift": "swift",
        ".lua": "lua",
        ".pl": "perl",
        ".pm": "perl",
        ".r": "r",
        ".jl": "julia",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".fish": "shell",
        ".ps1": "powershell",
        ".bat": "batch",
        ".cmd": "batch",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "css",
        ".sass": "css",
        ".less": "css",
        ".vue": "vue",
        ".svelte": "svelte",
        ".sql": "sql",
        ".graphql": "graphql",
        ".gql": "graphql",
        ".proto": "protobuf",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".jsonc": "json",
        ".xml": "xml",
        ".md": "markdown",
        ".mdx": "markdown",
        ".rst": "rst",
        ".txt": "text",
        ".tex": "latex",
        ".ipynb": "notebook",
        ".env": "dotenv",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "ini",
        ".gradle": "gradle",
        ".tf": "terraform",
        ".hcl": "hcl",
        ".nix": "nix",
        ".zig": "zig",
        ".ex": "elixir",
        ".exs": "elixir",
        ".erl": "erlang",
        ".hs": "haskell",
        ".ml": "ocaml",
        ".clj": "clojure",
        ".groovy": "groovy",
        ".dart": "dart",
        ".sol": "solidity",
        ".asm": "assembly",
        ".s": "assembly",
    }.get(ext, "text" if ext in {".log", ".csv"} else "other")


PROGRAMMING_LANGS = {
    "python",
    "javascript",
    "typescript",
    "go",
    "rust",
    "java",
    "kotlin",
    "scala",
    "ruby",
    "php",
    "csharp",
    "c",
    "cpp",
    "objective-c",
    "swift",
    "lua",
    "perl",
    "r",
    "julia",
    "shell",
    "powershell",
    "vue",
    "svelte",
    "dart",
    "elixir",
    "haskell",
    "clojure",
    "zig",
}


def module_label(path: str) -> str:
    """Human-friendly module name for a file path."""
    cleaned = posix(path)
    for prefix in ("src/", "lib/", "source/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    stem = cleaned.rsplit(".", 1)[0]
    stem = stem.replace("/", ".")
    if stem.endswith(".__init__"):
        stem = stem[: -len(".__init__")]
    return stem


def run_git(root: Path, *args: str, timeout: int = 8) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def slug(text: str, limit: int = 60) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:limit] or "report"


def which(program: str) -> str | None:
    from shutil import which as _which

    return _which(program)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def common_ancestor(paths: list[str]) -> str:
    if not paths:
        return ""
    parts = [p.split("/") for p in paths]
    prefix = parts[0]
    for other in parts[1:]:
        new: list[str] = []
        for left, right in zip(prefix, other):
            if left != right:
                break
            new.append(left)
        prefix = new
    return "/".join(prefix)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
