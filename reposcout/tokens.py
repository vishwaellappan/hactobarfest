"""Tokenization, light stemming and query expansion.

RepoScout deliberately avoids embeddings and external models for its core
retrieval.  A transparent lexical index means every ranking decision can be
traced back to a term that literally appears in the repository.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Task-tracker boilerplate and English function words.  These are removed from
# queries, not from the index: code may legitimately contain the word "issue".
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else when while of to in on at by for with from as is are was were be been being
    this that these those it its it's their there here you your yours we our ours they them he she his her
    i me my mine do does did done doing have has had having can could should would will shall may might must
    not no yes so such than too very just also only about into over under again further once all any both each
    few more most other some own same up down out off above below between through during before after
    what which who whom whose why how where
    please thanks thank hi hello regards
    issue bug feature request problem bugfix fixed fixes fixing
    add adds added adding support supports supported allow allows make makes making use uses used using
    implement implements implemented implementation provide provides provided
    expected actual behavior behaviour steps reproduce reproduction description context
    possible easy good first new need needs needed want wants like update updates updated change changes changed
    currently existing instead able unable unable work works working
    user users case cases example examples way ways thing things
    """.split()
)

# Words that are too common in code to carry retrieval signal on their own.
CODE_STOPWORDS: frozenset[str] = frozenset(
    """
    var let const function return import export from default class def self this super true false none null
    nil new public private protected static final void int float double bool string str list dict object
    print console log logging getter setter module package src lib test tests spec
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_ACRONYM_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_BACKTICK_RE = re.compile(r"`([^`\n]{2,120})`")
_QUOTE_RE = re.compile(r"[\"“]([^\"”\n]{4,160})[\"”]")
_PATH_RE = re.compile(
    r"(?<![\w/])((?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|js|mjs|cjs|jsx|ts|tsx|go|rs|java|kt|rb|php|cs|c|h|cc|cpp|hpp|"
    r"md|rst|toml|ya?ml|json|cfg|ini|txt|sh|ps1|html|css|scss|sql|proto))\b"
)
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(\s*\)")
_ERROR_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{2,}(?:Error|Exception|Warning))\b")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("crash", "exception", "traceback", "panic", "segfault", "failure", "fail"),
    ("error", "err", "exception", "traceback", "failure", "raise"),
    ("config", "configuration", "configure", "settings", "option", "options", "env", "environment"),
    ("login", "auth", "authentication", "signin", "token", "session"),
    ("test", "tests", "pytest", "unittest", "spec", "specs"),
    ("doc", "docs", "documentation", "readme", "guide", "tutorial"),
    ("slow", "performance", "perf", "latency", "timeout", "timing", "optimization", "optimize"),
    ("memory", "leak", "cache", "gc"),
    ("ui", "frontend", "view", "render", "template", "interface"),
    ("api", "endpoint", "route", "handler", "server", "http"),
    ("database", "db", "sql", "query", "sqlite", "postgres", "mysql"),
    ("file", "path", "filename", "filepath", "io", "filesystem"),
    ("install", "installation", "dependency", "dependencies", "requirements", "package", "pip"),
    ("windows", "win32", "nt", "powershell"),
    ("unicode", "utf", "utf8", "encoding", "decode", "encode"),
    ("print", "output", "stdout", "log", "logging", "console"),
    ("parse", "parser", "parsing", "tokenize", "tokenizer"),
    ("rank", "ranking", "ranked", "score", "scoring", "relevance", "search"),
    ("plan", "planning", "steps", "roadmap", "checklist"),
    ("document", "documentation", "docs", "readme", "typo", "wording"),
    ("build", "compile", "bundle", "package", "dist"),
    ("dependency", "dependencies", "import", "imports", "require"),
)


def split_identifier(word: str) -> list[str]:
    """Split ``loadConfigFile`` / ``load_config_file`` into lowercase words."""
    word = word.strip("`'\"()[]{}.,:;!?<>")
    pieces = re.split(r"[\s_\-./\\]+", word)
    parts: list[str] = []
    for piece in pieces:
        for match in _ACRONYM_RE.finditer(piece):
            parts.append(match.group(0).lower())
    return [part for part in parts if part]


def stem(word: str) -> str:
    """A conservative suffix stripper (not a full Porter stemmer)."""
    if len(word) <= 3 or word.isdigit():
        return word

    def undouble(base: str) -> str:
        if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            return base[:-1]
        return base

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 4:
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if len(base) >= 4:
            return undouble(base)
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if len(base) >= 4:
            return undouble(base)
    return word


def tokens(text: str, *, include_stopwords: bool = False) -> list[str]:
    """Tokenize prose or code into normalized, lightly stemmed terms."""
    output: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        for piece in split_identifier(raw):
            if len(piece) < 2 and not piece.isdigit():
                continue
            if not include_stopwords and piece in STOPWORDS:
                continue
            output.append(stem(piece))
    return output


def unique_tokens(text: str, *, include_stopwords: bool = False) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens(text, include_stopwords=include_stopwords):
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def raw_tokens(text: str, *, include_stopwords: bool = False) -> list[str]:
    """Lowercase surface words (no stemming), useful for human-facing output."""
    output: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        for piece in split_identifier(match.group(0)):
            if len(piece) < 2 and not piece.isdigit():
                continue
            if not include_stopwords and piece in STOPWORDS:
                continue
            output.append(piece.lower())
    return output


def unique_raw_tokens(text: str, *, include_stopwords: bool = False) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in raw_tokens(text, include_stopwords=include_stopwords):
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def path_tokens(path: str) -> list[str]:
    """Tokens from a repository path, with the filename stem repeated once."""
    cleaned = path.replace("\\", "/")
    parts = [part for part in re.split(r"[/._\-]+", cleaned) if part]
    output: list[str] = []
    for part in parts:
        output.extend(split_identifier(part))
    if parts:
        output.extend(split_identifier(parts[-1]))
    seen: set[str] = set()
    result: list[str] = []
    for token in output:
        if len(token) >= 2 and token not in STOPWORDS and token not in seen:
            seen.add(token)
            result.append(stem(token))
    return result


def _group_index() -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        normalized = {stem(item) for item in group}
        for item in normalized:
            index.setdefault(item, set()).update(normalized - {item})
    return index


_SYNONYMS = _group_index()
_SYNONYM_SURFACES: dict[str, list[str]] = {}
for _group in SYNONYM_GROUPS:
    for _item in _group:
        _SYNONYM_SURFACES.setdefault(stem(_item), [])
        for _other in _group:
            if _other != _item and _other not in _SYNONYM_SURFACES[stem(_item)]:
                _SYNONYM_SURFACES[stem(_item)].append(_other)


def expand_term(term: str) -> list[str]:
    """Return synonyms for a term (already stemmed, deduplicated)."""
    return sorted(_SYNONYMS.get(stem(term), set()))


def synonyms_for(term: str) -> list[str]:
    """Return human-readable synonyms for a surface term."""
    return sorted(_SYNONYM_SURFACES.get(stem(term), []))


@dataclass
class TaskSignals:
    identifiers: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "identifiers": self.identifiers,
            "phrases": self.phrases,
            "paths": self.paths,
            "terms": self.terms,
        }


def extract_task_signals(task: str) -> TaskSignals:
    """Pull the parts of a task description that deserve special weight."""
    identifiers: list[str] = []
    phrases: list[str] = []
    paths: list[str] = []

    for match in _BACKTICK_RE.finditer(task):
        span = match.group(1).strip()
        if " " in span or "/" in span:
            if span not in phrases:
                phrases.append(span)
            inner_paths = _PATH_RE.findall(span)
            for inner in inner_paths:
                if inner not in paths:
                    paths.append(inner)
        elif _IDENTIFIER_RE.match(span) and span not in identifiers:
            identifiers.append(span)

    for match in _QUOTE_RE.finditer(task):
        phrase = match.group(1).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    for match in _PATH_RE.finditer(task):
        path = match.group(1)
        if path not in paths:
            paths.append(path)
        stem_name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem_name and stem_name not in identifiers:
            identifiers.append(stem_name)

    for match in _CALL_RE.finditer(task):
        name = match.group(1)
        if name not in identifiers and name.lower() not in STOPWORDS:
            identifiers.append(name)

    for match in _ERROR_RE.finditer(task):
        name = match.group(1)
        if name not in identifiers:
            identifiers.append(name)

    for match in _TOKEN_RE.finditer(task):
        raw = match.group(0)
        if "_" in raw or re.search(r"[a-z][A-Z]", raw):
            if raw not in identifiers and raw.lower() not in STOPWORDS and len(raw) >= 4:
                identifiers.append(raw)

    terms = unique_tokens(task)
    return TaskSignals(identifiers=identifiers, phrases=phrases, paths=paths, terms=terms)


def looks_like_test_path(path: str) -> bool:
    lowered = "/" + path.lower()
    name = path.rsplit("/", 1)[-1].lower()
    if any(
        segment in lowered
        for segment in ("/tests/", "/test/", "/spec/", "/__tests__/", "/testdata/", "/fixtures/", "/e2e/")
    ):
        return True
    if name.startswith(("test_", "tests_")) or name.endswith(("_test.py", "_test.go")):
        return True
    if name in {"conftest.py", "pytest.ini", "tox.ini"}:
        return True
    if re.search(r"\.(test|spec)\.[a-z]+$", name):
        return True
    if re.search(r"_spec\.rb$", name):
        return True
    return False


def looks_like_test_case_path(path: str) -> bool:
    """True only for files a test runner would collect as tests.

    ``tests/support.py`` and ``conftest.py`` are test *support*: useful evidence,
    but "the test to extend" should usually be a file that actually contains
    test cases.
    """
    name = path.rsplit("/", 1)[-1].lower()
    if name.startswith(("test_", "tests_", "spec_")):
        return True
    if name.endswith(("_test.py", "_test.go", "_spec.rb")):
        return True
    return bool(re.search(r"\.(test|spec)\.[a-z]+$", name))


def looks_like_doc_path(path: str) -> bool:
    lowered = path.lower()
    name = lowered.rsplit("/", 1)[-1]
    if name.startswith(("readme", "changelog", "changes", "history", "contributing", "code_of_conduct",
                        "security", "support", "authors", "maintainers")):
        return name.endswith((".md", ".rst", ".txt", ".adoc")) or "." not in name
    if lowered.startswith(("docs/", "doc/", "documentation/")):
        return name.endswith((".md", ".rst", ".txt", ".adoc"))
    return name.endswith((".md", ".rst", ".adoc")) and "/" not in path


def looks_like_config_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    if name in {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "noxfile.py",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "cargo.toml",
        "cargo.lock",
        "go.mod",
        "go.sum",
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
        "tsconfig.json",
        ".pre-commit-config.yaml",
        ".editorconfig",
        "ruff.toml",
        ".ruff.toml",
        "mypy.ini",
        ".flake8",
    }:
        return True
    if name.endswith((".yml", ".yaml")) and ".github/workflows/" in path.replace("\\", "/").lower():
        return True
    if "/.github/" in "/" + path.lower():
        return True
    return False
