"""SQLite-backed repository index and BM25 retrieval primitives."""

from __future__ import annotations

import math
import os
import posixpath
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from . import parsing
from . import tokens as tok
from .models import FileInfo, Symbol
from .scanning import ScanResult, ScannedFile, discover
from .util import detect_language, now_iso, posix, read_text

SCHEMA_VERSION = "1"
INDEX_DIR = ".reposcout"
INDEX_FILE = "index.sqlite3"
FIELD_WEIGHTS = {"body": 1.0, "symbol": 2.2, "path": 1.8}
K1 = 1.2
B = 0.75

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    lang TEXT NOT NULL,
    lines INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    is_test INTEGER NOT NULL DEFAULT 0,
    is_doc INTEGER NOT NULL DEFAULT 0,
    is_config INTEGER NOT NULL DEFAULT 0,
    sha TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    line INTEGER NOT NULL,
    end_line INTEGER,
    signature TEXT NOT NULL DEFAULT '',
    doc TEXT NOT NULL DEFAULT '',
    exported INTEGER NOT NULL DEFAULT 1,
    container TEXT NOT NULL DEFAULT ''
);
CREATE TABLE imports (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    module TEXT NOT NULL,
    line INTEGER NOT NULL,
    raw TEXT NOT NULL DEFAULT '',
    resolved_id INTEGER
);
CREATE TABLE edges (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'import',
    PRIMARY KEY (src, dst, kind)
);
CREATE TABLE terms (
    file_id INTEGER NOT NULL,
    field TEXT NOT NULL,
    term TEXT NOT NULL,
    tf INTEGER NOT NULL,
    PRIMARY KEY (file_id, field, term)
) WITHOUT ROWID;
CREATE TABLE term_df (
    field TEXT NOT NULL,
    term TEXT NOT NULL,
    df INTEGER NOT NULL,
    PRIMARY KEY (field, term)
);
CREATE TABLE file_lens (
    file_id INTEGER NOT NULL,
    field TEXT NOT NULL,
    length INTEGER NOT NULL,
    PRIMARY KEY (file_id, field)
);
CREATE TABLE avgdl (field TEXT PRIMARY KEY, avg REAL NOT NULL);
CREATE INDEX idx_symbols_name ON symbols(name);
CREATE INDEX idx_symbols_file ON symbols(file_id);
CREATE INDEX idx_imports_file ON imports(file_id);
CREATE INDEX idx_imports_resolved ON imports(resolved_id);
CREATE INDEX idx_edges_dst ON edges(dst);
CREATE INDEX idx_edges_src ON edges(src);
CREATE INDEX idx_terms_term ON terms(term);
"""


class IndexUnavailable(RuntimeError):
    pass


@dataclass
class TermHit:
    term: str
    field: str
    tf: int
    contribution: float


@dataclass
class SearchHit:
    file_id: int
    path: str
    score: float
    term_hits: list[TermHit] = field(default_factory=list)

    def matched_terms(self) -> set[str]:
        return {hit.term for hit in self.term_hits}

    def fields_for(self, term: str) -> list[str]:
        return [hit.field for hit in self.term_hits if hit.term == term]


def default_db_path(root: Path) -> Path:
    return root / INDEX_DIR / INDEX_FILE


class Index:
    """Read-only view over a built index."""

    def __init__(self, connection: sqlite3.Connection, root: Path, db_path: Path) -> None:
        self.conn = connection
        self.root = root
        self.db_path = db_path
        self._meta: dict[str, str] | None = None
        self._symbol_by_name: dict[str, list[tuple[int, str, Symbol]]] | None = None

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, db_path: Path, root: Path | None = None) -> "Index":
        if not db_path.exists():
            raise IndexUnavailable(f"No index found at {db_path}")
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        index = cls(connection, root or db_path.parent.parent, db_path)
        version = index.meta("schema_version", "")
        if version != SCHEMA_VERSION:
            index.close()
            raise IndexUnavailable(
                f"Index schema {version or 'unknown'} does not match {SCHEMA_VERSION}; rebuild it."
            )
        return index

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- metadata ----------------------------------------------------------

    def meta(self, key: str, default: str = "") -> str:
        if self._meta is None:
            self._meta = {row["key"]: row["value"] for row in self.conn.execute("SELECT key, value FROM meta")}
        return self._meta.get(key, default)

    def meta_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.meta(key, str(default)))
        except ValueError:
            return default

    @property
    def file_count(self) -> int:
        return self.meta_int("file_count")

    @property
    def indexed_at(self) -> str:
        return self.meta("created_at")

    @property
    def commit(self) -> str:
        return self.meta("git_commit")

    def stats(self) -> dict[str, object]:
        return {
            "root": self.meta("root"),
            "files": self.file_count,
            "symbols": self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
            "imports": self.conn.execute("SELECT COUNT(*) FROM imports").fetchone()[0],
            "terms": self.conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0],
            "created_at": self.indexed_at,
            "commit": self.commit,
            "schema_version": self.meta("schema_version"),
            "db_path": str(self.db_path),
        }

    # -- file access -------------------------------------------------------

    def file_row(self, file_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        if row is None:
            raise KeyError(file_id)
        return row

    def file_id(self, path: str) -> int | None:
        row = self.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        return int(row["id"]) if row else None

    def file_paths(self) -> list[str]:
        return [row["path"] for row in self.conn.execute("SELECT path FROM files ORDER BY path")]

    def files_info(self) -> list[FileInfo]:
        infos: list[FileInfo] = []
        for row in self.conn.execute("SELECT * FROM files ORDER BY path"):
            infos.append(
                FileInfo(
                    path=row["path"],
                    lang=row["lang"],
                    lines=row["lines"],
                    bytes=row["bytes"],
                    is_test=bool(row["is_test"]),
                    is_doc=bool(row["is_doc"]),
                    is_config=bool(row["is_config"]),
                    sha=row["sha"],
                    summary=row["summary"],
                )
            )
        return infos

    def read(self, path: str, limit: int = 500_000) -> str:
        return read_text(self.root / path, limit=limit) or ""

    def symbols_for(self, file_id: int) -> list[Symbol]:
        rows = self.conn.execute(
            "SELECT name, kind, line, end_line, signature, doc, exported, container "
            "FROM symbols WHERE file_id = ? ORDER BY line",
            (file_id,),
        ).fetchall()
        return [
            Symbol(
                name=row["name"],
                kind=row["kind"],
                line=row["line"],
                end_line=row["end_line"],
                signature=row["signature"],
                doc=row["doc"],
                exported=bool(row["exported"]),
                container=row["container"],
            )
            for row in rows
        ]

    def symbols_for_path(self, path: str) -> list[Symbol]:
        file_id = self.file_id(path)
        return self.symbols_for(file_id) if file_id is not None else []

    def imports_for(self, file_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT module, line, raw, resolved_id FROM imports WHERE file_id = ? ORDER BY line",
                (file_id,),
            )
        )

    def imported_paths(self, file_id: int, limit: int = 30) -> list[str]:
        rows = self.conn.execute(
            "SELECT f.path FROM imports i JOIN files f ON f.id = i.resolved_id "
            "WHERE i.file_id = ? AND i.resolved_id IS NOT NULL GROUP BY f.path LIMIT ?",
            (file_id, limit),
        )
        return [row["path"] for row in rows]

    def importers(self, file_id: int, limit: int = 30) -> list[str]:
        rows = self.conn.execute(
            "SELECT f.path FROM edges e JOIN files f ON f.id = e.src "
            "WHERE e.dst = ? GROUP BY f.path LIMIT ?",
            (file_id, limit),
        )
        return [row["path"] for row in rows]

    def neighbor_ids(self, file_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT dst AS id FROM edges WHERE src = ? UNION SELECT src AS id FROM edges WHERE dst = ?",
            (file_id, file_id),
        )
        return [int(row["id"]) for row in rows]

    # -- symbols -----------------------------------------------------------

    def find_symbols(self, names: Iterable[str], limit: int = 30) -> list[tuple[str, Symbol]]:
        wanted = [name.lower() for name in names if name]
        if not wanted:
            return []
        if self._symbol_by_name is None:
            cache: dict[str, list[tuple[int, str, Symbol]]] = defaultdict(list)
            for row in self.conn.execute(
                "SELECT s.*, f.path FROM symbols s JOIN files f ON f.id = s.file_id"
            ):
                symbol = Symbol(
                    name=row["name"],
                    kind=row["kind"],
                    line=row["line"],
                    end_line=row["end_line"],
                    signature=row["signature"],
                    doc=row["doc"],
                    exported=bool(row["exported"]),
                    container=row["container"],
                )
                cache[row["name"].lower()].append((row["file_id"], row["path"], symbol))
            self._symbol_by_name = dict(cache)
        found: list[tuple[str, Symbol]] = []
        seen: set[tuple[str, int, str]] = set()
        for name in wanted:
            for file_id, path, symbol in self._symbol_by_name.get(name, []):
                key = (path, symbol.line, symbol.name)
                if key in seen:
                    continue
                seen.add(key)
                found.append((path, symbol))
                if len(found) >= limit:
                    return found
        return found

    def symbol_context(self, file_id: int, line: int) -> Symbol | None:
        """Innermost symbol whose body contains ``line``."""
        symbols = self.symbols_for(file_id)
        candidates = [
            symbol
            for symbol in symbols
            if symbol.line <= line and (symbol.end_line is None or line <= symbol.end_line)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda symbol: (symbol.line, symbol.end_line or symbol.line))

    # -- retrieval ---------------------------------------------------------

    def search(self, weights: dict[str, float], limit: int = 60) -> list[SearchHit]:
        """Weighted BM25 across the body, symbol and path fields."""
        terms = [term for term, weight in weights.items() if weight > 0]
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        total_docs = max(1, self.file_count)
        averages = {
            row["field"]: float(row["avg"])
            for row in self.conn.execute("SELECT field, avg FROM avgdl")
        }
        dfs = {
            (row["field"], row["term"]): int(row["df"])
            for row in self.conn.execute(
                f"SELECT field, term, df FROM term_df WHERE term IN ({placeholders})", terms
            )
        }
        rows = list(
            self.conn.execute(
                f"SELECT file_id, field, term, tf FROM terms WHERE term IN ({placeholders})", terms
            )
        )
        lengths = {
            (row["file_id"], row["field"]): int(row["length"])
            for row in self.conn.execute(f"SELECT file_id, field, length FROM file_lens")
        }
        scores: dict[int, float] = defaultdict(float)
        hits: dict[int, list[TermHit]] = defaultdict(list)
        for row in rows:
            term = row["term"]
            field = row["field"]
            query_weight = weights.get(term, 0.0)
            if query_weight <= 0:
                continue
            df = dfs.get((field, term), 1)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            average = averages.get(field, 1.0) or 1.0
            doc_length = lengths.get((row["file_id"], field), 0)
            denominator = row["tf"] + K1 * (1 - B + B * doc_length / average)
            if denominator <= 0:
                continue
            contribution = (
                query_weight
                * FIELD_WEIGHTS.get(field, 1.0)
                * idf
                * (row["tf"] * (K1 + 1))
                / denominator
            )
            scores[row["file_id"]] += contribution
            hits[row["file_id"]].append(
                TermHit(term=term, field=field, tf=row["tf"], contribution=contribution)
            )

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        output: list[SearchHit] = []
        for file_id, score in ordered:
            row = self.file_row(file_id)
            output.append(
                SearchHit(
                    file_id=file_id,
                    path=row["path"],
                    score=score,
                    term_hits=sorted(hits[file_id], key=lambda hit: -hit.contribution)[:24],
                )
            )
        return output

    def idf(self, field: str, term: str) -> float:
        row = self.conn.execute(
            "SELECT df FROM term_df WHERE field = ? AND term = ?", (field, term)
        ).fetchone()
        df = int(row["df"]) if row else 0
        total = max(1, self.file_count)
        return math.log(1 + (total - df + 0.5) / (df + 0.5))

    def has_term(self, term: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM term_df WHERE term = ? LIMIT 1", (term,)).fetchone()
        return row is not None

    def role_counts(self) -> dict[str, int]:
        row = self.conn.execute(
            "SELECT SUM(is_test) AS tests, SUM(is_doc) AS docs, SUM(is_config) AS configs FROM files"
        ).fetchone()
        return {
            "tests": int(row["tests"] or 0),
            "docs": int(row["docs"] or 0),
            "configs": int(row["configs"] or 0),
        }


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_index(root: Path, db_path: Path | None = None, *, scanned: ScanResult | None = None) -> Index:
    root = root.resolve()
    db_path = db_path or default_db_path(root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    scan = scanned or discover(root)
    tmp_path = db_path.with_suffix(".building")
    for candidate in (tmp_path, Path(str(tmp_path) + "-wal"), Path(str(tmp_path) + "-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass

    started = now_iso()
    connection = sqlite3.connect(str(tmp_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.executescript(SCHEMA)

    path_to_id: dict[str, int] = {}
    file_rows: list[tuple[object, ...]] = []
    symbol_rows: list[tuple[object, ...]] = []
    import_rows: list[tuple[object, ...]] = []
    term_rows: list[tuple[int, str, str, int]] = []
    pending_imports: list[tuple[int, int, str, int, str]] = []

    file_id = 0
    for item in scan.files:
        text = read_text(item.abs_path) or ""
        if not text:
            continue
        file_id += 1
        path_to_id[item.path] = file_id
        parsed = parsing.parse_file(item.lang, text, item.path)
        summary = parsed.summary or _fallback_summary(text, item.lang)
        file_rows.append(
            (
                file_id,
                item.path,
                item.lang,
                item.lines,
                item.bytes,
                int(item.is_test),
                int(item.is_doc),
                int(item.is_config),
                item.digest,
                summary,
            )
        )
        for symbol in parsed.symbols:
            symbol_rows.append(
                (
                    file_id,
                    symbol.name,
                    symbol.kind,
                    symbol.line,
                    symbol.end_line,
                    symbol.signature,
                    symbol.doc,
                    int(symbol.exported),
                    symbol.container,
                )
            )
        for ref in parsed.imports:
            pending_imports.append((file_id, item.path, ref.module, ref.line, ref.raw))

        body_counts = Counter(tok.tokens(text))
        symbol_text = " ".join(
            f"{symbol.name} {symbol.qualified} {symbol.doc} {symbol.signature}" for symbol in parsed.symbols
        )
        symbol_counts = Counter(tok.tokens(symbol_text))
        path_counts = Counter(tok.path_tokens(item.path))
        for field, counts in (("body", body_counts), ("symbol", symbol_counts), ("path", path_counts)):
            for term, tf in counts.items():
                if len(term) < 2 and not term.isdigit():
                    continue
                term_rows.append((file_id, field, term, tf))

    connection.executemany(
        "INSERT INTO files (id, path, lang, lines, bytes, is_test, is_doc, is_config, sha, summary) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        file_rows,
    )
    connection.executemany(
        "INSERT INTO symbols (file_id, name, kind, line, end_line, signature, doc, exported, container) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        symbol_rows,
    )
    connection.executemany(
        "INSERT INTO terms (file_id, field, term, tf) VALUES (?, ?, ?, ?)", term_rows
    )

    resolver = _ImportResolver(scan.files, path_to_id)
    edges: set[tuple[int, int, str]] = set()
    for src, src_path, module, line, raw in pending_imports:
        resolved_path = resolver.resolve(src_path, module)
        resolved_id = path_to_id.get(resolved_path) if resolved_path else None
        import_rows.append((src, module, line, raw, resolved_id))
        if resolved_id and resolved_id != src:
            edges.add((src, resolved_id, "import"))
    connection.executemany(
        "INSERT INTO imports (file_id, module, line, raw, resolved_id) VALUES (?, ?, ?, ?, ?)",
        import_rows,
    )
    connection.executemany("INSERT INTO edges (src, dst, kind) VALUES (?, ?, ?)", list(edges))

    connection.execute(
        "INSERT INTO term_df (field, term, df) "
        "SELECT field, term, COUNT(DISTINCT file_id) FROM terms GROUP BY field, term"
    )
    connection.execute(
        "INSERT INTO file_lens (file_id, field, length) "
        "SELECT file_id, field, SUM(tf) FROM terms GROUP BY file_id, field "
        "ON CONFLICT(file_id, field) DO UPDATE SET length = excluded.length"
    )
    connection.execute(
        "INSERT INTO avgdl (field, avg) "
        "SELECT field, AVG(length) FROM file_lens GROUP BY field"
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "created_at": started,
        "file_count": str(len(file_rows)),
        "git_commit": scan.profile.commit,
        "git_branch": scan.profile.branch,
        "builder": "reposcout",
    }
    connection.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items()))
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()

    os.replace(tmp_path, db_path)
    index = Index.open(db_path, root)
    return index


def _fallback_summary(text: str, lang: str) -> str:
    for line in text.split("\n")[:15]:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*", "import ", "from ", "use ", "package ")):
            continue
        return stripped[:200]
    return ""


def ensure_index(
    root: Path,
    *,
    refresh: bool = False,
    db_path: Path | None = None,
    scanned: ScanResult | None = None,
) -> tuple[Index, bool]:
    """Open an existing index or build one. Returns ``(index, built)``."""
    target = db_path or default_db_path(root)
    if refresh or not target.exists():
        index = build_index(root, target, scanned=scanned)
        return index, True
    try:
        return Index.open(target, root), False
    except IndexUnavailable:
        index = build_index(root, target, scanned=scanned)
        return index, True


def _module_key(path: str) -> str:
    cleaned = path
    for suffix in (".py", ".pyi"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    if cleaned.endswith("/__init__"):
        cleaned = cleaned[: -len("/__init__")]
    return cleaned.replace("/", ".")


class _ImportResolver:
    """Resolve imports to indexed files using language-specific conventions."""

    JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".json")
    PY_EXTENSIONS = (".py", ".pyi")

    def __init__(self, files: Sequence[ScannedFile], path_to_id: dict[str, int]) -> None:
        self.files = {item.path: item for item in files}
        self.path_to_id = path_to_id
        self.by_path = set(path_to_id)
        self.py_modules: dict[str, str] = {}
        self.by_basename: dict[str, list[str]] = defaultdict(list)
        self.by_dir: dict[str, list[str]] = defaultdict(list)
        for item in files:
            if item.lang in {"python"}:
                self.py_modules.setdefault(_module_key(item.path), item.path)
            self.by_basename[Path(item.path).stem.lower()].append(item.path)
            self.by_dir[posix(Path(item.path).parent)].append(item.path)

    def resolve(self, src_path: str, module: str) -> str | None:
        src = self.files.get(src_path)
        lang = src.lang if src else detect_language(src_path, Path(src_path).name)
        if not module:
            return None
        if lang == "python":
            return self._resolve_python(src_path, module)
        if lang in {"javascript", "typescript", "vue", "svelte"}:
            return self._resolve_js(src_path, module)
        if lang == "go":
            return self._resolve_by_suffix(module, {".go"})
        if lang == "rust":
            return self._resolve_rust(src_path, module)
        if lang in {"java", "kotlin", "scala"}:
            return self._resolve_by_class(module, {".java", ".kt", ".kts", ".scala"})
        if lang == "csharp":
            return self._resolve_by_class(module, {".cs"})
        if lang == "ruby":
            return self._resolve_relative_or_suffix(src_path, module, {".rb"})
        if lang == "php":
            return self._resolve_by_class(module.replace("\\", "/"), {".php"})
        if lang in {"c", "cpp", "objective-c"}:
            return self._resolve_relative_or_suffix(src_path, module, {".h", ".hpp"})
        if lang == "swift":
            return self._resolve_by_class(module, {".swift"})
        return self._resolve_by_class(module, {".py", ".js", ".ts", ".go", ".rs", ".rb"})

    def _resolve_python(self, src_path: str, module: str) -> str | None:
        if module.startswith("."):
            level = len(module) - len(module.lstrip("."))
            remainder = module.lstrip(".")
            base = posix(Path(src_path).parent)
            for _ in range(max(0, level - 1)):
                base = posix(Path(base).parent) if base not in ("", ".") else ""
            prefix = (base + "." if base else "") + remainder
            key = prefix if remainder else base
        else:
            key = module
        key = key.strip(".")
        if key in self.py_modules:
            return self.py_modules[key]
        # ``from package import name`` may target a submodule.
        for candidate in (f"{key}.__init__", f"{key}"):
            if candidate in self.py_modules:
                return self.py_modules[candidate]
        return self._resolve_by_suffix(key.replace(".", "/"), set(self.PY_EXTENSIONS))

    def _resolve_js(self, src_path: str, module: str) -> str | None:
        if not module.startswith((".", "/")):
            if module.startswith("@/"):
                base = "src/" + module[2:]
            else:
                return None
        else:
            base = posix(posixpath.normpath(posixpath.join(posix(Path(src_path).parent), module)))
        if base.startswith("/"):
            base = base.lstrip("/")
        candidates = [base]
        stem, ext = posixpath.splitext(base)
        if ext:
            candidates.extend([stem, stem + "/index"])
        for extension in self.JS_EXTENSIONS:
            candidates.append(base + extension)
            candidates.append(base + "/index" + extension)
        for candidate in candidates:
            if candidate in self.by_path:
                return candidate
        return None

    def _resolve_rust(self, src_path: str, module: str) -> str | None:
        cleaned = module.replace("::", "/")
        base = posix(Path(src_path).parent)
        if cleaned.startswith("crate/"):
            cleaned = cleaned[len("crate/") :]
            base = ""
        elif cleaned.startswith("self/"):
            cleaned = cleaned[len("self/") :]
        elif cleaned.startswith("super/"):
            while cleaned.startswith("super/"):
                cleaned = cleaned[len("super/") :]
                base = posix(Path(base).parent) if base not in ("", ".") else ""
        else:
            return None
        for candidate in (
            posixpath.join(base, cleaned + ".rs"),
            posixpath.join(base, cleaned, "mod.rs"),
            posixpath.join("src", cleaned + ".rs"),
            posixpath.join("src", cleaned, "mod.rs"),
        ):
            if candidate in self.by_path:
                return candidate
        return None

    def _resolve_by_class(self, module: str, extensions: set[str]) -> str | None:
        last = module.replace("\\", "/").split("/")[-1].split(".")[-1]
        for candidate in self.by_basename.get(last.lower(), []):
            if Path(candidate).suffix.lower() in extensions:
                return candidate
        return None

    def _resolve_relative_or_suffix(self, src_path: str, module: str, extensions: set[str]) -> str | None:
        base = posix(Path(src_path).parent)
        candidates = [posixpath.normpath(posixpath.join(base, module))]
        candidates.append(posixpath.normpath(module))
        for candidate in candidates:
            for extension in extensions:
                for option in (candidate + extension, candidate):
                    if option in self.by_path:
                        return option
        suffix = module.replace("\\", "/").lstrip("./")
        for path in self.by_path:
            if Path(path).suffix.lower() not in extensions:
                continue
            if path.endswith(suffix) or path.endswith("/" + suffix):
                return path
        return None

    def _resolve_by_suffix(self, module: str, extensions: set[str]) -> str | None:
        cleaned = module.strip("/").replace(".", "/") if extensions == self.PY_EXTENSIONS else module.strip("/")
        parts = [part for part in cleaned.split("/") if part]
        if not parts:
            return None
        for length in range(min(6, len(parts)), 0, -1):
            suffix = "/".join(parts[-length:])
            for item in self.files.values():
                if Path(item.path).suffix.lower() not in extensions:
                    continue
                directory = posix(Path(item.path).parent)
                if directory.endswith(suffix):
                    return item.path
        return None


def index_overview(index: Index) -> dict[str, object]:
    stats = index.stats()
    rows = list(
        index.conn.execute(
            "SELECT lang, COUNT(*) AS files, SUM(lines) AS lines FROM files "
            "GROUP BY lang ORDER BY lines DESC LIMIT 12"
        )
    )
    stats["languages"] = [
        {"lang": row["lang"], "files": row["files"], "lines": row["lines"]} for row in rows
    ]
    stats["roles"] = index.role_counts()
    return stats
