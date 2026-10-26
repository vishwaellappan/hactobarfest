"""Task-to-code mapping with line-level evidence.

Given a prose task, RepoScout combines:

* weighted BM25 over three fields (file text, symbol names/docs, path tokens);
* hard boosts for identifiers and paths that appear verbatim in the task;
* one hop of the resolved import graph;
* test-file association by module name and import edges.

Every ranked file carries the reasons it surfaced and, where possible, the
exact lines that support those reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from . import tokens as tok
from .indexing import Index, TermHit
from .models import (
    FileHit,
    Mapping,
    QueryTerm,
    Reason,
    Snippet,
    Source,
    Symbol,
)
from .util import clamp, module_label, split_lines, truncate

GENERIC_MODULE_TOKENS = {"test", "tests", "spec", "specs", "src", "lib", "main", "init", "index", "pkg"}
MAX_READ_BYTES = 400_000


@dataclass
class RetrievalOptions:
    max_primary: int = 5
    max_supporting: int = 6
    max_tests: int = 4
    max_docs: int = 4
    expand_graph: bool = True
    read_limit: int = 48


@dataclass
class _Candidate:
    file_id: int
    path: str
    row: object
    score: float = 0.0
    reasons: list[Reason] = field(default_factory=list)
    text: str = ""
    snippets: list[Snippet] = field(default_factory=list)
    term_hits: list[TermHit] = field(default_factory=list)
    graph_from: str | None = None
    relevant: list[Symbol] = field(default_factory=list)

    @property
    def is_test(self) -> bool:
        return bool(self.row["is_test"])

    @property
    def is_doc(self) -> bool:
        return bool(self.row["is_doc"])

    @property
    def is_config(self) -> bool:
        return bool(self.row["is_config"])


def build_query(
    index: Index, task: str, extra_terms: Iterable[str] | None = None
) -> tuple[object, list[QueryTerm], dict[str, float]]:
    """Turn a prose task into weighted, deduplicated index terms."""
    signals = tok.extract_task_signals(task)
    query_terms: list[QueryTerm] = []
    weights: dict[str, float] = {}
    seen_stems: dict[str, str] = {}

    def add(surface: str, weight: float, origin: str, note: str = "") -> None:
        surface = surface.strip()
        if not surface:
            return
        stem = tok.stem(surface.lower())
        if not stem or (len(stem) < 2 and not stem.isdigit()):
            return
        if stem in tok.STOPWORDS:
            return
        if stem in seen_stems:
            return
        seen_stems[stem] = surface
        query_terms.append(QueryTerm(surface, weight, origin, note))
        weights[stem] = weight

    for raw in tok.unique_raw_tokens(task):
        add(raw, 1.0, "task")

    for identifier in signals.identifiers:
        for piece in tok.split_identifier(identifier):
            add(piece, 1.25, "identifier", f"from `{identifier}`")

    expansion_pool: list[tuple[float, str, str]] = []
    for item in list(query_terms):
        for synonym in tok.synonyms_for(item.term):
            stem = tok.stem(synonym)
            if stem in seen_stems or not index.has_term(stem):
                continue
            expansion_pool.append((index.idf("body", stem), synonym, item.term))
    expansion_pool.sort(key=lambda entry: (-entry[0], entry[1]))
    for _, synonym, parent in expansion_pool[:16]:
        add(synonym, 0.45, "expansion", f"synonym of '{parent}' present in this repo")

    for raw in extra_terms or []:
        stem = tok.stem(raw)
        if stem in seen_stems or not index.has_term(stem):
            continue
        add(raw, 0.8, "llm", "model-suggested term; verified to exist in the index")

    return signals, query_terms, weights


def map_task(
    index: Index,
    task: str,
    *,
    options: RetrievalOptions | None = None,
    extra_terms: Iterable[str] | None = None,
) -> Mapping:
    options = options or RetrievalOptions()
    signals, query_terms, weights = build_query(index, task, extra_terms)
    mapping = Mapping(task=task, query_terms=query_terms, identifiers=signals.identifiers, phrases=signals.phrases)
    if not weights:
        mapping.notes.append(
            "The task contains no searchable terms after removing filler words. "
            "Try including a symbol name, filename or error message."
        )
        mapping.confidence_label = "low"
        mapping.confidence_rationale.append("No query terms were available.")
        return mapping

    hits = index.search(weights, limit=120)
    candidates: dict[str, _Candidate] = {}
    for hit in hits:
        row = index.file_row(hit.file_id)
        candidates[hit.path] = _Candidate(
            file_id=hit.file_id, path=hit.path, row=row, score=hit.score, term_hits=hit.term_hits
        )

    _apply_path_boosts(index, signals, candidates, mapping)
    _apply_symbol_boosts(index, signals, candidates)

    for candidate in candidates.values():
        _add_match_reason(candidate, query_terms, weights)

    if options.expand_graph:
        mapping.graph_expansions = _expand_graph(index, candidates, options)

    # Read the most promising files once; snippets and phrase bonuses need text.
    ordered = sorted(candidates.values(), key=lambda c: (-c.score, c.path))
    for candidate in ordered[: options.read_limit]:
        candidate.text = index.read(candidate.path, MAX_READ_BYTES)
        _apply_phrase_boosts(candidate, mapping)

    # Build evidence snippets and relevant symbols for the strongest files.
    for candidate in sorted(candidates.values(), key=lambda c: (-c.score, c.path))[: options.read_limit]:
        if not candidate.text:
            candidate.text = index.read(candidate.path, MAX_READ_BYTES)
        symbols = index.symbols_for(candidate.file_id)
        candidate.snippets = _extract_snippets(candidate, weights, signals.identifiers, signals.phrases, symbols)
        candidate.relevant = _relevant_symbols(
            symbols, weights, signals.identifiers, {snippet.line for snippet in candidate.snippets}
        )
        candidate.reasons.extend(_symbol_reasons(candidate, symbols, weights, signals.identifiers))

    _classify(index, candidates, mapping, options)

    _attach_fan_in(index, mapping)
    _link_tests(index, mapping, weights, signals.identifiers, signals.phrases, options)

    _score_confidence(mapping, weights, query_terms)
    _add_mapping_notes(mapping)
    return mapping


# ---------------------------------------------------------------------------
# Boosts
# ---------------------------------------------------------------------------


def _add_candidate(index: Index, candidates: dict[str, _Candidate], path: str) -> _Candidate | None:
    if path in candidates:
        return candidates[path]
    file_id = index.file_id(path)
    if file_id is None:
        return None
    row = index.file_row(file_id)
    candidate = _Candidate(file_id=file_id, path=path, row=row, score=0.0)
    candidates[path] = candidate
    return candidate


def _apply_path_boosts(
    index: Index, signals: object, candidates: dict[str, _Candidate], mapping: Mapping
) -> None:
    all_paths = index.file_paths()
    for referenced in getattr(signals, "paths", []):
        matches = [path for path in all_paths if path == referenced or path.endswith("/" + referenced)]
        if not matches:
            mapping.notes.append(f"The referenced path `{referenced}` is not in the index.")
            continue
        for path in matches[:3]:
            candidate = _add_candidate(index, candidates, path)
            if candidate is None:
                continue
            candidate.score += 40.0
            candidate.reasons.append(
                Reason(
                    text=f"Task references this path directly (`{referenced}`)",
                    kind="path",
                    strength=40.0,
                    sources=[Source(path=path)],
                )
            )

    identifiers = [name.lower() for name in getattr(signals, "identifiers", [])]
    if not identifiers:
        return
    for candidate in list(candidates.values()):
        filename = Path(candidate.path).name.lower()
        stem = Path(candidate.path).stem.lower()
        if filename in identifiers or stem in identifiers:
            candidate.score += 9.0
            candidate.reasons.append(
                Reason(
                    text=f"Filename matches task identifier `{stem}`",
                    kind="path",
                    strength=9.0,
                    sources=[Source(candidate.path)],
                )
            )


def _apply_symbol_boosts(index: Index, signals: object, candidates: dict[str, _Candidate]) -> None:
    identifiers = {name.lower() for name in getattr(signals, "identifiers", [])}
    if not identifiers:
        return
    found = index.find_symbols([name for name in identifiers], limit=80)
    per_file: dict[str, float] = {}
    for path, symbol in found:
        exact = symbol.name.lower() in identifiers or symbol.qualified.lower() in identifiers
        if not exact:
            continue
        candidate = _add_candidate(index, candidates, path)
        if candidate is None:
            continue
        boost = min(24.0, 13.0 + (symbol.end_line or symbol.line) / 400.0)
        added = per_file.get(path, 0.0)
        if added >= 30:
            continue
        boost = min(boost, 30.0 - added)
        per_file[path] = added + boost
        candidate.score += boost
        candidate.reasons.append(
            Reason(
                text=f"Defines task identifier `{symbol.qualified}` ({symbol.kind})",
                kind="symbol",
                strength=boost,
                sources=[Source(path=path, line=symbol.line, snippet=symbol.signature or symbol.name)],
            )
        )


def _expand_graph(index: Index, candidates: dict[str, _Candidate], options: RetrievalOptions) -> int:
    added = 0
    seeds = [
        candidate
        for candidate in sorted(candidates.values(), key=lambda c: -c.score)
        if candidate.score > 0 and not candidate.is_test and not candidate.is_doc
    ][:6]
    for seed in seeds:
        for neighbor_id in index.neighbor_ids(seed.file_id)[:6]:
            row = index.file_row(neighbor_id)
            if row["is_doc"] or row["is_config"]:
                continue
            path = row["path"]
            bonus = 0.8 + 0.22 * seed.score
            existing = candidates.get(path)
            if existing is None:
                if added >= 10:
                    continue
                existing = _Candidate(file_id=neighbor_id, path=path, row=row, score=bonus, graph_from=seed.path)
                candidates[path] = existing
                added += 1
            elif existing.score < bonus * 0.5:
                existing.score += bonus * 0.35
                existing.graph_from = existing.graph_from or seed.path
            else:
                continue
            existing.reasons.append(
                Reason(
                    text=f"Connected to `{seed.path}` through a resolved import",
                    kind="graph",
                    strength=round(bonus, 2),
                    sources=[Source(path=seed.path)],
                    inferred=True,
                )
            )
    return added


def _apply_phrase_boosts(candidate: _Candidate, mapping: Mapping) -> None:
    if not candidate.text or not mapping.phrases:
        return
    lowered = candidate.text.lower()
    for phrase in mapping.phrases:
        if phrase.lower() not in lowered:
            continue
        candidate.score += 3.0
        line = _line_containing(candidate.text, phrase)
        candidate.reasons.append(
            Reason(
                text=f"Contains quoted phrase \"{truncate(phrase, 80)}\"",
                kind="match",
                strength=3.0,
                sources=[Source(path=candidate.path, line=line, snippet=truncate(phrase, 120))],
            )
        )


def _line_containing(text: str, phrase: str) -> int | None:
    lowered = phrase.lower()
    for index, line in enumerate(split_lines(text), start=1):
        if lowered in line.lower():
            return index
    return None


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _add_match_reason(candidate: _Candidate, query_terms: Sequence[QueryTerm], weights: dict[str, float]) -> None:
    if not candidate.term_hits:
        return
    fields: dict[str, list[str]] = {}
    for hit in candidate.term_hits:
        surface = _surface_for(hit.term, query_terms)
        fields.setdefault(hit.field, []).append(surface)
    labels = {"body": "file text", "symbol": "symbols", "path": "path"}

    def unique(names: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered[:5]

    pieces = [f"{labels.get(field, field)}: {', '.join(unique(names))}" for field, names in fields.items()]
    candidate.reasons.append(
        Reason(
            text="Matched " + "; ".join(pieces),
            kind="match",
            strength=min(6.0, candidate.score / 6),
        )
    )


def _surface_for(term: str, query_terms: Sequence[QueryTerm]) -> str:
    for query_term in query_terms:
        if tok.stem(query_term.term) == term:
            return query_term.term
    return term


def _relevant_symbols(
    symbols: Sequence[Symbol], weights: dict[str, float], identifiers: Sequence[str], snippet_lines: set[int]
) -> list[Symbol]:
    identifier_set = {name.lower() for name in identifiers}
    scored: list[tuple[float, Symbol]] = []
    for symbol in symbols:
        name_tokens = set(tok.tokens(symbol.name + " " + symbol.qualified))
        overlap = len(name_tokens & set(weights))
        exact = symbol.name.lower() in identifier_set or symbol.qualified.lower() in identifier_set
        near = any(abs(symbol.line - line) <= 2 for line in snippet_lines)
        doc_overlap = len(set(tok.tokens(symbol.doc)) & set(weights))
        score = overlap * 2.5 + doc_overlap * 0.8 + (6.0 if exact else 0.0) + (1.0 if near else 0.0)
        if symbol.kind in {"h1", "h2", "h3", "section"}:
            score += 0.5
        scored.append((score, symbol))
    scored.sort(key=lambda pair: (-pair[0], pair[1].line))
    selected = [symbol for score, symbol in scored if score > 0][:6]
    if not selected:
        selected = sorted(symbols, key=lambda symbol: symbol.line)[:4]
    return sorted(selected, key=lambda symbol: symbol.line)


def _extract_snippets(
    candidate: _Candidate,
    weights: dict[str, float],
    identifiers: Sequence[str],
    phrases: Sequence[str],
    symbols: Sequence[Symbol],
) -> list[Snippet]:
    if not candidate.text:
        return []
    terms = set(weights)
    identifier_set = [name.lower() for name in identifiers]
    symbol_at = {symbol.line: symbol for symbol in symbols}
    lines = split_lines(candidate.text)
    scored: list[tuple[float, int, set[str], str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 1200:
            continue
        line_tokens = set(tok.tokens(line))
        matched = line_tokens & terms
        lowers = stripped.lower()
        phrase_hit = any(phrase.lower() in lowers for phrase in phrases)
        ident_hit = any(name in lowers for name in identifier_set)
        if not matched and not phrase_hit and not ident_hit:
            continue
        score = 2.0 * len(matched)
        if index + 1 in symbol_at:
            score += 2.5
        if phrase_hit:
            score += 3.0
        if ident_hit:
            score += 2.0
        if stripped.startswith(("#", "//", "/*", "*")):
            score -= 0.4
        scored.append((score, index, matched, stripped))

    scored.sort(key=lambda item: (-item[0], item[1]))
    snippets: list[Snippet] = []
    used_text: set[str] = set()
    covered: set[str] = set()
    for _, index, matched, stripped in scored:
        if len(snippets) >= 4:
            break
        if stripped in used_text:
            continue
        used_text.add(stripped)
        covered |= matched
        symbol = _symbol_containing(symbols, index + 1)
        snippets.append(
            Snippet(
                line=index + 1,
                text=truncate(stripped, 220),
                terms=sorted(matched)[:8],
                symbol=symbol.qualified if symbol else None,
            )
        )

    # Guarantee at least a little coverage of every meaningful query term.
    for term in sorted(terms):
        if term in covered or len(snippets) >= 4:
            continue
        for index, line in enumerate(lines):
            if term in set(tok.tokens(line)):
                snippets.append(
                    Snippet(
                        line=index + 1,
                        text=truncate(line.strip(), 220),
                        terms=[term],
                        symbol=(_symbol_containing(symbols, index + 1).qualified if _symbol_containing(symbols, index + 1) else None),
                    )
                )
                covered.add(term)
                break
    return sorted(snippets, key=lambda snippet: snippet.line)


def _symbol_containing(symbols: Sequence[Symbol], line: int) -> Symbol | None:
    matches = [
        symbol
        for symbol in symbols
        if symbol.line <= line and (symbol.end_line is None or line <= symbol.end_line)
    ]
    if not matches:
        return None
    return max(matches, key=lambda symbol: (symbol.line, symbol.end_line or symbol.line))


def _symbol_reasons(
    candidate: _Candidate, symbols: Sequence[Symbol], weights: dict[str, float], identifiers: Sequence[str]
) -> list[Reason]:
    reasons: list[Reason] = []
    identifier_set = {name.lower() for name in identifiers}
    for symbol in symbols:
        exact = symbol.name.lower() in identifier_set or symbol.qualified.lower() in identifier_set
        tokens = set(tok.tokens(symbol.name + " " + symbol.qualified)) & set(weights)
        if not exact and not tokens:
            continue
        if candidate.is_doc and symbol.kind not in {"h1", "h2", "h3", "section"}:
            continue
        reasons.append(
            Reason(
                text=f"Defines `{symbol.qualified}` ({symbol.kind}) at line {symbol.line}",
                kind="symbol",
                strength=3.0 if exact else 1.0,
                sources=[Source(candidate.path, symbol.line, symbol.signature or symbol.name)],
            )
        )
        if len(reasons) >= 4:
            break
    return reasons


# ---------------------------------------------------------------------------
# Classification and confidence
# ---------------------------------------------------------------------------


def _classify(
    index: Index,
    candidates: dict[str, _Candidate],
    mapping: Mapping,
    options: RetrievalOptions,
) -> None:
    ordered = sorted(candidates.values(), key=lambda c: (-c.score, c.path))
    threshold = max(0.9, (ordered[0].score * 0.06) if ordered else 0)

    code = [c for c in ordered if not c.is_test and not c.is_doc and not c.is_config and c.score >= threshold]
    docs = [c for c in ordered if c.is_doc and c.score >= threshold]
    configs = [c for c in ordered if c.is_config and c.score >= threshold]
    tests = [c for c in ordered if c.is_test and c.score >= threshold]

    # Documentation- or configuration-only tasks still deserve a target.
    if not code and docs:
        code = docs[:1]
    if not code and configs:
        code = configs[:1]

    selected = code[: options.max_primary]
    mapping.primaries = [
        _to_hit(hit, "doc" if hit.is_doc else "config" if hit.is_config else "primary", index)
        for hit in selected
    ]
    primary_paths = {hit.path for hit in mapping.primaries}

    remaining_code = [c for c in code[options.max_primary:] if c.score > threshold]
    supporting = list(remaining_code)
    for candidate in [*configs, *ordered]:
        if len(supporting) >= options.max_supporting:
            break
        if candidate.path in primary_paths or any(c.path == candidate.path for c in supporting):
            continue
        if candidate.is_test or candidate.is_doc:
            continue
        if candidate.score <= threshold:
            continue
        supporting.append(candidate)
    mapping.supporting = [_to_hit(c, "supporting", index) for c in supporting]

    mapping.tests = [_to_hit(c, "test", index) for c in tests[: options.max_tests]]
    seen = primary_paths | {hit.path for hit in mapping.tests} | {hit.path for hit in mapping.supporting}
    mapping.docs = [
        _to_hit(c, "doc", index) for c in docs if c.path not in seen
    ][: options.max_docs]


def _to_hit(candidate: _Candidate, role: str, index: Index) -> FileHit:
    relevant = candidate.relevant or index.symbols_for(candidate.file_id)[:4]
    return FileHit(
        path=candidate.path,
        role=role,  # type: ignore[arg-type]
        score=round(candidate.score, 3),
        lang=candidate.row["lang"],
        lines=candidate.row["lines"],
        reasons=candidate.reasons,
        snippets=candidate.snippets,
        symbols=relevant,
        imports=index.imported_paths(candidate.file_id, limit=8),
        imported_by=[],
        module=module_label(candidate.path),
    )


def _attach_fan_in(index: Index, mapping: Mapping) -> None:
    for hit in mapping.primaries + mapping.supporting:
        importers = index.importers(index.file_id(hit.path) or -1, limit=20)
        hit.imported_by = importers[:8]
        if len(importers) >= 2:
            hit.reasons.append(
                Reason(
                    text=f"Imported by {len(importers)} indexed file(s), including {', '.join(importers[:3])}",
                    kind="graph",
                    strength=min(4.0, len(importers) * 0.5),
                    sources=[Source(hit.path)],
                )
            )


def _link_tests(
    index: Index,
    mapping: Mapping,
    weights: dict[str, float],
    identifiers: Sequence[str],
    phrases: Sequence[str],
    options: RetrievalOptions,
) -> None:
    test_rows = list(index.conn.execute("SELECT id, path FROM files WHERE is_test = 1 ORDER BY path"))
    if not test_rows:
        return
    existing = {hit.path for hit in mapping.tests}
    for primary in list(mapping.primaries):
        if primary.role not in {"primary"}:
            continue
        primary_tokens = set(tok.path_tokens(primary.path)) - GENERIC_MODULE_TOKENS
        if not primary_tokens:
            continue
        importers = index.importers(index.file_id(primary.path) or -1, limit=30)
        importer_set = set(importers)
        for row in test_rows:
            path = row["path"]
            test_tokens = set(tok.path_tokens(path)) - {"tests", "test", "spec", "specs"}
            shared = primary_tokens & test_tokens
            imports_primary = path in importer_set
            if not shared and not imports_primary:
                continue
            if len(mapping.tests) >= options.max_tests and path not in existing:
                continue
            hit = next((item for item in mapping.tests if item.path == path), None)
            if hit is None:
                text = index.read(path, MAX_READ_BYTES)
                line = _first_import_line(index, path, primary.path)
                lines = split_lines(text)
                line_snippet = (
                    truncate(lines[line - 1].strip(), 160) if line and 1 <= line <= len(lines) else None
                )
                reason = Reason(
                    text=(
                        f"Imports the target module `{primary.module}`"
                        if imports_primary
                        else f"Test filename mirrors the target module ({', '.join(sorted(shared))})"
                    ),
                    kind="test-link",
                    strength=5.0 if imports_primary else 2.5,
                    sources=[Source(path, line, line_snippet)],
                    inferred=not imports_primary,
                )
                artifact = _Candidate(
                    file_id=row["id"], path=path, row=index.file_row(row["id"]), score=primary.score * 0.25 + 2
                )
                artifact.text = text
                artifact.snippets = _extract_snippets(
                    artifact, weights, [*identifiers, Path(primary.path).stem], phrases, index.symbols_for(row["id"])
                )[:3]
                artifact.reasons = [reason]
                hit = _to_hit(artifact, "test", index)
                mapping.tests.append(hit)
                existing.add(path)
            else:
                hit.reasons.append(
                    Reason(
                        text=f"Covers target module `{primary.module}`",
                        kind="test-link",
                        strength=2.0,
                        sources=[Source(path)],
                        inferred=True,
                    )
                )
            primary.reasons.append(
                Reason(
                    text=f"Existing test file: `{path}`",
                    kind="test-link",
                    strength=2.0,
                    sources=[Source(path)],
                )
            )
        if len(mapping.tests) >= options.max_tests:
            break


def _first_import_line(index: Index, test_path: str, target_path: str) -> int | None:
    test_id = index.file_id(test_path)
    target_id = index.file_id(target_path)
    if test_id is None or target_id is None:
        return None
    for row in index.imports_for(test_id):
        if row["resolved_id"] == target_id:
            return int(row["line"])
    return None


def _score_confidence(mapping: Mapping, weights: dict[str, float], query_terms: Sequence[QueryTerm]) -> None:
    if not mapping.primaries:
        mapping.confidence = 0.0
        mapping.confidence_label = "low"
        mapping.confidence_rationale.append("No source file accumulated enough evidence to be a target.")
        return
    top = mapping.primaries[0]
    important = {tok.stem(term.term) for term in query_terms if term.weight >= 0.9}
    covered: set[str] = set()
    for snippet in top.snippets:
        covered |= {tok.stem(term) for term in snippet.terms}
    covered |= {tok.stem(symbol.name) for symbol in top.symbols}
    coverage = len(important & covered) / max(1, len(important))

    second = mapping.primaries[1].score if len(mapping.primaries) > 1 else 0.0
    separation = clamp((top.score / max(1.0, second) - 1) / 2) if second else 1.0
    normalized = clamp(top.score / 30)
    confidence = 0.45 * normalized + 0.40 * coverage + 0.15 * separation

    explicit = any(
        reason.kind in {"path", "symbol"} and reason.strength >= 9.0
        for reason in top.reasons
    )
    if explicit:
        confidence = max(confidence, 0.68)
    if mapping.graph_expansions and not top.snippets and not explicit:
        confidence *= 0.7
    mapping.confidence = round(clamp(confidence), 3)
    mapping.confidence_label = "high" if mapping.confidence >= 0.7 else "medium" if mapping.confidence >= 0.45 else "low"
    mapping.confidence_rationale.extend(
        [
            f"Top candidate scored {top.score:.1f} (weighted BM25).",
            f"{len(important & covered)}/{len(important)} high-weight task terms appear in the top candidate's evidence.",
            "Clear separation from the runner-up."
            if separation >= 0.6
            else "Runner-up is close; compare candidates before editing.",
        ]
    )
    if explicit:
        mapping.confidence_rationale.append("The task names a path or symbol that exists in this repository.")


def _add_mapping_notes(mapping: Mapping) -> None:
    mapping.notes.append(
        "Ranking is lexical (BM25 over text, symbols and paths), not semantic. "
        "Tasks that paraphrase code vocabulary may rank poorly; adding an error message, symbol or filename helps."
    )
    if not mapping.identifiers:
        mapping.notes.append(
            "No explicit symbol or filename was found in the task text, so no identifier-level boost was applied."
        )
    if mapping.graph_expansions:
        mapping.notes.append(
            f"Added {mapping.graph_expansions} file(s) through one hop of the resolved import graph; those links are inferred, not declared by the task."
        )
    if mapping.confidence_label == "low":
        mapping.notes.append(
            "Low confidence: treat the ordering below as a shortlist to investigate, not as an answer."
        )


__all__ = [
    "RetrievalOptions",
    "build_query",
    "map_task",
]
