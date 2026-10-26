"""Typed data model shared by every stage of RepoScout.

The central design rule of this project is that facts and guesses never share
a shelf.  A :class:`Claim` is either ``verified`` (backed by at least one
:class:`Source` pointing at a file/line) or ``suggested`` (backed by an
explicit rationale in ``basis``).  Constructing an unsupported claim raises,
so the distinction cannot silently rot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["verified", "suggested"]
Role = Literal["primary", "supporting", "test", "doc", "config"]


@dataclass(frozen=True)
class Source:
    """A concrete pointer into the repository."""

    path: str
    line: int | None = None
    snippet: str | None = None

    def label(self) -> str:
        if self.line is None:
            return self.path
        return f"{self.path}:{self.line}"


@dataclass
class Claim:
    """A statement that is explicitly either grounded or speculative."""

    text: str
    status: Status
    sources: list[Source] = field(default_factory=list)
    basis: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("verified", "suggested"):
            raise ValueError(f"unknown claim status: {self.status!r}")
        if self.status == "verified" and not self.sources:
            raise ValueError(f"verified claim requires a source: {self.text!r}")
        if self.status == "suggested" and not self.basis:
            raise ValueError(f"suggested claim requires a basis: {self.text!r}")


@dataclass
class Check:
    """A command a contributor could run, with provenance."""

    id: str
    command: str
    kind: str  # test | lint | typecheck | format | build | docs | security
    status: Status
    sources: list[Source] = field(default_factory=list)
    basis: str = ""
    scope: list[str] = field(default_factory=list)
    declared_in: str | None = None
    runnable: bool = False
    tool_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status == "verified" and not self.sources:
            raise ValueError(f"verified check requires a source: {self.command!r}")
        if self.status == "suggested" and not self.basis:
            raise ValueError(f"suggested check requires a basis: {self.command!r}")


@dataclass
class Symbol:
    name: str
    kind: str
    line: int
    end_line: int | None = None
    signature: str = ""
    doc: str = ""
    exported: bool = True
    container: str = ""

    @property
    def qualified(self) -> str:
        return f"{self.container}.{self.name}" if self.container else self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "line": self.line,
            "end_line": self.end_line,
            "signature": self.signature,
            "doc": self.doc,
            "exported": self.exported,
            "container": self.container,
            "qualified": self.qualified,
        }


@dataclass
class ImportRef:
    module: str
    line: int
    raw: str
    resolved: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "line": self.line,
            "raw": self.raw,
            "resolved": self.resolved,
        }


@dataclass
class FileInfo:
    """One indexed file and everything we observed about it."""

    path: str
    lang: str
    lines: int
    bytes: int
    is_test: bool = False
    is_doc: bool = False
    is_config: bool = False
    sha: str = ""
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[ImportRef] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "lang": self.lang,
            "lines": self.lines,
            "bytes": self.bytes,
            "is_test": self.is_test,
            "is_doc": self.is_doc,
            "is_config": self.is_config,
            "sha": self.sha,
            "summary": self.summary,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": [i.to_dict() for i in self.imports],
        }


@dataclass
class Manifest:
    path: str
    kind: str
    name: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
        }


@dataclass
class RepoProfile:
    root: str
    name: str
    files_indexed: int = 0
    total_lines: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    manifests: list[Manifest] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    guidelines: list[str] = field(default_factory=list)
    issue_templates: list[str] = field(default_factory=list)
    ci_workflows: list[str] = field(default_factory=list)
    test_files: int = 0
    entry_points: list[str] = field(default_factory=list)
    package_name: str = ""
    description: str = ""
    license: str = ""
    is_git: bool = False
    branch: str = ""
    commit: str = ""
    remote: str = ""
    top_level: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()}
        data["manifests"] = [m.to_dict() for m in self.manifests]
        return data


@dataclass
class QueryTerm:
    term: str
    weight: float = 1.0
    origin: str = "task"  # task | expansion | identifier | llm
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "weight": self.weight,
            "origin": self.origin,
            "note": self.note,
        }


@dataclass
class Snippet:
    line: int
    text: str
    terms: list[str] = field(default_factory=list)
    symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "text": self.text,
            "terms": self.terms,
            "symbol": self.symbol,
        }


@dataclass
class Reason:
    text: str
    kind: str  # match | symbol | graph | role | test-link | path
    strength: float = 0.0
    sources: list[Source] = field(default_factory=list)
    inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "strength": round(self.strength, 4),
            "sources": [s.__dict__ for s in self.sources],
            "inferred": self.inferred,
        }


@dataclass
class FileHit:
    path: str
    role: Role
    score: float
    lang: str = ""
    lines: int = 0
    reasons: list[Reason] = field(default_factory=list)
    snippets: list[Snippet] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    module: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "score": round(self.score, 3),
            "lang": self.lang,
            "lines": self.lines,
            "reasons": [r.to_dict() for r in self.reasons],
            "snippets": [s.to_dict() for s in self.snippets],
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": self.imports,
            "imported_by": self.imported_by,
            "module": self.module,
        }


@dataclass
class Mapping:
    task: str
    primaries: list[FileHit] = field(default_factory=list)
    supporting: list[FileHit] = field(default_factory=list)
    tests: list[FileHit] = field(default_factory=list)
    docs: list[FileHit] = field(default_factory=list)
    query_terms: list[QueryTerm] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str = "low"
    confidence_rationale: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    graph_expansions: int = 0

    def all_hits(self) -> list[FileHit]:
        seen: set[str] = set()
        ordered: list[FileHit] = []
        for hit in [*self.primaries, *self.supporting, *self.tests, *self.docs]:
            if hit.path not in seen:
                seen.add(hit.path)
                ordered.append(hit)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "primaries": [h.to_dict() for h in self.primaries],
            "supporting": [h.to_dict() for h in self.supporting],
            "tests": [h.to_dict() for h in self.tests],
            "docs": [h.to_dict() for h in self.docs],
            "query_terms": [t.to_dict() for t in self.query_terms],
            "identifiers": self.identifiers,
            "phrases": self.phrases,
            "confidence": round(self.confidence, 3),
            "confidence_label": self.confidence_label,
            "confidence_rationale": self.confidence_rationale,
            "notes": self.notes,
            "graph_expansions": self.graph_expansions,
        }


@dataclass
class Action:
    kind: str  # read | run | edit | inspect | ask
    detail: str
    status: Status = "suggested"
    sources: list[Source] = field(default_factory=list)
    basis: str | None = None
    command: str | None = None
    check_id: str | None = None

    def __post_init__(self) -> None:
        if self.status == "verified" and not self.sources:
            raise ValueError(f"verified action requires a source: {self.detail!r}")
        if self.status == "suggested" and not self.basis:
            raise ValueError(f"suggested action requires a basis: {self.detail!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "status": self.status,
            "sources": [s.__dict__ for s in self.sources],
            "basis": self.basis,
            "command": self.command,
            "check_id": self.check_id,
        }


@dataclass
class PlanStep:
    number: int
    title: str
    goal: str
    duration_minutes: int = 15
    actions: list[Action] = field(default_factory=list)
    rationale: list[Claim] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "goal": self.goal,
            "duration_minutes": self.duration_minutes,
            "actions": [a.to_dict() for a in self.actions],
            "rationale": [
                {
                    "text": c.text,
                    "status": c.status,
                    "sources": [s.__dict__ for s in c.sources],
                    "basis": c.basis,
                }
                for c in self.rationale
            ],
            "risks": self.risks,
        }


@dataclass
class ContributionPlan:
    task: str
    difficulty_score: int = 0
    difficulty_label: str = "moderate"
    estimated_minutes: tuple[int, int] = (30, 120)
    summary: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    checklist: list[Claim] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    files_to_touch: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "difficulty_score": self.difficulty_score,
            "difficulty_label": self.difficulty_label,
            "estimated_minutes": list(self.estimated_minutes),
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "checklist": [
                {
                    "text": c.text,
                    "status": c.status,
                    "sources": [s.__dict__ for s in c.sources],
                    "basis": c.basis,
                }
                for c in self.checklist
            ],
            "unknowns": self.unknowns,
            "files_to_touch": self.files_to_touch,
            "test_files": self.test_files,
        }


@dataclass
class TraceStep:
    index: int
    tool: str
    input: str
    output: str
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "input": self.input,
            "output": self.output,
            "duration_ms": self.duration_ms,
        }


@dataclass
class CheckResult:
    """A command that was actually executed, as opposed to just declared."""

    check_id: str
    command: str
    returncode: int
    duration_ms: int = 0
    timed_out: bool = False
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "command": self.command,
            "returncode": self.returncode,
            "ok": self.ok,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "output": self.output,
        }


@dataclass
class Report:
    task: str
    profile: RepoProfile
    mapping: Mapping
    checks: list[Check]
    plan: ContributionPlan
    trace: list[TraceStep] = field(default_factory=list)
    check_results: list[CheckResult] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    llm_used: bool = False
    llm_notes: list[str] = field(default_factory=list)
    generated_at: str = ""
    version: str = ""
    repo_root: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "generated_at": self.generated_at,
            "version": self.version,
            "repo_root": self.repo_root,
            "profile": self.profile.to_dict(),
            "mapping": self.mapping.to_dict(),
            "checks": [
                {
                    "id": c.id,
                    "command": c.command,
                    "kind": c.kind,
                    "status": c.status,
                    "sources": [s.__dict__ for s in c.sources],
                    "basis": c.basis,
                    "scope": c.scope,
                    "declared_in": c.declared_in,
                    "runnable": c.runnable,
                    "tool_notes": c.tool_notes,
                }
                for c in self.checks
            ],
            "plan": self.plan.to_dict(),
            "trace": [t.to_dict() for t in self.trace],
            "check_results": [r.to_dict() for r in self.check_results],
            "limitations": self.limitations,
            "llm_used": self.llm_used,
            "llm_notes": self.llm_notes,
        }
