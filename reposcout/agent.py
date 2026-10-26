"""The agent loop: scan, retrieve, inspect, verify, plan.

The loop is deterministic by default; an optional LLM (any OpenAI-compatible
chat endpoint) can propose extra search terms and a short grounded narrative.
Model output is accepted only when it points at retrieved evidence.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .checks import CheckRun, CheckSet, discover_checks, run_check
from .indexing import Index, ensure_index
from .models import CheckResult, Report, TraceStep
from .planning import build_plan
from .retrieval import RetrievalOptions, map_task
from .scanning import ScanResult, detect_repo_root, discover
from .tokens import looks_like_test_path
from .util import now_iso

MAX_NARRATIVE = 1500


@dataclass
class AgentOptions:
    refresh: bool = False
    use_llm: bool = False
    expand_graph: bool = True
    db_path: Path | None = None
    max_primary: int = 5
    run_checks: bool = False
    timeout: int = 180
    llm_timeout: int = 45


@dataclass
class _Timer:
    started: float = field(default_factory=time.perf_counter)

    def ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


class LLMClient:
    """Minimal OpenAI-compatible client; no third-party dependencies."""

    def __init__(self, timeout: int = 45) -> None:
        self.base_url = (os.environ.get("REPOSCOUT_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("REPOSCOUT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.model = os.environ.get("REPOSCOUT_LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("no choices in LLM response")
        return str(choices[0].get("message", {}).get("content", ""))


@dataclass
class AgentRun:
    report: Report
    index: Index
    checks: CheckSet
    scan: ScanResult
    check_runs: list[CheckRun] = field(default_factory=list)

    def close(self) -> None:
        self.index.close()


class RepoAgent:
    def __init__(self, root: Path, options: AgentOptions | None = None) -> None:
        self.root = detect_repo_root(root)
        self.options = options or AgentOptions()
        self.trace: list[TraceStep] = []
        self.llm = LLMClient(timeout=self.options.llm_timeout) if self.options.use_llm else None
        self.llm_notes: list[str] = []

    def _log(self, tool: str, tool_input: str, output: str, timer: _Timer) -> None:
        self.trace.append(
            TraceStep(
                index=len(self.trace) + 1,
                tool=tool,
                input=tool_input,
                output=output,
                duration_ms=timer.ms(),
            )
        )

    def run(self, task: str, *, run_checks: bool = False) -> AgentRun:
        task = " ".join(task.split())
        if not task:
            raise ValueError("task description is empty")

        timer = _Timer()
        scan = discover(self.root)
        index, built = ensure_index(
            self.root, refresh=self.options.refresh, db_path=self.options.db_path, scanned=scan
        )
        stats = index.stats()
        self._log(
            "index",
            f"root={self.root}",
            f"{'built' if built else 'loaded'} {stats['files']} files, {stats['symbols']} symbols, "
            f"{stats['imports']} imports ({stats['created_at']})",
            timer,
        )

        timer = _Timer()
        mapping = map_task(
            index,
            task,
            options=RetrievalOptions(
                max_primary=self.options.max_primary, expand_graph=self.options.expand_graph
            ),
        )
        top = ", ".join(f"{hit.path} ({hit.score:.1f})" for hit in mapping.primaries[:3]) or "none"
        self._log(
            "map_task",
            task,
            f"confidence={mapping.confidence_label} ({mapping.confidence:.2f}); top: {top}",
            timer,
        )

        if self.llm and self.llm.available:
            timer = _Timer()
            try:
                terms, narrative = self._llm_pass(task, index, mapping)
                if terms:
                    mapping = map_task(
                        index,
                        task,
                        options=RetrievalOptions(
                            max_primary=self.options.max_primary, expand_graph=self.options.expand_graph
                        ),
                        extra_terms=terms,
                    )
                    mapping.notes.append(
                        "An optional LLM proposed extra search terms; only terms already present in the index were used."
                    )
                if narrative:
                    self.llm_notes.append(narrative)
                self._log(
                    "llm",
                    f"model={self.llm.model}",
                    f"{len(terms)} grounded term(s), narrative={'yes' if narrative else 'no'}",
                    timer,
                )
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
                self.llm_notes.append(f"LLM pass skipped: {type(exc).__name__}: {exc}")
                self._log("llm", f"model={self.llm.model}", f"skipped ({type(exc).__name__})", timer)
        elif self.options.use_llm:
            self.llm_notes.append(
                "LLM requested but no API key was found (set REPOSCOUT_LLM_API_KEY or OPENAI_API_KEY). "
                "The deterministic pipeline produced this report."
            )
            self._log("llm", "api key check", "skipped: no credentials", _Timer())

        timer = _Timer()
        paths = [item.path for item in scan.files]
        check_set = discover_checks(self.root, scan.profile, paths)
        self._log(
            "discover_checks",
            f"{len(paths)} files",
            f"{len(check_set.verified)} verified, {len(check_set.suggested)} suggested, "
            f"{sum(1 for check in check_set.checks if check.runnable)} runnable here",
            timer,
        )

        check_runs: list[CheckRun] = []
        if run_checks:
            timer = _Timer()
            for check in check_set.verified:
                if not check.runnable:
                    continue
                try:
                    check_runs.append(run_check(check, self.root, timeout=self.options.timeout))
                except (ValueError, OSError) as exc:
                    check.tool_notes.append(f"execution failed: {exc}")
            passed = sum(1 for run in check_runs if run.ok)
            self._log("run_checks", "all verified + runnable", f"{passed}/{len(check_runs)} passed", timer)

        timer = _Timer()
        plan = build_plan(
            task,
            scan.profile,
            mapping,
            check_set,
            test_paths=[path for path in paths if looks_like_test_path(path)],
            root=self.root,
        )
        self._log(
            "build_plan",
            f"{len(mapping.primaries)} primaries",
            f"{len(plan.steps)} steps, difficulty={plan.difficulty_label} ({plan.difficulty_score}), "
            f"{plan.estimated_minutes[0]}-{plan.estimated_minutes[1]} min",
            timer,
        )

        report = Report(
            task=task,
            profile=scan.profile,
            mapping=mapping,
            checks=check_set.checks,
            plan=plan,
            trace=self.trace,
            check_results=[
                CheckResult(
                    check_id=run.check_id,
                    command=run.command,
                    returncode=run.returncode,
                    duration_ms=run.duration_ms,
                    timed_out=run.timed_out,
                    output=run.output,
                )
                for run in check_runs
            ],
            llm_used=bool(self.llm and self.llm.available),
            llm_notes=self.llm_notes,
            generated_at=now_iso(),
            version=__version__,
            repo_root=str(self.root),
        )
        self._self_check(report, index.indexed_at)
        return AgentRun(report=report, index=index, checks=check_set, scan=scan, check_runs=check_runs)

    # -- LLM grounding -----------------------------------------------------

    def _llm_pass(self, task: str, index: Index, mapping: object):
        assert self.llm is not None
        evidence = _evidence_pack(mapping)
        system = (
            "You are RepoScout, a repository analysis assistant. You receive retrieved evidence only. "
            "Never invent file paths, symbols or commands. Return strict JSON with keys "
            "'terms' (array of at most 6 additional search words that are likely to appear verbatim in this repository) "
            "and 'narrative' (at most 4 sentences; every claim must cite [path:line] taken from the evidence). "
            "If the evidence is insufficient, say so in the narrative."
        )
        user = f"TASK:\n{task}\n\nEVIDENCE:\n{evidence}"
        raw = self.llm.complete(system, user)
        data = _parse_json_object(raw)
        raw_terms = data.get("terms", [])
        terms = [str(term) for term in raw_terms if isinstance(term, str)][:6] if isinstance(raw_terms, list) else []
        # Grounding check: a term is usable only if it occurs in the index.
        usable: list[str] = []
        from .tokens import stem as _stem

        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            if index.has_term(_stem(cleaned.lower())) or index.has_term(cleaned.lower()):
                usable.append(cleaned)
        narrative = data.get("narrative")
        narrative_text = str(narrative).strip() if isinstance(narrative, str) else ""
        if narrative_text:
            narrative_text = _validate_narrative(narrative_text, mapping)
        return usable, narrative_text

    def _self_check(self, report: Report, indexed_at: str) -> None:
        limitations = [
            f"The index was built at {indexed_at}; changes made after indexing are invisible. "
            "Re-run `reposcout index --refresh` after editing.",
            "Python symbols come from the AST; other languages use line-based heuristics, so a missed symbol is possible.",
            "Only static evidence is used: dynamic imports, generated code and files outside the index are not analysed.",
            "Ranking is lexical (BM25 and import proximity), not an embedding model; paraphrased tasks can rank poorly.",
        ]
        if not report.mapping.primaries:
            limitations.append("The mapping produced no confident code target.")
        for hit in report.mapping.primaries:
            if not hit.snippets and not any(reason.kind in {"path", "symbol"} for reason in hit.reasons):
                limitations.append(
                    f"`{hit.path}` is ranked, but no line-level evidence was captured for it; treat the suggestion as unverified."
                )
        if not report.checks:
            limitations.append("No verification command exists in the repository, so acceptance criteria are unverified.")
        report.limitations = limitations
        for item in limitations:
            if "index" not in item.lower() and item not in report.plan.unknowns:
                report.plan.unknowns.append(item)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evidence_pack(mapping: object) -> str:
    primaries = getattr(mapping, "primaries", [])
    supporting = getattr(mapping, "supporting", [])
    lines: list[str] = []
    for hit in [*primaries, *supporting][:6]:
        lines.append(f"FILE {hit.path} (role={hit.role}, score={hit.score})")
        for reason in hit.reasons[:3]:
            lines.append(f"  why: {reason.text}")
        for snippet in hit.snippets[:3]:
            lines.append(f"  {snippet.line}: {snippet.text}")
        lines.append("")
    pack = "\n".join(lines)
    return pack[:6000]


_CITATION_RE = re.compile(r"\[([\w./\\-]+?):(\d+)\]")


def _validate_narrative(text: str, mapping: object) -> str:
    text = text.replace("```", "").strip()
    if len(text) > MAX_NARRATIVE:
        text = text[:MAX_NARRATIVE].rsplit(" ", 1)[0] + "..."
    allowed = {hit.path for hit in getattr(mapping, "all_hits", lambda: [])()}
    citations = _CITATION_RE.findall(text)
    if not citations:
        return ""
    if any(path.replace("\\", "/") not in allowed for path, _ in citations):
        return ""
    return text


def _parse_json_object(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in LLM response")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")
    return data


__all__ = ["AgentOptions", "AgentRun", "RepoAgent", "LLMClient"]
