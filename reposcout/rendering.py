"""Human-readable and machine-readable report rendering."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .models import Claim, FileHit, Report
from .tokens import looks_like_test_case_path

BADGE = {"verified": "verified", "suggested": "suggested"}
STATUS_LABEL = {"verified": "VERIFIED", "suggested": "SUGGESTED"}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: Report) -> str:
    out: list[str] = []
    profile = report.profile
    mapping = report.mapping

    out.append("# RepoScout contribution brief")
    out.append("")
    out.append(f"**Task:** {_md(report.task)}  ")
    repo_line = f"**Repository:** `{profile.name}`"
    if profile.is_git and profile.branch:
        repo_line += f" on `{profile.branch}`"
        if profile.commit:
            repo_line += f" @ `{profile.commit}`"
    repo_line += f" · {profile.files_indexed} files · {profile.total_lines:,} lines"
    out.append(repo_line + "  ")
    out.append(
        f"**Generated:** {report.generated_at} · RepoScout v{report.version} · "
        f"**Confidence:** `{mapping.confidence_label}` ({mapping.confidence:.2f})"
    )
    out.append("")
    out.append(f"> {_md(report.plan.summary)}")
    out.append("")
    out.append(
        "**How to read this report:** `VERIFIED` means a file/line in this repository states it. "
        "`SUGGESTED` means RepoScout inferred it. Declared checks have not been executed unless you asked for that."
    )
    out.append("")
    out.append("## At a glance")
    out.append("")
    if mapping.primaries:
        top = mapping.primaries[0]
        out.append(f"- **Start here:** `{top.path}` (line {_anchor_line(top)}), score {top.score:.1f}")
    else:
        out.append("- **Start here:** no confident target was found; see the shortlist below.")
    if mapping.tests:
        test_hits = [hit for hit in mapping.tests if looks_like_test_case_path(hit.path)] or mapping.tests
        out.append("- **Test to extend:** " + ", ".join(f"`{hit.path}`" for hit in test_hits[:3]))
    else:
        out.append("- **Test to extend:** none mapped")
    verified_checks = [check for check in report.checks if check.status == "verified"]
    suggested_checks = [check for check in report.checks if check.status == "suggested"]
    out.append(
        f"- **Checks:** {len(verified_checks)} declared, {len(suggested_checks)} suggested, "
        f"{sum(1 for check in report.checks if check.runnable)} runnable on this machine"
    )
    out.append(
        f"- **Effort:** {report.plan.estimated_minutes[0]}-{report.plan.estimated_minutes[1]} minutes "
        f"(`{report.plan.difficulty_label}`)"
    )
    out.append(f"- **Files likely to change:** {len(report.plan.files_to_touch)}")
    out.append("")

    out.extend(_md_confidence(mapping))
    out.extend(_md_query(mapping))
    out.extend(_md_code_map(mapping))
    out.extend(_md_checks(report))
    out.extend(_md_plan(report))
    out.extend(_md_checklist(report))
    out.extend(_md_unknowns(report))
    out.extend(_md_trace(report))
    out.extend(_md_method(report))
    return "\n".join(out).rstrip() + "\n"


def _md_confidence(mapping) -> list[str]:
    if not mapping.confidence_rationale:
        return []
    lines = ["## Why this confidence", ""]
    for item in mapping.confidence_rationale:
        lines.append(f"- {_md(item)}")
    for note in mapping.notes:
        lines.append(f"- {_md(note)}")
    lines.append("")
    return lines


def _md_query(mapping) -> list[str]:
    if not mapping.query_terms and not mapping.identifiers:
        return []
    lines = ["## How the task was interpreted", ""]
    if mapping.identifiers:
        lines.append("**Identifiers given extra weight:** " + ", ".join(f"`{name}`" for name in mapping.identifiers))
        lines.append("")
    if mapping.phrases:
        lines.append("**Quoted phrases:** " + ", ".join(f"\"{_md(phrase)}\"" for phrase in mapping.phrases))
        lines.append("")
    if mapping.query_terms:
        terms = mapping.query_terms[:18]
        rendered = ", ".join(f"`{_md(term.term)}`" for term in terms)
        lines.append(f"**Search terms:** {rendered}")
        expansions = [term for term in mapping.query_terms if term.origin == "expansion"][:8]
        if expansions:
            lines.append(
                "**Expanded via repository vocabulary:** "
                + ", ".join(f"`{_md(term.term)}`" for term in expansions)
            )
        lines.append("")
    return lines


def _md_code_map(mapping) -> list[str]:
    lines = ["## Relevant code (with evidence)", ""]
    if not mapping.primaries and not mapping.supporting:
        lines.append("No file matched the task strongly enough to be recommended.")
        lines.append("")
        return lines

    sections = [
        ("Primary targets", mapping.primaries),
        ("Supporting files", mapping.supporting),
        ("Existing tests", mapping.tests),
        ("Documentation and configuration", mapping.docs),
    ]
    for title, hits in sections:
        if not hits:
            continue
        lines.append(f"### {title}")
        lines.append("")
        for index, hit in enumerate(hits, start=1):
            lines.append(f"#### {index}. `{hit.path}` · score {hit.score:.1f} · {hit.role}")
            lines.append("")
            if hit.reasons:
                for reason in hit.reasons[:6]:
                    suffix = " _(inferred)_" if reason.inferred else ""
                    lines.append(f"- {_md(reason.text)}{suffix}")
                lines.append("")
            if hit.snippets:
                lines.append("Evidence:")
                lines.append("")
                lines.append("```text")
                for snippet in hit.snippets[:4]:
                    prefix = f"{hit.path}:{snippet.line}"
                    lines.append(prefix)
                    lines.append(f"  {snippet.text}")
                lines.append("```")
                lines.append("")
            if hit.symbols:
                rendered = ", ".join(
                    f"`{_md(symbol.qualified)}` ({symbol.kind}, L{symbol.line})" for symbol in hit.symbols[:5]
                )
                lines.append(f"Symbols: {rendered}")
                lines.append("")
            if hit.imports or hit.imported_by:
                relation: list[str] = []
                if hit.imports:
                    relation.append("imports " + ", ".join(f"`{path}`" for path in hit.imports[:4]))
                if hit.imported_by:
                    relation.append("imported by " + ", ".join(f"`{path}`" for path in hit.imported_by[:4]))
                lines.append("Graph: " + "; ".join(relation))
                lines.append("")
    return lines


def _md_checks(report: Report) -> list[str]:
    lines = ["## Checks", ""]
    verified = [check for check in report.checks if check.status == "verified"]
    suggested = [check for check in report.checks if check.status == "suggested"]

    lines.append("### Verified: declared inside this repository")
    lines.append("")
    if verified:
        lines.append("| Command | Kind | Declared at | Runnable here | Notes |")
        lines.append("| --- | --- | --- | --- | --- |")
        for check in verified:
            sources = ", ".join(source.label() for source in check.sources[:2])
            notes = "; ".join(check.tool_notes) or ""
            lines.append(
                f"| `{_md(check.command)}` | {check.kind} | {sources} | "
                f"{'yes' if check.runnable else 'no'} | {_md(notes)} |"
            )
    else:
        lines.append("None. The repository does not declare how to test or lint it.")
    lines.append("")
    if report.check_results:
        lines.append("### Execution results (explicitly requested)")
        lines.append("")
        lines.append("| Command | Result | Exit | Time | Output (truncated) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for result in report.check_results:
            status = "PASS" if result.ok else ("TIMEOUT" if result.timed_out else "FAIL")
            command = _md(result.command).replace("|", "\\|")
            output = _md(_short(result.output, 300)).replace("|", "\\|")
            lines.append(
                f"| `{command}` | {status} | "
                f"{result.returncode} | {result.duration_ms} ms | {output} |"
            )
        lines.append("")
    lines.append("### Suggested: inferred, not declared")
    lines.append("")
    if suggested:
        lines.append("| Command | Kind | Why it is only a suggestion |")
        lines.append("| --- | --- | --- |")
        for check in suggested:
            basis = check.basis or "inferred from repository contents"
            lines.append(f"| `{_md(check.command)}` | {check.kind} | {_md(basis)} |")
    else:
        lines.append("None.")
    lines.append("")
    for note in _check_notes(report):
        lines.append(f"- {_md(note)}")
    if _check_notes(report):
        lines.append("")
    if report.check_results:
        lines.append(
            "Execution results above come from an explicit run. Commands outside that table are still "
            "declared only, not verified by execution."
        )
    else:
        lines.append(
            "Verified commands are executed only when you explicitly run `reposcout checks --run --yes`; "
            "this report does not claim they passed."
        )
    lines.append("")
    if report.llm_notes:
        lines.append("### AI narrative")
        lines.append("")
        for note in report.llm_notes:
            lines.append(f"- {_md(note)}")
        lines.append("")
        lines.append("_Citations were validated against the retrieved evidence; the narrative can still be wrong._")
        lines.append("")
    return lines


def _md_plan(report: Report) -> list[str]:
    plan = report.plan
    lines = [
        "## Contribution plan",
        "",
        f"Difficulty: **{plan.difficulty_label}** ({plan.difficulty_score}/100) · "
        f"estimate {plan.estimated_minutes[0]}-{plan.estimated_minutes[1]} minutes · "
        f"{sum(step.duration_minutes for step in plan.steps)} minutes of guided steps",
        "",
    ]
    for step in plan.steps:
        lines.append(f"### Step {step.number}: {_md(step.title)}")
        lines.append("")
        lines.append(f"_{_md(step.goal)}_ · ~{step.duration_minutes} min")
        lines.append("")
        for action in step.actions:
            marker = "VERIFIED" if action.status == "verified" else "SUGGESTED"
            detail = _md(action.detail)
            if action.sources and action.status == "verified":
                detail += " — " + ", ".join(f"`{source.label()}`" for source in action.sources[:2])
            lines.append(f"- `{marker}` **{action.kind}**: {detail}")
            if action.basis and action.status == "suggested":
                lines.append(f"  - _basis: {_md(action.basis)}_")
        if step.rationale:
            lines.append("")
            lines.append("Why:")
            for claim in step.rationale:
                lines.append(f"- {_md_claim(claim)}")
        if step.risks:
            lines.append("")
            lines.append("Risks:")
            for risk in step.risks:
                lines.append(f"- {_md(risk)}")
        lines.append("")
    return lines


def _md_checklist(report: Report) -> list[str]:
    lines = ["## Pull request checklist", ""]
    for claim in report.plan.checklist:
        marker = "VERIFIED" if claim.status == "verified" else "SUGGESTED"
        suffix = ""
        if claim.sources:
            suffix = " — " + ", ".join(f"`{source.label()}`" for source in claim.sources[:2])
        elif claim.basis:
            suffix = f" — _{_md(claim.basis)}_"
        lines.append(f"- [ ] `{marker}` {_md(claim.text)}{suffix}")
    lines.append("")
    return lines


def _md_unknowns(report: Report) -> list[str]:
    items: list[str] = []
    for item in [*report.plan.unknowns, *report.limitations]:
        if item not in items:
            items.append(item)
    if not items:
        return []
    lines = ["## What this report does not verify", ""]
    for item in items:
        lines.append(f"- {_md(item)}")
    lines.append("")
    return lines


def _md_trace(report: Report) -> list[str]:
    if not report.trace:
        return []
    lines = ["<details>", "<summary>Agent trace</summary>", ""]
    lines.append("| # | Tool | Input | Output | ms |")
    lines.append("| --- | --- | --- | --- | --- |")
    for step in report.trace:
        lines.append(
            f"| {step.index} | `{step.tool}` | {_md(_short(step.input, 80))} | "
            f"{_md(_short(step.output, 120))} | {step.duration_ms} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


def _md_method(report: Report) -> list[str]:
    languages = ", ".join(f"{lang} ({lines:,})" for lang, lines in list(report.profile.languages.items())[:5])
    lines = [
        "## Method",
        "",
        "- Walked the working tree, honoured `.gitignore`, skipped vendored and generated directories.",
        f"- Indexed {report.profile.files_indexed} files ({languages or 'languages unknown'}) into SQLite.",
        "- Scored files with weighted BM25 over body text, symbols and paths; expanded one hop through resolved imports.",
        "- Extracted every verification command with its declaration site; unmarked ecosystem defaults stay suggestions.",
        "- Generated the plan from observed symbols, snippets, tests and check declarations.",
        "",
        f"RepoScout v{report.version} · index root `{report.repo_root}` · source available under the MIT licence.",
    ]
    return lines


def _check_notes(report: Report) -> list[str]:
    notes = []
    if not any(check.status == "verified" for check in report.checks):
        notes.append("Nothing in the repository declares a verification command; every check is a suggestion.")
    if not any(check.runnable for check in report.checks):
        notes.append("No check is runnable on this machine (tools or dependencies are missing).")
    return notes


def _md_claim(claim: Claim) -> str:
    marker = "VERIFIED" if claim.status == "verified" else "SUGGESTED"
    text = f"`{marker}` {_md(claim.text)}"
    if claim.sources:
        text += " — " + ", ".join(f"`{source.label()}`" for source in claim.sources[:2])
    elif claim.basis:
        text += f" — _{_md(claim.basis)}_"
    return text


def _md(text: str) -> str:
    return text.replace("\n", " ").strip()


def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _anchor_line(hit: FileHit) -> int:
    if hit.snippets:
        return hit.snippets[0].line
    if hit.symbols:
        return hit.symbols[0].line
    return 1


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def render_json(report: Report) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #f7f7f4; --panel: #ffffff; --ink: #1b1d1c; --muted: #5c6360; --line: #e2e3dd;
  --accent: #145c46; --accent-soft: #e6f2ec; --warn: #8a5200; --warn-soft: #fdf1dc;
  --code: #f0f1ea; --shadow: 0 1px 2px rgba(20,30,25,.06), 0 8px 24px rgba(20,30,25,.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141715; --panel: #1b1f1c; --ink: #e8ece7; --muted: #a4ada7; --line: #2c332e;
    --accent: #7fd0ad; --accent-soft: #1d332a; --warn: #f0b45c; --warn-soft: #332816;
    --code: #222823; --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.25);
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1000px; margin: 0 auto; padding: 32px 20px 80px; }
header.hero {
  background: linear-gradient(140deg, var(--accent-soft), transparent 70%);
  border: 1px solid var(--line); border-radius: 18px; padding: 30px 30px 24px; margin-bottom: 26px;
  box-shadow: var(--shadow);
}
.eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: 11px; font-weight: 700; color: var(--accent); }
h1 { font-size: clamp(26px, 4vw, 38px); line-height: 1.15; margin: 8px 0 12px; letter-spacing: -.02em; }
h2 { font-size: 22px; margin: 42px 0 14px; letter-spacing: -.01em; }
h3 { font-size: 17px; margin: 26px 0 10px; }
h4 { font-size: 15px; margin: 18px 0 8px; }
p { margin: 10px 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }
code { background: var(--code); padding: 1px 5px; border-radius: 5px; font-size: .88em; overflow-wrap: anywhere; }
pre { background: var(--code); padding: 14px 16px; border-radius: 10px; overflow-x: auto; font-size: 13px; line-height: 1.5; }
pre code { background: none; padding: 0; }
.task { font-size: 18px; margin: 0 0 14px; max-width: 70ch; }
.meta { color: var(--muted); font-size: 13.5px; display: flex; flex-wrap: wrap; gap: 6px 16px; }
.meta strong { color: var(--ink); font-weight: 600; }
.summary { margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 15.5px; }
.legend { display: flex; flex-wrap: wrap; gap: 10px 22px; margin: 16px 0 0; font-size: 13px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
nav.toc { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 8px; }
nav.toc a {
  border: 1px solid var(--line); background: var(--panel); border-radius: 999px;
  padding: 6px 12px; font-size: 13px; color: var(--muted);
}
nav.toc a:hover { color: var(--accent); border-color: var(--accent); text-decoration: none; }
.badge {
  display: inline-block; border-radius: 999px; padding: 2px 9px; font-size: 10.5px;
  font-weight: 700; letter-spacing: .06em; text-transform: uppercase; vertical-align: 1px;
}
.badge.verified { background: var(--accent-soft); color: var(--accent); border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent); }
.badge.suggested { background: var(--warn-soft); color: var(--warn); border: 1px solid color-mix(in srgb, var(--warn) 30%, transparent); }
.badge.role { background: var(--code); color: var(--muted); border: 1px solid var(--line); }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 20px 22px; box-shadow: var(--shadow); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }
.stat { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 13px 15px; }
.stat .k { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 700; }
.stat .v { font-size: 17px; font-weight: 650; margin-top: 4px; }
.meter { height: 7px; background: var(--code); border-radius: 99px; overflow: hidden; margin-top: 8px; }
.meter > i { display: block; height: 100%; background: var(--accent); border-radius: 99px; }
.file { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; margin: 14px 0; overflow: hidden; }
.file > summary { cursor: pointer; padding: 14px 18px; display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; list-style: none; }
.file > summary::-webkit-details-marker { display: none; }
.file > summary .path { font-weight: 650; font-family: ui-monospace, Consolas, monospace; font-size: 14px; }
.file > summary .score { color: var(--muted); font-size: 12.5px; margin-left: auto; }
.file .body { padding: 0 18px 18px; border-top: 1px solid var(--line); }
ul.why { margin: 12px 0; padding-left: 20px; }
ul.why li { margin: 5px 0; }
ul.why .inferred { color: var(--warn); font-size: 11.5px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.evidence { margin: 12px 0; border-left: 3px solid var(--accent); background: var(--code); padding: 10px 14px; border-radius: 0 8px 8px 0; }
.evidence .loc { color: var(--muted); font-size: 11.5px; font-family: ui-monospace, Consolas, monospace; }
.evidence .line { font-family: ui-monospace, Consolas, monospace; font-size: 12.5px; margin-top: 3px; word-break: break-word; }
.symbols { color: var(--muted); font-size: 13.5px; margin: 10px 0; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13.5px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; }
tr:last-child td { border-bottom: none; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
.tablewrap table { margin: 0; }
.step { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; margin: 12px 0; }
.step h3 { margin-top: 0; display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; }
.step h3 .num {
  display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
  border-radius: 50%; background: var(--accent-soft); color: var(--accent); font-size: 13px; font-weight: 800;
}
.step h3 .mins { margin-left: auto; color: var(--muted); font-size: 12px; font-weight: 500; }
.goal { color: var(--muted); margin: -4px 0 12px; }
.action { padding: 7px 0; border-bottom: 1px dashed var(--line); }
.action:last-child { border-bottom: 0; }
.action .kind { font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; margin-right: 5px; }
.basis { color: var(--muted); font-size: 12.5px; margin: 3px 0 0 0; }
.risk { color: var(--warn); background: var(--warn-soft); border-radius: 9px; padding: 9px 12px; font-size: 13px; margin-top: 8px; }
.checklist { list-style: none; padding: 0; }
.checklist li { padding: 8px 0 8px 28px; position: relative; border-bottom: 1px solid var(--line); }
.checklist li::before { content: ""; position: absolute; left: 0; top: 12px; width: 15px; height: 15px; border: 1.5px solid var(--muted); border-radius: 4px; }
.note { color: var(--muted); font-size: 13px; }
details.trace summary, details.more summary { cursor: pointer; color: var(--accent); font-weight: 600; font-size: 14px; padding: 6px 0; }
.ok { color: var(--accent); font-weight: 700; }
.no { color: var(--warn); font-weight: 700; }
footer { margin-top: 50px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12.5px; }
@media print {
  body { background: #fff; } .wrap { padding: 0; max-width: none; }
  .file, .panel, .step { box-shadow: none; break-inside: avoid; }
  nav.toc, .legend { display: none; }
}
"""


def render_html(report: Report) -> str:
    profile = report.profile
    mapping = report.mapping
    plan = report.plan
    e = html.escape

    def badge(status: str) -> str:
        return f'<span class="badge {status}">{STATUS_LABEL[status]}</span>'

    def inline(text: str) -> str:
        escaped = e(text)
        escaped = re.sub(r"`([^`]+)`", lambda match: f"<code>{match.group(1)}</code>", escaped)
        return escaped

    verified = [check for check in report.checks if check.status == "verified"]
    suggested = [check for check in report.checks if check.status == "suggested"]

    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>RepoScout - {e(profile.name)}</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append('<div class="wrap">')

    # Hero
    parts.append('<header class="hero">')
    parts.append('<div class="eyebrow">RepoScout contribution brief</div>')
    parts.append(f"<h1>{inline(report.task)}</h1>")
    parts.append('<div class="meta">')
    repo_meta = f"<strong>{e(profile.name)}</strong>"
    if profile.is_git and profile.branch:
        repo_meta += f" on <code>{e(profile.branch)}</code>"
        if profile.commit:
            repo_meta += f" @ <code>{e(profile.commit)}</code>"
    parts.append(f"<span>{repo_meta}</span>")
    parts.append(f"<span>{profile.files_indexed:,} files</span>")
    parts.append(f"<span>{profile.total_lines:,} lines</span>")
    parts.append(f"<span>{len(profile.languages)} languages</span>")
    parts.append(f"<span>Generated {e(report.generated_at)}</span>")
    parts.append(f"<span>v{e(report.version)}</span>")
    parts.append("</div>")
    parts.append(f'<div class="summary">{inline(plan.summary)}</div>')
    parts.append(
        '<div class="legend">'
        f'<span>{badge("verified")} backed by a file and line in this repository</span>'
        f'<span>{badge("suggested")} inferred by RepoScout; confirm before acting</span>'
        "</div>"
    )
    parts.append("</header>")

    parts.append(
        '<nav class="toc">'
        '<a href="#overview">Overview</a><a href="#code">Code map</a><a href="#checks">Checks</a>'
        '<a href="#plan">Plan</a><a href="#checklist">PR checklist</a><a href="#limits">Limits</a>'
        "</nav>"
    )

    # Overview stats
    parts.append('<section id="overview"><h2>Overview</h2>')
    parts.append('<div class="grid">')
    confidence_pct = int(round(mapping.confidence * 100))
    parts.append(
        f'<div class="stat"><div class="k">Mapping confidence</div>'
        f'<div class="v">{e(mapping.confidence_label)} · {confidence_pct}%</div>'
        f'<div class="meter"><i style="width:{confidence_pct}%"></i></div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="k">Primary targets</div><div class="v">{len(mapping.primaries)} files</div>'
        f'<div class="note">{len(plan.files_to_touch)} likely to change</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="k">Mapped tests</div><div class="v">{len(mapping.tests)}</div>'
        f'<div class="note">{"ready to extend" if mapping.tests else "no test found"}</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="k">Declared checks</div><div class="v">{len(verified)}</div>'
        f'<div class="note">{sum(1 for check in verified if check.runnable)} runnable here</div></div>'
    )
    parts.append(
        f'<div class="stat"><div class="k">Estimated effort</div>'
        f'<div class="v">{plan.estimated_minutes[0]}-{plan.estimated_minutes[1]} min</div>'
        f'<div class="note">{e(plan.difficulty_label)} · score {plan.difficulty_score}/100</div></div>'
    )
    parts.append("</div>")

    if mapping.confidence_rationale or mapping.notes:
        parts.append('<div class="panel">')
        parts.append("<h3>Why this confidence</h3><ul class=\"why\">")
        for item in mapping.confidence_rationale:
            parts.append(f"<li>{inline(item)}</li>")
        for note in mapping.notes:
            parts.append(f'<li class="note">{inline(note)}</li>')
        parts.append("</ul></div>")

    if mapping.identifiers or mapping.query_terms:
        parts.append("<details class=\"more\"><summary>How the task was interpreted</summary><div class=\"panel\">")
        if mapping.identifiers:
            parts.append(
                "<p><strong>Identifiers:</strong> "
                + " ".join(f"<code>{e(name)}</code>" for name in mapping.identifiers)
                + "</p>"
            )
        if mapping.phrases:
            parts.append(
                "<p><strong>Quoted phrases:</strong> "
                + " ".join(f"<code>{e(phrase)}</code>" for phrase in mapping.phrases)
                + "</p>"
            )
        parts.append(
            "<p><strong>Index terms:</strong> "
            + " ".join(
                f'<code title="{e(term.origin)} · weight {term.weight}">{e(term.term)}</code>'
                for term in mapping.query_terms[:24]
            )
            + "</p>"
        )
        parts.append("</div></details>")
    parts.append("</section>")

    # Code map
    parts.append('<section id="code"><h2>Relevant code</h2>')
    if not mapping.all_hits():
        parts.append('<div class="panel">No file matched strongly enough. Narrow the task with a filename, symbol or error text.</div>')
    for role_title, hits in (
        ("Primary targets", mapping.primaries),
        ("Supporting files", mapping.supporting),
        ("Existing tests", mapping.tests),
        ("Docs and configuration", mapping.docs),
    ):
        if not hits:
            continue
        parts.append(f"<h3>{role_title}</h3>")
        for number, hit in enumerate(hits, start=1):
            open_attr = " open" if number == 1 and role_title == "Primary targets" else ""
            parts.append(f'<details class="file"{open_attr}>')
            parts.append("<summary>")
            parts.append(f'<span class="path">{e(hit.path)}</span>')
            parts.append(f'<span class="badge role">{e(hit.role)}</span>')
            parts.append(f'<span class="badge role">{e(hit.lang or "text")}</span>')
            if hit.module and hit.module != Path(hit.path).stem:
                parts.append(f'<span class="badge role">{e(hit.module)}</span>')
            parts.append(f'<span class="score">score {hit.score:.1f} · {hit.lines:,} lines</span>')
            parts.append("</summary>")
            parts.append('<div class="body">')
            if hit.reasons:
                parts.append('<ul class="why">')
                for reason in hit.reasons[:7]:
                    inferred = ' <span class="inferred">inferred</span>' if reason.inferred else ""
                    parts.append(f"<li>{inline(reason.text)}{inferred}</li>")
                parts.append("</ul>")
            for snippet in hit.snippets[:4]:
                parts.append('<div class="evidence">')
                parts.append(f'<div class="loc">{e(hit.path)}:{snippet.line}'
                             + (f" · {e(snippet.symbol)}" if snippet.symbol else "")
                             + "</div>")
                parts.append(f'<div class="line">{e(snippet.text)}</div>')
                parts.append("</div>")
            if hit.symbols:
                parts.append('<div class="symbols"><strong>Symbols:</strong> ')
                parts.append(
                    ", ".join(
                        f"<code>{e(symbol.qualified)}</code> <span class=\"note\">{e(symbol.kind)} · L{symbol.line}</span>"
                        for symbol in hit.symbols[:6]
                    )
                )
                parts.append("</div>")
            relations: list[str] = []
            if hit.imports:
                relations.append("imports " + ", ".join(f"<code>{e(path)}</code>" for path in hit.imports[:4]))
            if hit.imported_by:
                relations.append("imported by " + ", ".join(f"<code>{e(path)}</code>" for path in hit.imported_by[:4]))
            if relations:
                parts.append('<div class="symbols">' + "; ".join(relations) + "</div>")
            parts.append("</div></details>")
    parts.append("</section>")

    # Checks
    parts.append('<section id="checks"><h2>Checks</h2>')
    if report.check_results:
        parts.append(
            '<p class="note">Verified means the command is written down in this repository. '
            "The execution table below lists commands you explicitly asked RepoScout to run; "
            "everything else is declared but unexecuted.</p>"
        )
    else:
        parts.append(
            '<p class="note">Verified means the command is written down in this repository, not that it passed. '
            "RepoScout executes nothing unless you run <code>reposcout checks --run --yes</code>.</p>"
        )
    parts.append(f"<h3>{badge('verified')} Declared in this repository</h3>")
    if verified:
        parts.append('<div class="tablewrap"><table><thead><tr>'
                     "<th>Command</th><th>Kind</th><th>Declared at</th><th>Runnable here</th><th>Notes</th>"
                     "</tr></thead><tbody>")
        for check in verified:
            sources = "<br>".join(
                f'<code>{e(source.path)}:{source.line}</code>' for source in check.sources[:3]
            )
            runnable = f'<span class="ok">yes</span>' if check.runnable else '<span class="no">no</span>'
            notes = "; ".join(check.tool_notes)
            parts.append(
                f"<tr><td><code>{e(check.command)}</code></td><td>{e(check.kind)}</td><td>{sources}</td>"
                f"<td>{runnable}</td><td class=\"note\">{e(notes)}</td></tr>"
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append('<div class="panel">None. This repository declares no verification command.</div>')
    if report.check_results:
        parts.append("<h3>Execution results (explicitly requested)</h3>")
        parts.append(
            '<div class="tablewrap"><table><thead><tr>'
            "<th>Result</th><th>Command</th><th>Exit</th><th>Time</th><th>Output (truncated)</th>"
            "</tr></thead><tbody>"
        )
        for result in report.check_results:
            if result.ok:
                status = '<span class="ok">PASS</span>'
            elif result.timed_out:
                status = '<span class="no">TIMEOUT</span>'
            else:
                status = '<span class="no">FAIL</span>'
            parts.append(
                f"<tr><td>{status}</td><td><code>{e(result.command)}</code></td>"
                f"<td>{result.returncode}</td><td>{result.duration_ms} ms</td>"
                f"<td class=\"note\">{e(_short(result.output, 240))}</td></tr>"
            )
        parts.append("</tbody></table></div>")
    parts.append(f"<h3>{badge('suggested')} Inferred, not declared</h3>")
    if suggested:
        parts.append('<div class="tablewrap"><table><thead><tr>'
                     "<th>Command</th><th>Kind</th><th>Why it stays a suggestion</th>"
                     "</tr></thead><tbody>")
        for check in suggested:
            parts.append(
                f"<tr><td><code>{e(check.command)}</code></td><td>{e(check.kind)}</td>"
                f"<td>{inline(check.basis)}</td></tr>"
            )
        parts.append("</tbody></table></div>")
    else:
        parts.append('<div class="panel">No inferred commands.</div>')
    for note in _check_notes(report):
        parts.append(f'<p class="note">{inline(note)}</p>')
    if report.llm_notes:
        parts.append('<div class="panel"><h3>AI narrative</h3>')
        for note in report.llm_notes:
            parts.append(f"<p>{inline(note)}</p>")
        parts.append(
            '<p class="note">Citations were validated against the retrieved evidence. '
            "The narrative is optional and can still be wrong.</p></div>"
        )
    parts.append("</section>")

    # Plan
    parts.append('<section id="plan"><h2>Contribution plan</h2>')
    parts.append(
        f'<p class="note">Difficulty <strong>{e(plan.difficulty_label)}</strong> '
        f"({plan.difficulty_score}/100) · estimate {plan.estimated_minutes[0]}-{plan.estimated_minutes[1]} minutes.</p>"
    )
    for step in plan.steps:
        parts.append('<div class="step">')
        parts.append(
            f'<h3><span class="num">{step.number}</span> {inline(step.title)}'
            f'<span class="mins">~{step.duration_minutes} min</span></h3>'
        )
        parts.append(f'<p class="goal">{inline(step.goal)}</p>')
        for action in step.actions:
            parts.append('<div class="action">')
            parts.append(f"{badge(action.status)} <span class=\"kind\">{e(action.kind)}</span>")
            parts.append(inline(action.detail))
            if action.sources and action.status == "verified":
                parts.append(
                    ' <span class="note">→ '
                    + ", ".join(f'<code>{e(source.label())}</code>' for source in action.sources[:2])
                    + "</span>"
                )
            if action.basis and action.status == "suggested":
                parts.append(f'<div class="basis">basis: {inline(action.basis)}</div>')
            parts.append("</div>")
        if step.rationale:
            parts.append('<div class="basis"><strong>Why:</strong><ul class="why">')
            for claim in step.rationale:
                parts.append(f"<li>{_html_claim(claim, badge)}</li>")
            parts.append("</ul></div>")
        for risk in step.risks:
            parts.append(f'<div class="risk">{inline(risk)}</div>')
        parts.append("</div>")
    parts.append("</section>")

    # Checklist
    parts.append('<section id="checklist"><h2>Pull request checklist</h2><ul class="checklist">')
    for claim in plan.checklist:
        source_note = ""
        if claim.sources:
            source_note = " <span class=\"note\">— " + ", ".join(
                f"<code>{e(source.label())}</code>" for source in claim.sources[:2]
            ) + "</span>"
        elif claim.basis:
            source_note = f' <span class="note">— {inline(claim.basis)}</span>'
        parts.append(f"<li>{badge(claim.status)} {inline(claim.text)}{source_note}</li>")
    parts.append("</ul></section>")

    # Limits
    parts.append('<section id="limits"><h2>What this report does not verify</h2><div class="panel">')
    seen: set[str] = set()
    for item in [*plan.unknowns, *report.limitations]:
        if item in seen:
            continue
        seen.add(item)
        parts.append(f"<p>• {inline(item)}</p>")
    if not seen:
        parts.append("<p>No additional caveats were recorded.</p>")
    parts.append("</div></section>")

    # Trace
    if report.trace:
        parts.append('<details class="trace"><summary>Agent trace</summary><div class="tablewrap"><table>')
        parts.append("<thead><tr><th>#</th><th>Tool</th><th>Input</th><th>Output</th><th>ms</th></tr></thead><tbody>")
        for step in report.trace:
            parts.append(
                f"<tr><td>{step.index}</td><td><code>{e(step.tool)}</code></td>"
                f"<td>{inline(_short(step.input, 90))}</td>"
                f"<td>{inline(_short(step.output, 160))}</td><td>{step.duration_ms}</td></tr>"
            )
        parts.append("</tbody></table></div></details>")

    parts.append("<footer>")
    parts.append(
        f"RepoScout v{e(report.version)} · generated {e(report.generated_at)} · "
        f"index root <code>{e(report.repo_root)}</code><br>"
        "Lexical retrieval + static analysis. Verified means declared in the repository; "
        "suggested means inferred. Nothing was executed without an explicit request."
    )
    parts.append("</footer>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


def _html_claim(claim: Claim, badge) -> str:
    escaped = html.escape(claim.text)
    escaped = re.sub(r"`([^`]+)`", lambda match: f"<code>{match.group(1)}</code>", escaped)
    suffix = ""
    if claim.sources:
        suffix = " <span class=\"note\">→ " + ", ".join(
            f"<code>{html.escape(source.label())}</code>" for source in claim.sources[:2]
        ) + "</span>"
    elif claim.basis:
        suffix = f' <span class="note">— {html.escape(claim.basis)}</span>'
    return f'{badge(claim.status)} {escaped}{suffix}'


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_report(report: Report, path: Path, fmt: str) -> Path:
    fmt = fmt.lower().lstrip(".")
    renderers = {"md": render_markdown, "markdown": render_markdown, "html": render_html, "json": render_json}
    if fmt not in renderers:
        raise ValueError(f"unsupported format: {fmt}")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(renderers[fmt](report), encoding="utf-8")
    return path


def suggest_output_path(root: Path, task: str, default: Path | None = None) -> Path:
    from .util import slug

    if default is not None:
        return default
    return root / ".reposcout" / "reports" / f"{slug(task)}.html"
