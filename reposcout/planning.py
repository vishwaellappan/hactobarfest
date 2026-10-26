"""Turn a mapping plus declared checks into a beginner-sized contribution plan.

Every step contains actions whose status is explicit.  Reads and citations are
usually ``verified`` because the file/line exists; proposed edits and strategy
advice are ``suggested`` because no static tool can prove the right fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .checks import CheckSet
from .models import (
    Action,
    Check,
    Claim,
    ContributionPlan,
    FileHit,
    Mapping,
    PlanStep,
    RepoProfile,
    Source,
)
from .util import clamp, common_ancestor, read_text, split_lines, truncate
from .tokens import looks_like_test_case_path

DEFAULT_TOTAL_MINUTES = (45, 120)


def build_plan(
    task: str,
    profile: RepoProfile,
    mapping: Mapping,
    checks: CheckSet,
    *,
    test_paths: Iterable[str] = (),
    root: Path | None = None,
) -> ContributionPlan:
    root = root or Path(profile.root)
    plan = ContributionPlan(task=task)
    plan.difficulty_score = _difficulty(mapping, checks, profile)
    plan.difficulty_label = (
        "starter" if plan.difficulty_score <= 30 else "moderate" if plan.difficulty_score <= 60 else "advanced"
    )
    low, high = _estimate_minutes(plan.difficulty_score)
    plan.estimated_minutes = (low, high)
    plan.files_to_touch = [hit.path for hit in mapping.primaries]
    plan.test_files = sorted(
        (hit.path for hit in mapping.tests),
        key=lambda path: (not looks_like_test_case_path(path), path),
    )
    plan.summary = _summary(mapping, checks, plan)

    steps = [
        _step_orient(profile, mapping, root),
        _step_confirm(profile, mapping, checks, test_paths, root),
        _step_read(mapping),
        _step_change(mapping),
        _step_tests(mapping, checks, profile, test_paths, root),
        _step_verify(mapping, checks, profile),
        _step_submit(mapping, profile, root),
    ]
    for index, step in enumerate(steps, start=1):
        step.number = index
    plan.steps = steps
    plan.checklist = _checklist(profile, root)
    plan.unknowns = _unknowns(mapping, checks, profile)
    return plan


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _step_orient(profile: RepoProfile, mapping: Mapping, root: Path) -> PlanStep:
    actions: list[Action] = []
    rationale: list[Claim] = []

    readme = next((path for path in profile.docs if Path(path).name.lower().startswith("readme")), None)
    if readme:
        actions.append(
            Action(
                kind="read",
                detail=f"Skim `{readme}` for setup, scope and support channels.",
                status="verified",
                sources=[Source(readme)],
            )
        )
    else:
        actions.append(
            Action(
                kind="ask",
                detail="No README is indexed. Ask the maintainers where setup instructions live.",
                status="suggested",
                basis="no README file was found in the indexed tree",
            )
        )

    guide = next((path for path in profile.guidelines if "contribut" in path.lower()), None)
    if guide:
        actions.append(
            Action(
                kind="read",
                detail=f"Read `{guide}` before writing code; it is the project's own contribution contract.",
                status="verified",
                sources=[Source(guide)],
            )
        )
        rationale.append(
            Claim(
                text=f"The project publishes contribution guidance at `{guide}`.",
                status="verified",
                sources=[Source(guide)],
            )
        )
    else:
        rationale.append(
            Claim(
                text="No CONTRIBUTING guide was found, so the review expectations below are generic.",
                status="suggested",
                basis="no contributing guide exists in the indexed tree",
            )
        )

    if mapping.primaries:
        top = mapping.primaries[0]
        anchor = _anchor(top)
        actions.append(
            Action(
                kind="read",
                detail=(
                    f"Open `{top.path}` at {anchor.label()} first"
                    + (f" (symbol `{top.symbols[0].qualified}`)" if top.symbols else "")
                    + "."
                ),
                status="verified",
                sources=[anchor],
            )
        )

    rationale.append(
        Claim(
            text=f"The repository has {profile.files_indexed} indexed files across {len(profile.languages)} language(s), {profile.test_files} of them test files.",
            status="verified",
            sources=[Source(_profile_source(profile))],
        )
    )
    return PlanStep(
        number=0,
        title="Orient yourself",
        goal="Understand the project's rules and where the task lives before touching code.",
        duration_minutes=10,
        actions=actions,
        rationale=rationale,
        risks=["Guides and READMEs can lag behind the code; trust the source you cite in the PR."],
    )


def _step_confirm(
    profile: RepoProfile,
    mapping: Mapping,
    checks: CheckSet,
    test_paths: Iterable[str],
    root: Path,
) -> PlanStep:
    actions: list[Action] = []
    rationale: list[Claim] = []
    test_checks = [check for check in checks.checks if check.kind == "test"]
    verified_tests = [check for check in test_checks if check.status == "verified"]
    suggested_tests = [check for check in test_checks if check.status == "suggested"]

    if verified_tests:
        check = verified_tests[0]
        actions.append(_run_action(check))
        rationale.append(
            Claim(
                text=f"The project declares a test command: `{check.command}` ({check.declared_in}).",
                status="verified",
                sources=check.sources[:1],
            )
        )
    elif suggested_tests:
        check = suggested_tests[0]
        actions.append(
            Action(
                kind="run",
                detail=f"Try `{check.command}` once to see the current baseline (unverified suggestion).",
                status="suggested",
                basis=check.basis,
                command=check.command,
                check_id=check.id,
            )
        )
    else:
        rationale.append(
            Claim(
                text="No test command could be found or inferred, so reproducing the issue is an open question.",
                status="suggested",
                basis="the repository declares no test command and no test files were detected",
            )
        )

    if mapping.tests:
        test = mapping.tests[0]
        anchor = _anchor(test)
        actions.append(
            Action(
                kind="read",
                detail=f"Read `{test.path}` near {anchor.label()} to see the existing expectations and fixtures.",
                status="verified",
                sources=[anchor],
            )
        )
        rationale.append(
            Claim(
                text=f"An existing test file is associated with the target: `{test.path}`.",
                status="verified",
                sources=[Source(test.path)],
            )
        )
    else:
        actions.append(
            Action(
                kind="ask",
                detail="Ask in the issue thread how maintainers would like a regression test; none was detected.",
                status="suggested",
                basis="no test file matched the target module and no test directory was detected",
            )
        )

    return PlanStep(
        number=0,
        title="Reproduce or pin down the behavior",
        goal="Get a concrete before/after signal that the change can be judged against.",
        duration_minutes=15,
        actions=actions,
        rationale=rationale,
        risks=[
            "Running tests may need dependencies that are not installed; use the project's documented setup.",
        ],
    )


def _step_read(mapping: Mapping) -> PlanStep:
    actions: list[Action] = []
    rationale: list[Claim] = []
    for hit in mapping.primaries[:4]:
        anchor = _anchor(hit)
        symbol = hit.symbols[0].qualified if hit.symbols else None
        detail = f"Read `{hit.path}` starting at {anchor.label()}"
        if symbol:
            detail += f"; follow `{symbol}` and its callers."
        else:
            detail += "; follow the matching lines into their enclosing functions."
        actions.append(Action(kind="read", detail=detail, status="verified", sources=[anchor]))
        if hit.role == "doc":
            rationale.append(
                Claim(
                    text=f"The task vocabulary matches the documentation file `{hit.path}` (score {hit.score:.1f}).",
                    status="verified",
                    sources=[anchor],
                )
            )
        elif hit.snippets:
            snippet = hit.snippets[0]
            rationale.append(
                Claim(
                    text=f"`{hit.path}:{snippet.line}` is direct evidence for the task: \"{truncate(snippet.text, 70)}\".",
                    status="verified",
                    sources=[Source(hit.path, snippet.line, snippet.text)],
                )
            )
        else:
            graph_reason = next((reason for reason in hit.reasons if reason.kind == "graph"), None)
            rationale.append(
                Claim(
                    text=(
                        f"`{hit.path}` is relevant through the import graph"
                        + (f" ({graph_reason.text})" if graph_reason else "")
                        + ", not through a direct line match."
                    ),
                    status="suggested",
                    basis="the file was reached by graph expansion, so relevance is inferred rather than observed",
                )
            )

    for hit in mapping.supporting[:2]:
        actions.append(
            Action(
                kind="read",
                detail=f"Read `{hit.path}` for surrounding context before editing; it is connected to the targets.",
                status="verified",
                sources=[Source(hit.path)],
            )
        )

    risks = []
    if any(hit.imported_by for hit in mapping.primaries):
        risks.append("Callers exist in the index; changing a signature can break files outside this list.")
    if any(reason.inferred for hit in mapping.primaries for reason in hit.reasons):
        risks.append("Some links are inferred from imports or file names; confirm them with search before relying on them.")

    return PlanStep(
        number=0,
        title="Read the code in evidence order",
        goal="Build a mental model from the files that actually matched the task.",
        duration_minutes=20,
        actions=actions,
        rationale=rationale,
        risks=risks,
    )


def _step_change(mapping: Mapping) -> PlanStep:
    actions: list[Action] = []
    rationale: list[Claim] = []
    for hit in mapping.primaries[:4]:
        anchor = _anchor(hit)
        symbol = next(
            (item for item in hit.symbols if any(snippet.line >= item.line and (item.end_line is None or snippet.line <= item.end_line) for snippet in hit.snippets)),
            hit.symbols[0] if hit.symbols else None,
        )
        location = f"`{symbol.qualified}`" if symbol else "the matched block"
        actions.append(
            Action(
                kind="edit",
                detail=f"Change {location} in `{hit.path}` at {anchor.label()}, keeping the diff limited to this behavior.",
                status="suggested",
                sources=[anchor],
                basis="proposed edit site derived from ranked evidence; the correct fix still needs human judgement",
            )
        )
        if hit.role == "doc":
            rationale.append(
                Claim(
                    text=f"Documentation target: `{hit.path}`.",
                    status="verified",
                    sources=[anchor],
                )
            )
        else:
            rationale.append(
                Claim(
                    text=f"Primary implementation target: `{hit.path}` ({hit.lines} lines, {hit.lang}).",
                    status="verified",
                    sources=[anchor],
                )
            )

    actions.append(
        Action(
            kind="inspect",
            detail="Search the repository for other call sites before renaming or changing a public signature.",
            status="suggested",
            basis="static evidence is limited to indexed files and one import hop",
        )
    )
    rationale.append(
        Claim(
            text="Keep the patch minimal and behavior-preserving; a focused diff is easier for maintainers to review.",
            status="suggested",
            basis="review heuristic, not a repository rule",
        )
    )

    risks = [
        "Avoid drive-by refactors; they raise review cost and hide the behavioral change.",
    ]
    if any("__init__" in hit.path for hit in mapping.primaries):
        risks.append("This is a package entry point; changing exports can break downstream imports.")
    return PlanStep(
        number=0,
        title="Make the smallest change that satisfies the task",
        goal="Move from reading to one coherent, reviewable edit.",
        duration_minutes=25,
        actions=actions,
        rationale=rationale,
        risks=risks,
    )


def _step_tests(
    mapping: Mapping,
    checks: CheckSet,
    profile: RepoProfile,
    test_paths: Iterable[str],
    root: Path,
) -> PlanStep:
    actions: list[Action] = []
    rationale: list[Claim] = []
    target = mapping.primaries[0] if mapping.primaries else None

    if mapping.tests:
        for test in mapping.tests[:2]:
            actions.append(
                Action(
                    kind="edit",
                    detail=f"Extend `{test.path}` with a focused regression test for the task behavior.",
                    status="suggested",
                    sources=[Source(test.path)],
                    basis="existing test file mapped to the target module; the exact assertion depends on the intended behavior",
                )
            )
        test = mapping.tests[0]
        command = _targeted_test_command(test, checks)
        actions.append(
            Action(
                kind="run",
                detail=f"Run the focused test first: `{command}`.",
                status="suggested",
                basis="targeted invocation derived from the mapped test file; not declared as a repository command",
                command=command,
            )
        )
        rationale.append(
            Claim(
                text=f"A matching test exists at `{test.path}`; extending it is cheaper than creating a new suite.",
                status="verified",
                sources=[Source(test.path)],
            )
        )
    elif target:
        suggestion = _suggest_test_path(target.path, target.lang, profile, list(test_paths))
        actions.append(
            Action(
                kind="edit",
                detail=f"Add a focused regression test at the conventional path `{suggestion}`.",
                status="suggested",
                basis="derived from the target filename and language convention; not an existing file",
            )
        )
        rationale.append(
            Claim(
                text="No existing test maps to this target, so a new regression test is likely required.",
                status="suggested",
                basis="no test file shared a module name or imported the target",
            )
        )

    rationale.append(
        Claim(
            text="A regression test should fail before the fix and pass after it.",
            status="suggested",
            basis="testing heuristic, independent of this repository",
        )
    )
    return PlanStep(
        number=0,
        title="Add or extend a regression test",
        goal="Prove the behavior change and protect it from regressions.",
        duration_minutes=20,
        actions=actions,
        rationale=rationale,
        risks=[
            "If the project has no tests, maintainers may prefer a manual reproduction; confirm before adding a framework.",
        ],
    )


def _step_verify(mapping: Mapping, checks: CheckSet, profile: RepoProfile) -> PlanStep:
    actions: list[Action] = []
    rationale: list[Claim] = []
    verified = [check for check in checks.checks if check.status == "verified"][:8]
    suggested = [check for check in checks.checks if check.status == "suggested"][:4]

    for check in verified:
        actions.append(_run_action(check))
        note = f" ({'; '.join(check.tool_notes)})" if check.tool_notes else ""
        rationale.append(
            Claim(
                text=f"`{check.command}` is declared at {check.declared_in or check.sources[0].label()}.{note}",
                status="verified",
                sources=check.sources[:1],
            )
        )
    for check in suggested:
        actions.append(
            Action(
                kind="run",
                detail=f"Optional, unverified: `{check.command}`. {check.basis}.",
                status="suggested",
                basis=check.basis,
                command=check.command,
                check_id=check.id,
            )
        )
    if not verified:
        rationale.append(
            Claim(
                text="Nothing in the repository declares how to validate the change. Treat every command here as a suggestion.",
                status="suggested",
                basis="no verification command was found in config, CI or contributor docs",
            )
        )

    return PlanStep(
        number=0,
        title="Run the declared checks",
        goal="Separate what the repository requires from what we merely think you should run.",
        duration_minutes=15,
        actions=actions,
        rationale=rationale,
        risks=[
            "A command can be declared but fail for environmental reasons (missing tools, dependencies or services).",
        ],
    )


def _step_submit(mapping: Mapping, profile: RepoProfile, root: Path) -> PlanStep:
    actions: list[Action] = []
    rationale: list[Claim] = []

    template = next(
        (path for path in profile.guidelines if "pull_request" in path.lower() or "contributing" in path.lower()),
        None,
    )
    if template:
        actions.append(
            Action(
                kind="read",
                detail=f"Copy the pull-request expectations from `{template}` into your own checklist.",
                status="verified",
                sources=[Source(template)],
            )
        )
        rationale.append(
            Claim(
                text=f"The repository publishes a pull-request/contribution process at `{template}`.",
                status="verified",
                sources=[Source(template)],
            )
        )
    else:
        rationale.append(
            Claim(
                text="No pull-request template or contributing guide was found; use the generic checklist below.",
                status="suggested",
                basis="no contribution documents were detected in the repository",
            )
        )

    if any(hit.role == "doc" for hit in mapping.primaries):
        rationale.append(
            Claim(
                text="Because the target is documentation, check spelling and links as well as the rendered formatting.",
                status="suggested",
                basis="documentation review heuristic",
            )
        )

    actions.append(
        Action(
            kind="ask",
            detail="Open the pull request against the default branch, reference the issue, and describe the verification you ran.",
            status="suggested",
            basis="standard open-source workflow; the exact branch and template come from the repository guidelines",
        )
    )
    return PlanStep(
        number=0,
        title="Prepare the pull request",
        goal="Make the contribution easy to review and easy to trust.",
        duration_minutes=10,
        actions=actions,
        rationale=rationale,
        risks=[],
    )


# ---------------------------------------------------------------------------
# Difficulty, estimates and checklist
# ---------------------------------------------------------------------------


def _difficulty(mapping: Mapping, checks: CheckSet, profile: RepoProfile) -> int:
    score = 12
    primaries = mapping.primaries
    score += 6 * max(0, len(primaries) - 1)
    line_total = sum(hit.lines for hit in primaries)
    score += min(18, line_total / 120)
    fan_in = sum(len(hit.imported_by) for hit in primaries)
    score += min(15, fan_in * 2)
    if not mapping.tests:
        score += 9
    if not checks.verified:
        score += 10
    if not profile.guidelines:
        score += 4
    if mapping.confidence_label == "low":
        score += 8
    if len({hit.lang for hit in primaries}) > 1:
        score += 5
    if mapping.primaries and all(hit.role == "doc" for hit in mapping.primaries):
        score -= 8
    if checks.verified:
        score -= 5
    return int(clamp(score, 5, 100))


def _estimate_minutes(difficulty: int) -> tuple[int, int]:
    base = 20 + difficulty * 1.6
    low = max(15, int(round(base * 0.6 / 15) * 15))
    high = max(low + 15, int(round(base * 1.4 / 15) * 15))
    return low, high


def _summary(mapping: Mapping, checks: CheckSet, plan: ContributionPlan) -> str:
    if not mapping.primaries:
        return (
            "No confident code target was found. Start by narrowing the task with a filename, symbol or error message; "
            "the evidence below is a shortlist, not an answer."
        )
    top = mapping.primaries[0]
    anchor = _anchor(top)
    parts = [
        f"Start with `{top.path}` ({anchor.label()})",
        f"{len(mapping.primaries)} primary target(s)",
    ]
    if mapping.tests:
        parts.append(f"{len(mapping.tests)} mapped test file(s)")
    verified_checks = len(checks.verified)
    parts.append(f"{verified_checks} declared check(s)" if verified_checks else "no declared checks")
    parts.append(
        f"estimated {plan.estimated_minutes[0]}-{plan.estimated_minutes[1]} min"
    )
    parts.append(f"confidence: {mapping.confidence_label}")
    return "; ".join(parts) + "."


def _checklist(profile: RepoProfile, root: Path) -> list[Claim]:
    claims: list[Claim] = []
    # Only read checklists from contribution material.  README task lists and
    # issue templates are not PR checklists, and treating them as such would
    # turn unrelated prose into "verified" project requirements.
    candidates = [
        path
        for path in profile.guidelines
        if path.lower().endswith((".md", ".rst", ".txt", ".adoc", ""))
        and any(
            marker in path.lower()
            for marker in ("contribut", "pull_request", "pull-request", "pr_template", "review")
        )
    ]
    seen: set[str] = set()
    for path in candidates[:6]:
        text = read_text(root / path, limit=200_000) or ""
        lines = split_lines(text)
        for index, line in enumerate(lines):
            match = re.match(r"^\s*[-*]\s*\[[ xX]\]\s+(.+?)\s*$", line)
            if not match:
                continue
            item = truncate(match.group(1), 160)
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                Claim(
                    text=f"Project checklist: {item}",
                    status="verified",
                    sources=[Source(path, index + 1, item)],
                )
            )
            if len(claims) >= 10:
                break
        if len(claims) >= 10:
            break

    generic = [
        Claim(
            text="The change includes or references a regression test (or explains why one is not practical).",
            status="suggested",
            basis="review heuristic; no checklist item for this was found in the repository",
        ),
        Claim(
            text="The pull request is scoped to one behavior and contains no unrelated refactors.",
            status="suggested",
            basis="review heuristic",
        ),
        Claim(
            text="Commands you ran are listed, including which ones were declared by the project and which were your own.",
            status="suggested",
            basis="transparency heuristic; RepoScout itself separates verified from suggested checks",
        ),
    ]
    for claim in generic:
        if not any(claim.text.lower() == existing.text.lower() for existing in claims):
            claims.append(claim)
    return claims


def _unknowns(mapping: Mapping, checks: CheckSet, profile: RepoProfile) -> list[str]:
    unknowns: list[str] = []
    if mapping.confidence_label == "low":
        unknowns.append("The target files are a low-confidence shortlist; confirm with a maintainer before coding.")
    if not mapping.tests:
        unknowns.append("No test file maps to the target, so the project's expected test style is unknown.")
    if not checks.verified:
        unknowns.append("No verification command is declared, so acceptance criteria are unknown.")
    if not profile.guidelines:
        unknowns.append("No contribution guide or PR template was found.")
    if not profile.is_git:
        unknowns.append("The directory is not a git repository, so branch/PR conventions cannot be inferred.")
    if profile.test_files == 0:
        unknowns.append("No test files were detected at all; validation may be manual.")
    if any(check.status == "verified" and not check.runnable for check in checks.checks):
        unknowns.append("Some declared checks cannot run here because their tools are not installed.")
    return unknowns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anchor(hit: FileHit) -> Source:
    if hit.snippets:
        snippet = hit.snippets[0]
        return Source(hit.path, snippet.line, snippet.text)
    if hit.symbols:
        symbol = hit.symbols[0]
        return Source(hit.path, symbol.line, symbol.signature)
    return Source(hit.path, 1)


def _run_action(check: Check) -> Action:
    detail = f"Run `{check.command}`"
    if check.tool_notes:
        detail += f" ({'; '.join(check.tool_notes)})"
    return Action(
        kind="run",
        detail=detail,
        status=check.status,
        sources=check.sources[:1],
        basis=check.basis if check.status == "suggested" else None,
        command=check.command,
        check_id=check.id,
    )


def _targeted_test_command(test: FileHit, checks: CheckSet) -> str:
    commands = " ".join(check.command for check in checks.checks if check.status == "verified").lower()
    if test.lang == "python":
        if "unittest" in commands:
            module = test.path[:-3].replace("/", ".") if test.path.endswith(".py") else test.path
            return f"python -m unittest {module} -v"
        return f"python -m pytest {test.path} -q"
    if test.lang in {"javascript", "typescript", "vue", "svelte"}:
        if "vitest" in commands:
            return f"npx vitest run {test.path}"
        return f"npx jest {test.path}"
    if test.lang == "go":
        return f"go test ./{test.path.rsplit('/', 1)[0] if '/' in test.path else '.'}"
    if test.lang == "rust":
        return f"cargo test --test {Path(test.path).stem}"
    if test.lang == "ruby":
        return f"bundle exec rspec {test.path}"
    if test.lang == "php":
        return f"php vendor/bin/phpunit {test.path}"
    return f"run the focused test in {test.path}"


def _suggest_test_path(primary: str, lang: str, profile: RepoProfile, test_paths: list[str]) -> str:
    stem = Path(primary).stem
    existing_dirs = sorted({str(Path(path).parent).replace("\\", "/") for path in test_paths})
    base = "tests" if not existing_dirs else (common_ancestor(existing_dirs) or existing_dirs[0])
    if base == ".":
        base = ""
    if lang == "python":
        candidate = f"{base + '/' if base else ''}test_{stem}.py"
    elif lang in {"javascript", "typescript", "vue", "svelte"}:
        extension = ".ts" if lang == "typescript" else ".js"
        candidate = f"{base + '/' if base else ''}{stem}.test{extension}"
    elif lang == "go":
        candidate = str(Path(primary).with_name(f"{stem}_test.go")).replace("\\", "/")
    elif lang == "rust":
        candidate = f"tests/{stem}.rs"
    elif lang in {"java", "kotlin", "csharp"}:
        candidate = f"{base + '/' if base else ''}{stem}Test{Path(primary).suffix}"
    elif lang == "ruby":
        candidate = f"spec/{stem}_spec.rb"
    else:
        candidate = f"{base + '/' if base else ''}{stem}_test{Path(primary).suffix or '.txt'}"
    return candidate


def _profile_source(profile: RepoProfile) -> str:
    for manifest in profile.manifests:
        return manifest.path
    return profile.docs[0] if profile.docs else "."
