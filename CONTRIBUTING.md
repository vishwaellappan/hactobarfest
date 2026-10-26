# Contributing to RepoScout

Thanks for helping improve RepoScout. The project has no third-party runtime
dependencies, so setup is just a Python 3.11+ interpreter.

## Ground rules

- Keep the deterministic pipeline dependency-free.
- Never blur provenance. New claims, checks and actions must be `verified` with
  a real `Source`, or `suggested` with an explicit `basis`.
- Never execute a command unless the user passed `--run --yes`.
- Add a test for behavior changes. Prefer a focused regression test over a broad
  refactor.

## Development workflow

1. Fork the repository and create a topic branch.
2. Run the test suite before and after your change:

```bash
python -m unittest discover -s tests -v
```

3. If you changed retrieval or parsing, add a fixture repository under `tests/`
   and assert on ranking or extracted symbols, not just on "no exception".
4. Update `README.md` when user-facing behavior changes.

## Pull request checklist

- [ ] Tests cover the new behavior, or the PR explains why a test is impractical.
- [ ] The change is scoped to one behavior and contains no drive-by refactors.
- [ ] New user-facing strings avoid promising certainty that the code cannot prove.
- [ ] Verified output still cites a repository file and line.
- [ ] Suggested output still explains its basis.
- [ ] `python -m unittest discover -s tests -v` passes locally.
- [ ] The PR description lists the commands you ran and their results.

## Adding a language parser

Add a parser function in `reposcout/parsing.py`, register it in `parse_file`,
and make sure every emitted `Symbol` and `ImportRef` carries a real line number.
Heuristics are welcome; silently guessing line numbers is not.

## Reporting a mapping bug

Include the task text, the repository revision, the command you ran, and the
expected vs actual top-ranked files. If possible, attach the `--json` mapping
output; it contains the query terms and per-file reasons needed to reproduce a
ranking problem.
