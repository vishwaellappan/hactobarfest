# RepoScout

RepoScout is a small, dependency-free Python agent that answers three questions
about an unfamiliar open-source repository:

1. **Which code does this task touch?** It ranks files with weighted BM25 over
   file text, symbols and paths, then adds one hop of the resolved import graph.
2. **What is the evidence?** Every ranked file carries reasons and, where
   possible, exact `path:line` snippets.
3. **How do I know a change is acceptable?** It separates commands **declared by
   the repository** from commands it merely **suggests**, and it never runs
   anything without `--yes`.

It also emits a beginner-sized contribution plan with a PR checklist, time
estimate and an explicit list of unknowns.

```
VERIFIED  = a file and line in this repository states it
SUGGESTED = RepoScout inferred it; confirm before acting
```

That distinction is enforced in code: `Claim`, `Check` and `Action` raise
`ValueError` when a verified item has no `Source` or a suggested item has no
`basis`.

## Install

RepoScout targets **Python 3.11+** and uses only the standard library.

```bash
git clone <your-fork>
cd reposcout
python -m reposcout --version
```

There is nothing to install. Add the repository root to `PYTHONPATH` or run the
commands from the checkout. A `pyproject.toml` is included if you prefer
`pip install -e .`.

## Quick start

```bash
# Build .reposcout/index.sqlite3
python -m reposcout index .

# Describe the project and its verification setup
python -m reposcout overview

# Map an issue to files, with evidence
python -m reposcout map "config loader crashes when the file is missing"

# Produce a Markdown brief, or a self-contained HTML report
python -m reposcout plan "add a timeout option to the HTTP client"
python -m reposcout plan "add a timeout option to the HTTP client" --format html
python -m reposcout plan "add a timeout option to the HTTP client" -o brief.html

# List declared vs suggested checks
python -m reposcout checks

# Execute declared+runnable checks only, after explicit confirmation
python -m reposcout checks --run --yes
```

Use `-C path` to point at another checkout, `--json` on `index`, `overview`,
`map` and `checks` for machine-readable output, and `--refresh` to rebuild the
index after editing.

`checks --run --yes` exits non-zero when any executed check fails (or times
out), so it can gate a script. Listing checks, or running them without
`--run`, always exits 0 unless there is an argument or repository error.

## What "verified" means here

| Item | Verified means | It does not mean |
| --- | --- | --- |
| File mapping | The file exists and the cited `path:line` produced the match | The file is the right place for the fix |
| Check | The command is written in a config, CI workflow, Makefile, package script or contributor doc | The command passed, or that its tools are installed |
| Read action | The file/line exists in the index | Reading it will explain the bug |
| Edit action | Never verified | A proposed change site is always a hypothesis |
| Checklist item | A contribution document contains that checkbox | The item applies to your specific PR |

Check discovery covers `pyproject.toml`, `setup.cfg`, `tox.ini`, `noxfile.py`,
`Makefile`, `package.json`, `Cargo.toml`, `go.mod`, `pom.xml`, Gradle,
`composer.json`, common CI providers, and fenced commands in contribution or
development docs. Example blocks and directory trees are ignored, and
ecosystem defaults (for example `cargo test` because a `Cargo.toml` exists) are
always `suggested`.

## Output

`plan` renders Markdown to stdout, JSON, or a **single-file HTML** report with
embedded CSS, light/dark support, evidence blocks, confidence meters, check
tables and the agent trace. HTML mode without `-o` writes to
`.reposcout/reports/<task-slug>.html`.

Pass `--run-checks --yes` to execute declared **and** runnable checks; the report
then includes a real execution table (PASS/FAIL, exit code, duration, truncated
output) alongside the declaration table. Without that flag, the report says
plainly that nothing was executed. The JSON shape includes `check_results` for
the same data.

The JSON schema is the `Report.to_dict()` shape in `reposcout/models.py`, so it
is stable enough to feed into another tool.

## How it works

```
discover ──► parse ──► SQLite index ──► BM25 + boosts ──► graph hop ──► evidence
                                                   │
                                                   ├──► check discovery (verified/suggested)
                                                   └──► contribution plan + report
```

- **Discovery** walks the tree, honours `.gitignore`, skips vendored/generated
  directories, lockfiles and credential files, and profiles manifests, docs,
  contribution guides, CI and tests.
- **Parsing** uses Python's `ast`; other languages use conservative line
  patterns with exact line numbers. Unsupported languages are tokenized but not
  symbol-parsed.
- **Indexing** stores files, symbols, imports, resolved edges and per-field term
  frequencies in SQLite. No server, no embeddings, no network.
- **Retrieval** combines BM25 (`body`, `symbol`, `path` fields), exact
  identifier/path boosts, one-hop import expansion, test association and
  phrase boosts.
- **Planning** turns the mapping and checks into seven steps and records each
  action's provenance.

### Optional language model

The deterministic pipeline is the default. `--llm` enables an OpenAI-compatible
chat endpoint; it can propose additional search terms and a short narrative.
Terms are used only if they already exist in the index, and narrative citations
must reference retrieved files, otherwise the text is discarded.

```bash
export REPOSCOUT_LLM_BASE_URL=https://api.openai.com/v1
export REPOSCOUT_LLM_API_KEY=...
export REPOSCOUT_LLM_MODEL=gpt-4o-mini
python -m reposcout plan "..." --llm
```

No key? RepoScout records a note and continues deterministically.

## Limitations

- Retrieval is lexical, not semantic. Paraphrased tasks can rank poorly; adding a
  filename, symbol, error type or error text helps a lot.
- Symbol extraction outside Python is heuristic. Dynamic imports, generated
  files and anything outside the index are invisible.
- The import graph is one hop and resolves common conventions only. It is
  evidence for investigation, not proof of a dependency.
- The plan is advice. It cannot know product intent, maintainer preferences or
  hidden acceptance criteria; those are listed under unknowns.
- Declared commands can be stale. Runnability is reported separately, and a
  command is never executed without `--run --yes`.

## Development

Run the test suite with the same command the project declares in CI:

```bash
python -m unittest discover -s tests -v
```

Layout:

```
reposcout/
  agent.py       orchestration and optional LLM grounding
  checks.py      command discovery, auditing, execution
  cli.py         argparse interface
  indexing.py    SQLite schema, import resolution, BM25
  models.py      provenance-enforcing dataclasses
  parsing.py     Python AST + heuristic language parsers
  planning.py    contribution plan and PR checklist
  rendering.py   Markdown, JSON, self-contained HTML
  retrieval.py   task-to-code mapping and evidence
  scanning.py    tree walk, ignore rules, repository profile
  tokens.py      tokenizing, stemming, synonyms, task signals
  util.py        small shared helpers
tests/
```

## License

MIT. See [LICENSE](LICENSE).
