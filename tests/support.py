"""Shared fixtures for the RepoScout test suite.

Fixtures are created in the system temporary directory and removed by the test
that created them.  Nothing here touches the RepoScout checkout itself.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RepoFixture:
    def __init__(self, prefix: str = "reposcout-test-") -> None:
        self.root = Path(tempfile.mkdtemp(prefix=prefix))

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def python_repo() -> RepoFixture:
    """A tiny Python project with a known ranking target and declared checks."""
    repo = RepoFixture("reposcout-py-")
    repo.write(
        "README.md",
        "# SampleLib\n\n"
        "SampleLib loads JSON configuration files and makes small HTTP requests.\n\n"
        "## Usage\n\n"
        "Run the command line interface to load a configuration.\n",
    )
    repo.write(
        "pyproject.toml",
        '[project]\n'
        'name = "samplelib"\n'
        'version = "0.3.0"\n'
        'description = "Load JSON configuration files with clear errors."\n'
        'license = { text = "MIT" }\n'
        "\n"
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n'
        "\n"
        "[tool.ruff]\n"
        "line-length = 100\n",
    )
    repo.write(
        "CONTRIBUTING.md",
        "# Contributing\n\n"
        "Use a topic branch and keep patches small.\n\n"
        "## Pull request checklist\n\n"
        "- [ ] A regression test is included.\n"
        "- [ ] The change is limited to one behavior.\n"
        "- [ ] The PR description lists the commands that were run.\n\n"
        "Run the suite with:\n\n"
        "```bash\n"
        "python -m unittest discover -s tests -v\n"
        "```\n",
    )
    repo.write(
        ".github/workflows/tests.yml",
        "name: tests\n"
        "on: [push, pull_request]\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n"
        '          python-version: "3.12"\n'
        "      - name: Unit tests\n"
        "        run: python -m unittest discover -s tests -v\n",
    )
    repo.write(
        "samplelib/__init__.py",
        '"""SampleLib package."""\n\n'
        "from .config import ConfigError, load_config\n\n"
        '__all__ = ["ConfigError", "load_config"]\n',
    )
    repo.write(
        "samplelib/config.py",
        '"""Configuration loading helpers."""\n\n'
        "import json\n"
        "from pathlib import Path\n\n"
        "DEFAULT_TIMEOUT = 30\n\n\n"
        "class ConfigError(Exception):\n"
        '    """Raised when a configuration file cannot be read."""\n\n\n'
        "def load_config(path):\n"
        '    """Load a JSON configuration file from disk."""\n'
        "    file_path = Path(path)\n"
        "    if not file_path.exists():\n"
        '        raise ConfigError(f"configuration file not found: {file_path}")\n'
        '    with file_path.open("r", encoding="utf-8") as handle:\n'
        "        return json.load(handle)\n",
    )
    repo.write(
        "samplelib/client.py",
        '"""HTTP client with a configurable timeout."""\n\n'
        "from .config import DEFAULT_TIMEOUT\n\n\n"
        "class HttpClient:\n"
        '    """Minimal client wrapper used by the tests."""\n\n'
        "    def __init__(self, timeout=DEFAULT_TIMEOUT):\n"
        "        self.timeout = timeout\n\n"
        "    def get(self, url):\n"
        '        """Return the request metadata for a URL."""\n'
        '        return {"url": url, "timeout": self.timeout}\n',
    )
    repo.write(
        "samplelib/cli.py",
        '"""Command line entry point."""\n\n'
        "from .client import HttpClient\n"
        "from .config import load_config\n\n\n"
        "def main(argv):\n"
        '    """Load a configuration file and make one request."""\n'
        "    config = load_config(argv[1])\n"
        '    client = HttpClient(timeout=config.get("timeout", 30))\n'
        '    return client.get(config["url"])\n',
    )
    repo.write(
        "tests/test_config.py",
        "import json\n"
        "import tempfile\n"
        "import unittest\n"
        "from pathlib import Path\n\n"
        "from samplelib.config import ConfigError, load_config\n\n\n"
        "class LoadConfigTests(unittest.TestCase):\n"
        "    def test_missing_file_raises_config_error(self):\n"
        '        with self.assertRaises(ConfigError):\n'
        '            load_config("/definitely/not/here.json")\n\n'
        "    def test_reads_json_object(self):\n"
        "        with tempfile.TemporaryDirectory() as directory:\n"
        '            path = Path(directory) / "config.json"\n'
        '            path.write_text(json.dumps({"timeout": 5}), encoding="utf-8")\n'
        '            self.assertEqual(load_config(path)["timeout"], 5)\n\n\n'
        'if __name__ == "__main__":\n'
        "    unittest.main()\n",
    )
    repo.write(
        "tests/test_client.py",
        "import unittest\n\n"
        "from samplelib.client import HttpClient\n\n\n"
        "class HttpClientTests(unittest.TestCase):\n"
        "    def test_timeout_is_kept(self):\n"
        "        client = HttpClient(timeout=12)\n"
        '        self.assertEqual(client.get("https://example.test")["timeout"], 12)\n\n\n'
        'if __name__ == "__main__":\n'
        "    unittest.main()\n",
    )
    repo.write("docs/usage.md", "# Usage\n\nCall `load_config` before constructing the client.\n")
    repo.write(".gitignore", ".venv/\n*.log\n.reposcout/\n")
    repo.write(".env", "SAMPLELIB_API_KEY=super-secret-value\n")
    repo.write(".env.example", "SAMPLELIB_API_KEY=replace-me\n")
    repo.write("logs/run.log", "this file is ignored\n")
    return repo


def js_repo() -> RepoFixture:
    repo = RepoFixture("reposcout-js-")
    repo.write("README.md", "# Tiny JS project\n\nA package used to test script discovery.\n")
    repo.write(
        "package.json",
        json.dumps(
            {
                "name": "tiny-js",
                "version": "1.0.0",
                "description": "Tiny JavaScript fixture",
                "license": "MIT",
                "scripts": {
                    "test": "jest",
                    "lint": "eslint .",
                    "build": "tsc",
                },
                "devDependencies": {"jest": "^29.0.0", "eslint": "^8.0.0"},
            },
            indent=2,
        ),
    )
    repo.write("src/math.js", "export function add(a, b) {\n  return a + b;\n}\n")
    repo.write(
        "src/math.test.js",
        'import { add } from "./math.js";\n\n'
        'test("adds numbers", () => {\n'
        "  expect(add(1, 2)).toBe(3);\n"
        "});\n",
    )
    repo.write("yarn.lock", "# lockfile placeholder\n")
    return repo
