"""End-to-end tests for the command line interface.

These tests spawn ``python -m reposcout`` as a subprocess so the argparse
wiring, exit codes and JSON envelopes are exercised the same way a user would
exercise them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

from support import PROJECT_ROOT, RepoFixture, python_repo


def run_cli(*arguments: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "reposcout", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = python_repo()

    @classmethod
    def tearDownClass(cls):
        cls.repo.cleanup()

    def test_version(self):
        result = run_cli("--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RepoScout 0.1.0", result.stdout)

    def test_index_accepts_a_positional_repository(self):
        result = run_cli("index", str(self.repo.root), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertGreater(data["files"], 5)
        self.assertGreater(data["symbols"], 0)
        self.assertEqual(data["languages"][0]["lang"], "python")

    def test_map_json_exposes_evidence(self):
        result = run_cli(
            "map",
            "-C",
            str(self.repo.root),
            "--json",
            "missing configuration file raises ConfigError",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        mapping = data["mapping"]
        self.assertEqual(data["task"], "missing configuration file raises ConfigError")
        self.assertTrue(mapping["primaries"])
        self.assertEqual(mapping["primaries"][0]["path"], "samplelib/config.py")
        self.assertIn(mapping["confidence_label"], {"medium", "high"})

    def test_plan_json_contains_checks_and_plan(self):
        result = run_cli(
            "plan",
            "-C",
            str(self.repo.root),
            "--format",
            "json",
            "missing configuration file raises ConfigError",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["task"], "missing configuration file raises ConfigError")
        self.assertEqual(len(data["plan"]["steps"]), 7)
        commands = {
            check["command"] for check in data["checks"] if check["status"] == "verified"
        }
        self.assertIn("python -m unittest discover -s tests -v", commands)

    def test_plan_writes_html_report(self):
        target = self.repo.root / "brief.html"
        result = run_cli(
            "plan",
            "-C",
            str(self.repo.root),
            "--format",
            "html",
            "-o",
            str(target),
            "missing configuration file raises ConfigError",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(target.exists())
        document = target.read_text(encoding="utf-8")
        self.assertIn("<!doctype html>", document)
        self.assertIn("samplelib/config.py", document)

    def test_checks_json_separates_declared_commands(self):
        result = run_cli("checks", "-C", str(self.repo.root), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        verified = {check["command"] for check in data["verified"]}
        self.assertIn("python -m unittest discover -s tests -v", verified)
        for check in data["verified"]:
            self.assertTrue(check["sources"], check["command"])
        for check in data["suggested"]:
            self.assertTrue(check["basis"], check["command"])

    def test_run_requires_confirmation(self):
        result = run_cli("checks", "-C", str(self.repo.root), "--run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--yes", result.stderr)

    def test_missing_task_is_rejected(self):
        result = run_cli("map", "-C", str(self.repo.root))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("task", result.stderr.lower())

    def test_unknown_directory_is_rejected(self):
        result = run_cli("index", str(self.repo.root / "does-not-exist"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a directory", result.stderr)


class CliCheckExecutionTests(unittest.TestCase):
    """Executed checks must fail the command, not just print FAIL."""

    def setUp(self):
        self.repo = RepoFixture("reposcout-fail-")
        self.addCleanup(self.repo.cleanup)
        self.repo.write(
            ".github/workflows/tests.yml",
            "name: tests\n"
            "on: [push, pull_request]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - name: Unit tests\n"
            "        run: python -m unittest discover -s tests -v\n",
        )
        self.repo.write(
            "tests/test_regression.py",
            "import unittest\n\n\n"
            "class RegressionTests(unittest.TestCase):\n"
            "    def test_expected_failure(self):\n"
            "        self.assertEqual(1, 2)\n\n\n"
            'if __name__ == "__main__":\n'
            "    unittest.main()\n",
        )

    def test_failing_declared_check_exits_non_zero(self):
        result = run_cli("checks", "-C", str(self.repo.root), "--run", "--yes")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[FAIL]", result.stdout)
        self.assertIn("python -m unittest discover -s tests -v", result.stdout)

    def test_failing_declared_check_is_fail_in_json(self):
        result = run_cli("checks", "-C", str(self.repo.root), "--run", "--yes", "--json")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(data["runs"])
        self.assertFalse(all(run["ok"] for run in data["runs"]))


if __name__ == "__main__":
    unittest.main()
