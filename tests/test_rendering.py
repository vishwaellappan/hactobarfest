import json
import tempfile
import unittest
from pathlib import Path

from reposcout.agent import AgentOptions, RepoAgent
from reposcout.models import CheckResult
from reposcout.rendering import render_html, render_json, render_markdown, write_report
from support import python_repo


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = python_repo()
        cls.agent = RepoAgent(cls.repo.root, AgentOptions(max_primary=3))
        cls.agent_run = cls.agent.run("configuration file missing should raise ConfigError")

    @classmethod
    def tearDownClass(cls):
        cls.agent_run.close()
        cls.repo.cleanup()

    def test_markdown_separates_provenance(self):
        markdown = render_markdown(self.agent_run.report)
        self.assertIn("VERIFIED", markdown)
        self.assertIn("SUGGESTED", markdown)
        self.assertIn("## Contribution plan", markdown)
        self.assertIn("## Checks", markdown)
        self.assertIn("samplelib/config.py", markdown)

    def test_html_is_self_contained_and_escapes_content(self):
        report = self.agent_run.report
        original = report.task
        report.task = '<script>alert("xss")</script> load_config'
        try:
            document = render_html(report)
        finally:
            report.task = original
        self.assertIn("<!doctype html>", document)
        self.assertIn("<style>", document)
        self.assertNotIn('<script>alert("xss")</script>', document)
        self.assertIn("&lt;script&gt;", document)
        self.assertNotIn("http://", document.split("<style>")[0])

    def test_json_roundtrip(self):
        data = json.loads(render_json(self.agent_run.report))
        self.assertEqual(data["version"], "0.1.0")
        self.assertIn("mapping", data)
        self.assertIn("checks", data)
        self.assertIn("plan", data)
        self.assertIn("check_results", data)
        self.assertTrue(data["checks"])
        for check in data["checks"]:
            self.assertIn(check["status"], {"verified", "suggested"})

    def test_execution_results_are_rendered(self):
        report = self.agent_run.report
        report.check_results = [
            CheckResult(
                check_id="test-1",
                command="python -m unittest discover -s tests -v",
                returncode=1,
                duration_ms=12,
                output="AssertionError: boom",
            )
        ]
        try:
            markdown = render_markdown(report)
            document = render_html(report)
        finally:
            report.check_results = []
        self.assertIn("Execution results", markdown)
        self.assertIn("FAIL", markdown)
        self.assertIn("AssertionError: boom", markdown)
        self.assertIn("Execution results", document)
        self.assertIn("FAIL", document)
        self.assertIn("AssertionError: boom", document)

    def test_write_report_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "brief.html"
            written = write_report(self.agent_run.report, target, "html")
            self.assertTrue(written.exists())
            self.assertGreater(written.stat().st_size, 5000)
            text = written.read_text(encoding="utf-8")
            self.assertIn("RepoScout", text)

    def test_write_report_rejects_unknown_format(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_report(self.agent_run.report, Path(directory) / "x.txt", "txt")


if __name__ == "__main__":
    unittest.main()
