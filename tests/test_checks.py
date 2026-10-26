import unittest

from reposcout.checks import CheckSet, discover_checks, run_check
from reposcout.models import Check, Source
from reposcout.scanning import discover
from support import RepoFixture, js_repo, python_repo


class PythonCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = python_repo()
        cls.scan = discover(cls.repo.root)
        cls.checks = discover_checks(
            cls.repo.root, cls.scan.profile, [item.path for item in cls.scan.files]
        )

    @classmethod
    def tearDownClass(cls):
        cls.repo.cleanup()

    def test_commands_are_found_and_provenanced(self):
        commands = {check.command for check in self.checks.checks}
        self.assertIn("python -m unittest discover -s tests -v", commands)
        self.assertIn("python -m pytest", commands)
        self.assertIn("ruff check .", commands)
        for check in self.checks.verified:
            self.assertTrue(check.sources, check.command)
            self.assertEqual(check.status, "verified")
        for check in self.checks.suggested:
            self.assertTrue(check.basis, check.command)

    def test_ci_and_docs_merge_into_one_declared_test_command(self):
        check = next(
            check
            for check in self.checks.verified
            if check.command == "python -m unittest discover -s tests -v"
        )
        paths = {source.path for source in check.sources}
        self.assertIn(".github/workflows/tests.yml", paths)
        self.assertIn("CONTRIBUTING.md", paths)
        self.assertEqual(check.kind, "test")

    def test_pytest_config_is_verified_even_without_pytest_installed(self):
        check = next(check for check in self.checks.verified if check.command == "python -m pytest")
        self.assertEqual(check.declared_in.split(":")[0], "pyproject.toml")
        self.assertIn("pyproject.toml", {source.path for source in check.sources})

    def test_suggestion_cannot_be_executed(self):
        suggestion = Check(
            "test-1",
            "python -m pytest",
            "test",
            "suggested",
            basis="inferred from file names",
        )
        with self.assertRaises(ValueError):
            run_check(suggestion, self.repo.root)

    def test_declared_unittest_command_runs(self):
        check = next(
            check
            for check in self.checks.verified
            if check.command == "python -m unittest discover -s tests -v"
        )
        self.assertTrue(check.runnable, check.tool_notes)
        result = run_check(check, self.repo.root, timeout=120)
        self.assertTrue(result.ok, result.output)
        self.assertEqual(result.returncode, 0)
        self.assertIn("OK", result.output)


class JavaScriptCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = js_repo()
        cls.scan = discover(cls.repo.root)
        cls.checks = discover_checks(
            cls.repo.root, cls.scan.profile, [item.path for item in cls.scan.files]
        )

    @classmethod
    def tearDownClass(cls):
        cls.repo.cleanup()

    def test_package_scripts_use_lockfile_runner(self):
        commands = {check.command for check in self.checks.verified}
        self.assertIn("yarn run test", commands)
        self.assertIn("yarn run lint", commands)
        self.assertIn("yarn run build", commands)
        check = next(check for check in self.checks.verified if check.command == "yarn run test")
        self.assertEqual(check.kind, "test")
        self.assertIn("package.json", {source.path for source in check.sources})

    def test_js_runner_is_only_suggested(self):
        suggested = {check.command for check in self.checks.suggested}
        self.assertIn("npx jest", suggested)
        check = next(check for check in self.checks.suggested if check.command == "npx jest")
        self.assertTrue(check.basis)


class DocumentationCheckTests(unittest.TestCase):
    """Fenced blocks are full of examples and directory trees, not commands."""

    def setUp(self):
        self.repo = RepoFixture("reposcout-docs-")
        self.addCleanup(self.repo.cleanup)

    def discover(self):
        scan = discover(self.repo.root)
        return discover_checks(
            self.repo.root, scan.profile, [item.path for item in scan.files]
        )

    def test_usage_examples_are_not_declared_checks(self):
        self.repo.write(
            "README.md",
            "# Tool\n\n"
            "## Quick start\n\n"
            "```bash\n"
            "python -m tool plan --format html\n"
            "```\n",
        )
        self.repo.write("tool/__init__.py", "")
        commands = {check.command for check in self.discover().verified}
        self.assertNotIn("python -m tool plan --format html", commands)

    def test_directory_trees_are_not_commands(self):
        self.repo.write(
            "README.md",
            "# Tool\n\n"
            "## Development\n\n"
            "```bash\n"
            "python -m unittest discover -s tests -v\n"
            "```\n\n"
            "Layout:\n\n"
            "```\n"
            "tool/\n"
            "  core.py\n"
            "```\n",
        )
        self.repo.write("tool/__init__.py", "")
        self.repo.write("tests/test_core.py", "import unittest\n")
        commands = {check.command for check in self.discover().verified}
        self.assertIn("python -m unittest discover -s tests -v", commands)
        self.assertNotIn("tool/", commands)
        self.assertNotIn("core.py", commands)

    def test_contribution_docs_do_not_need_a_heading(self):
        self.repo.write(
            "CONTRIBUTING.md",
            "# Contributing\n\n"
            "```bash\n"
            "python -m unittest discover -s tests\n"
            "```\n",
        )
        self.repo.write("tests/test_core.py", "import unittest\n")
        commands = {check.command for check in self.discover().verified}
        self.assertIn("python -m unittest discover -s tests", commands)


class CheckSetTests(unittest.TestCase):
    def test_partition_helpers(self):
        verified = Check(
            "test-1",
            "python -m unittest",
            "test",
            "verified",
            sources=[Source("README.md", 3)],
        )
        suggested = Check(
            "lint-1", "ruff check .", "lint", "suggested", basis="ruff.toml exists"
        )
        check_set = CheckSet(checks=[verified, suggested])
        self.assertEqual([check.id for check in check_set.verified], ["test-1"])
        self.assertEqual([check.id for check in check_set.suggested], ["lint-1"])
        self.assertEqual(check_set.get("lint-1"), suggested)
        self.assertEqual(set(check_set.by_kind()), {"test", "lint"})


if __name__ == "__main__":
    unittest.main()
