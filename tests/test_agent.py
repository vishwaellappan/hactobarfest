import unittest

from reposcout.agent import AgentOptions, RepoAgent
from support import python_repo


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = python_repo()
        cls.agent = RepoAgent(cls.repo.root, AgentOptions(max_primary=4))
        cls.agent_run = cls.agent.run(
            "missing configuration file should raise a clear ConfigError with the path"
        )

    @classmethod
    def tearDownClass(cls):
        cls.agent_run.close()
        cls.repo.cleanup()

    def test_report_has_all_sections(self):
        report = self.agent_run.report
        self.assertEqual(report.task, "missing configuration file should raise a clear ConfigError with the path")
        self.assertTrue(report.mapping.primaries)
        self.assertEqual(report.mapping.primaries[0].path, "samplelib/config.py")
        self.assertEqual(len(report.plan.steps), 7)
        self.assertTrue(report.plan.checklist)
        self.assertTrue(report.limitations)
        self.assertTrue(report.trace)
        self.assertEqual(report.version, "0.1.0")

    def test_every_action_and_claim_has_provenance(self):
        for step in self.agent_run.report.plan.steps:
            self.assertTrue(step.actions, f"step {step.number} has no actions")
            for action in step.actions:
                if action.status == "verified":
                    self.assertTrue(action.sources, action.detail)
                else:
                    self.assertTrue(action.basis, action.detail)
            for claim in step.rationale:
                if claim.status == "verified":
                    self.assertTrue(claim.sources, claim.text)
                else:
                    self.assertTrue(claim.basis, claim.text)
        for claim in self.agent_run.report.plan.checklist:
            if claim.status == "verified":
                self.assertTrue(claim.sources, claim.text)
            else:
                self.assertTrue(claim.basis, claim.text)

    def test_plan_uses_real_files_and_checks(self):
        plan = self.agent_run.report.plan
        self.assertIn("samplelib/config.py", plan.files_to_touch)
        self.assertIn("tests/test_config.py", plan.test_files)
        verified_commands = {
            check.command for check in self.agent_run.report.checks if check.status == "verified"
        }
        self.assertIn("python -m unittest discover -s tests -v", verified_commands)
        self.assertGreaterEqual(plan.difficulty_score, 5)
        self.assertLessEqual(plan.difficulty_score, 100)
        self.assertGreaterEqual(plan.estimated_minutes[0], 15)

    def test_checklist_includes_project_items(self):
        texts = [claim.text for claim in self.agent_run.report.plan.checklist]
        self.assertTrue(any("regression test is included" in text for text in texts))
        self.assertTrue(any(claim.status == "suggested" for claim in self.agent_run.report.plan.checklist))

    def test_trace_records_the_loop(self):
        tools = [step.tool for step in self.agent_run.report.trace]
        self.assertIn("index", tools)
        self.assertIn("map_task", tools)
        self.assertIn("discover_checks", tools)
        self.assertIn("build_plan", tools)
        self.assertTrue(all(step.index == position for position, step in enumerate(self.agent_run.report.trace, start=1)))

    def test_no_llm_without_opt_in(self):
        self.assertFalse(self.agent_run.report.llm_used)
        self.assertFalse(any("LLM" in note for note in self.agent_run.report.llm_notes))

    def test_requested_checks_are_executed_and_recorded(self):
        run = RepoAgent(self.repo.root, AgentOptions(max_primary=2)).run(
            "missing configuration file raises ConfigError", run_checks=True
        )
        try:
            results = run.report.check_results
            self.assertTrue(results)
            self.assertTrue(any(result.command.startswith("python -m unittest") for result in results))
            self.assertTrue(all(result.ok for result in results), [r.output for r in results])
        finally:
            run.close()


if __name__ == "__main__":
    unittest.main()
