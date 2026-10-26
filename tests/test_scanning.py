import unittest

from reposcout.scanning import discover, is_secret_filename
from support import RepoFixture, python_repo


class SecretTests(unittest.TestCase):
    def test_secret_files_are_rejected(self):
        self.assertTrue(is_secret_filename(".env"))
        self.assertTrue(is_secret_filename(".env.production"))
        self.assertTrue(is_secret_filename("id_rsa"))
        self.assertTrue(is_secret_filename("server.pem"))

    def test_documentation_variants_are_allowed(self):
        self.assertFalse(is_secret_filename(".env.example"))
        self.assertFalse(is_secret_filename(".env.sample"))
        self.assertFalse(is_secret_filename("secrets.md"))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.repo = python_repo()
        self.addCleanup(self.repo.cleanup)
        self.scan = discover(self.repo.root)

    def test_skips_secrets_and_gitignored_files(self):
        paths = {item.path for item in self.scan.files}
        self.assertNotIn(".env", paths)
        self.assertIn(".env.example", paths)
        self.assertNotIn("logs/run.log", paths)
        self.assertGreater(self.scan.skipped["ignored"], 0)

    def test_profile_identity_and_structure(self):
        profile = self.scan.profile
        self.assertEqual(profile.name, "samplelib")
        self.assertIn("Load JSON configuration", profile.description)
        self.assertEqual(profile.test_files, 2)
        self.assertTrue(profile.docs)
        self.assertIn("README.md", profile.docs)
        self.assertIn("CONTRIBUTING.md", profile.guidelines)
        self.assertIn(".github/workflows/tests.yml", profile.ci_workflows)
        self.assertEqual(list(profile.languages)[0], "python")
        self.assertFalse(profile.is_git)

    def test_manifest_detection(self):
        kinds = {manifest.kind for manifest in self.scan.profile.manifests}
        self.assertIn("pyproject", kinds)

    def test_test_classification(self):
        by_path = {item.path: item for item in self.scan.files}
        self.assertTrue(by_path["tests/test_config.py"].is_test)
        self.assertFalse(by_path["samplelib/config.py"].is_test)
        self.assertTrue(by_path["README.md"].is_doc)


class GuidelineTests(unittest.TestCase):
    def test_test_support_module_is_not_a_guide(self):
        repo = RepoFixture("reposcout-guides-")
        self.addCleanup(repo.cleanup)
        repo.write("SUPPORT.md", "# Support\n")
        repo.write("tests/support.py", "VALUE = 1\n")
        scan = discover(repo.root)
        self.assertIn("SUPPORT.md", scan.profile.guidelines)
        self.assertNotIn("tests/support.py", scan.profile.guidelines)


if __name__ == "__main__":
    unittest.main()
