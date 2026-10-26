import unittest

from reposcout.indexing import ensure_index, index_overview
from reposcout.retrieval import RetrievalOptions, map_task
from support import python_repo


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = python_repo()
        cls.index, cls.built = ensure_index(cls.repo.root)
        cls.indexed_at = cls.index.indexed_at

    @classmethod
    def tearDownClass(cls):
        cls.index.close()
        cls.repo.cleanup()

    def test_index_contains_files_and_symbols(self):
        paths = self.index.file_paths()
        self.assertIn("samplelib/config.py", paths)
        self.assertIn("tests/test_config.py", paths)
        self.assertNotIn(".env", paths)
        self.assertGreater(self.index.file_count, 5)
        found = self.index.find_symbols(["load_config", "ConfigError"])
        names = {symbol.name for _path, symbol in found}
        self.assertIn("load_config", names)
        self.assertIn("ConfigError", names)

    def test_relative_imports_resolve(self):
        cli_id = self.index.file_id("samplelib/cli.py")
        self.assertIsNotNone(cli_id)
        imported = self.index.imported_paths(cli_id or -1)
        self.assertIn("samplelib/config.py", imported)
        self.assertIn("samplelib/client.py", imported)
        importers = self.index.importers(self.index.file_id("samplelib/config.py") or -1)
        self.assertIn("samplelib/__init__.py", importers)

    def test_search_and_stats(self):
        hits = self.index.search({"configuration": 1.0, "missing": 0.8})
        self.assertTrue(hits)
        self.assertIn("samplelib/config.py", [hit.path for hit in hits[:5]])
        stats = index_overview(self.index)
        self.assertGreater(stats["symbols"], 0)
        self.assertEqual(stats["languages"][0]["lang"], "python")

    def test_ensure_index_reuses_existing_database(self):
        reopened, built = ensure_index(self.repo.root)
        try:
            self.assertFalse(built)
            self.assertEqual(reopened.indexed_at, self.indexed_at)
        finally:
            reopened.close()

    def test_map_task_returns_evidence_and_tests(self):
        mapping = map_task(
            self.index,
            "missing configuration file raises ConfigError when loading JSON",
            options=RetrievalOptions(max_primary=4),
        )
        self.assertTrue(mapping.primaries)
        self.assertEqual(mapping.primaries[0].path, "samplelib/config.py")
        self.assertIn("samplelib/config.py", [hit.path for hit in mapping.primaries])
        self.assertTrue(mapping.tests)
        self.assertIn("tests/test_config.py", [hit.path for hit in mapping.tests])
        top = mapping.primaries[0]
        self.assertTrue(top.snippets, "the top target must carry line-level evidence")
        self.assertTrue(any(reason.kind in {"match", "symbol", "path"} for reason in top.reasons))
        self.assertGreaterEqual(mapping.confidence, 0.45)
        self.assertIn(mapping.confidence_label, {"medium", "high"})

    def test_map_task_explains_missing_terms(self):
        mapping = map_task(self.index, "the and of to a")
        self.assertFalse(mapping.primaries)
        self.assertEqual(mapping.confidence_label, "low")
        self.assertTrue(mapping.notes)

    def test_confidence_does_not_claim_a_missing_identifier(self):
        mapping = map_task(self.index, "make loading configuration friendlier for newcomers")
        if mapping.identifiers:
            self.skipTest("task unexpectedly produced explicit identifiers")
        rationale = " ".join(mapping.confidence_rationale)
        self.assertNotIn("names a path or symbol", rationale)
        self.assertIn("No explicit symbol or filename", " ".join(mapping.notes))


if __name__ == "__main__":
    unittest.main()
