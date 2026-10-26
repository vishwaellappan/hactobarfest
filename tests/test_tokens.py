import unittest

from reposcout import tokens as tok


class TokenTests(unittest.TestCase):
    def test_split_identifier(self):
        self.assertEqual(tok.split_identifier("loadConfigFile"), ["load", "config", "file"])
        self.assertEqual(tok.split_identifier("HTTPClient"), ["http", "client"])
        self.assertEqual(tok.split_identifier("snake_case_name"), ["snake", "case", "name"])

    def test_stem_is_conservative(self):
        self.assertEqual(tok.stem("configurations"), "configuration")
        self.assertEqual(tok.stem("loading"), "load")
        self.assertEqual(tok.stem("files"), "file")
        self.assertEqual(tok.stem("class"), "class")

    def test_tokens_drop_filler(self):
        self.assertNotIn("the", tok.tokens("the config file"))
        self.assertIn("config", tok.tokens("the config file"))
        self.assertIn("the", tok.tokens("the config file", include_stopwords=True))

    def test_raw_tokens_keep_surface_forms(self):
        self.assertEqual(tok.raw_tokens("LoadConfig configs"), ["load", "config", "configs"])

    def test_synonyms_are_human_readable(self):
        synonyms = tok.synonyms_for("config")
        self.assertIn("settings", synonyms)
        self.assertIn("configuration", synonyms)

    def test_task_signals(self):
        task = (
            "`load_config` crashes in `samplelib/config.py` with ConfigError"
        )
        signals = tok.extract_task_signals(task)
        self.assertIn("load_config", signals.identifiers)
        self.assertIn("ConfigError", signals.identifiers)
        self.assertIn("samplelib/config.py", signals.paths)
        self.assertIn("config", signals.identifiers)

    def test_quoted_phrase(self):
        signals = tok.extract_task_signals('show "configuration file not found" instead')
        self.assertIn("configuration file not found", signals.phrases)

    def test_test_case_classifier_excludes_helpers(self):
        self.assertTrue(tok.looks_like_test_case_path("tests/test_config.py"))
        self.assertTrue(tok.looks_like_test_case_path("src/math.test.js"))
        self.assertTrue(tok.looks_like_test_case_path("internal/store/store_test.go"))
        self.assertFalse(tok.looks_like_test_case_path("tests/support.py"))
        self.assertFalse(tok.looks_like_test_case_path("tests/conftest.py"))

    def test_path_classifiers(self):
        self.assertTrue(tok.looks_like_test_path("tests/test_config.py"))
        self.assertTrue(tok.looks_like_test_path("src/math.test.js"))
        self.assertTrue(tok.looks_like_test_path("internal/store/store_test.go"))
        self.assertFalse(tok.looks_like_test_path("src/contest.py"))
        self.assertTrue(tok.looks_like_doc_path("README.md"))
        self.assertTrue(tok.looks_like_doc_path("docs/usage.md"))
        self.assertTrue(tok.looks_like_doc_path("SUPPORT.md"))
        self.assertFalse(tok.looks_like_doc_path("tests/support.py"))
        self.assertTrue(tok.looks_like_config_path("pyproject.toml"))
        self.assertTrue(tok.looks_like_config_path(".github/workflows/tests.yml"))


if __name__ == "__main__":
    unittest.main()
