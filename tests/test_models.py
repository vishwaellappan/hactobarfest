import unittest

from reposcout.models import Action, Check, Claim, Source


class SourceTests(unittest.TestCase):
    def test_label_uses_line_when_present(self):
        self.assertEqual(Source("a/b.py").label(), "a/b.py")
        self.assertEqual(Source("a/b.py", 12).label(), "a/b.py:12")


class ClaimTests(unittest.TestCase):
    def test_verified_requires_source(self):
        with self.assertRaises(ValueError):
            Claim("the parser exists", "verified")

    def test_suggested_requires_basis(self):
        with self.assertRaises(ValueError):
            Claim("the parser should move", "suggested")

    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):
            Claim("x", "probably", sources=[Source("x")])

    def test_valid_claims(self):
        verified = Claim("exists", "verified", sources=[Source("a.py", 1)])
        suggested = Claim("could move", "suggested", basis="review heuristic")
        self.assertEqual(verified.sources[0].line, 1)
        self.assertEqual(suggested.status, "suggested")


class CheckTests(unittest.TestCase):
    def test_verified_check_requires_source(self):
        with self.assertRaises(ValueError):
            Check("test-1", "python -m unittest", "test", "verified")

    def test_suggested_check_requires_basis(self):
        with self.assertRaises(ValueError):
            Check("test-1", "pytest", "test", "suggested")

    def test_valid_check(self):
        check = Check(
            "test-1",
            "python -m unittest",
            "test",
            "verified",
            sources=[Source("pyproject.toml", 4)],
            declared_in="pyproject.toml:4",
        )
        self.assertEqual(check.declared_in, "pyproject.toml:4")


class ActionTests(unittest.TestCase):
    def test_verified_action_requires_source(self):
        with self.assertRaises(ValueError):
            Action("edit", "change it", status="verified")

    def test_suggested_action_requires_basis(self):
        with self.assertRaises(ValueError):
            Action("edit", "change it")

    def test_valid_action(self):
        action = Action(
            "read", "read it", status="verified", sources=[Source("a.py", 3)]
        )
        self.assertEqual(action.sources[0].line, 3)


if __name__ == "__main__":
    unittest.main()
