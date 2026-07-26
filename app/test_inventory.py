import importlib
import json
from pathlib import Path
import subprocess
import unittest


INVENTORY_PATH = Path(__file__).with_name("unit_test_inventory.json")
EXCLUDED_TEST_MODULES = {"test_api"}  # Script-style live API suite, not unittest.


def _tracked_test_modules():
    """Module stems for test files git tracks, or None if git cannot answer.

    This working tree is shared between concurrent sessions on different
    branches, so an untracked test file belonging to someone else's branch sits
    beside ours. Discovering from the filesystem swept those into the checked-in
    inventory, and CI - which only has the committed files - then reported every
    one of them as "no longer discovered". That broke the build three times in
    one day. Only files under version control can be in a checked-in inventory.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "test*.py"],
            cwd=Path(__file__).parent, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return {Path(name).stem
            for name in result.stdout.decode("utf-8").split("\0") if name}


def _iter_tests(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _iter_tests(test)
        else:
            yield test


def get_unit_test_inventory():
    """Return every discoverable unittest module and its stable test IDs."""
    tracked = _tracked_test_modules()
    modules = sorted(
        path.stem for path in Path(__file__).parent.glob("test*.py")
        if path.stem not in EXCLUDED_TEST_MODULES
        # Fall back to the filesystem when git cannot answer (a source export,
        # a container without git) rather than reporting an empty inventory.
        and (tracked is None or path.stem in tracked)
    )
    inventory = {}
    for module_name in modules:
        module = importlib.import_module(module_name)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        inventory[module_name] = sorted(test.id() for test in _iter_tests(suite))
    return inventory


class TestInventoryTests(unittest.TestCase):
    def test_unit_test_inventory_is_stable(self):
        from update_test_inventory import get_inventory_differences

        expected = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        actual = get_unit_test_inventory()

        self.assertNotIn("test_api", actual)
        differences = get_inventory_differences(expected, actual)
        self.assertEqual([], differences, "Unit test inventory drift:\n" + "\n".join(differences))

    def test_inventory_differences_are_compact_and_actionable(self):
        from update_test_inventory import get_inventory_differences

        differences = get_inventory_differences(
            {"test_example": ["test_example.Example.test_removed"]},
            {"test_example": ["test_example.Example.test_added"]},
        )
        self.assertEqual([
            "Missing from inventory: test_example.Example.test_added",
            "No longer discovered: test_example.Example.test_removed",
        ], differences)

    def test_inventory_writer_is_deterministic_and_check_is_read_only(self):
        import tempfile
        from unittest.mock import patch
        import update_test_inventory as updater

        inventory = {"test_z": ["z"], "test_a": ["a"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "inventory.json")
            with patch.object(updater, "get_unit_test_inventory", return_value=inventory):
                updater.write_inventory(path)
                first = path.read_bytes()
                self.assertLess(first.index(b'"test_a"'), first.index(b'"test_z"'))
                updater.write_inventory(path)
                self.assertEqual(first, path.read_bytes())
                self.assertEqual([], updater.check_inventory(path))
                self.assertEqual(first, path.read_bytes())


class TrackedOnlyDiscoveryTest(unittest.TestCase):
    """The inventory is checked in, so it can only contain tracked files.

    This tree is shared between concurrent sessions on different branches. An
    untracked test file from another branch used to be swept into the inventory
    here and then reported missing by CI, which only sees committed files.
    """

    def test_an_untracked_test_file_is_not_inventoried(self):
        untracked = Path(__file__).with_name("test_zz_untracked_probe.py")
        untracked.write_text(
            "import unittest\n\n\n"
            "class Probe(unittest.TestCase):\n"
            "    def test_probe(self):\n        pass\n",
            encoding="utf-8")
        try:
            self.assertNotIn("test_zz_untracked_probe", get_unit_test_inventory())
        finally:
            untracked.unlink()

    def test_tracked_files_are_still_discovered(self):
        self.assertIn("test_inventory", get_unit_test_inventory())

    def test_discovery_falls_back_when_git_is_unavailable(self):
        # A source export without git must still produce an inventory rather
        # than an empty one.
        import test_inventory
        original = test_inventory._tracked_test_modules
        test_inventory._tracked_test_modules = lambda: None
        try:
            self.assertIn("test_inventory", get_unit_test_inventory())
        finally:
            test_inventory._tracked_test_modules = original


class UntrackedTestWarningTest(unittest.TestCase):
    """Excluding untracked files from the inventory created an ordering trap.

    Write a new test file, regenerate the inventory, then git add and commit:
    the inventory was generated while the file was still untracked, so CI
    discovers tests it does not list and the build goes red. This happened on
    PR #233, one PR after the exclusion was introduced.
    """

    def test_a_new_untracked_test_file_is_reported(self):
        from update_test_inventory import find_untracked_test_modules
        probe = Path(__file__).with_name("test_zz_untracked_warning_probe.py")
        probe.write_text("import unittest\n", encoding="utf-8")
        try:
            self.assertIn(probe.name, find_untracked_test_modules())
        finally:
            probe.unlink()

    def test_a_tracked_file_is_not_reported(self):
        from update_test_inventory import find_untracked_test_modules
        # Asserting the whole tree is clean would fail during normal work -
        # a new test file is untracked exactly when you are writing it, which
        # is when you run the suite most. Assert the behaviour instead.
        self.assertNotIn("test_inventory.py", find_untracked_test_modules())


if __name__ == "__main__":
    unittest.main()
