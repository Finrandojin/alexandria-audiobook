"""No absolute path to one person's machine may appear in executable source.

WHY THIS IS A TEST AND NOT A CONVENTION. 77 experiment scripts opened with

    REPO = "/home/fakemitch/pinokio/api/alexandria-audiobook2.git"

on a project whose GPU work runs on a disposable cloud instance where that
directory does not exist. It worked there only through a symlink recorded
nowhere, and on 2026-08-04 a failure inside a file physically living under
/home/ubuntu printed a /home/fakemitch traceback.

That was fixed. Then the fix was described as complete, and it was not: the
`REPO` assignment had been migrated while **41 other absolute references
survived**, 33 of them `sys.path.insert` lines. An external reviewer found them
the same day. A grep is exactly the kind of check a person does once and a test
does every time, which is why this exists rather than another careful sweep.

WHAT COUNTS AS MACHINE-SPECIFIC. A path rooted at a named user's home
directory. Not `~`, not `os.path.expanduser`, not an environment variable with
a sensible default - those relocate. A literal `/home/<someone>/...` does not.

Documentation is deliberately NOT covered. A work log recording that a
traceback showed `/home/fakemitch/...` is reporting evidence, and rewriting
history to look portable would be worse than the original problem.
"""
import os
import re
import unittest

APP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(APP)

# `/home/<user>/` and the macOS equivalent. `/home/runner` and similar CI roots
# are not exempted; if one ever appears it should be made relocatable too.
MACHINE_PATH = re.compile(r"['\"](/home/[a-z][\w.-]*|/Users/[a-z][\w.-]*)/")

# Every TRACKED Python file, not a hand-listed set of directories. The first
# version of this test named three directories and therefore could not see
# `ab_test_runtime/pipeline_repeats/score_repeats.py`, which had the same
# hard-coded root - the reviewer pointed that out. "Directories I thought of"
# is the same failure mode as the manual sweep this test replaces.
#
# `git ls-files` is the definition of tracked. Falling back to a walk keeps the
# test meaningful in an exported tree with no git, and vendored environments
# are excluded there because they are not ours to fix.
EXCLUDED_DIRS = {"env", "venv", ".analysis_env", "preparer_env", "booknlp",
                 "node_modules", ".git", "__pycache__", "site-packages"}

# Paths that are genuinely outside the repository and cannot be derived from
# __file__. Each must still be overridable; the test enforces that rather than
# waving it through.
ALLOWED_IF_OVERRIDABLE = {"profile_vram.py"}

# Files whose path literals ARE the test data. `redact_text` cannot be
# tested for collapsing home directories without writing one down.
PATHS_ARE_THE_SUBJECT = {"test_voicelab_diagnostics.py"}


def source_files():
    import subprocess
    try:
        out = subprocess.run(["git", "-C", REPO, "ls-files", "*.py"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            for rel in out.stdout.splitlines():
                path = os.path.join(REPO, rel)
                if os.path.exists(path):
                    yield path
            return
    except Exception:                                   # noqa: BLE001
        pass
    for root, dirs, names in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for name in sorted(names):
            if name.endswith(".py"):
                yield os.path.join(root, name)


class TestNoMachinePaths(unittest.TestCase):

    def test_no_home_directory_literals_in_source(self):
        offenders = []
        for path in source_files():
            base = os.path.basename(path)
            if base == os.path.basename(__file__) or \
                    base in PATHS_ARE_THE_SUBJECT:
                continue          # these quote them on purpose
            with open(path, encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if line.lstrip().startswith("#"):
                        continue  # a comment recording history is evidence
                    if MACHINE_PATH.search(line):
                        rel = os.path.relpath(path, REPO)
                        if os.path.basename(path) in ALLOWED_IF_OVERRIDABLE:
                            continue
                        offenders.append(f"{rel}:{n}: {line.strip()[:90]}")
        self.assertEqual(
            [], offenders,
            "absolute machine paths in source - derive from __file__, or take "
            "an environment variable with a relocatable default:\n  "
            + "\n  ".join(offenders))

    def test_allowed_files_take_an_environment_override(self):
        """An exemption is only acceptable while it stays configurable."""
        for name in ALLOWED_IF_OVERRIDABLE:
            path = os.path.join(REPO, "app", "experiments", name)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            self.assertTrue(
                "os.environ.get" in src or "expanduser" in src,
                f"{name} is exempted from the path rule but offers no override")

    def test_the_pattern_actually_matches_the_defect(self):
        """A guard that cannot fail is not a guard.

        This is the literal line that shipped in 77 files.
        """
        shipped = ('REPO = "/home/fakemitch/pinokio/api/'
                   'alexandria-audiobook2.git"')
        self.assertTrue(MACHINE_PATH.search(shipped))
        self.assertTrue(MACHINE_PATH.search(
            'sys.path.insert(0, "/home/fakemitch/pinokio/api/x.git/app")'))

    def test_the_pattern_does_not_match_relocatable_forms(self):
        for ok in ['os.path.expanduser("~/.lmstudio/bin/lms")',
                   'os.environ.get("LMS_BIN", "")',
                   'REPO = os.path.dirname(os.path.abspath(__file__))',
                   'path = os.path.join(REPO, "app")',
                   'url = "http://127.0.0.1:8090/v1"']:
            self.assertIsNone(MACHINE_PATH.search(ok), ok)


if __name__ == "__main__":
    unittest.main()
