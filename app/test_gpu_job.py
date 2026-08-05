"""Behavioural tests for gpu_job.sh, the thing that stops two jobs sharing a GPU.

WHY THESE EXIST. The lock has failed twice in ways that cost real time, and
both times it was verified by hand afterwards rather than by a test:

  - the script called `flock 9` and never checked it, so a failed acquisition
    fell straight through to running the command. `set -e` is deliberately off
    (the wrapped command's exit code has to survive), which is exactly what
    made the unchecked call dangerous.
  - the fixed version was committed locally and the CLOUD BOX KEPT RUNNING THE
    OLD ONE for hours, unnoticed, because nothing compared them.

Manual verification catches a bug once. This catches it every run, on whichever
machine runs the suite.

Each test drives the real script in a temporary directory with its own lock and
queue log, so nothing here touches the actual GPU lock.
"""
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_JOB = os.path.join(REPO, "gpu_job.sh")


@unittest.skipUnless(os.path.exists(GPU_JOB), "gpu_job.sh not present")
class GpuJobTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock = os.path.join(self.tmp.name, "gpu.lock")
        self.qlog = os.path.join(self.tmp.name, "queue.log")

    def tearDown(self):
        self.tmp.cleanup()

    def run_job(self, *argv, path_prefix=None, lock=None, timeout=30):
        env = dict(os.environ, GPU_LOCK=lock or self.lock, GPU_QLOG=self.qlog)
        if path_prefix:
            env["PATH"] = path_prefix + os.pathsep + env["PATH"]
        return subprocess.run(["bash", GPU_JOB, *argv], env=env,
                              capture_output=True, text=True, timeout=timeout)

    def log(self):
        if not os.path.exists(self.qlog):
            return ""
        with open(self.qlog, encoding="utf-8") as fh:
            return fh.read()

    def fake_bin(self, name, body):
        """A directory holding one executable that shadows the real one."""
        d = os.path.join(self.tmp.name, "bin_" + name)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(p, 0o755)
        return d

    # ---------------------------------------------------------------- basics

    def test_a_successful_job_runs_and_logs_ok(self):
        marker = os.path.join(self.tmp.name, "ran")
        r = self.run_job("good", "touch", marker)
        self.assertEqual(r.returncode, 0)
        self.assertTrue(os.path.exists(marker))
        self.assertIn("START    good", self.log())
        self.assertIn("OK       good", self.log())

    def test_queue_log_order_is_queued_then_ident_then_start_then_result(self):
        # IDENT sits between QUEUED and START on purpose: it must describe the
        # code that is ABOUT to run. Recorded after the fact it would be a
        # post-mortem, which is what reading logs already gave us.
        self.run_job("ordered", "true")
        lines = [l.split()[1] for l in self.log().splitlines() if l.strip()]
        self.assertEqual(lines, ["QUEUED", "IDENT", "START", "OK"])

    # ------------------------------------------------------- failure surfaces

    def test_wrapped_failure_propagates_the_exit_code(self):
        """A chained job that fails quietly gets read as a result."""
        r = self.run_job("bad", "bash", "-c", "exit 37")
        self.assertEqual(r.returncode, 37)
        self.assertIn("FAILED   bad rc=37", self.log())
        self.assertNotIn("OK ", self.log())

    def test_failure_is_announced_on_stderr(self):
        r = self.run_job("bad", "bash", "-c", "exit 1")
        self.assertIn("FAILED", r.stderr)

    @unittest.skipUnless(shutil.which("timeout"), "timeout command unavailable")
    def test_timeout_kills_job_releases_lock_and_logs_failure(self):
        """A timed-out GPU job must not poison the queue behind it."""
        r = self.run_job("timed", "timeout", "0.2", "sleep", "10")
        self.assertEqual(124, r.returncode)
        self.assertIn("FAILED   timed rc=124", self.log())
        next_job = self.run_job("after_timeout", "true")
        self.assertEqual(0, next_job.returncode)
        self.assertIn("OK       after_timeout", self.log())

    # ------------------------------------------------------------ the gate

    def test_a_failed_flock_refuses_to_run_the_command(self):
        """THE DEFECT. An unchecked `flock` ran the command anyway."""
        marker = os.path.join(self.tmp.name, "must_not_exist")
        d = self.fake_bin("flock", "#!/bin/bash\nexit 73\n")
        r = self.run_job("gated", "touch", marker, path_prefix=d)
        self.assertFalse(os.path.exists(marker),
                         "command ran despite the lock failing")
        self.assertEqual(r.returncode, 4)
        self.assertIn("LOCK_FAILED", self.log())
        self.assertNotIn("START", self.log())
        self.assertNotIn("OK", self.log())

    def test_an_unopenable_lock_file_refuses_to_run(self):
        marker = os.path.join(self.tmp.name, "must_not_exist")
        bad = os.path.join(self.tmp.name, "no_such_dir", "gpu.lock")
        r = self.run_job("nolock", "touch", marker, lock=bad)
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(r.returncode, 4)
        self.assertIn("LOCK_FAILED", self.log())

    # --------------------------------------------------------- serialisation

    def test_two_jobs_do_not_overlap(self):
        """The whole point. Overlap cost 42 minutes of A6000 training once.

        Each job appends to a witness file on entry and exit; interleaved
        markers would mean both held the GPU at once.
        """
        witness = os.path.join(self.tmp.name, "witness")
        script = textwrap.dedent(f"""
            echo in >> {witness}
            sleep 1
            echo out >> {witness}
        """)
        env = dict(os.environ, GPU_LOCK=self.lock, GPU_QLOG=self.qlog)
        procs = [subprocess.Popen(["bash", GPU_JOB, f"j{i}", "bash", "-c",
                                   script], env=env,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
                 for i in range(2)]
        for p in procs:
            p.wait(timeout=60)
        with open(witness, encoding="utf-8") as fh:
            seq = [l.strip() for l in fh if l.strip()]
        self.assertEqual(seq, ["in", "out", "in", "out"],
                         f"jobs overlapped: {seq}")

    def test_the_second_job_waits_rather_than_failing(self):
        """Blocking is correct; a queued job must not be dropped."""
        env = dict(os.environ, GPU_LOCK=self.lock, GPU_QLOG=self.qlog)
        slow = subprocess.Popen(["bash", GPU_JOB, "slow", "sleep", "2"],
                                env=env, stdout=subprocess.DEVNULL)
        time.sleep(0.4)
        r = self.run_job("waiter", "true", timeout=60)
        slow.wait(timeout=60)
        self.assertEqual(r.returncode, 0)
        self.assertIn("OK       waiter", self.log())

    def test_interrupting_a_waiter_releases_nothing_and_logs_no_start(self):
        """A job killed while queued must not appear to have run."""
        env = dict(os.environ, GPU_LOCK=self.lock, GPU_QLOG=self.qlog)
        holder = subprocess.Popen(["bash", GPU_JOB, "holder", "sleep", "3"],
                                  env=env, stdout=subprocess.DEVNULL)
        time.sleep(0.4)
        marker = os.path.join(self.tmp.name, "waiter_ran")
        waiter = subprocess.Popen(["bash", GPU_JOB, "waiter", "touch", marker],
                                  env=env, stdout=subprocess.DEVNULL)
        time.sleep(0.4)
        waiter.kill()
        waiter.wait(timeout=30)
        holder.wait(timeout=60)
        self.assertFalse(os.path.exists(marker))
        self.assertNotIn("START    waiter", self.log())
        self.assertIn("OK       holder", self.log())

    # -------------------------------------------------- deployment identity

    def test_identity_is_logged_before_start(self):
        """Written BEFORE the job runs, or it is a post-mortem, not a record.

        Two jobs died on 2026-08-04 because a box was running a superseded copy
        of this script. Nothing announced it; it was found by reading logs
        afterwards.
        """
        self.run_job("ident", "true")
        kinds = [l.split()[1] for l in self.log().splitlines() if l.strip()]
        self.assertEqual(kinds, ["QUEUED", "IDENT", "START", "OK"])

    def test_identity_carries_what_is_needed_to_tell_two_runs_apart(self):
        self.run_job("ident", "echo", "hello")
        line = [l for l in self.log().splitlines() if " IDENT " in l][0]
        for field in ("commit=", "tree=", "gpu_job_sha=", "host=", "gpu=",
                      "cmd="):
            self.assertIn(field, line)
        self.assertIn("echo hello", line)

    def test_the_script_hash_actually_identifies_the_script(self):
        """The field that would have caught the stale cloud copy."""
        self.run_job("ident", "true")
        line = [l for l in self.log().splitlines() if " IDENT " in l][0]
        logged = line.split("gpu_job_sha=")[1].split()[0]
        import hashlib
        with open(GPU_JOB, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()[:12]
        self.assertEqual(logged, actual)

    def test_identity_degrades_rather_than_blocking_the_job(self):
        """Identity is evidence, not a gate.

        With git and both smi tools absent, the job must still run - a
        provenance record that can refuse to start work is worse than none.
        """
        d = self.tmp.name + "/emptybin"
        os.makedirs(d, exist_ok=True)
        for tool in ("git", "nvidia-smi", "rocm-smi", "sha256sum", "hostname"):
            p = os.path.join(d, tool)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/bash\nexit 127\n")
            os.chmod(p, 0o755)
        marker = os.path.join(self.tmp.name, "ran_anyway")
        r = self.run_job("degraded", "touch", marker, path_prefix=d)
        self.assertTrue(os.path.exists(marker), "job blocked by identity capture")
        self.assertEqual(r.returncode, 0)
        self.assertIn("IDENT", self.log())
        self.assertIn("unknown", self.log())

    # ------------------------------------------------------------- misuse

    def test_no_command_is_rejected(self):
        self.assertEqual(self.run_job("nameonly").returncode, 2)

    def test_no_name_is_rejected(self):
        r = subprocess.run(["bash", GPU_JOB], capture_output=True, text=True,
                           env=dict(os.environ, GPU_LOCK=self.lock,
                                    GPU_QLOG=self.qlog))
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
