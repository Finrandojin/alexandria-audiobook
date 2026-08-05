"""Run one repository GPU job with the approved lock and queue log."""
import os
import subprocess


def run_gpu_job(repo, app, python, name, timeout_seconds, script, arguments,
                log_name):
    env = os.environ.copy()
    env["GPU_LOCK"] = os.path.expanduser("~/.alexandria_gpu.lock")
    env["GPU_QLOG"] = os.path.join(
        repo, "ab_test_runtime", "logs", "gpu_jobq.log")
    log_path = os.path.join(repo, "ab_test_runtime", "logs", log_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    command = [os.path.join(repo, "gpu_job.sh"), name, "timeout",
               str(timeout_seconds), python, "-u", script, *arguments]
    print("+", " ".join(command), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        subprocess.run(command, cwd=app, check=True, stdout=log,
                       stderr=subprocess.STDOUT, env=env)
