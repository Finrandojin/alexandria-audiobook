module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "app/env",
      path: ".",
      env: {
        PYTHONUNBUFFERED: "1",
        GPU_QLOG: "{{path.resolve(cwd, 'ab_test_runtime', 'logs', 'gpu_jobq.log')}}"
      },
      message: "python -u run_stage4_checkpoint.py"
    }
  }]
}
