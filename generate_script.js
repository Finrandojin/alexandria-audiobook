module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "python generate_script.py --file ../{{args.file}}",
    }
  }]
}
