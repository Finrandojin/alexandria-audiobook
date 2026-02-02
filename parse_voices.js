module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "env",
      // No 'path' attribute, so it runs from the project root
      message: "python app/parse_voices.py",
    }
  }]
}
