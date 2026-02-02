module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "env",
      // No 'path', so it runs from the project root
      message: "python app/generate_audiobook.py",
    }
  }]
}
