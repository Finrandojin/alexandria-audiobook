module.exports = {
  run: [{
    method: "browser.open",
    params: {
      uri: "http://127.0.0.1:4200",
      target: "_blank"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "python app.py",
      on: [{ "event": "/http:\/\/[0-9.:]+/" }]
    }
  }]
}
