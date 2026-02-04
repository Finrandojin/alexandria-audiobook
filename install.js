module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "uv cache clean"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "python -m venv env"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip uninstall google-genai",
        "uv pip install -r requirements.txt"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "python app.py",
      on: [{ "event": "/http:\/\/[0-9.:]+/", "done": true }]
    }
  }, {
    method: "local.set",
    params: {
      url: "{{input.event[0]}}"
    }
  }, {
    method: "browser.open",
    params: {
      uri: "{{local.url}}",
      target: "_blank"
    }
  }]
}
