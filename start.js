module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "python app.py",
        on: [{
          // Capture the URL when the server prints it
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      // Set the local variable 'url' for pinokio.js to display "Open Web UI"
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
