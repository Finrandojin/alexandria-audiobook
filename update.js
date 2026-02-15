module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "script.start",
    params: {
      uri: "install.js"
    }
  }]
}
