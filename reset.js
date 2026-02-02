module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      paths: [
        "annotated_script.txt",
        "voices.json",
        "voice_config.json"
      ]
    }
  }]
}
