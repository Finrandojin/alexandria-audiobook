module.exports = {
  run: [{
    method: "input",
    params: {
      title: "Configure Voice for {{args.speaker}}",
      form: [{
        name: "ref_text",
        label: "Transcript of the audio file",
        type: "text"
      }]
    }
  }, {
    method: "filepicker",
    params: {
      title: "Select Reference Audio File",
      path: "{{kernel.homedir}}"
    }
  }, {
    method: "json.set",
    params: {
      path: "voice_config.json",
      key: "{{args.speaker}}",
      value: {
        ref_audio: "{{input.file}}",
        ref_text: "{{input.ref_text}}"
      }
    }
  }]
}
