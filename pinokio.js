module.exports = async (kernel) => {
  let installed = await kernel.exists(__dirname, "app", "venv")
  let script_exists = await kernel.exists(__dirname, "annotated_script.txt")
  let voices_exist = await kernel.exists(__dirname, "voices.json")

  if (installed) {
    if (script_exists) {
      if (voices_exist) {
        // Step 3: Configure voices
        let voices = await kernel.fs.read("voices.json").then(r => r.json())
        let voice_config = await kernel.fs.read("voice_config.json").then(r => r.json()).catch(e => ({}))
        
        let menu = []
        for(let voice of voices) {
          let configured = voice_config[voice]
          menu.push({
            icon: `fa-solid ${configured ? 'fa-check' : 'fa-microphone-lines'}`,
            text: `Configure Voice: ${voice}`,
            href: "configure_voice.js",
            params: {
              run: true,
              speaker: voice
            }
          })
        }
        
        menu.push({
          icon: "fa-solid fa-play-circle",
          text: "Generate Audiobook",
          href: "generate_audiobook.js", // Will create this next
          params: {
            run: true
          }
        })
        
        menu.push({
          icon: "fa-solid fa-trash",
          text: "Reset",
          href: "reset.js",
          params: {
            run: true
          }
        })

        return menu

      } else {
        // Step 2: Parse the script to find voices
        return [{
          icon: "fa-solid fa-user-plus",
          text: "Parse Voices",
          href: "parse_voices.js",
          params: {
            run: true,
          }
        }, {
          icon: "fa-solid fa-trash",
          text: "Reset",
          href: "reset.js",
           params: {
            run: true,
          }
        }]
      }
    } else {
      // Step 1: Generate the script
      return [{
        icon: "fa-solid fa-file-lines",
        text: "Generate Script",
        href: "generate_script.js",
        params: {
          run: true,
          requires: "file"
        }
      }, {
        icon: "fa-solid fa-wrench",
        text: "Configure",
        href: "configure.js",
        params: {
          run: true,
        }
      }, {
        icon: "fa-solid fa-plug",
        text: "Reinstall",
        href: "install.js",
        params: {
          run: true,
        }
      }]
    }
  } else {
    // Not installed yet
    return [{
      icon: "fa-solid fa-plug",
      text: "Install",
      href: "install.js",
      params: {
        run: true,
      }
    }]
  }
}
