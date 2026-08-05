const fs = require('fs')
const path = require('path')

module.exports = {
  version: "5.0",
  title: "Alexandria",
  description: "A tool that takes a text document containing a book or a novel, ingests it with an LLM to produce an annotated script, and then uses a TTS API to generate the voice lines, finally stitching them together into an audiobook in MP3 format.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    // Check running states
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      start_llm: info.running("start_llm.js"),
      stage4_checkpoint: info.running("run_stage4_checkpoint.js"),
      reset: info.running("reset.js"),
      update: info.running("update.js")
    }

    // Check file existence states
    let installed = info.exists("app/env")

    // Handle running states first
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js"
      }]
    }

    if (running.start) {
      let local = info.local("start.js")
      let items = []
      if (local && local.url) {
        items.push({
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: local.url,
        })
        items.push({
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        })
      } else {
        items.push({
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Starting",
          href: "start.js",
        })
      }
      if (running.start_llm) {
        items.push({
          icon: "fa-solid fa-brain",
          text: "LLM Server",
          href: "start_llm.js",
        })
      }
      return items
    }

    if (running.start_llm && !running.start) {
      return [{
        default: true,
        icon: "fa-solid fa-brain",
        text: "LLM Server",
        href: "start_llm.js",
      }, {
        icon: "fa-solid fa-power-off",
        text: "Start App",
        href: "start.js",
      }]
    }

    if (running.stage4_checkpoint) {
      return [{
        default: true,
        icon: "fa-solid fa-flask",
        text: "Stage 4 Checkpoint Running",
        href: "run_stage4_checkpoint.js"
      }]
    }

    if (running.reset) {
      return [{
        default: true,
        icon: "fa-solid fa-rotate-left",
        text: "Resetting",
        href: "reset.js"
      }]
    }

    if (running.update) {
      return [{
        default: true,
        icon: "fa-solid fa-arrows-rotate",
        text: "Updating",
        href: "update.js"
      }]
    }

    // STATE: NOT_INSTALLED - auto-run install
    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js"
      }]
    }

    // STATE: INSTALLED
    return [{
      default: true,
      icon: "fa-solid fa-power-off",
      text: "Start",
      href: "start.js"
    }, {
      icon: "fa-solid fa-flask",
      text: "Run Stage 4 Checkpoint",
      href: "run_stage4_checkpoint.js"
    }, {
      icon: "fa-solid fa-brain",
      text: "Start LLM: Gemma 4",
      href: "start_llm.js",
      params: {
        model: "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
      }
    }, {
      icon: "fa-solid fa-brain",
      text: "Start LLM: Qwen 2.5 14B",
      href: "start_llm.js",
      params: {
        model: "Qwen2.5-14B-Instruct-Q6_K.gguf"
      }
    }, {
      icon: "fa-solid fa-folder-open",
      text: "Open Voicelines",
      href: "voicelines"
    }, {
      icon: "fa-solid fa-arrows-rotate",
      text: "Update",
      href: "update.js"
    }, {
      icon: "fa-solid fa-plug",
      text: "Reinstall",
      href: "install.js"
    }, {
      icon: "fa-solid fa-rotate-left",
      text: "Reset",
      href: "reset.js"
    }]
  }
}
