module.exports = {
  run: [{
    method: "input",
    params: {
      title: "Configure Alexandria",
      description: "Alexandria uses an LLM to convert your book into an annotated script, then a TTS server to generate voice lines for each character.",
      form: [{
        key: "llm_base_url",
        title: "LLM API Base URL",
        description: "Base URL for OpenAI-compatible API. Use http://localhost:1234/v1 for LM Studio, http://localhost:11434/v1 for Ollama, or https://api.openai.com/v1 for OpenAI.",
        type: "text",
        default: "http://localhost:11434/v1",
        placeholder: "http://localhost:11434/v1",
        required: true
      }, {
        key: "llm_api_key",
        title: "LLM API Key",
        description: "API key for the LLM server. For local servers, you can often use any value (e.g., 'local'). For OpenAI, use your actual API key.",
        type: "password",
        default: "local",
        placeholder: "sk-... or local",
        required: true
      }, {
        key: "llm_model_name",
        title: "LLM Model Name",
        description: "The model to use. For local servers, check your server's loaded model name. For OpenAI: gpt-4o, gpt-4o-mini, etc.",
        type: "text",
        default: "richardyoung/qwen3-14b-abliterated:Q8_0",
        placeholder: "gpt-4o-mini or local-model",
        required: true
      }, {
        key: "tts_url",
        title: "TTS Server URL",
        description: "URL of your Qwen3 TTS Gradio server. Start the TTS server first, then enter its URL here.",
        type: "text",
        default: "http://127.0.0.1:7860",
        placeholder: "http://127.0.0.1:7860",
        required: true
      }]
    }
  }, {
    method: "fs.write",
    params: {
      path: "app/config.json",
      json2: {
        llm: {
          base_url: "{{input.llm_base_url}}",
          api_key: "{{input.llm_api_key}}",
          model_name: "{{input.llm_model_name}}"
        },
        tts: {
          url: "{{input.tts_url}}"
        }
      }
    }
  }]
}
