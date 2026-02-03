# Alexandria Audiobook Generator

Transform any book or novel into a fully-voiced audiobook using AI-powered script annotation and voice cloning.

## Features

- **LLM Script Annotation** - Google Gemini parses your text, identifying dialogue, speakers, and narration
- **Voice Cloning** - Qwen3 TTS 1.7B creates consistent character voices from short audio samples
- **Multi-Speaker Support** - Configure unique voices for each character
- **Smart Chunking** - Groups consecutive lines by speaker (up to 500 chars) for natural flow
- **Natural Pauses** - Automatic delays between speakers and segments
- **Reproducible Output** - Optional per-voice seed for consistent generation

## Requirements

- [Pinokio](https://pinokio.computer/)
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))
- Qwen3 TTS server running locally (Gradio interface)

## Installation

1. Install [Pinokio](https://pinokio.computer/) if you haven't already
2. In Pinokio, click **Download** and paste this URL:
   ```
   https://github.com/Finrandojin/alexandria-audiobook
   ```
3. Click **Install** to set up dependencies
4. Click **Configure** and enter:
   - Your Gemini API key
   - TTS server URL (default: `http://127.0.0.1:7860`)

## Usage

1. **Select File** - Choose your book/novel text file (.txt or .md)
2. **Generate Script** - LLM creates speaker-annotated script
3. **Parse Voices** - Extracts unique speakers from script
4. **Configure Voices** - For each speaker, provide:
   - A reference audio clip (~5-15 seconds)
   - The exact transcript of that audio
   - Optional: seed for reproducible output
5. **Generate Audiobook** - Creates final MP3

## Output

The generated audiobook is saved as `cloned_audiobook.mp3` in the project root.

## License

MIT
