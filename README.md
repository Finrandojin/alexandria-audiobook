<img width="475" height="467" alt="Gemini_Generated_Image_yk5astyk5astyk5a" src="https://github.com/user-attachments/assets/fa2c36d3-a5f3-49ab-9dfe-30933359dfbd" />


# Alexandria Audiobook Generator

Transform any book or novel into a fully-voiced audioplay using AI-powered script annotation and styled TTS.

## Features

- **Local & Cloud LLM Support** - Use any OpenAI-compatible API (LM Studio, Ollama, OpenAI, etc.)
- **LLM Script Annotation** - Parses your text into a JSON script with speakers, dialogue, and style directions
- **Non-verbal Sounds** - Supports `[laughs]`, `[sighs]`, `[gasps]`, and other vocalizations inline with dialogue
- **Style Instructions** - Per-line delivery directions like "nervous, whispered" or "angry, shouting"
- **Two Voice Modes:**
  - **Custom Voices** - 9 pre-trained voices with style direction support
  - **Voice Cloning** - Clone any voice from a reference audio sample
- **Smart Chunking** - Groups consecutive lines by speaker (up to 500 chars) for natural flow
- **Natural Pauses** - Automatic delays between speakers and segments
- **Audioplay Export** - Individual voiceline files for audio editing (Audacity, etc.)
- **Robust Parsing** - Handles thinking tags, markdown, and control characters from local LLMs

## Requirements

- [Pinokio](https://pinokio.computer/)
- LLM server (one of the following):
  - [LM Studio](https://lmstudio.ai/) (local) - recommended: Qwen3 or similar
  - [Ollama](https://ollama.ai/) (local)
  - [OpenAI API](https://platform.openai.com/) (cloud)
  - Any OpenAI-compatible API
- [Qwen3 TTS](https://github.com/Qwen/Qwen3-TTS) server running locally (Gradio interface)

## Installation

1. Install [Pinokio](https://pinokio.computer/) if you haven't already
2. In Pinokio, click **Download** and paste this URL:
   ```
   https://github.com/Finrandojin/alexandria-audiobook
   ```
3. Click **Install** to set up dependencies
4. Click **Configure** and enter:
   - **LLM Base URL**:
     - LM Studio: `http://localhost:1234/v1`
     - Ollama: `http://localhost:11434/v1`
     - OpenAI: `https://api.openai.com/v1`
   - **LLM API Key**: Your API key (use `local` for local servers)
   - **LLM Model Name**: The model to use (check your server's loaded model)
   - **TTS Server URL**: Default `http://127.0.0.1:7860`

## Usage

1. **Select File** - Choose your book/novel text file (.txt or .md)
2. **Generate Script** - LLM creates JSON script with speakers, text, style, and non-verbals
3. **Parse Voices** - Extracts unique speakers from script
4. **Configure Voices** - For each speaker, choose:
   - **Custom Voice**: Pick from 9 pre-trained voices + style direction
   - **Clone Voice**: Provide reference audio + transcript (style ignored)
5. **Generate Audiobook** - Creates final MP3 and individual voicelines

## Voice Options

### Custom Voices (with style direction)
Pre-trained voices that respond to style instructions from the script:
- Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian

Configuration:
- Voice selection
- Default style (e.g., "calm, professional narrator")
- Seed for reproducible output (-1 for random)

### Clone Voices (from reference audio)
Clone any voice using a short audio sample:
- Provide 5-15 seconds of clear speech
- Include exact transcript of the reference audio
- Voice characteristics captured from sample

Note: Style directions from the script are ignored for cloned voices.

## Script Format

The generated script is a JSON array with style directions and non-verbal cues:

```json
[
  {"speaker": "NARRATOR", "text": "The door creaked open slowly.", "style": "tense, suspenseful"},
  {"speaker": "ELENA", "text": "[gasps] Who's there?", "style": "frightened, whispered"},
  {"speaker": "MARCUS", "text": "[chuckles] Did you miss me?", "style": "smug, menacing"}
]
```

### Supported Non-verbal Sounds
`[laughs]`, `[chuckles]`, `[giggles]`, `[sighs]`, `[gasps]`, `[groans]`, `[moans]`, `[whimpers]`, `[sobs]`, `[cries]`, `[sniffs]`, `[whispers]`, `[shouts]`, `[screams]`, `[clears throat]`, `[coughs]`, `[pauses]`, `[hesitates]`, `[stammers]`, `[gulps]`

## Output

**Combined Audiobook:**
- `cloned_audiobook.mp3` - Full audiobook with natural pauses

**Individual Voicelines (for audio editing):**
- `voicelines/voiceline_0001_narrator.mp3`
- `voicelines/voiceline_0002_elena.mp3`
- `voicelines/voiceline_0003_marcus.mp3`
- ...

Files are numbered in timeline order and include the speaker name, making it easy to:
- Import into Audacity or other DAWs
- Place each character on separate tracks
- Color-code by speaker
- Fine-tune timing and effects

## Recommended Local Models

For script generation, non-thinking models work best:
- **Qwen2.5** (any size)
- **Qwen3** (non-thinking variant)
- **Llama 3.1/3.2**
- **Mistral/Mixtral**

Avoid "thinking" models (DeepSeek-R1, GLM4-air, etc.) as they can interfere with JSON output.

## License

MIT
