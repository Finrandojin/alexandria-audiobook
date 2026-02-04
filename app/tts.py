import os
import re
import json
from pydub import AudioSegment
from gradio_client import Client, handle_file
import shutil

DEFAULT_PAUSE_MS = 500  # Pause between different speakers
SAME_SPEAKER_PAUSE_MS = 250  # Shorter pause for same speaker continuing

def sanitize_filename(name):
    """Make a string safe for use in filenames"""
    name = re.sub(r'[^\w\-]', '_', name)
    return name.lower()

def preprocess_text_for_tts(text):
    """Extract non-verbal cues and prepare text for TTS.

    Returns: (processed_text, nonverbal_instructions)

    Example:
        "[laughs] That's hilarious!" -> ("That's hilarious!", "laughing")
        "[sighs] I'm so tired" -> ("sighs... I'm so tired", "sighing, weary")
        "[gasps] What?!" -> ("What?!", "gasping, shocked")
    """
    # Map of non-verbals to TTS style instructions
    nonverbal_to_style = {
        'laughs': 'laughing',
        'laugh': 'laughing',
        'chuckles': 'chuckling, amused',
        'chuckle': 'chuckling',
        'giggles': 'giggling, amused',
        'giggle': 'giggling',
        'scoffs': 'scoffing, dismissive',
        'scoff': 'scoffing',
        'sighs': 'sighing',
        'sigh': 'sighing',
        'gasps': 'gasping, shocked',
        'gasp': 'gasping',
        'groans': 'groaning',
        'groan': 'groaning',
        'moans': 'moaning',
        'moan': 'moaning',
        'whimpers': 'whimpering, distressed',
        'whimper': 'whimpering',
        'sobs': 'sobbing, crying',
        'sob': 'sobbing',
        'cries': 'crying',
        'cry': 'crying',
        'sniffs': 'sniffling',
        'sniff': 'sniffling',
        'whispers': 'whispering, quiet',
        'whisper': 'whispering',
        'shouts': 'shouting, loud',
        'shout': 'shouting',
        'screams': 'screaming',
        'scream': 'screaming',
        'yells': 'yelling, loud',
        'yell': 'yelling',
        'clears throat': 'clearing throat',
        'coughs': 'coughing',
        'cough': 'coughing',
        'pauses': 'with a pause',
        'pause': 'with a pause',
        'hesitates': 'hesitant, uncertain',
        'hesitate': 'hesitant',
        'stammers': 'stammering, nervous',
        'stammer': 'stammering',
        'gulps': 'gulping, nervous',
        'gulp': 'gulping',
        'snorts': 'snorting, derisive',
        'snort': 'snorting',
        'hums': 'humming',
        'hum': 'humming',
        'growls': 'growling, menacing',
        'growl': 'growling',
        'purrs': 'purring, satisfied',
        'purr': 'purring',
        'shivers': 'shivering, cold or scared',
        'shiver': 'shivering',
    }

    # Extract all non-verbals
    nonverbals_found = re.findall(r'\[([^\]]+)\]', text.lower())

    # Build style instructions from non-verbals
    style_additions = []
    for nv in nonverbals_found:
        nv_clean = nv.strip().lower()
        if nv_clean in nonverbal_to_style:
            style_additions.append(nonverbal_to_style[nv_clean])
        else:
            # Generic handling for unknown non-verbals
            style_additions.append(nv_clean)

    # Process the text - remove brackets but keep the word for some, remove entirely for others
    # For vocalizations that should be heard, keep them: laughs, sighs, etc.
    # For pure actions, remove them: pauses, hesitates
    action_only = {'pauses', 'pause', 'hesitates', 'hesitate', 'clears throat'}

    def replace_nonverbal(match):
        nv = match.group(1).strip().lower()
        if nv in action_only:
            return ''  # Remove completely, style will handle it
        else:
            return match.group(1) + '...'  # Keep as vocalization

    processed = re.sub(r'\[([^\]]+)\]', replace_nonverbal, text)

    # Clean up multiple ellipsis, spaces, and leading/trailing
    processed = re.sub(r'\.{4,}', '...', processed)
    processed = re.sub(r'\s+', ' ', processed).strip()
    processed = re.sub(r'^\.\.\.', '', processed).strip()  # Remove leading ellipsis

    nonverbal_style = ', '.join(style_additions) if style_additions else ''

    return processed, nonverbal_style

def test_tts_connection(tts_url, voice_config):
    """Test the TTS connection with the first configured voice"""
    print(f"Testing TTS connection to {tts_url}...")

    speaker = list(voice_config.keys())[0] if voice_config else None
    if not speaker:
        print("Error: No voices configured in voice_config.json")
        return False

    voice_data = voice_config[speaker]
    voice = voice_data.get("voice", "Ryan")
    seed = int(voice_data.get("seed", -1))

    print(f"  Voice: {voice}")
    print(f"  Seed: {seed}")

    try:
        client = Client(tts_url)

        result = client.predict(
            text="Testing, one two three.",
            language="Auto",
            speaker=voice,
            instruct="neutral, clear",
            model_size="1.7B",
            seed=seed,
            api_name="/generate_custom_voice"
        )
        print(f"  Test successful! Output: {result[0]}")
        return True
    except Exception as e:
        print(f"  TTS Test FAILED: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Make sure the TTS server is running")
        print("  2. Check if the CustomVoice model is loaded")
        return False

def generate_custom_voice(text, style, speaker, voice_config, output_path, client):
    """Generate audio using CustomVoice model"""
    try:
        voice_data = voice_config.get(speaker)
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        voice = voice_data.get("voice", "Ryan")
        default_style = voice_data.get("default_style", "")
        seed = int(voice_data.get("seed", -1))

        # Preprocess text and extract non-verbal style cues
        processed_text, nonverbal_style = preprocess_text_for_tts(text)

        # Build the full style instruction:
        # 1. Non-verbal cues take priority (laughing, sighing, etc.)
        # 2. Then per-line style from script
        # 3. Then default character style
        style_parts = []
        if nonverbal_style:
            style_parts.append(nonverbal_style)
        if style:
            style_parts.append(style)
        elif default_style:
            style_parts.append(default_style)

        instruct = ', '.join(style_parts) if style_parts else "neutral"

        result = client.predict(
            text=processed_text,
            language="Auto",
            speaker=voice,
            instruct=instruct,
            model_size="1.7B",
            seed=seed,
            api_name="/generate_custom_voice"
        )

        generated_audio_filepath = result[0]
        if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
            print(f"Error: No audio file generated for: '{text[:50]}...'")
            return False

        # Check file size
        if os.path.getsize(generated_audio_filepath) == 0:
            print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
            return False

        shutil.copy(generated_audio_filepath, output_path)
        return True

    except Exception as e:
        print(f"Error generating custom voice for '{speaker}': {e}")
        return False

def generate_clone_voice(text, speaker, voice_config, output_path, client):
    """Generate audio using voice cloning from reference audio"""
    try:
        voice_data = voice_config.get(speaker)
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        ref_audio = voice_data.get("ref_audio")
        ref_text = voice_data.get("ref_text")
        seed = int(voice_data.get("seed", -1))

        if not ref_audio or not ref_text:
            print(f"Warning: Clone voice for '{speaker}' missing ref_audio or ref_text. Skipping.")
            return False

        if not os.path.exists(ref_audio):
            print(f"Warning: Reference audio not found for '{speaker}': {ref_audio}")
            return False

        # Preprocess text (strip non-verbals but don't use style since clone doesn't support it)
        processed_text, _ = preprocess_text_for_tts(text)

        result = client.predict(
            handle_file(ref_audio),  # Reference audio file path (wrapped for Gradio)
            ref_text,            # Transcript of reference audio
            processed_text,      # Text to generate
            "Auto",              # Language detection
            False,               # use_xvector_only
            "1.7B",              # Model size
            200,                 # max_chunk_chars
            0,                   # chunk_gap
            seed,                # seed
            api_name="/generate_voice_clone"
        )

        generated_audio_filepath = result[0]
        if not generated_audio_filepath or not os.path.exists(generated_audio_filepath):
            print(f"Error: No audio file generated for: '{text[:50]}...'")
            return False

        # Check file size
        if os.path.getsize(generated_audio_filepath) == 0:
            print(f"Error: Generated audio file is empty for: '{text[:50]}...'")
            return False

        shutil.copy(generated_audio_filepath, output_path)
        return True

    except Exception as e:
        print(f"Error generating clone voice for '{speaker}': {e}")
        return False

def generate_voice(text, style, speaker, voice_config, output_path, client):
    """Generate audio using either custom voice or clone voice based on config"""
    voice_data = voice_config.get(speaker)
    if not voice_data:
        print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
        return False

    voice_type = voice_data.get("type", "custom")

    if voice_type == "clone":
        # Clone voice ignores style
        return generate_clone_voice(text, speaker, voice_config, output_path, client)
    else:
        # Custom voice uses style directions
        return generate_custom_voice(text, style, speaker, voice_config, output_path, client)

def combine_audio_with_pauses(audio_segments, speakers, pause_ms=DEFAULT_PAUSE_MS, same_speaker_pause_ms=SAME_SPEAKER_PAUSE_MS):
    """Combine audio segments with pauses between them"""
    if not audio_segments:
        return None

    silence_between_speakers = AudioSegment.silent(duration=pause_ms)
    silence_same_speaker = AudioSegment.silent(duration=same_speaker_pause_ms)

    combined = audio_segments[0]
    prev_speaker = speakers[0]

    for segment, speaker in zip(audio_segments[1:], speakers[1:]):
        if speaker == prev_speaker:
            combined += silence_same_speaker + segment
        else:
            combined += silence_between_speakers + segment
        prev_speaker = speaker

    return combined
