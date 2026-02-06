import os
import re
import json
import base64
from pydub import AudioSegment
from gradio_client import Client, handle_file
import shutil

DEFAULT_PAUSE_MS = 500  # Pause between different speakers
SAME_SPEAKER_PAUSE_MS = 250  # Shorter pause for same speaker continuing

def sanitize_filename(name):
    """Make a string safe for use in filenames"""
    name = re.sub(r'[^\w\-]', '_', name)
    return name.lower()


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

def generate_custom_voice(text, instruct_text, speaker, voice_config, output_path, client):
    """Generate audio using CustomVoice model"""
    try:
        voice_data = voice_config.get(speaker)
        if not voice_data:
            print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
            return False

        voice = voice_data.get("voice", "Ryan")
        default_style = voice_data.get("default_style", "")
        seed = int(voice_data.get("seed", -1))

        # Build instruct: prefer chunk instruct, fall back to voice_config default_style
        instruct = instruct_text if instruct_text else (default_style if default_style else "neutral")

        print(f"TTS generating with instruct='{instruct}' for text='{text[:50]}...'")

        result = client.predict(
            text=text,
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

        result = client.predict(
            handle_file(ref_audio),  # Reference audio file path (wrapped for Gradio)
            ref_text,            # Transcript of reference audio
            text,                # Text to generate
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

def generate_voice(text, instruct_text, speaker, voice_config, output_path, client):
    """Generate audio using either custom voice or clone voice based on config"""
    voice_data = voice_config.get(speaker)
    if not voice_data:
        print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
        return False

    voice_type = voice_data.get("type", "custom")

    if voice_type == "clone":
        return generate_clone_voice(text, speaker, voice_config, output_path, client)
    else:
        return generate_custom_voice(text, instruct_text, speaker, voice_config, output_path, client)


def generate_batch(chunks, voice_config, output_dir, client, batch_seed=-1):
    """Generate multiple audio files in a single batch API call.

    Args:
        chunks: List of dicts with 'text', 'instruct', 'speaker', 'index' keys
        voice_config: Voice configuration dict
        output_dir: Directory to save output files
        client: Gradio client instance
        batch_seed: Single seed for all generations (-1 for random)

    Returns:
        dict with 'completed' (list of indices) and 'failed' (list of (index, error) tuples)
    """
    results = {"completed": [], "failed": []}

    if not chunks:
        return results

    # Separate custom voice chunks from clone voice chunks
    # Clone voices don't support batching, process them individually
    custom_chunks = []
    clone_chunks = []

    for chunk in chunks:
        speaker = chunk.get("speaker")
        voice_data = voice_config.get(speaker, {})
        voice_type = voice_data.get("type", "custom")

        if voice_type == "clone":
            clone_chunks.append(chunk)
        else:
            custom_chunks.append(chunk)

    # Process custom voice chunks in batch
    if custom_chunks:
        batch_results = _generate_custom_voice_batch(custom_chunks, voice_config, output_dir, client, batch_seed)
        results["completed"].extend(batch_results["completed"])
        results["failed"].extend(batch_results["failed"])

    # Process clone voice chunks individually (no batch support)
    for chunk in clone_chunks:
        idx = chunk["index"]
        output_path = os.path.join(output_dir, f"temp_batch_{idx}.wav")
        try:
            success = generate_clone_voice(
                chunk["text"], chunk["speaker"], voice_config, output_path, client
            )
            if success:
                results["completed"].append(idx)
            else:
                results["failed"].append((idx, "Clone voice generation failed"))
        except Exception as e:
            results["failed"].append((idx, str(e)))

    return results


def _generate_custom_voice_batch(chunks, voice_config, output_dir, client, batch_seed=-1):
    """Internal: Generate custom voice audio in batch with single seed."""
    results = {"completed": [], "failed": []}

    # Build batch config
    texts = []
    speakers = []
    instructs = []
    indices = []

    for chunk in chunks:
        idx = chunk["index"]
        text = chunk.get("text", "")
        instruct_text = chunk.get("instruct", "")
        speaker_name = chunk.get("speaker", "")

        voice_data = voice_config.get(speaker_name, {})
        voice = voice_data.get("voice", "Ryan")
        default_style = voice_data.get("default_style", "")

        # Build instruct: prefer chunk instruct, fall back to voice_config default_style
        instruct = instruct_text if instruct_text else (default_style if default_style else "neutral")

        texts.append(text)
        speakers.append(voice)
        instructs.append(instruct)
        indices.append(idx)

    # Build config JSON with single seed
    config = {
        "mode": "custom_voice",
        "texts": texts,
        "speaker": speakers,
        "instruct": instructs,
        "seed": batch_seed,
        "language": "en",
        "model_size": "1.7B"
    }

    print(f"Sending batch request for {len(texts)} chunks...")

    try:
        result = client.predict(
            config_json=json.dumps(config),
            api_name="/generate_batch"
        )

        # Parse response
        response = json.loads(result)

        if not response.get("success"):
            error_msg = response.get("error", "Unknown batch error")
            print(f"Batch generation failed: {error_msg}")
            for idx in indices:
                results["failed"].append((idx, error_msg))
            return results

        audio_files = response.get("audio_files", [])
        sample_rate = response.get("sample_rate", 24000)

        print(f"Received {len(audio_files)} audio files from batch")

        # Process each audio file
        for audio_item in audio_files:
            item_index = audio_item.get("index")
            audio_base64 = audio_item.get("audio_base64")

            # Map batch index back to chunk index
            if item_index is not None and item_index < len(indices):
                chunk_idx = indices[item_index]
            else:
                print(f"Warning: Invalid index {item_index} in batch response")
                continue

            if not audio_base64:
                results["failed"].append((chunk_idx, "No audio data in response"))
                continue

            try:
                # Decode base64 and save as WAV
                audio_bytes = base64.b64decode(audio_base64)
                output_path = os.path.join(output_dir, f"temp_batch_{chunk_idx}.wav")

                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

                # Verify file was written
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    results["completed"].append(chunk_idx)
                    print(f"Batch chunk {chunk_idx} saved: {os.path.getsize(output_path)} bytes")
                else:
                    results["failed"].append((chunk_idx, "Audio file empty after decode"))

            except Exception as e:
                print(f"Error decoding audio for chunk {chunk_idx}: {e}")
                results["failed"].append((chunk_idx, f"Decode error: {e}"))

    except Exception as e:
        print(f"Batch API call failed: {e}")
        for idx in indices:
            results["failed"].append((idx, f"Batch API error: {e}"))

    return results


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
