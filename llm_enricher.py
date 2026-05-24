#!/usr/bin/env python3

import json
import os
import re
import logging
import traceback
from typing import Dict, List, Any

from llama_cpp import Llama

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LLMEnricher:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        try:
            logger.info(f"Loading LLM model from: {self.model_path}")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False
            )
            logger.info("LLM model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM model from {self.model_path}: {e}")
            logger.debug(traceback.format_exc())
            raise

    def enrich_transcript_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enriches a transcript chunk with metadata using the LLM."""
        if not self.llm:
            logger.error("LLM model not loaded. Cannot enrich transcript.")
            return chunk

        prompt = self._create_prompt(chunk)

        try:
            logger.info(f"Enriching chunk: {chunk.get('start', 'N/A'):.2f}s - {chunk.get('end', 'N/A'):.2f}s")
            output = self.llm(
                prompt,
                max_tokens=150,
                stop=["</s>"],
                temperature=0.7
            )

            enriched_data = self._parse_llm_output(output['choices'][0]['text'])

            chunk.update(enriched_data)
            return chunk

        except Exception as e:
            logger.error(f"Error during LLM enrichment for chunk {chunk.get('start', 'N/A')}: {e}")
            logger.debug(traceback.format_exc())
            return chunk

    def _create_prompt(self, chunk: Dict[str, Any]) -> str:
        """Creates a prompt for the LLM to extract metadata."""
        text = chunk.get('text', '')
        speaker = chunk.get('speaker', 'UNKNOWN')
        start = chunk.get('start', 0.0)
        end = chunk.get('end', 0.0)

        prompt = f"""Analyze the following transcript segment and extract metadata:

Transcript Segment:
"{text}"

Speaker: {speaker}
Start Time: {start:.2f}s
End Time: {end:.2f}s

Extract the following metadata:
- Speaker Attribution (e.g., "main character", "narrator", "secondary character")
- Narration Style (e.g., "calm", "energetic", "sad", "questioning")
- Emotional Tone (e.g., "happy", "anxious", "neutral", "excited")

Provide the output as a JSON object with keys: "speaker_attribution", "narration_style", "emotional_tone". If any information cannot be determined, use 'N/A'.

Example Output Format:
{{"speaker_attribution": "main character", "narration_style": "calm", "emotional_tone": "neutral"}}

Output JSON: """
        return prompt

    def _parse_llm_output(self, output_text: str) -> Dict[str, str]:
        """Parses the LLM's output to extract metadata."""
        # Try to find JSON in markdown code blocks first
        json_match = re.search(r'```json\s*\n({.*?})\n\s*```', output_text, re.DOTALL)
        if not json_match:
            # Fallback to finding standalone JSON
            json_match = re.search(r'(\{"speaker_attribution".*?\})', output_text, re.DOTALL)
        
        if json_match:
            try:
                metadata = json.loads(json_match.group(1))
                return metadata
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM output as JSON: {json_match.group(1)[:100]}...")

        logger.warning(f"Could not parse LLM output as JSON: {output_text[:200]}")
        return {
            "speaker_attribution": "N/A",
            "narration_style": "N/A",
            "emotional_tone": "N/A"
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Transcript Enricher")
    parser.add_argument("--model-path", required=True, help="Path to the GGUF LLM model file.")
    parser.add_argument("--input-file", required=True, help="Path to the input JSON file containing transcript segments.")
    parser.add_argument("--output-file", required=True, help="Path to save the enriched transcript JSON file.")

    args = parser.parse_args()

    try:
        enricher = LLMEnricher(args.model_path)
    except Exception as e:
        logger.error(f"Exiting: Could not initialize LLMEnricher: {e}")
        exit(1)

    try:
        with open(args.input_file, 'r') as f:
            transcript_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        exit(1)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from input file: {args.input_file}")
        exit(1)

    enriched_data = []
    for i, chunk in enumerate(transcript_data):
        try:
            enriched_chunk = enricher.enrich_transcript_chunk(chunk)
            enriched_data.append(enriched_chunk)
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            logger.debug(traceback.format_exc())
            enriched_data.append(chunk)

    try:
        with open(args.output_file, 'w') as f:
            json.dump(enriched_data, f, indent=2)
        logger.info(f"Enriched transcript saved to: {args.output_file}")
    except IOError as e:
        logger.error(f"Failed to write output file {args.output_file}: {e}")
        exit(1)

if __name__ == "__main__":
    main()
