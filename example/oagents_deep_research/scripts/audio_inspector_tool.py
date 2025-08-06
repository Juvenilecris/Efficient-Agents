#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from smolagents import Tool
from smolagents.models import MessageRole, Model

import openai
import os

# Placeholder for a function to get audio duration. You'll need to implement this.
# For example, using a library like pydub: from pydub import AudioSegment; duration = AudioSegment.from_file(file_path).duration_seconds
def get_audio_duration_seconds(file_path: str) -> float:
    """Placeholder: Implement this function to get audio duration in seconds."""
    # Example implementation (replace with actual logic):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return audio.duration_seconds
    except Exception as e:
        logger.warning(f"Could not get audio duration for {file_path} using pydub: {e}. Returning 0.")
        return 0.0 

logger = logging.getLogger(__name__)

# Module-level tracker for Whisper costs
whisper_cost_tracker = {
    "model_id": "whisper-1",
    "total_minutes_processed": 0.0,
    "total_cost": 0.0,
    "api_calls": 0,
}

WHISPER_PRICE_PER_MINUTE = 0.006

def reset_whisper_cost_tracker():
    global whisper_cost_tracker
    whisper_cost_tracker = {
        "model_id": "whisper-1",
        "total_minutes_processed": 0.0,
        "total_cost": 0.0,
        "api_calls": 0,
    }

def get_cumulative_whisper_cost_details() -> dict:
    return whisper_cost_tracker.copy()

class AudioInspectorTool(Tool):
    name = "inspect_file_as_audio"
    description = """
You cannot load files directly: use this tool to process audio files and answer related questions.
This tool supports the following audio formats: [".mp3", ".m4a", ".wav"]. For other file types, use the appropriate inspection tool."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as audio. Must be a '.something' file, like '.mp3','.m4a','.wav'. If it is an text, use the text_inspector tool instead! If it is an image, use the visual_inspector tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the audio content. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, text_limit: int):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")

    def _validate_file_type(self, file_path: str):
        """Validate if the file type is a supported audio format"""
        if not any(file_path.endswith(ext) for ext in [".mp3", ".m4a", ".wav"]):
            raise ValueError("Unsupported file type. Use the appropriate tool for text/image files.")

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio using OpenAI Whisper API"""
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            # Cost tracking for Whisper
            audio_duration_seconds = get_audio_duration_seconds(file_path)
            audio_duration_minutes = audio_duration_seconds / 60.0
            current_whisper_cost = audio_duration_minutes * WHISPER_PRICE_PER_MINUTE

            global whisper_cost_tracker
            whisper_cost_tracker["total_minutes_processed"] += audio_duration_minutes
            whisper_cost_tracker["total_cost"] += current_whisper_cost
            whisper_cost_tracker["api_calls"] += 1

            logger.debug(
                f"Whisper Call - File: {file_path}, Duration (min): {audio_duration_minutes:.2f}, Cost: ${current_whisper_cost:.6f}"
            )
            logger.debug(
                f"Whisper Cumulative - Total Calls: {whisper_cost_tracker['api_calls']}, Total Minutes: {whisper_cost_tracker['total_minutes_processed']:.2f}, Total Cost: ${whisper_cost_tracker['total_cost']:.6f}"
            )
            return transcription.text
        except Exception as e:
            raise RuntimeError(f"Speech recognition failed: {str(e)}") from e

    def forward(self, file_path: str, question: Optional[str] = None) -> str:
        self._validate_file_type(file_path)
        
        try:
            transcript = self.transcribe_audio(file_path)
        except Exception as e:
            return f"Audio processing error: {str(e)}"
        
        if not question:
            return f"Audio transcription:\n{transcript[:self.text_limit]}"
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [{
                    "type": "text",
                    "text": f"Here is the an audio transcription:\n{transcript[:self.text_limit]}\n"
                            "Answer the following question based on the audio content using the format:1. Brief answer\n2. Detailed analysis\n3. Relevant context\n\n"
                }]
            },
            {
                "role": MessageRole.USER,
                "content": [{
                    "type": "text",
                    "text": f"Please answer the question: {question}"
                }]
            }
        ]
        
        return self.model(messages).content