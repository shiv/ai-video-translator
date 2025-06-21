import os
import warnings

from typing import Final, Mapping, Sequence

import torch

from pyannote.audio import Pipeline

from app import logger
from app.services.processing.utils.pydub_audio_segment import AudioSegment

_DEFAULT_DUBBED_AUDIO_FILE: Final[str] = "dubbed_audio"
_DEFAULT_OUTPUT_FORMAT: Final[str] = ".mp3"


def create_pyannote_timestamps(
    *,
    audio_file: str,
    pipeline: Pipeline,
    device: str = "cpu",
) -> Sequence[Mapping[str, float]]:
    """Create timestamps from audio using Pyannote speaker diarization."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if device == "cuda":
            pipeline.to(torch.device("cuda"))
        diarization = pipeline(audio_file)
        utterance_metadata = [
            {"start": segment.start, "end": segment.end, "speaker_id": speaker}
            for segment, _, speaker in diarization.itertracks(yield_label=True)
        ]
        return utterance_metadata


def _cut_and_save_audio(
    *,
    audio: AudioSegment,
    utterance: Mapping[str, str | float],
    prefix: str,
    output_directory: str,
) -> str:
    """Cut audio segment and save as MP3."""
    start_time_ms = int(utterance["start"] * 1000)
    end_time_ms = int(utterance["end"] * 1000)
    chunk = audio[start_time_ms:end_time_ms]
    chunk_filename = f"{prefix}_{utterance['start']}_{utterance['end']}.mp3"
    chunk_path = os.path.join(output_directory, chunk_filename)
    chunk.export(chunk_path, format="mp3")
    return chunk_path


def run_cut_and_save_audio(
    *,
    utterance_metadata: Sequence[Mapping[str, float]],
    audio_file: str,
    output_directory: str,
) -> Sequence[Mapping[str, float]]:
    """Cut audio into chunks based on timestamps and save each chunk."""

    audio = AudioSegment.from_file(audio_file)
    key = "path"
    prefix = "chunk"
    updated_utterance_metadata = []
    for utterance in utterance_metadata:
        chunk_path = _cut_and_save_audio(
            audio=audio,
            utterance=utterance,
            prefix=prefix,
            output_directory=output_directory,
        )
        utterance_copy = utterance.copy()
        utterance_copy[key] = chunk_path
        updated_utterance_metadata.append(utterance_copy)
    return updated_utterance_metadata


def create_dubbed_audio_track(
    *,
    utterance_metadata: Sequence[Mapping[str, str | float]],
    original_audio_file: str,
    output_directory: str,
    target_language: str,
) -> str:
    """Create complete dubbed audio track by replacing speech segments."""
    
    original_audio = AudioSegment.from_file(original_audio_file)
    total_duration = len(original_audio)
    
    dubbed_audio = AudioSegment.silent(duration=total_duration)
    
    for item in utterance_metadata:
        try:
            for_dubbing = item["for_dubbing"]
            
            if for_dubbing:
                dubbed_file = item["dubbed_path"]
                start_time = int(item["start"] * 1000)
                
                logger().debug(f"create_dubbed_audio_track. Inserting dubbed audio: {dubbed_file}")
                dubbed_chunk = AudioSegment.from_mp3(dubbed_file)
                dubbed_audio = dubbed_audio.overlay(dubbed_chunk, position=start_time)
            else:
                start_time = int(item["start"] * 1000)
                end_time = int(item["end"] * 1000)
                
                original_chunk = original_audio[start_time:end_time]
                dubbed_audio = dubbed_audio.overlay(original_chunk, position=start_time)
                
        except Exception as e:
            start = int(item["start"])
            end = int(item["end"])
            dubbed_file = item.get("dubbed_path", "N/A")
            logger().error(
                f"create_dubbed_audio_track. Error processing segment at {start}-{end}s, file: {dubbed_file}, error: {e}"
            )

    target_language_suffix = "_" + target_language.replace("-", "_").lower()
    dubbed_audio_file = os.path.join(
        output_directory,
        _DEFAULT_DUBBED_AUDIO_FILE + target_language_suffix + _DEFAULT_OUTPUT_FORMAT,
    )
    dubbed_audio.export(dubbed_audio_file, format="mp3")
    return dubbed_audio_file
