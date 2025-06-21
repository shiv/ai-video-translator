import array

import numpy as np

from faster_whisper import WhisperModel

from app import logger
from app.services.stt.speech_to_text import SpeechToText


class SpeechToTextFasterWhisper(SpeechToText):

    def __init__(self, *, model_name="medium", device="cpu", cpu_threads=0, vad=False):
        super().__init__(device=device, model_name=model_name, cpu_threads=cpu_threads)
        self.vad = vad

    def load_model(self):
        self._model = WhisperModel(
            model_size_or_path=self.model_name,
            device=self.device,
            cpu_threads=self.cpu_threads,
            compute_type="float16" if self.device == "cuda" else "int8",
        )

    def get_languages(self):
        iso_639_3 = []
        for language in self.model.supported_languages:
            pt3 = self._get_iso_639_3(language)
            iso_639_3.append(pt3)
        return iso_639_3

    def _transcribe(
        self,
        *,
        audio_filepath: str,
        source_language_iso_639_1: str,
    ) -> str:
        segments, _ = self.model.transcribe(
            audio_filepath, source_language_iso_639_1, vad_filter=self.vad
        )
        return " ".join(segment.text for segment in segments)

    def _get_audio_language(self, audio: array.array) -> str:
        audio_input = np.array(audio).astype(np.float32) / 32768.0
        _, info = self.model.transcribe(audio_input)
        detected_language = self._get_iso_639_3(info.language)
        logger().debug(
            f"speech_to_text_faster_whisper._get_audio_language. Detected language: {detected_language}"
        )
        return detected_language
