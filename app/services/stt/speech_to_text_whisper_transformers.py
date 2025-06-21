import array

import numpy as np
import torch

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from app import logger
from app.services.processing.utils.pydub_audio_segment import AudioSegment
from app.services.stt.speech_to_text import SpeechToText


class SpeechToTextWhisperTransformers(SpeechToText):

    def __init__(self, *, model_name="medium", device="cpu", cpu_threads=0):
        super().__init__(device=device, model_name=model_name, cpu_threads=cpu_threads)
        self._processor = None

    def load_model(self):
        full_model_name = f"openai/whisper-{self.model_name}"
        self._processor = WhisperProcessor.from_pretrained(full_model_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(full_model_name)

    def _transcribe(
        self,
        *,
        audio_filepath: str,
        source_language_iso_639_1: str,
    ) -> str:
        audio = AudioSegment.from_file(audio_filepath)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set the frame rate to 16kHz
        # Convert the audio to a numpy array
        audio_input = (
            np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        )  # Normalize

        # Preprocess the audio input
        input_features = self._processor(
            audio_input, sampling_rate=16000, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_features, language=source_language_iso_639_1
            )
        transcription = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        logger().debug(
            f"speech_to_text_whisper_transfomers._transcribe. transcription: {transcription}, file {audio_filepath}"
        )
        return transcription

    def _get_audio_language(self, audio: array.array) -> str:
        audio_input = np.array(audio).astype(np.float32) / 32768.0

        # Preprocess the audio input
        input_features = self._processor(
            audio_input, sampling_rate=16000, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            generated_ids = self._model.generate(input_features)

        # Decode the transcription including special tokens to capture the language token
        transcription_with_tokens = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        detected_language = None
        if "|" in transcription_with_tokens:
            for token in transcription_with_tokens.split("|"):
                if (len(token) == 2 or len(token) == 3) and token.isalpha():
                    detected_language = self._get_iso_639_3(token)
                    break
        logger().debug(
            f"speech_to_text_whisper_transfomers._get_audio_language. Detected language: {detected_language}"
        )

        return detected_language

    def get_languages(self):
        languages = [
            "af",
            "am",
            "ar",
            "as",
            "az",
            "ba",
            "be",
            "bg",
            "bn",
            "bo",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "gl",
            "gu",
            "ha",
            "haw",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "is",
            "it",
            "ja",
            "jw",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "la",
            "lb",
            "ln",
            "lo",
            "lt",
            "lv",
            "mg",
            "mi",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "pa",
            "pl",
            "ps",
            "pt",
            "ro",
            "ru",
            "sa",
            "sd",
            "si",
            "sk",
            "sl",
            "sn",
            "so",
            "sq",
            "sr",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tk",
            "tl",
            "tr",
            "tt",
            "uk",
            "ur",
            "uz",
            "vi",
            "yi",
            "yo",
            "zh",
            "yue",
        ]

        iso_639_3 = []
        for language in languages:
            pt3 = self._get_iso_639_3(language)
            iso_639_3.append(pt3)
        return iso_639_3
