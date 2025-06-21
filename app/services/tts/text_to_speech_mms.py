from typing import List

import numpy as np
import scipy.io.wavfile
import torch

from transformers import AutoTokenizer, VitsModel

from app import logger
from app.services.tts.text_to_speech import TextToSpeech, Voice


class TextToSpeechMMS(TextToSpeech):

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def get_available_voices(self, language_code: str) -> List[Voice]:
        return [Voice(name="voice", gender=self._SSML_MALE)]

    def _convert_text_to_speech(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
    ) -> str:

        logger().debug(f"TextToSpeechMMS._convert_text_to_speech: {text}")
        local_files_only = False

        # Load pre-trained model and tokenizer
        model = VitsModel.from_pretrained(
            f"facebook/mms-tts-{target_language}", local_files_only=local_files_only
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            f"facebook/mms-tts-{target_language}", local_files_only=local_files_only
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        # Model returns for some sequences of tokens no result
        if inputs["input_ids"].shape[1] == 0:
            sampling_rate = 16000
            duration_seconds = 1
            # If we fill the array with (np.zeros) the ffmpeg process later fails
            output_np = np.ones(sampling_rate * duration_seconds, dtype=np.int16)
            logger().warning(
                f"TextToSpeechMMS._convert_text_to_speech. Model returns input tokens for text '{text}', generating an empty WAV file."
            )
        else:
            with torch.no_grad():
                output = model(**inputs).waveform

            # Convert waveform to NumPy array and scale to 16-bit PCM
            # Assuming `output` is a 2D tensor with shape (batch_size, samples)
            output_np = (
                output.squeeze().cpu().numpy()
            )  # Remove batch dimension if present
            output_np = np.clip(output_np, -1, 1)  # Clip values to be between -1 and 1
            output_np = (output_np * 32767).astype(np.int16)  # Scale to 16-bit PCM

            # Get the sampling rate
            sampling_rate = model.config.sampling_rate

        # Write to WAV file
        wav_file = output_filename.replace(".mp3", ".wav")
        scipy.io.wavfile.write(wav_file, rate=sampling_rate, data=output_np)

        self._convert_to_mp3(wav_file, output_filename)
        logger().debug(
            f"text_to_speech.client.synthesize_speech: output_filename: '{output_filename}'"
        )
        return output_filename

    def get_languages(self):
        return [
            "eng",
            "spa",
            "fra",
            "deu",
            "ita",
            "por",
            "jpn",
            "kor",
            "cmn",
            "hin",
        ]
