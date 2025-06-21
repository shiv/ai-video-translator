import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from app import logger
from app.services.processing.utils.pydub_audio_segment import AudioSegment


class ModelHead(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)  # Original model has 3 classes
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender


class VoiceGenderClassifier:

    FEMALE = "Female"
    MALE = "Male"

    def __init__(self, device="cpu"):
        self.device = device
        model_name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name).to(self.device)

    def load_audio_file(self, file_path, target_sampling_rate=16000):
        max_duration = 10

        audio = AudioSegment.from_file(file_path, format="mp3")

        if len(audio) > max_duration * 1000:
            audio = audio[: max_duration * 1000]

        audio = audio.set_frame_rate(target_sampling_rate)

        samples = np.array(audio.get_array_of_samples())

        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        samples = samples.astype(np.float32) / np.iinfo(np.int16).max

        samples = np.expand_dims(samples, axis=0)

        return samples, target_sampling_rate

    def _predict(
        self,
        x: np.ndarray,
        sampling_rate: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict age and gender from raw audio signal."""

        y = self.processor(x, sampling_rate=sampling_rate)
        y = y["input_values"][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.device)

        with torch.no_grad():
            y = self.model(y)
            logits_age, logits_gender = y[1], y[2]
            return logits_age, logits_gender

    def _interpret_gender(self, logits_gender):
        """Convert gender logits into male/female label."""
        male_female_logits = logits_gender[:, :2]  # Take first two logits (female, male)

        gender_labels = [self.FEMALE, self.MALE]

        predicted_gender_idx = torch.argmax(male_female_logits, dim=1).item()
        probabilities = F.softmax(male_female_logits, dim=1).cpu()
        prob_female, prob_male = probabilities[0].tolist()

        return gender_labels[predicted_gender_idx]

    def get_gender_for_file(self, file_path):
        try:
            signal, sampling_rate = self.load_audio_file(file_path)

            _, gender_logits = self._predict(signal, sampling_rate)
            predicted_gender = self._interpret_gender(gender_logits)
        except Exception as e:
            logger().error(
                f"voice_gender_classifier. get_gender_for_file. Error '{e}' processing {file_path}"
            )
            predicted_gender = self.FEMALE

        logger().debug(
            f"The audio from {os.path.basename(file_path)} is predicted {predicted_gender}"
        )
        return predicted_gender
