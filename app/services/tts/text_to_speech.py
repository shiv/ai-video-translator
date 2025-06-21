import math
import os

from abc import ABC, abstractmethod
from typing import Final, List, Mapping, NamedTuple, Sequence

from app import logger
from app.services.processing.ffmpeg import FFmpeg
from app.services.processing.utils.pydub_audio_segment import AudioSegment
from app.services.processing.utterance import Utterance


class Voice(NamedTuple):
    name: str
    gender: str


class TextToSpeech(ABC):

    def __init__(self):
        self._SSML_MALE: Final[str] = "Male"
        self._SSML_FEMALE: Final[str] = "Female"
        self._DEFAULT_SPEED: Final[float] = 1.0

    @abstractmethod
    def get_available_voices(self, language_code: str) -> List[Voice]:
        pass

    def assign_voices(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        target_language: str,
    ) -> Mapping[str, str | None]:

        voices = self.get_available_voices(target_language)

        voice_assignment = {}
        used_voices = set()
        for chunk in utterance_metadata:
            speaker_id = chunk["speaker_id"]
            if speaker_id in voice_assignment:
                continue

            gender = chunk["gender"]
            for voice in voices:  # Try to use an unused voice of the same gender
                if (
                    voice.name not in used_voices
                    and voice.gender.lower() == gender.lower()
                ):
                    voice_assignment[speaker_id] = voice.name
                    used_voices.add(voice.name)
                    break
            else:
                for (
                    voice
                ) in voices:  # Try to use an already used voice of same gender
                    if voice.gender.lower() == gender.lower():
                        voice_assignment[speaker_id] = voice.name
                        used_voices.add(voice.name)
                        break
                else:  # Try to use any other voice of any gender even if it used
                    for voice in voices:
                        voice_assignment[speaker_id] = voice.name
                        used_voices.add(voice.name)
                        break

        logger().info(f"text_to_speech.assign_voices. Returns: {voice_assignment}")
        return voice_assignment

    def _convert_to_mp3(self, input_file, output_mp3):
        FFmpeg().convert_to_format(source=input_file, target=output_mp3)
        os.remove(input_file)

    def _add_text_to_speech_properties(
        self,
        *,
        utterance_metadata: Mapping[str, str | float],
    ) -> Mapping[str, str | float]:
        """Updates utterance metadata with Text-To-Speech properties."""
        utterance_metadata_copy = utterance_metadata.copy()
        voice_properties = dict(
            speed=self._DEFAULT_SPEED,
        )
        utterance_metadata_copy.update(voice_properties)
        return utterance_metadata_copy

    def update_utterance_metadata(
        self,
        *,
        utterance: Utterance | None = None,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        assigned_voices: Mapping[str, str] | None,
    ) -> Sequence[Mapping[str, str | float]]:
        """Updates utterance metadata with assigned voices."""
        updated_utterance_metadata = []
        for metadata_item in utterance_metadata:
            new_utterance = metadata_item.copy()

            fields = (
                utterance.get_modified_utterance_fields(new_utterance)
                if utterance
                else []
            )
            # If "assigned_voice" has changed we give it priority and not overwrite it
            # by recalculating from speaker_id/gender
            if not utterance or (
                "speaker_id" in fields and "assigned_voice" not in fields
            ):
                speaker_id = new_utterance.get("speaker_id")
                new_utterance["assigned_voice"] = assigned_voices.get(speaker_id)
                new_utterance = self._add_text_to_speech_properties(
                    utterance_metadata=new_utterance
                )
            updated_utterance_metadata.append(new_utterance)
        return updated_utterance_metadata

    @abstractmethod
    def get_languages(self):
        pass

    """ TTS add silence at the end that we want to remove to prevent increasing the speech of next
        segments if is not necessary."""

    def _convert_text_to_speech_without_end_silence(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
    ) -> str:

        dubbed_file = self._convert_text_to_speech(
            assigned_voice=assigned_voice,
            target_language=target_language,
            output_filename=output_filename,
            text=text,
            speed=speed,
        )

        dubbed_audio = AudioSegment.from_file(dubbed_file)
        pre_duration = len(dubbed_audio)

        FFmpeg().remove_silence(filename=dubbed_file)
        dubbed_audio = AudioSegment.from_file(dubbed_file)
        post_duration = len(dubbed_audio)
        if pre_duration != post_duration:
            logger().debug(
                f"text_to_speech._convert_text_to_speech_without_end_silence. File {dubbed_file} shorten from {pre_duration} to {post_duration}"
            )

        return dubbed_file

    @abstractmethod
    def _convert_text_to_speech(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
    ) -> str:
        pass

    def _calculate_target_utterance_speed(
        self,
        *,
        start: float,
        end: float,
        dubbed_file: str,
        utterance_metadata: Sequence[Mapping[str, float | str]],
        audio_file=str,
    ) -> float:
        """Returns the ratio between the reference and target duration."""

        end = self.get_start_time_of_next_speech_utterance(
            utterance_metadata=utterance_metadata,
            start=start,
            end=end,
            audio_file=audio_file,
        )

        reference_length = end - start
        dubbed_audio = AudioSegment.from_file(dubbed_file)
        dubbed_duration = dubbed_audio.duration_seconds
        r = (
            math.ceil(dubbed_duration / reference_length * 10) / 10
        )  # Rounds up with .1 decimal precision
        logger().debug(f"text_to_speech._calculate_target_utterance_speed: {r}")
        return r

    def _does_voice_supports_speeds(self):
        return False

    def get_start_time_of_next_speech_utterance(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        start: float,
        end: float,
        audio_file: str,
    ) -> float:
        result = None
        for utterance in utterance_metadata:
            u_start = utterance["start"]
            if u_start <= start:
                continue

            for_dubbing = utterance["for_dubbing"]
            if not for_dubbing:
                continue

            result = u_start
            break

        if not result:
            try:
                background_audio = AudioSegment.from_mp3(audio_file)
                total_duration = background_audio.duration_seconds
                logger().debug(
                    f"get_start_time_of_next_speech_utterance. File duration: {total_duration}"
                )
                result = total_duration
            except Exception as e:
                logger().error(f"Error '{e}' reading {audio_file}")

        if not result:
            result = end

        logger().debug(
            f"get_start_time_of_next_speech_utterance from_time: {start}, result: {result}"
        )
        return result

    def dub_utterances(
        self,
        *,
        utterance_metadata: Sequence[Mapping[str, str | float]],
        output_directory: str,
        target_language: str,
        audio_file: str,
    ) -> Sequence[Mapping[str, str | float]]:
        """Processes a list of utterance metadata, generating dubbed audio files."""

        updated_utterance_metadata = []
        for utterance in utterance_metadata:
            utterance_copy = utterance.copy()
            if not utterance_copy["for_dubbing"]:
                try:
                    dubbed_path = utterance_copy["path"]
                except KeyError:
                    dubbed_path = f"chunk_{utterance['start']}_{utterance['end']}.mp3"
            else:
                assigned_voice = utterance_copy["assigned_voice"]
                text = utterance_copy["translated_text"]
                try:
                    path = utterance_copy["path"]
                    base_filename = os.path.splitext(os.path.basename(path))[0]
                    output_filename = os.path.join(
                        output_directory, f"dubbed_{base_filename}.mp3"
                    )
                except KeyError:
                    output_filename = os.path.join(
                        output_directory,
                        f"dubbed_chunk_{utterance['start']}_{utterance['end']}.mp3",
                    )

                speed = utterance_copy["speed"]
                dubbed_path = self._convert_text_to_speech_without_end_silence(
                    assigned_voice=assigned_voice,
                    target_language=target_language,
                    output_filename=output_filename,
                    text=text,
                    speed=speed,
                )
                assigned_voice = utterance_copy.get("assigned_voice", None)
                assigned_voice = assigned_voice if assigned_voice else ""
                support_speeds = self._does_voice_supports_speeds()

                start = utterance["start"]
                end = utterance["end"]
                speed = self._calculate_target_utterance_speed(
                    start=start,
                    end=end,
                    dubbed_file=dubbed_path,
                    utterance_metadata=utterance_metadata,
                    audio_file=audio_file,
                )

                logger().debug(f"support_speeds: {support_speeds}, speed: {speed}")

                if speed > 1.0:
                    translated_text = utterance_copy["translated_text"]
                    logger().debug(
                        f"text_to_speech.dub_utterances. Need to increase speed for '{translated_text}'"
                    )

                    MAX_SPEED = 1.3
                    if speed > MAX_SPEED:
                        logger().debug(
                            f"text_to_speech.dub_utterances: Reduced speed from {speed} to {MAX_SPEED}"
                        )
                        speed = MAX_SPEED

                    translated_text = utterance_copy["translated_text"]
                    logger().debug(
                        f"text_to_speech.dub_utterances: Adjusting speed to {speed} for '{translated_text}'"
                    )

                    utterance_copy["speed"] = speed
                    if support_speeds:
                        dubbed_path = self._convert_text_to_speech_without_end_silence(
                            assigned_voice=assigned_voice,
                            target_language=target_language,
                            output_filename=output_filename,
                            text=text,
                            speed=speed,
                        )
                    else:
                        FFmpeg().adjust_audio_speed(
                            filename=dubbed_path,
                            speed=speed,
                        )
                        logger().debug(
                            f"text_to_speech.adjust_audio_speed: dubbed_audio: {dubbed_path}, speed: {speed}"
                        )
                else:
                    utterance_copy["speed"] = self._DEFAULT_SPEED

            utterance_copy["dubbed_path"] = dubbed_path
            updated_utterance_metadata.append(utterance_copy)
        return updated_utterance_metadata
