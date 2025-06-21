import dataclasses
import hashlib
import json
import os
import shutil
import tempfile

from typing import Any, Dict, Final, List, Tuple

from app import logger
from app.services.processing.preprocessing import PreprocessingArtifacts


class Utterance:

    def __init__(self, target_language: str, output_directory: str):
        self.target_language = target_language
        self.output_directory = output_directory

    def _get_file_name(self):
        _UTTERNACE_METADATA_FILE_NAME: Final[str] = "utterance_metadata"

        target_language_suffix = "_" + self.target_language.replace("-", "_").lower()
        utterance_metadata_file = os.path.join(
            self.output_directory,
            _UTTERNACE_METADATA_FILE_NAME + target_language_suffix + ".json",
        )
        return utterance_metadata_file

    def load_utterances(self) -> tuple[Any, PreprocessingArtifacts, Any]:
        utterance_metadata_file = self._get_file_name()

        with open(utterance_metadata_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            utterances = data["utterances"]
            preprocessing_output = PreprocessingArtifacts(
                **data["PreprocessingArtifacts"]
            )
            metadata = data["metadata"]

        return utterances, preprocessing_output, metadata

    def save_utterances(
        self,
        *,
        utterance_metadata: str,
        preprocessing_output: str,
        metadata: Dict[str, str],
        do_hash: bool = True,
        unique_id: bool = True,
    ) -> None:
        """Save utterance metadata to JSON file."""
        utterance_metadata_file = self._get_file_name()

        try:
            all_data = {}
            if unique_id:
                utterance_metadata = self._add_unique_ids(utterance_metadata)

            if do_hash:
                utterance_metadata = self._hash_utterances(utterance_metadata)

            all_data["utterances"] = utterance_metadata
            if preprocessing_output:
                all_data["PreprocessingArtifacts"] = dataclasses.asdict(
                    preprocessing_output
                )
            all_data["metadata"] = metadata

            json_data = json.dumps(all_data, ensure_ascii=False, indent=4)
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, encoding="utf-8"
            ) as temporary_file:

                temporary_file.write(json_data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            shutil.copy(temporary_file.name, utterance_metadata_file)
            os.remove(temporary_file.name)
            logger().debug(
                "Utterance metadata saved successfully to"
                f" '{utterance_metadata_file}'"
            )
        except Exception as e:
            logger().warning(f"Error saving utterance metadata: {e}")

    def _get_utterance_fields_to_hash(self, utterance):
        filtered_fields = {
            key: value for key, value in utterance.items() if not key.startswith("_")
        }
        return filtered_fields

    def _hash_utterances(self, utterance_metadata):
        for utterance in utterance_metadata:
            filtered_fields = self._get_utterance_fields_to_hash(utterance)
            dict_str = json.dumps(filtered_fields, sort_keys=True)
            _hash = hashlib.sha256(dict_str.encode()).hexdigest()
            utterance["_hash"] = _hash

            for field in ["assigned_voice", "speaker_id"]:
                value = utterance.get(field)
                if value:
                    utterance[f"_{field}_hash"] = hashlib.sha256(
                        value.encode()
                    ).hexdigest()

        return utterance_metadata

    def _add_unique_ids(self, utterance_metadata):
        for idx, utterance in enumerate(utterance_metadata, start=1):
            new_utterance = {"id": idx}
            new_utterance.update(utterance)
            utterance_metadata[idx - 1] = new_utterance

        return utterance_metadata

    def get_files_paths(self, utterance_metadata) -> Tuple[List[str], List[str]]:
        dubbed_paths = []
        paths = []
        for chunk in utterance_metadata:
            if "path" in chunk:
                paths.append(chunk["path"])
            if "dubbed_path" in chunk:
                dubbed_paths.append(chunk["dubbed_path"])

        return paths, dubbed_paths

    def get_modified_utterance_fields(self, utterance):
        modified = []
        for field in utterance:
            field_hash = utterance.get(f"_{field}_hash")
            if not field_hash:
                continue

            field_value = utterance[field]
            current_hash = hashlib.sha256(field_value.encode()).hexdigest()

            if current_hash != field_hash:
                modified.append(field)

        return modified

    def get_without_empty_blocks(self, utterance_metadata):
        new_utterance = []

        for utterance in utterance_metadata:
            text = utterance["text"]
            if len(text) == 0:
                logger().debug(f"Removing empty block: {utterance}")
                continue

            new_utterance.append(utterance)

        return new_utterance
