import dataclasses


@dataclasses.dataclass
class PreprocessingArtifacts:
    """Preprocessing outputs."""
    video_file: str | None
    audio_file: str
