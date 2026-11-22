import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from speech_to_text import constants as CONST


def _is_hallucination(segment: Segment) -> bool:
    """Detects text Whisper invents for noise, based on its own confidence values."""
    return (
        segment.no_speech_prob > CONST.NO_SPEECH_PROB_MAX
        and segment.avg_logprob < CONST.AVG_LOGPROB_MIN
    )


class Transcriber:
    """Turns audio into text with a faster-whisper model running on the CPU."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self.model_name: str | None = None

    def load(self, model_name: str) -> None:
        """Loads a model, which faster-whisper downloads on first use."""
        self._model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=CONST.COMPUTE_TYPE,
            cpu_threads=CONST.CPU_THREADS,
        )
        self.model_name = model_name

    def transcribe(self, audio: np.ndarray, language: str | None) -> str:
        """Returns the text spoken in a 16 kHz mono recording (`None` detects it)."""
        if self._model is None:
            raise RuntimeError("No model loaded")

        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=CONST.BEAM_SIZE,
            vad_filter=True,  # removes silence, the main source of invented text
            condition_on_previous_text=False,
        )
        texts = [
            segment.text.strip()
            for segment in segments
            if not _is_hallucination(segment)
        ]
        return " ".join(text for text in texts if text)
