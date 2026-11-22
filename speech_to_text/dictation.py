import queue
import threading

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .logger import logger
from .recorder import MicrophoneRecorder
from .segmenter import SpeechSegmenter
from .transcriber import Transcriber

QUEUE_POLL_S = 0.1


class DictationEngine(QObject):
    """Records and transcribes in background threads and reports via Qt signals.

    While recording, every pause in speech completes a segment, which is
    transcribed right away. After `stop()`, the remaining audio is transcribed
    before `finished` is emitted.
    """

    model_loaded = pyqtSignal(str)
    model_failed = pyqtSignal(str, str)  # model name, error
    text_ready = pyqtSignal(str)
    transcribing_changed = pyqtSignal(bool)
    transcription_failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._recorder = MicrophoneRecorder()
        self._transcriber = Transcriber()
        self._stop_requested = threading.Event()
        self.language: str | None = None

    @property
    def level(self) -> float:
        """Returns the current microphone level as RMS."""
        return self._recorder.level

    def load_model(self, model_name: str) -> None:
        """Loads a model in the background and emits `model_loaded` when done."""
        threading.Thread(
            target=self._load_model, args=(model_name,), daemon=True
        ).start()

    def start(self) -> None:
        """Starts recording. Raises `sd.PortAudioError` without a microphone."""
        self._recorder.start()
        self._stop_requested.clear()
        threading.Thread(target=self._process_audio, daemon=True).start()

    def stop(self) -> None:
        """Stops recording. Speech already recorded is still transcribed."""
        self._recorder.stop()
        self._stop_requested.set()

    def _load_model(self, model_name: str) -> None:
        """Loads the model and reports the result."""
        logger.info(f"Loading model '{model_name}'")
        try:
            self._transcriber.load(model_name)
        except Exception as error:  # download, disk and format errors share no base
            logger.error(f"Failed to load model '{model_name}': {error}", exc_info=True)
            self.model_failed.emit(model_name, str(error))
            return
        logger.info(f"Model '{model_name}' loaded")
        self.model_loaded.emit(model_name)

    def _process_audio(self) -> None:
        """Splits the recorded blocks into segments and transcribes each one."""
        segmenter = SpeechSegmenter()
        while True:
            try:
                block = self._recorder.blocks.get(timeout=QUEUE_POLL_S)
            except queue.Empty:
                if self._stop_requested.is_set():
                    break
                continue
            segment = segmenter.process(block)
            if segment is not None:
                self._transcribe(segment)

        segment = segmenter.flush()
        if segment is not None:
            self._transcribe(segment)
        self.finished.emit()

    def _transcribe(self, audio: np.ndarray) -> None:
        """Transcribes one segment and emits the text, if there is any."""
        self.transcribing_changed.emit(True)
        try:
            text = self._transcriber.transcribe(audio, self.language)
        except Exception as error:  # one failed segment must not end the recording
            logger.error(f"Failed to transcribe segment: {error}", exc_info=True)
            self.transcription_failed.emit(str(error))
            return
        finally:
            self.transcribing_changed.emit(False)

        if text:
            self.text_ready.emit(text)
