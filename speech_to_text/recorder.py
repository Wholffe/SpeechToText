import queue

import numpy as np
import sounddevice as sd

from speech_to_text import constants as CONST

from .logger import logger
from .segmenter import rms


class MicrophoneRecorder:
    """Records mono 16 kHz audio from the default input device into a block queue."""

    def __init__(self) -> None:
        self.blocks: queue.Queue[np.ndarray] = queue.Queue()
        self.level = 0.0  # RMS of the latest block, read by the level meter
        self._stream: sd.InputStream | None = None

    @property
    def is_recording(self) -> bool:
        """Returns whether the microphone stream is open."""
        return self._stream is not None

    def start(self) -> None:
        """Opens the default microphone. Raises `sd.PortAudioError` on failure."""
        self.blocks = queue.Queue()
        self.level = 0.0
        stream = sd.InputStream(
            samplerate=CONST.SAMPLE_RATE,
            blocksize=CONST.BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        try:
            stream.start()
        except sd.PortAudioError:
            stream.close()
            raise
        self._stream = stream

    def stop(self) -> None:
        """Closes the microphone. Blocks already queued stay available."""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.level = 0.0

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Copies a block from the audio thread into the queue."""
        if status:
            logger.warning(f"Audio input reported '{status}'")
        block = indata[:, 0].copy()
        self.level = rms(block)
        self.blocks.put(block)
