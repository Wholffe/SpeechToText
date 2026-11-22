from collections import deque

import numpy as np

from speech_to_text import constants as CONST


def rms(block: np.ndarray) -> float:
    """Returns the root mean square (loudness) of an audio block."""
    return float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))


def _ms_to_blocks(duration_ms: float, block_ms: float) -> int:
    """Converts a duration into a number of blocks, at least one."""
    return max(1, round(duration_ms / block_ms))


class SpeechSegmenter:
    """Splits a stream of microphone blocks into speech segments at pauses.

    A block counts as speech when it is clearly louder than the background noise,
    which is estimated from a low percentile of the recent block levels. A segment
    ends after a pause of `SILENCE_MS` or when it reaches `MAX_SEGMENT_S`.
    """

    def __init__(
        self,
        block_size: int = CONST.BLOCK_SIZE,
        sample_rate: int = CONST.SAMPLE_RATE,
    ) -> None:
        block_ms = block_size * 1000 / sample_rate
        self._silence_blocks = _ms_to_blocks(CONST.SILENCE_MS, block_ms)
        self._min_speech_blocks = _ms_to_blocks(CONST.MIN_SPEECH_MS, block_ms)
        self._max_blocks = _ms_to_blocks(CONST.MAX_SEGMENT_S * 1000, block_ms)
        self._pre_roll: deque[np.ndarray] = deque(
            maxlen=_ms_to_blocks(CONST.PRE_ROLL_MS, block_ms)
        )
        self._levels: deque[float] = deque(
            maxlen=_ms_to_blocks(CONST.NOISE_WINDOW_S * 1000, block_ms)
        )
        self._segment: list[np.ndarray] = []
        self._speech_blocks = 0
        self._silent_blocks = 0

    @property
    def threshold(self) -> float:
        """Returns the level above which a block counts as speech."""
        if not self._levels:
            return CONST.MIN_THRESHOLD
        noise_floor = float(np.percentile(self._levels, CONST.NOISE_PERCENTILE))
        return max(noise_floor * CONST.THRESHOLD_FACTOR, CONST.MIN_THRESHOLD)

    @property
    def is_in_speech(self) -> bool:
        """Returns whether a segment is currently being collected."""
        return bool(self._segment)

    def process(self, block: np.ndarray) -> np.ndarray | None:
        """Adds a block and returns the finished segment once it is complete."""
        level = rms(block)
        is_loud = level >= self.threshold
        self._levels.append(level)

        if not self.is_in_speech:
            if is_loud:
                self._segment = [*self._pre_roll, block]
                self._pre_roll.clear()
                self._speech_blocks = 1
                self._silent_blocks = 0
            else:
                self._pre_roll.append(block)
            return None

        self._segment.append(block)
        if is_loud:
            self._speech_blocks += 1
            self._silent_blocks = 0
        else:
            self._silent_blocks += 1

        if (
            self._silent_blocks >= self._silence_blocks
            or len(self._segment) >= self._max_blocks
        ):
            return self._finish_segment()
        return None

    def flush(self) -> np.ndarray | None:
        """Returns the unfinished segment, e.g. when the recording stops."""
        if not self.is_in_speech:
            return None
        return self._finish_segment()

    def _finish_segment(self) -> np.ndarray | None:
        """Resets the state and returns the segment, unless it was too short."""
        segment = self._segment
        speech_blocks = self._speech_blocks
        self._segment = []
        self._speech_blocks = 0
        self._silent_blocks = 0

        if speech_blocks < self._min_speech_blocks:
            return None
        return np.concatenate(segment)
