import unittest

import numpy as np

from speech_to_text import constants as CONST
from speech_to_text.segmenter import SpeechSegmenter, rms

SILENCE_LEVEL = 0.001
NOISE_LEVEL = 0.02
SPEECH_LEVEL = 0.1


def _blocks_for(duration_ms: int) -> int:
    """Returns the number of blocks that cover a duration."""
    return duration_ms // CONST.BLOCK_MS


def _block(level: float) -> np.ndarray:
    """Returns a block whose RMS equals `level`."""
    return np.full(CONST.BLOCK_SIZE, level, dtype=np.float32)


class TestSpeechSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.segmenter = SpeechSegmenter()

    def _feed(self, level: float, duration_ms: int) -> list[np.ndarray]:
        """Feeds blocks of one level and returns the finished segments."""
        segments = []
        for _ in range(_blocks_for(duration_ms)):
            segment = self.segmenter.process(_block(level))
            if segment is not None:
                segments.append(segment)
        return segments

    def test_rms_of_constant_block(self) -> None:
        self.assertAlmostEqual(rms(_block(0.5)), 0.5, places=6)

    def test_silence_yields_no_segment(self) -> None:
        self.assertEqual(self._feed(SILENCE_LEVEL, 3000), [])
        self.assertIsNone(self.segmenter.flush())

    def test_speech_followed_by_pause_yields_segment(self) -> None:
        self._feed(SILENCE_LEVEL, 1000)
        self.assertEqual(self._feed(SPEECH_LEVEL, 1000), [])

        segments = self._feed(SILENCE_LEVEL, 1000)

        self.assertEqual(len(segments), 1)
        expected_blocks = (
            _blocks_for(CONST.PRE_ROLL_MS)
            + _blocks_for(1000)
            + _blocks_for(CONST.SILENCE_MS)
        )
        self.assertEqual(len(segments[0]), expected_blocks * CONST.BLOCK_SIZE)

    def test_segment_starts_with_pre_roll(self) -> None:
        self._feed(SILENCE_LEVEL, 1000)
        self._feed(SPEECH_LEVEL, 1000)

        segment = self._feed(SILENCE_LEVEL, 1000)[0]

        pre_roll_samples = _blocks_for(CONST.PRE_ROLL_MS) * CONST.BLOCK_SIZE
        self.assertAlmostEqual(rms(segment[:pre_roll_samples]), SILENCE_LEVEL, 6)
        self.assertAlmostEqual(rms(segment[pre_roll_samples:][:100]), SPEECH_LEVEL, 6)

    def test_two_utterances_yield_two_segments(self) -> None:
        self._feed(SILENCE_LEVEL, 1000)
        segments = []
        for _ in range(2):
            segments += self._feed(SPEECH_LEVEL, 800)
            segments += self._feed(SILENCE_LEVEL, 1000)

        self.assertEqual(len(segments), 2)

    def test_short_click_is_discarded(self) -> None:
        self._feed(SILENCE_LEVEL, 1000)
        self._feed(SPEECH_LEVEL, CONST.BLOCK_MS * 2)

        self.assertEqual(self._feed(SILENCE_LEVEL, 1000), [])

    def test_long_speech_is_split_at_max_length(self) -> None:
        self._feed(SILENCE_LEVEL, 1000)

        # Real speech has short dips between words, which keep the noise floor low
        segments = []
        for _ in range((CONST.MAX_SEGMENT_S + 1) * 1000 // 300):
            segments += self._feed(SPEECH_LEVEL, 240)
            segments += self._feed(SILENCE_LEVEL, 60)

        self.assertEqual(len(segments), 1)
        max_samples = CONST.MAX_SEGMENT_S * 1000 // CONST.BLOCK_MS * CONST.BLOCK_SIZE
        self.assertEqual(len(segments[0]), max_samples)
        self.assertTrue(self.segmenter.is_in_speech)

    def test_flush_returns_unfinished_segment(self) -> None:
        self._feed(SILENCE_LEVEL, 1000)
        self._feed(SPEECH_LEVEL, 1000)

        segment = self.segmenter.flush()

        self.assertIsNotNone(segment)
        self.assertFalse(self.segmenter.is_in_speech)
        self.assertIsNone(self.segmenter.flush())

    def test_constant_background_noise_is_ignored(self) -> None:
        self.assertEqual(self._feed(NOISE_LEVEL, 3000), [])
        self.assertGreater(self.segmenter.threshold, NOISE_LEVEL)

    def test_speech_over_background_noise_is_detected(self) -> None:
        self._feed(NOISE_LEVEL, 3000)
        self._feed(SPEECH_LEVEL, 1000)

        segments = self._feed(NOISE_LEVEL, 1000)

        self.assertEqual(len(segments), 1)
