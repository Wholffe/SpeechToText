import os
from typing import Final

# region Info
NAME: Final = "Speech to Text"
VERSION: Final = "0.1.0"
SETTINGS_KEY: Final = "SpeechToText"  # QSettings organization and application name

# region Audio
SAMPLE_RATE: Final = 16_000  # Whisper expects 16 kHz mono audio
BLOCK_MS: Final = 30  # length of one microphone block
BLOCK_SIZE: Final = SAMPLE_RATE * BLOCK_MS // 1000

# region Segmentation
SILENCE_MS: Final = 700  # pause that ends a segment and triggers its transcription
PRE_ROLL_MS: Final = 300  # audio kept before speech starts, so no syllable is cut off
MIN_SPEECH_MS: Final = 250  # shorter loud bursts (clicks, coughs) are discarded
MAX_SEGMENT_S: Final = 25  # Whisper processes at most 30 s at once
NOISE_WINDOW_S: Final = 10  # recent audio used to estimate the background noise
NOISE_PERCENTILE: Final = 10  # pauses between words keep this percentile near the noise
THRESHOLD_FACTOR: Final = 3.0  # speech must be this much louder than the noise floor
MIN_THRESHOLD: Final = 0.006  # lowest RMS that counts as speech

# region Transcription
MODELS: Final = ("tiny", "base", "small", "medium", "large-v3-turbo")
DEFAULT_MODEL: Final = "small"
LANGUAGES: Final[dict[str, str | None]] = {
    "German": "de",
    "English": "en",
    "Auto-detect": None,
}
DEFAULT_LANGUAGE: Final = "German"
COMPUTE_TYPE: Final = "int8"
CPU_THREADS: Final = max(1, (os.cpu_count() or 2) // 2)
BEAM_SIZE: Final = 5
# Whisper invents text for silence; segments beyond both limits are dropped
NO_SPEECH_PROB_MAX: Final = 0.6
AVG_LOGPROB_MIN: Final = -1.0

# region GUI
LEVEL_UPDATE_MS: Final = 50
LEVEL_FLOOR_DB: Final = -60.0  # level meter shows the range from here to 0 dB
RECORD_SHORTCUT: Final = "Ctrl+R"
