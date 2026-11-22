# Speech to Text

A small desktop tool for dictation: press **Record**, speak, and the text appears in the window after every short pause. Speech recognition runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the CPU, so it works offline and no audio leaves the machine.

## Features

- Live transcription: each pause of about 0.7 s completes a segment, which is transcribed right away
- Models from `tiny` to `large-v3-turbo`, selectable in the window
- Languages: German, English or automatic detection
- Level meter to check the microphone
- Editable text, **Copy all** and **Clear** buttons
- Optional: copy each dictation to the clipboard when the recording stops
- Keyboard shortcut `Ctrl+R` to start and stop
- Model, language, options and window size are remembered

## Requirements

- Linux with PipeWire or PulseAudio (tested on KDE Plasma, Wayland)
- Python 3.12 or newer (tested with 3.14)
- PortAudio (`pacman -S portaudio`)
- A microphone, set as the default input device in the system settings

## Setup

### 1. Create a virtual environment

If PyQt6 and NumPy are installed through the system package manager (`python-pyqt6`, `python-numpy`), reuse them:

```bash
python3 -m venv --system-site-packages .venv
```

Otherwise create a plain environment and let pip install PyQt6 as well:

```bash
python3 -m venv .venv
```

### 2. Install the dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

### 3. Start the application

```bash
.venv/bin/python main.py
```

On first use, the selected model is downloaded from Hugging Face into `~/.cache/huggingface` (see the table below). Later starts load it from there in a few seconds.

## Usage

1. Choose model and language.
2. Click **Record** or press `Ctrl+R` and start speaking.
3. After each short pause, the spoken text is appended. Every recording starts on a new line.
4. Click **Stop**. The last words are still transcribed before the button is available again.

## Models

| Model | Download | Speed | Accuracy |
|---|---|---|---|
| `tiny` | ~75 MB | fastest | low |
| `base` | ~145 MB | very fast | moderate |
| `small` | ~480 MB | fast (default) | good |
| `medium` | ~1.5 GB | slow | very good |
| `large-v3-turbo` | ~1.6 GB | moderate | best |

With `small`, a sentence of a few seconds takes about half a second on a modern 12-core CPU.

## How it works

```
main.py                         -> entry point
speech_to_text/gui.py           -> MainWindow (PyQt6): controls, level meter, text
speech_to_text/dictation.py     -> DictationEngine: background threads, Qt signals
speech_to_text/recorder.py      -> MicrophoneRecorder: 16 kHz mono blocks via sounddevice
speech_to_text/segmenter.py     -> SpeechSegmenter: splits the audio stream at pauses
speech_to_text/transcriber.py   -> Transcriber: faster-whisper model (CPU, int8)
speech_to_text/constants.py     -> all tunable values
```

The segmenter compares the loudness of each 30 ms block with the background noise, estimated from a low percentile of the last ten seconds. Whisper's own voice activity filter and confidence values then remove the text it would otherwise invent for silence.

## Configuration

All tunable values live in `speech_to_text/constants.py`. The most useful ones:

| Constant | Default | Effect |
|---|---|---|
| `SILENCE_MS` | `700` | Pause that ends a segment. Higher values give longer, more coherent segments but more delay. |
| `THRESHOLD_FACTOR` | `3.0` | How much louder than the background speech must be. Lower it if quiet speech is missed. |
| `MIN_THRESHOLD` | `0.006` | Lowest level that counts as speech. |
| `DEFAULT_MODEL` | `small` | Model selected on the very first start. |
| `CPU_THREADS` | half of the cores | Threads used for transcription. |

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
.venv/bin/ruff format .
.venv/bin/ruff check .
```

## Troubleshooting

- **The level meter does not move:** check the default input device in the system settings (KDE: *System Settings > Sound*) and whether the microphone is muted on the device itself.
- **"Cannot open microphone":** no input device is available, or PortAudio is missing.
- **Wrong language or odd words:** select the language explicitly instead of *Auto-detect*, or try a larger model.
