import math
import sys

import sounddevice as sd
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtGui import QCloseEvent, QGuiApplication, QIcon, QKeySequence, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from speech_to_text import constants as CONST

from .dictation import DictationEngine
from .logger import logger


def level_to_percent(level: float) -> int:
    """Maps an RMS level onto the 0–100 range of the level meter (logarithmic)."""
    decibels = 20 * math.log10(max(level, 1e-9))
    ratio = (decibels - CONST.LEVEL_FLOOR_DB) / -CONST.LEVEL_FLOOR_DB
    return round(100 * min(max(ratio, 0.0), 1.0))


class MainWindow(QMainWindow):
    """Main window: record button, level meter and the transcribed text."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{CONST.NAME} {CONST.VERSION}")
        self.setWindowIcon(QIcon.fromTheme("audio-input-microphone"))
        self.resize(640, 420)

        self._settings = QSettings(CONST.SETTINGS_KEY, CONST.SETTINGS_KEY)
        self._engine = DictationEngine(self)
        self._is_model_ready = False
        self._is_recording = False
        self._is_finishing = False
        self._dictation_texts: list[str] = []  # text of the current recording

        self._build_ui()
        self._restore_settings()
        self._connect_signals()

        self._level_timer = QTimer(self)
        self._level_timer.setInterval(CONST.LEVEL_UPDATE_MS)
        self._level_timer.timeout.connect(self._update_level)

        self._engine.language = CONST.LANGUAGES[self.language_box.currentText()]
        self._load_model(self.model_box.currentText())

    # region UI
    def _build_ui(self) -> None:
        """Creates the widgets and layouts."""
        self.model_box = QComboBox()
        self.model_box.addItems(CONST.MODELS)
        self.model_box.setToolTip("Larger models are more accurate but slower")
        self.language_box = QComboBox()
        self.language_box.addItems(CONST.LANGUAGES)

        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Model:"))
        settings_row.addWidget(self.model_box)
        settings_row.addSpacing(16)
        settings_row.addWidget(QLabel("Language:"))
        settings_row.addWidget(self.language_box)
        settings_row.addStretch()

        self.record_button = QPushButton()
        self.record_button.setMinimumHeight(44)
        self.record_button.setShortcut(QKeySequence(CONST.RECORD_SHORTCUT))
        self.record_button.setToolTip(f"Start or stop ({CONST.RECORD_SHORTCUT})")

        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setTextVisible(False)
        self.level_meter.setMaximumHeight(10)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "Press Record and start speaking. Text appears after each short pause."
        )

        self.auto_copy_box = QCheckBox("Copy each dictation to clipboard")
        self.copy_button = QPushButton(QIcon.fromTheme("edit-copy"), "Copy all")
        self.clear_button = QPushButton(QIcon.fromTheme("edit-clear"), "Clear")

        actions_row = QHBoxLayout()
        actions_row.addWidget(self.auto_copy_box)
        actions_row.addStretch()
        actions_row.addWidget(self.copy_button)
        actions_row.addWidget(self.clear_button)

        layout = QVBoxLayout()
        layout.addLayout(settings_row)
        layout.addWidget(self.record_button)
        layout.addWidget(self.level_meter)
        layout.addWidget(self.text_edit)
        layout.addLayout(actions_row)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        """Wires widgets and engine signals to their handlers."""
        self.record_button.clicked.connect(self._toggle_recording)
        self.model_box.currentTextChanged.connect(self._load_model)
        self.language_box.currentTextChanged.connect(self._change_language)
        self.copy_button.clicked.connect(self._copy_all)
        self.clear_button.clicked.connect(self.text_edit.clear)

        self._engine.model_loaded.connect(self._on_model_loaded)
        self._engine.model_failed.connect(self._on_model_failed)
        self._engine.text_ready.connect(self._on_text_ready)
        self._engine.transcribing_changed.connect(self._on_transcribing_changed)
        self._engine.transcription_failed.connect(self._on_transcription_failed)
        self._engine.finished.connect(self._on_finished)

    def _update_controls(self) -> None:
        """Enables and labels the controls according to the current state."""
        is_idle = not self._is_recording and not self._is_finishing
        self.model_box.setEnabled(is_idle)
        self.language_box.setEnabled(is_idle)
        self.record_button.setEnabled(self._is_model_ready and not self._is_finishing)

        if self._is_recording:
            self.record_button.setText("Stop")
            self.record_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        elif self._is_finishing:
            self.record_button.setText("Finishing…")
        else:
            self.record_button.setText("Record")
            self.record_button.setIcon(QIcon.fromTheme("media-record"))

    def _show_status(self, message: str) -> None:
        """Shows a message in the status bar until the next one replaces it."""
        self.statusBar().showMessage(message)

    # region Settings
    def _restore_settings(self) -> None:
        """Applies the settings saved by the last session."""
        model = self._settings.value("model", CONST.DEFAULT_MODEL)
        if model in CONST.MODELS:
            self.model_box.setCurrentText(model)
        language = self._settings.value("language", CONST.DEFAULT_LANGUAGE)
        if language in CONST.LANGUAGES:
            self.language_box.setCurrentText(language)
        self.auto_copy_box.setChecked(
            self._settings.value("auto_copy", False, type=bool)
        )
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _save_settings(self) -> None:
        """Saves model, language, options and window geometry."""
        self._settings.setValue("model", self.model_box.currentText())
        self._settings.setValue("language", self.language_box.currentText())
        self._settings.setValue("auto_copy", self.auto_copy_box.isChecked())
        self._settings.setValue("geometry", self.saveGeometry())

    # region Actions
    def _load_model(self, model_name: str) -> None:
        """Starts loading a model; recording stays disabled until it is ready."""
        self._is_model_ready = False
        self._update_controls()
        self.model_box.setEnabled(False)
        self._show_status(
            f"Loading model '{model_name}' (downloaded once on first use)…"
        )
        self._engine.load_model(model_name)

    def _change_language(self, language: str) -> None:
        """Sets the language used for the next segments."""
        self._engine.language = CONST.LANGUAGES[language]

    def _toggle_recording(self) -> None:
        """Starts or stops the recording."""
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Opens the microphone and starts live transcription."""
        try:
            self._engine.start()
        except sd.PortAudioError as error:
            logger.error(f"Failed to open microphone: {error}")
            QMessageBox.warning(self, CONST.NAME, f"Cannot open microphone:\n{error}")
            return
        self._is_recording = True
        self._dictation_texts = []
        self._level_timer.start()
        self._update_controls()
        self._show_status("Listening…")

    def _stop_recording(self) -> None:
        """Stops the microphone; the last words are still transcribed."""
        self._engine.stop()
        self._is_recording = False
        self._is_finishing = True
        self._level_timer.stop()
        self.level_meter.setValue(0)
        self._update_controls()
        self._show_status("Transcribing the last words…")

    def _copy_all(self) -> None:
        """Copies the whole text to the clipboard."""
        QGuiApplication.clipboard().setText(self.text_edit.toPlainText())
        self._show_status("Copied all text to the clipboard")

    def _append_text(self, text: str) -> None:
        """Appends text at the end; each recording starts on a new line."""
        existing = self.text_edit.toPlainText()
        separator = ""
        if existing and not existing[-1].isspace():
            separator = " " if self._dictation_texts else "\n"

        cursor = QTextCursor(self.text_edit.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(separator + text)

        scroll_bar = self.text_edit.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _update_level(self) -> None:
        """Refreshes the level meter from the latest microphone block."""
        self.level_meter.setValue(level_to_percent(self._engine.level))

    # region Engine signals
    def _on_model_loaded(self, model_name: str) -> None:
        """Enables recording once the selected model is ready."""
        if model_name != self.model_box.currentText():
            return  # another model was selected while this one was loading
        self._is_model_ready = True
        self._update_controls()
        self._show_status(f"Ready (model '{model_name}')")

    def _on_model_failed(self, model_name: str, error: str) -> None:
        """Reports a model that could not be loaded."""
        if model_name != self.model_box.currentText():
            return
        self._update_controls()
        self._show_status(f"Model '{model_name}' could not be loaded")
        QMessageBox.warning(
            self, CONST.NAME, f"Cannot load model '{model_name}':\n{error}"
        )

    def _on_text_ready(self, text: str) -> None:
        """Shows a transcribed segment."""
        self._append_text(text)
        self._dictation_texts.append(text)

    def _on_transcribing_changed(self, is_transcribing: bool) -> None:
        """Shows whether a segment is being transcribed."""
        if is_transcribing:
            self._show_status("Transcribing…")
        elif self._is_recording:
            self._show_status("Listening…")

    def _on_transcription_failed(self, error: str) -> None:
        """Reports a segment that could not be transcribed."""
        self._show_status(f"Transcription failed: {error}")

    def _on_finished(self) -> None:
        """Completes a recording once its last segment is transcribed."""
        self._is_finishing = False
        self._update_controls()
        if self.auto_copy_box.isChecked() and self._dictation_texts:
            QGuiApplication.clipboard().setText(" ".join(self._dictation_texts))
            self._show_status("Done, dictation copied to the clipboard")
        else:
            self._show_status("Done")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stops a running recording and saves the settings."""
        if self._is_recording:
            self._engine.stop()
        self._save_settings()
        super().closeEvent(event)


def run() -> int:
    """Starts the application and returns its exit code."""
    app = QApplication(sys.argv)
    app.setApplicationName(CONST.NAME)
    window = MainWindow()
    window.show()
    return app.exec()
