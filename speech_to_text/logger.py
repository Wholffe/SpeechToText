import logging


def setup_logger() -> logging.Logger:
    """Configures and returns the console logger of the application."""
    logger = logging.getLogger("SpeechToText")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()
