import logging
from pathlib import Path

_LOGGER_CREATED = False

def setup_logger(log_file: Path | None = None) -> logging.Logger:
    global _LOGGER_CREATED

    logger = logging.getLogger("hotel_booking_ml")
    if _LOGGER_CREATED:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGER_CREATED = True
    return logger
