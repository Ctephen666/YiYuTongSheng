from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str = "yiyu_tongsheng", log_file: Path | None = None) -> logging.Logger:
    """Create a console logger used by app.py and pipeline stages.

    Input:
        name: Logger name.
        log_file: Optional log file path.
    Output:
        Configured logging.Logger.
    TODO:
        Add per-run log directories and structured JSON logs.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
