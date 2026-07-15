"""
logging_config.py

Shared logger factory. Every agent/module calls get_logger(__name__)
so all logs share the same format and land in ./logs/agentic_rag.log
(and also print to console for prototype convenience).
"""

import logging
import os
from config import LOG_FOLDER

os.makedirs(LOG_FOLDER, exist_ok=True)

_LOG_FILE = os.path.join(LOG_FOLDER, "agentic_rag.log")

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (avoid duplicate handlers on re-import)
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(_FORMAT)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
