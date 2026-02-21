from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure structured logging for the full runtime."""
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "root": {
                "level": level,
                "handlers": ["console"],
            },
        }
    )
