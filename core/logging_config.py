"""Basic logging configuration shared by the whole project."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger.

    force=True replaces existing handlers, so calling this repeatedly is safe.
    That matters because Streamlit re-runs the whole script on every interaction,
    and without force=True each re-run would add a duplicate handler (double log lines).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )