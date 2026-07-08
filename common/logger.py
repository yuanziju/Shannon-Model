"""
Unified logging setup for distributed training.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "shannon",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """Set up logger with optional file output. Only logs on rank 0."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger