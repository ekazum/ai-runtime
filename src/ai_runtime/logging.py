"""Shared logging setup for every entry point.

``setup_logging`` configures the root logger once, at the start of an entry
point; modules only call ``logging.getLogger(__name__)`` and never add
handlers. Output goes to stdout so a batch scheduler routes log lines to the
job's output file rather than its error file.
"""

import logging
import os
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def setup_logging(level: Optional[int] = None) -> None:
    """Configure the root logger to write ``level`` and above to stdout.

    ``level=None`` reads the ``LOG_LEVEL`` env var (default ``INFO``); set
    ``LOG_LEVEL=DEBUG`` to surface debug-level diagnostics without a code
    change.
    """
    if level is None:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stdout)
