"""``ServiceEndpoint`` — where a running service says it is.

The service writes one JSON file (host, port, started_at, plus any extra
fields) on a shared filesystem; clients on any machine read it to dial.
A missing file reads as "no service".
"""

import json
import logging
import socket
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ServiceEndpoint:
    """Writes, reads and clears the service's endpoint file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def write(self, port: int, extra: Optional[dict[str, Any]] = None) -> None:
        """Publish this host and ``port``; the host is the node's FQDN."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **(extra or {}),
            "host": socket.getfqdn(),
            "port": int(port),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("ServiceEndpoint: %s:%d published to %s.",
                    payload["host"], payload["port"], self._path)

    def read(self) -> Optional[dict[str, Any]]:
        """The published endpoint, or ``None`` when there is no service."""
        try:
            with open(self._path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or "host" not in payload:
            return None
        return payload

    def clear(self) -> None:
        """Remove the file, so consumers stop dialling a gone service."""
        self._path.unlink(missing_ok=True)
        logger.info("ServiceEndpoint: %s cleared.", self._path)
