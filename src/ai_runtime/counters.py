"""``LockedCounter`` — an int that many threads advance.

``+=`` is a read-modify-write and is not atomic under the GIL, so increments
and reads go through a lock.
"""

import threading


class LockedCounter:
    """Locked multi-writer counter."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def add(self, n: int) -> int:
        """Advance by ``n``; return the new total.

        A caller can derive unique indices from it (``total - n + 1 .. total``).
        """
        with self._lock:
            self._value += n
            return self._value

    @property
    def value(self) -> int:
        """The current total."""
        with self._lock:
            return self._value

    def load(self, value: int) -> None:
        """Restore the counter, e.g. from a checkpoint."""
        with self._lock:
            self._value = int(value)
